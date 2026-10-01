"""MaizeGuard heads on frozen CLIP ViT-B/32 features.
Components: lesion-aware token typing (L), symptom-grounded descriptor alignment (S), quality-gated fusion (Q),
severity read-out, symptom-evidence profile (for evidence-consistent conformal decisions)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
TILE = slice(49, 245)  # tokens of the four 448-tiles (finer grid used for severity)

def nrm(x, d=-1): return F.normalize(x, dim=d)

class MaizeGuard(nn.Module):
    def __init__(self, text, use_lesion=True, use_desc=True, use_quality=True, n_cls=4, adapter=False):
        super().__init__()
        self.use_lesion, self.use_desc, self.use_quality = use_lesion, use_desc, use_quality
        self.register_buffer("P", torch.tensor(text["tok"], dtype=torch.float32))            # 3x512 token-type probes
        self.register_buffer("Qp", torch.tensor(text["qual"], dtype=torch.float32))          # 6x512 quality probes
        desc = torch.tensor(text["desc"], dtype=torch.float32)                                # K,M,512
        if not use_desc:  # class-name prompt only (no symptom descriptors)
            desc = torch.tensor(text["cname"], dtype=torch.float32)[:, None, :]
        self.register_buffer("D0", desc)
        K, M, d = desc.shape
        self.delta = nn.Parameter(torch.zeros(K, M, d))                                      # descriptor offsets
        self.rho = nn.Parameter(torch.zeros(K, M))                                           # descriptor relevance
        self.tok_lin = nn.Linear(d, 3); nn.init.zeros_(self.tok_lin.weight); nn.init.zeros_(self.tok_lin.bias)
        self.log_tau_tok = nn.Parameter(torch.tensor(np.log(30.0), dtype=torch.float32))
        self.gate = nn.Linear(6 + 1, 1); nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, 0.0)
        self.adapter = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(), nn.Linear(256, d)) if adapter else None
        if adapter: nn.init.zeros_(self.adapter[2].weight); nn.init.zeros_(self.adapter[2].bias)
        self.log_tau = nn.Parameter(torch.tensor(np.log(50.0), dtype=torch.float32))
        self.register_buffer("log_prior", torch.zeros(n_cls))

    def descriptors(self):
        return nrm(self.D0 + self.delta)

    def forward(self, G, D):
        """G: (B,5,512) view embeddings, D: (B,245,512) dense tokens (5 views x 49)."""
        rg = nrm(nrm(G).mean(1))
        t = nrm(D.float())
        u = torch.exp(self.log_tau_tok) * (t @ self.P.T) + self.tok_lin(t)                   # B,T,3
        pi = u.softmax(-1); les, tis = pi[..., 0], pi[..., 1]
        sev = les[:, TILE].sum(1) / (les[:, TILE].sum(1) + tis[:, TILE].sum(1) + 1e-6)
        mass = les.mean(1, keepdim=True)
        rl = nrm((les.unsqueeze(-1) * t).sum(1) / (les.sum(1, keepdim=True) + 1e-6))
        q = (100 * nrm(G[:, 0]) @ self.Qp.T).softmax(-1)                                       # language quality probe
        if self.use_lesion:
            gin = torch.cat([q if self.use_quality else torch.zeros_like(q), mass], 1)
            g = torch.sigmoid(self.gate(gin) + (0.0 if self.use_quality else 0.0))
            z = (1 - g) * rg + g * rl
            if self.adapter is not None: z = z + self.adapter(torch.cat([rg, rl], 1))
        else:
            g = torch.zeros(G.shape[0], 1, device=G.device); z = rg
            if self.adapter is not None: z = z + self.adapter(torch.cat([rg, rg], 1))
        z = nrm(z)
        Dk = self.descriptors()                                                               # K,M,d
        w = self.rho.softmax(-1)                                                              # K,M
        sims = torch.einsum("bd,kmd->bkm", z, Dk)
        logits = torch.exp(self.log_tau) * (w.unsqueeze(0) * sims).sum(-1)
        # symptom-evidence profile: lesion-weighted token support for every descriptor + fused-embedding support
        tok_sims = torch.einsum("btd,nd->btn", t, Dk.reshape(-1, Dk.shape[-1]))               # B,T,KM
        ev_les = (les.unsqueeze(-1) * tok_sims).sum(1) / (les.sum(1, keepdim=True) + 1e-6)
        ev = torch.cat([ev_les, sims.reshape(sims.shape[0], -1)], 1)
        return dict(logits=logits, sev=sev, pi=pi, z=z, rg=rg, rl=rl, q=q, gate=g, mass=mass.squeeze(1), ev=ev)

class LinearProbe(nn.Module):
    def __init__(self, d=512, K=4, init=None):
        super().__init__(); self.fc = nn.Linear(d, K)
        if init is not None:
            with torch.no_grad(): self.fc.weight.copy_(torch.tensor(init) * 50); self.fc.bias.zero_()
    def forward(self, G, D):
        rg = nrm(nrm(G).mean(1)); return dict(logits=self.fc(rg), rg=rg, z=rg)

class ClipAdapter(nn.Module):
    """CLIP-Adapter: residual MLP on the image embedding, zero-shot text classifier."""
    def __init__(self, zs, ratio=0.2):
        super().__init__(); d = zs.shape[1]
        self.register_buffer("W", torch.tensor(zs, dtype=torch.float32)); self.ratio = ratio
        self.mlp = nn.Sequential(nn.Linear(d, d // 4, bias=False), nn.ReLU(), nn.Linear(d // 4, d, bias=False), nn.ReLU())
    def forward(self, G, D):
        rg = nrm(nrm(G).mean(1)); x = nrm(self.ratio * self.mlp(rg) + (1 - self.ratio) * rg)
        return dict(logits=100 * x @ self.W.T, rg=rg, z=x)

class TipAdapterF(nn.Module):
    """Tip-Adapter-F: cache model of training embeddings (keys fine-tuned) + zero-shot classifier."""
    def __init__(self, zs, keys, onehot, alpha=1.0, beta=5.5):
        super().__init__()
        self.register_buffer("W", torch.tensor(zs, dtype=torch.float32))
        self.keys = nn.Parameter(torch.tensor(keys, dtype=torch.float32))
        self.register_buffer("V", torch.tensor(onehot, dtype=torch.float32)); self.alpha, self.beta = alpha, beta
    def forward(self, G, D):
        rg = nrm(nrm(G).mean(1)); A = torch.exp(-self.beta * (1 - rg @ nrm(self.keys).T))
        return dict(logits=100 * rg @ self.W.T + self.alpha * 100 * (A @ self.V) / 10, rg=rg, z=rg)

class ABMIL(nn.Module):
    """Gated attention MIL over dense tokens (no language grounding) + global embedding."""
    def __init__(self, d=512, K=4, h=128):
        super().__init__(); self.V = nn.Linear(d, h); self.U = nn.Linear(d, h); self.w = nn.Linear(h, 1)
        self.fc = nn.Linear(2 * d, K)
    def forward(self, G, D):
        t = nrm(D.float()); a = self.w(torch.tanh(self.V(t)) * torch.sigmoid(self.U(t))).squeeze(-1).softmax(1)
        r = nrm((a.unsqueeze(-1) * t).sum(1)); rg = nrm(nrm(G).mean(1))
        return dict(logits=self.fc(torch.cat([rg, r], 1)), rg=rg, z=nrm(torch.cat([rg, r], 1)), att=a)
