"""Progression forecasting on the weather-driven benchmark.
Observation model inputs come from the trained vision model (severity read-out, class posterior, quality probe,
conformal abstention) evaluated on real held-out PlantVillage test images (clean or degraded)."""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
import progression as P
GRID = torch.linspace(-7.5, 3.0, 106)  # logit lesion-fraction grid
H_LIST = (7, 14)

def logit(y): y = np.clip(y, 1e-3, 0.97); return np.log(y / (1 - y))

class ObsBank:
    """maps latent severity to a real held-out image and returns the vision model's outputs for it"""
    def __init__(self, vis, labels, true_sev, rng):
        self.vis = vis; self.labels = labels; self.sev = true_sev; self.rng = rng
    def draw(self, d, y, corr):
        if y < 0.01 or d == "healthy": cand = np.where(self.labels == 0)[0]
        else:
            lab = {"gls": 1, "nlb": 2, "rust": 3}[d]; cand = np.where(self.labels == lab)[0]
            cand = cand[np.argsort(np.abs(self.sev[cand] - y))[:6]]
        i = self.rng.choice(cand); v = self.vis[corr]
        return dict(sev=v["sev"][i], p=v["p"][i], degr=v["degr"][i], abst=v["abst"][i], img=i)

def make_sequences(seasons, bank, rng, p_corrupt=0.35):
    kinds = [k for k in bank.vis if k != "clean"]
    seqs = []
    for s in seasons:
        T = len(s["y"]); obs = np.zeros((T, 9), np.float32)  # flag, logit sev, degr, abst, p(4), quality weight placeholder
        for t in s["visits"]:
            corr = kinds[rng.integers(len(kinds))] if rng.random() < p_corrupt else "clean"
            o = bank.draw(s["disease"], s["y"][t], corr)
            obs[t, 0] = 1; obs[t, 1] = logit(o["sev"]); obs[t, 2] = o["degr"]; obs[t, 3] = o["abst"]; obs[t, 4:8] = o["p"]
        seqs.append(dict(obs=obs, wf=s["wf"].astype(np.float32), y=s["y"].astype(np.float32), visits=s["visits"], disease=s["disease"]))
    return seqs

def batchify(seqs):
    return (torch.tensor(np.stack([s["obs"] for s in seqs])), torch.tensor(np.stack([s["wf"] for s in seqs])),
            torch.tensor(np.stack([s["y"] for s in seqs])))

class EnvFilter(nn.Module):
    """Grid Bayes filter on logit lesion fraction (probability space, O(BK) per day).
    Daily growth = softplus(MLP(weather, disease belief)) applied as a fractional grid shift, followed by Gaussian
    diffusion; observation likelihood Gaussian in logit space with a learned quality/abstention precision weight."""
    def __init__(self, use_env=True, use_quality=True, use_disease=True):
        super().__init__(); self.use_env, self.use_quality, self.use_disease = use_env, use_quality, use_disease
        self.g = nn.Sequential(nn.Linear(3 + 3, 32), nn.Tanh(), nn.Linear(32, 1))
        self.log_sq = nn.Parameter(torch.tensor(-2.0)); self.a = nn.Parameter(torch.tensor(0.0)); self.b = nn.Parameter(torch.tensor(1.0))
        self.log_so = nn.Parameter(torch.tensor(0.0)); self.qw = nn.Linear(2, 1)
        nn.init.zeros_(self.qw.weight); nn.init.constant_(self.qw.bias, 2.0)
        self.mu0 = nn.Parameter(torch.tensor(-6.0)); self.log_s0 = nn.Parameter(torch.tensor(0.3))
        self.dx = float(GRID[1] - GRID[0]); self.K = len(GRID)
    def growth(self, wf, belief):
        x = torch.cat([wf if self.use_env else torch.zeros_like(wf), belief if self.use_disease else torch.zeros_like(belief)], -1)
        return F.softplus(self.g(x)).squeeze(-1) * 0.5
    def step(self, p, gr):
        s = gr / self.dx; fl = torch.floor(s); f = (s - fl).unsqueeze(1); ar = torch.arange(self.K)
        i0 = ar[None, :] - fl[:, None].long(); i1 = i0 - 1
        g0 = torch.gather(p, 1, i0.clamp(0, self.K - 1)) * (i0 >= 0); g1 = torch.gather(p, 1, i1.clamp(0, self.K - 1)) * (i1 >= 0)
        q = (1 - f) * g0 + f * g1
        lost = (1 - q.sum(1, keepdim=True)).clamp(min=0); q = torch.cat([q[:, :-1], q[:, -1:] + lost], 1)
        sig = torch.exp(self.log_sq) / self.dx; r = 4
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / (sig + 1e-3)) ** 2); k = k / k.sum()
        q = F.conv1d(F.pad(q[:, None, :], (r, r), mode="replicate"), k.view(1, 1, -1))[:, 0]
        return q / (q.sum(1, keepdim=True) + 1e-12)
    def forward(self, obs, wf):
        B, T, _ = obs.shape; G = GRID
        prior = torch.distributions.Normal(self.mu0, torch.exp(self.log_s0)).log_prob(G).expand(B, self.K)
        p = torch.softmax(prior, 1)
        belief = torch.full((B, 3), 1 / 3.); cnt = torch.zeros(B, 1); filt = []; beliefs = []
        for t in range(T):
            o = obs[:, t]; has = o[:, 0] > 0
            if has.any():
                pd = o[:, 5:8] / (o[:, 5:8].sum(1, keepdim=True) + 1e-6)
                belief = torch.where(has[:, None], (belief * cnt + pd) / (cnt + 1), belief); cnt = cnt + has[:, None].float()
                so = torch.exp(self.log_so)
                w = torch.sigmoid(self.qw(o[:, 2:4])).squeeze(-1) if self.use_quality else torch.ones(B)
                ll = -0.5 * w[:, None] * ((o[:, 1:2] - (self.a + self.b * G[None, :])) / so) ** 2
                lp = torch.log(p + 1e-12) + ll; lp = lp - torch.logsumexp(lp, 1, keepdim=True)
                p = torch.where(has[:, None], lp.exp(), p)
            filt.append(p); beliefs.append(belief)
            p = self.step(p, self.growth(wf[:, t], belief))
        self._beliefs = torch.stack(beliefs, 1)
        return torch.stack(filt, 1)
    def forecast(self, p_t, wf, t, h, belief):
        p = p_t
        for k in range(h):
            tt = min(t + k, wf.shape[1] - 1); p = self.step(p, self.growth(wf[:, tt], belief))
        return torch.log(p + 1e-12)

def bin_index(y):
    e = torch.log(torch.clamp(y, 1e-3, 0.97) / (1 - torch.clamp(y, 1e-3, 0.97)))
    return torch.argmin((e[..., None] - GRID) ** 2, -1)

def train_filter(model, seqs, epochs=40, bs=64, lr=0.03, seed=0):
    torch.manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs, wf, y = batchify(seqs); n = len(seqs)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            i = perm[s:s + bs]; filt = model(obs[i], wf[i])
            # multi-step predictive loss at visit days: predicted distribution of y(t+h) from filtered state at t
            loss = 0; cnt = 0
            for h in H_LIST:
                # cheap approximation during training: roll filtered distributions forward using cumulative growth
                T = obs.shape[1]
                for t0 in range(10, T - h, 14):
                    lp = model.forecast(filt[:, t0], wf[i], t0, h, model._beliefs[:, t0])
                    tgt = bin_index(y[i][:, t0 + h])
                    loss = loss - lp.gather(1, tgt[:, None]).mean(); cnt += 1
            loss = loss / cnt
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model

class GRUForecaster(nn.Module):
    def __init__(self, use_env=True, hid=64):
        super().__init__(); self.use_env = use_env
        self.gru = nn.GRU(9 + 3, hid, batch_first=True); self.head = nn.Linear(hid + 3 * 14 * 0, 2 * len(H_LIST))
    def forward(self, obs, wf):
        x = torch.cat([obs, wf if self.use_env else torch.zeros_like(wf)], -1)
        h, _ = self.gru(x); out = self.head(h)
        return out.view(*out.shape[:2], len(H_LIST), 2)  # mean, log-sd of logit y(t+h)

def train_gru(model, seqs, epochs=60, bs=64, lr=3e-3, seed=0):
    torch.manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs, wf, y = batchify(seqs); n = len(seqs); T = obs.shape[1]
    ly = torch.log(torch.clamp(y, 1e-3, 0.97) / (1 - torch.clamp(y, 1e-3, 0.97)))
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            i = perm[s:s + bs]; out = model(obs[i], wf[i]); loss = 0
            for k, h in enumerate(H_LIST):
                tgt = torch.cat([ly[i][:, h:], ly[i][:, -1:].expand(-1, h)], 1)
                m, ls = out[..., k, 0], out[..., k, 1].clamp(-4, 3)
                loss = loss + (0.5 * ((tgt - m) / torch.exp(ls)) ** 2 + ls)[:, 5:T - h].mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model
