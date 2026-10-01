import numpy as np, torch, csv, os
from splits import load_index, seed_split
class Store:
    def __init__(self):
        self.rows = load_index()
        self.G = torch.from_numpy(np.load("feats/pvpd_clean_G.npy"))
        self.D = np.load("feats/pvpd_clean_D.npy", mmap_mode="r")
        self.y = torch.tensor([["healthy", "gls", "nlb", "rust"].index(r["label"]) for r in self.rows])
        self.ds = np.array([r["ds"] for r in self.rows])
        tt = np.load("feats/pv_token_targets.npz")
        leaf = tt["leaf"].reshape(-1, 245); les = tt["lesion"].reshape(-1, 245)
        self.tok_t = torch.tensor(np.stack([les, np.clip(leaf - les, 0, 1), 1 - leaf], -1), dtype=torch.float32)
        self.sev = torch.tensor(tt["sev"], dtype=torch.float32)                      # NaN for PlantDoc
        self.has_tok = torch.tensor(~np.isnan(tt["sev"]))
        self.Ga = None
        if os.path.exists("feats/nontest_aug.npz"):
            nt = np.load("feats/nontest_idx.npy")
            self.aug_pos = -np.ones(len(self.rows), int); self.aug_pos[nt] = np.arange(len(nt))
            self.Ga = torch.from_numpy(np.load("feats/nontest_aug_G.npy")); self.Da = np.load("feats/nontest_aug_D.npy", mmap_mode="r")
    def batch(self, idx, corrupt_mask=None, need_D=True):
        idx_np = np.asarray(idx); G = self.G[idx_np].clone()
        D = torch.from_numpy(np.ascontiguousarray(self.D[np.sort(idx_np)][np.argsort(np.argsort(idx_np))])) if need_D else None
        if corrupt_mask is not None and self.Ga is not None and corrupt_mask.any():
            cm = np.asarray(corrupt_mask); p = self.aug_pos[idx_np[cm]]
            G[torch.tensor(cm)] = self.Ga[p]
            if need_D:
                o = np.argsort(p); Dp = np.asarray(self.Da[p[o]])[np.argsort(o)]
                D[torch.tensor(cm)] = torch.from_numpy(Dp)
        return G.float(), D
