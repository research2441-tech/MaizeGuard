"""Shared evaluation logic: temperature scaling, open-set scores, conformal sets (baselines and the dual test)."""
import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.covariance import LedoitWolf
from metrics import fit_temperature, pvalues, conformal_quantile

def load_pred(path, prefix=None):
    z = np.load(path)
    if prefix is None: return {k: z[k] for k in z.files}
    return {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(prefix + "__")}

class Calibrated:
    """holds validation/calibration material for one trained model"""
    def __init__(self, val, cal, cal_aug, y_val, y_cal, q_val=None, q_cal=None, q_cal_aug=None):
        self.T = fit_temperature(val["logits"], y_val)
        self.val, self.y_val = val, y_val
        # calibration pool = clean calibration images + their degraded copies
        self.cal = {k: np.concatenate([cal[k], cal_aug[k]]) for k in cal if k in cal_aug}
        self.y_cal = np.concatenate([y_cal, y_cal])
        self.psi_cal = None if q_cal is None else 1 - np.concatenate([q_cal, q_cal_aug])[:, 5]
        self.has_ev = "ev" in val
        if self.has_ev:
            self.gauss = {}
            for k in range(4):
                E = val["ev"][y_val == k]; lw = LedoitWolf().fit(E)
                self.gauss[k] = (lw.location_, np.linalg.inv(lw.covariance_ + 1e-4 * np.eye(E.shape[1])))
        # feature-space detectors fitted on validation embeddings
        self.zval = val["z"] / np.linalg.norm(val["z"], axis=1, keepdims=True)
        self.class_means = np.stack([self.zval[y_val == k].mean(0) for k in range(4)])
        cov = sum(np.cov((self.zval[y_val == k] - self.class_means[k]).T) * (y_val == k).sum() for k in range(4)) / len(y_val)
        self.prec = np.linalg.inv(cov + 1e-3 * np.eye(cov.shape[0]))
        self.strata = None
        if self.psi_cal is not None:
            self.strata = np.quantile(self.psi_cal, [1 / 3, 2 / 3])
    def probs(self, logits): return softmax(logits / self.T, axis=1)
    def ev_score(self, ev):
        out = np.zeros((len(ev), 4))
        for k, (m, P) in self.gauss.items():
            d = ev - m; out[:, k] = np.einsum("ij,jk,ik->i", d, P, d)
        return out
    def stratum(self, psi):
        return np.digitize(psi, self.strata) if self.strata is not None else np.zeros(len(psi), int)

def open_scores(C, pred, zs_logits=None):
    """larger = more likely unknown"""
    lg = pred["logits"]; p = C.probs(lg); z = pred["z"] / np.linalg.norm(pred["z"], axis=1, keepdims=True)
    s = dict(msp=1 - p.max(1), maxlogit=-lg.max(1), energy=-C.T * logsumexp(lg / C.T, axis=1))
    d = z[:, None, :] - C.class_means[None]; s["mahalanobis"] = np.einsum("nkd,de,nke->nk", d, C.prec, d).min(1)
    sims = z @ C.zval.T; s["knn"] = -np.sort(sims, 1)[:, -10]
    if zs_logits is not None: s["mcm"] = -softmax(zs_logits / 100.0, axis=1).max(1)
    return s

class Conformal:
    """label test (Mondrian over quality strata) + class-conditional evidence test"""
    def __init__(self, C, cal=None, y_cal=None, psi_cal=None, mondrian=True):
        cal = C.cal if cal is None else cal; y_cal = C.y_cal if y_cal is None else y_cal
        psi_cal = C.psi_cal if psi_cal is None else psi_cal
        self.C = C; self.mondrian = mondrian and psi_cal is not None
        p = C.probs(cal["logits"]); self.alab = 1 - p[np.arange(len(y_cal)), y_cal]
        self.st_cal = C.stratum(psi_cal) if self.mondrian else np.zeros(len(y_cal), int)
        self.y_cal = y_cal
        if C.has_ev and "ev" in cal:
            E = C.ev_score(cal["ev"]); self.aev = E[np.arange(len(y_cal)), y_cal]
        # lesion-mass threshold for unknown vs re-image
        self.mu0 = np.quantile(cal["mass"][y_cal > 0], 0.05) if "mass" in cal else 0.0
        # APS / RAPS scores
        self.p_cal = p
    def p_lab(self, pred, psi):
        A = 1 - self.C.probs(pred["logits"]); out = np.zeros_like(A)
        st = self.C.stratum(psi) if self.mondrian else np.zeros(len(A), int)
        for b in np.unique(st):
            m = st == b; ref = self.alab[self.st_cal == b]
            if len(ref) < 20: ref = self.alab
            out[m] = pvalues(ref, A[m])
        return out
    def p_ev(self, pred):
        E = self.C.ev_score(pred["ev"]); out = np.zeros_like(E)
        for k in range(4):
            ref = self.aev[self.y_cal == k]
            out[:, k] = pvalues(ref, E[:, k]) if len(ref) else 1.0
        return out
    def dual_sets(self, pred, psi, alpha, beta):
        pl = self.p_lab(pred, psi); pe = self.p_ev(pred)
        S = (pl > alpha) & (pe > beta)
        return S, pl, pe
    def lac_sets(self, pred, alpha, classwise=False):
        A = 1 - self.C.probs(pred["logits"])
        if not classwise:
            q = conformal_quantile(self.alab, alpha); return A <= q
        S = np.zeros_like(A, bool)
        for k in range(4):
            ref = self.alab[self.y_cal == k]
            S[:, k] = A[:, k] <= (conformal_quantile(ref, alpha) if len(ref) else np.inf)
        return S
    def aps_sets(self, pred, alpha, lam=0.0, kreg=0, seed=0):
        """APS (lam=0) and RAPS (lam>0) with randomisation"""
        rng = np.random.default_rng(seed)
        def all_scores(p):
            o = np.argsort(-p, 1); ps = np.take_along_axis(p, o, 1); cs = np.cumsum(ps, 1)
            u = rng.random((len(p), 1)); pen = lam * np.maximum(0, np.arange(1, p.shape[1] + 1) - kreg)
            sc_sorted = cs - u * ps + pen[None]
            sc = np.empty_like(sc_sorted); np.put_along_axis(sc, o, sc_sorted, 1); return sc
        cal = all_scores(self.p_cal)[np.arange(len(self.y_cal)), self.y_cal]
        q = conformal_quantile(cal, alpha)
        return all_scores(self.C.probs(pred["logits"])) <= q

def set_stats(S, y=None):
    size = S.sum(1); out = dict(size=size.mean(), single=(size == 1).mean(), empty=(size == 0).mean())
    if y is not None:
        out["cov"] = S[np.arange(len(y)), y].mean()
        sg = size == 1
        out["single_acc"] = (S[sg].argmax(1) == y[sg]).mean() if sg.any() else np.nan
    return out
