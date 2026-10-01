"""Dual conformal decision used in the paper (final design).
Label test: softmax nonconformity, Mondrian over quality strata (tertiles of psi on the calibration pool).
Evidence test: class-conditional Mahalanobis distance of the lesion-pooled embedding r_l (or any feature),
Gaussians fitted on a fitting split disjoint from the calibration split."""
import numpy as np
from sklearn.covariance import LedoitWolf
from metrics import pvalues, conformal_quantile

def fit_gauss(F, y, classes):
    G = {}
    for k in classes:
        E = F[y == k]
        if len(E) < 5: continue
        lw = LedoitWolf(store_precision=False).fit(E); G[k] = (lw.location_, np.linalg.inv(lw.covariance_ + 1e-4 * np.eye(E.shape[1])))
    return G

def maha(G, F, K=4):
    out = np.full((len(F), K), np.inf)
    for k, (m, P) in G.items():
        d = F - m; out[:, k] = np.einsum("ij,jk,ik->i", d, P, d)
    return out

class Dual:
    def __init__(self, P_cal, y_cal, psi_cal, E_cal, strata=True, fallback=None):
        """P_cal: calibrated softmax (n,K); E_cal: evidence distances (n,K) to every class Gaussian"""
        self.alab = 1 - P_cal[np.arange(len(y_cal)), y_cal]; self.y = y_cal
        self.cut = np.quantile(psi_cal, [1 / 3, 2 / 3]) if strata else None
        self.st = np.digitize(psi_cal, self.cut) if strata else np.zeros(len(y_cal), int)
        self.aev = E_cal[np.arange(len(y_cal)), y_cal]; self.fallback = fallback or {}
    def p_lab(self, P, psi):
        A = 1 - P; out = np.zeros_like(A)
        st = np.digitize(psi, self.cut) if self.cut is not None else np.zeros(len(A), int)
        for b in np.unique(st):
            ref = self.alab[self.st == b]
            if len(ref) < 15: ref = self.alab
            out[st == b] = pvalues(ref, A[st == b])
        return out
    def p_ev(self, E):
        out = np.ones_like(E)
        for k in range(E.shape[1]):
            ref = self.aev[self.y == k]
            if len(ref) == 0 and k in self.fallback: ref = self.fallback[k]
            if len(ref): out[:, k] = pvalues(ref, E[:, k])
        return out
    def sets(self, P, psi, E, alpha, beta):
        pl = self.p_lab(P, psi); pe = self.p_ev(E) if beta > 0 else np.ones_like(P)
        return (pl > alpha) & (pe > beta), pl, pe

def stats(S, y=None, mass=None, mu0=None):
    size = S.sum(1); r = dict(size=float(size.mean()), single=float((size == 1).mean()), empty=float((size == 0).mean()))
    if y is not None:
        r["cov"] = float(S[np.arange(len(y)), y].mean()); sg = size == 1
        r["single_acc"] = float((S[sg].argmax(1) == y[sg]).mean()) if sg.any() else float("nan")
    return r
