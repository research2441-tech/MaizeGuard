"""Statistical comparisons (E8): per-seed differences + image-level paired bootstrap pooled over seeds, Holm correction."""
import json, csv, numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score, roc_auc_score
from splits import load_index, seed_split
from analysis_core import load_pred, Calibrated
from conformal2 import Dual, fit_gauss, maha

CL = ["healthy", "gls", "nlb", "rust"]; rows = load_index(); Y = np.array([CL.index(r["label"]) for r in rows])
cds = list(csv.DictReader(open("index_cds.csv"))); ycds = np.array([CL.index(r["label"]) if r["label"] in CL else -1 for r in cds])
I = lambda f: np.array([i for i, r in enumerate(cds) if f(r)])
tk = I(lambda r: r["ds"] == "cds" and r["split"] == "test" and r["label"] in ("gls", "nlb"))
tn = I(lambda r: r["ds"] == "cds" and r["split"] == "test" and r["label"] == "nls")
trk = I(lambda r: r["ds"] == "cds" and r["split"] == "train" and r["label"] in ("gls", "nlb"))
SEEDS = range(5); B = 1000; rng = np.random.default_rng(7)

def calib(m, s):
    sp = seed_split(rows, s); pr = lambda k: load_pred(f"runs/{m}_s{s}_pred.npz", k); mg = lambda k: load_pred(f"runs/mg_s{s}_pred.npz", k)
    return Calibrated(pr("val"), pr("cal"), pr("cal_aug"), Y[sp["val"]], Y[sp["cal"]], mg("val")["q"], mg("cal")["q"], mg("cal_aug")["q"]), sp

def flags(m, s, key):
    C, sp = calib(m, s); D = load_pred(f"runs/{m}_s{s}_cds.npz"); psi = 1 - load_pred(f"runs/mg_s{s}_cds.npz")["q"][:, 5]
    P = C.probs(D["logits"]); r = np.random.default_rng(s); perm = r.permutation(trk); h = len(perm) // 2
    G = fit_gauss(C.val[key], C.y_val, [0, 3]); G.update(fit_gauss(D[key][perm[:h]], ycds[perm[:h]], [1, 2]))
    need = np.concatenate([perm[h:], tk, tn]); E = np.zeros((len(P), 4)); E[need] = maha(G, D[key][need])
    Es = maha(G, C.cal[key]); fb = {k: Es[C.y_cal == k, k] for k in (0, 3)}
    du = Dual(P[perm[h:]], ycds[perm[h:]], psi[perm[h:]], E[perm[h:]], fallback=fb)
    Sk, _, _ = du.sets(P[tk], psi[tk], E[tk], 0.05, 0.05); Sn, _, _ = du.sets(P[tn], psi[tn], E[tn], 0.05, 0.05)
    return (Sn.sum(1) == 0).astype(float), Sk[np.arange(len(tk)), ycds[tk]].astype(float)

def compare(name, fa, fb, n, metric):
    """fa/fb: per-seed functions returning per-image arrays; metric(arrays, idx)"""
    diffs = []; boots = np.zeros(B)
    for s in SEEDS:
        a, b = fa(s), fb(s)
        diffs.append(metric(a, np.arange(n)) - metric(b, np.arange(n)))
        for j in range(B):
            idx = rng.integers(0, n, n); boots[j] += (metric(a, idx) - metric(b, idx)) / len(SEEDS)
    d = np.array(diffs); t = ttest_rel(d, np.zeros_like(d)) if d.std() > 0 else None
    p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(name=name, mean=float(d.mean()), sd=float(d.std(ddof=1)), ci=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                p_boot=float(max(p_boot, 1 / B)), p_t=float(t.pvalue) if t else 1.0, dz=float(d.mean() / (d.std(ddof=1) + 1e-12)))

if __name__ == "__main__":
    out = []
    lg = lambda m, s, sub="cds": load_pred(f"runs/{m}_s{s}_cds.npz")["logits"]
    f1m = lambda arr, idx: f1_score(ycds[tk][idx], arr[idx].argmax(1), labels=[1, 2], average="macro")
    for b in ["lp", "clipad", "tipf"]:
        out.append(compare(f"CD&S macro-F1: MG - {b}", lambda s: lg("mg", s)[tk], lambda s, b=b: lg(b, s)[tk], len(tk), f1m))
    def pd_logits(m, s): return load_pred(f"runs/{m}_s{s}_pred.npz", "test_pd")["logits"]
    ypd = Y[seed_split(rows, 0)["test_pd"]]
    f1pd = lambda arr, idx: f1_score(ypd[idx], arr[idx].argmax(1), labels=[1, 2, 3], average="macro")
    for b in ["lp", "clipad", "tipf"]:
        out.append(compare(f"PlantDoc macro-F1: MG - {b}", lambda s: pd_logits("mg", s), lambda s, b=b: pd_logits(b, s), len(ypd), f1pd))
    # novelty AUROC (no target labels): MaxLogit of MG vs best baseline score
    def ml(m, s): L = lg(m, s); return np.r_[-L[tk].max(1), -L[tn].max(1)]
    yy = np.r_[np.zeros(len(tk)), np.ones(len(tn))]
    au = lambda arr, idx: roc_auc_score(yy[idx], arr[idx]) if 0 < yy[idx].mean() < 1 else 0.5
    for b in ["clipad", "lp", "zsdesc"]:
        out.append(compare(f"NLS AUROC (MaxLogit): MG - {b}", lambda s: ml("mg", s), lambda s, b=b: ml(b, s), len(yy), au))
    # few-label target dual test: NLS flag rate
    fr = lambda arr, idx: arr[idx].mean()
    out.append(compare("NLS flagged: MG(r_l) - LP(r_g)", lambda s: flags("mg", s, "rl")[0], lambda s: flags("lp", s, "z")[0], len(tn), fr))
    out.append(compare("NLS flagged: MG(r_l) - MG(r_g)", lambda s: flags("mg", s, "rl")[0], lambda s: flags("mg", s, "rg")[0], len(tn), fr))
    out.append(compare("Known coverage: MG(r_l) - LP(r_g)", lambda s: flags("mg", s, "rl")[1], lambda s: flags("lp", s, "z")[1], len(tk), fr))
    # Holm over bootstrap p-values
    ps = np.array([o["p_boot"] for o in out]); order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(order): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    for o, a in zip(out, adj): o["p_holm"] = float(a)
    json.dump(out, open("results/stats.json", "w"), indent=1)
    for o in out: print(f"{o['name']:40s} {o['mean']:+.3f} ±{o['sd']:.3f} CI[{o['ci'][0]:+.3f},{o['ci'][1]:+.3f}] p_holm={o['p_holm']:.4f} dz={o['dz']:.2f}")
