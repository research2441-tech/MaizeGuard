"""Degraded field images (CD&S test under blur/dark/JPEG level 2): classification and dual decision with
target calibration on clean CD&S training images."""
import json, csv, numpy as np
from sklearn.metrics import f1_score
from splits import load_index, seed_split
from analysis_core import load_pred, Calibrated
from conformal2 import Dual, fit_gauss, maha
CL = ["healthy", "gls", "nlb", "rust"]; rows = load_index(); Y = np.array([CL.index(r["label"]) for r in rows])
cds = list(csv.DictReader(open("index_cds.csv"))); ycds = np.array([CL.index(r["label"]) if r["label"] in CL else -1 for r in cds])
test_rows = [i for i, r in enumerate(cds) if r["ds"] == "cds" and r["split"] == "test"]
pos = {r: j for j, r in enumerate(test_rows)}
I = lambda f: np.array([i for i, r in enumerate(cds) if f(r)])
tk = I(lambda r: r["ds"] == "cds" and r["split"] == "test" and r["label"] in ("gls", "nlb"))
tn = I(lambda r: r["ds"] == "cds" and r["split"] == "test" and r["label"] == "nls")
trk = I(lambda r: r["ds"] == "cds" and r["split"] == "train" and r["label"] in ("gls", "nlb"))
out = {}
for c in ["clean", "blur2", "dark2", "jpeg2"]:
    for m in ["lp", "clipad", "tipf", "mg"]:
        for s in range(5):
            sp = seed_split(rows, s); pr = lambda k: load_pred(f"runs/{m}_s{s}_pred.npz", k); mg = lambda k: load_pred(f"runs/mg_s{s}_pred.npz", k)
            C = Calibrated(pr("val"), pr("cal"), pr("cal_aug"), Y[sp["val"]], Y[sp["cal"]], mg("val")["q"], mg("cal")["q"], mg("cal_aug")["q"])
            Dc = load_pred(f"runs/{m}_s{s}_cds.npz"); qc = load_pred(f"runs/mg_s{s}_cds.npz")["q"]
            if c == "clean": Dt = {k: v[test_rows] for k, v in Dc.items()}; qt = qc[test_rows]
            else: Dt = load_pred(f"runs/{m}_s{s}_cdst_{c}.npz"); qt = load_pred(f"runs/mg_s{s}_cdst_{c}.npz")["q"]
            ik = np.array([pos[i] for i in tk]); iN = np.array([pos[i] for i in tn])
            f1 = f1_score(ycds[tk], Dt["logits"][ik].argmax(1), labels=[1, 2], average="macro")
            key = "rl" if m == "mg" else "z"
            perm = np.random.default_rng(s).permutation(trk); h = len(perm) // 2
            G = fit_gauss(C.val[key], C.y_val, [0, 3]); G.update(fit_gauss(Dc[key][perm[:h]], ycds[perm[:h]], [1, 2]))
            psic = 1 - qc[:, 5]; psit = 1 - qt[:, 5]
            Es = maha(G, C.cal[key]); fb = {k: Es[C.y_cal == k, k] for k in (0, 3)}
            du = Dual(C.probs(Dc["logits"][perm[h:]]), ycds[perm[h:]], psic[perm[h:]], maha(G, Dc[key][perm[h:]]), fallback=fb)
            P = C.probs(Dt["logits"]); E = maha(G, Dt[key])
            Sk, _, _ = du.sets(P[ik], psit[ik], E[ik], 0.05, 0.05); Sn, _, _ = du.sets(P[iN], psit[iN], E[iN], 0.05, 0.05)
            r = dict(f1=f1, cov=float(Sk[np.arange(len(ik)), ycds[tk]].mean()), size=float(Sk.sum(1).mean()), flag=float((Sn.sum(1) == 0).mean()),
                     known_empty=float((Sk.sum(1) == 0).mean()))
            out.setdefault(f"{m}|{c}", []).append(r)
res = {k: {q: (float(np.mean([x[q] for x in v])), float(np.std([x[q] for x in v]))) for q in v[0]} for k, v in out.items()}
json.dump(res, open("results/cdst.json", "w"), indent=1)
for k, v in res.items(): print(k, {q: round(a, 3) for q, (a, b) in v.items()})
