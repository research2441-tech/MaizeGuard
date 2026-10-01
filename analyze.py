"""Aggregate all vision experiments into results/vision.json (E1-E5, E7-E8)."""
import os, json, glob, csv, numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
from splits import load_index, seed_split
from metrics import cls_metrics, ood_metrics
from analysis_core import load_pred, Calibrated, Conformal, open_scores, set_stats
from conformal2 import Dual, fit_gauss, maha, stats

CL = ["healthy", "gls", "nlb", "rust"]
rows = load_index(); Y = np.array([CL.index(r["label"]) for r in rows])
cds = list(csv.DictReader(open("index_cds.csv")))
ci = {k: np.array([i for i, r in enumerate(cds) if f(r)]) for k, f in dict(
    test_known=lambda r: r["ds"] == "cds" and r["split"] == "test" and r["label"] in ("gls", "nlb"),
    test_nls=lambda r: r["ds"] == "cds" and r["split"] == "test" and r["label"] == "nls",
    train_known=lambda r: r["ds"] == "cds" and r["split"] == "train" and r["label"] in ("gls", "nlb"),
    train_nls=lambda r: r["ds"] == "cds" and r["split"] == "train" and r["label"] == "nls",
    sev=lambda r: r["ds"] == "cds_sev", ann=lambda r: r["ds"] == "cds_ann").items()}
ycds = np.array([CL.index(r["label"]) if r["label"] in CL else -1 for r in cds])
cds_test_rows = [i for i, r in enumerate(cds) if r["ds"] == "cds" and r["split"] == "test"]  # order of cdst_* files
SEEDS = sorted({int(os.path.basename(p).split("_s")[1].split("_")[0].split(".")[0]) for p in glob.glob("runs/mg_s*.pt")})
MAIN = ["zs", "zsdesc", "lp", "clipad", "tipf", "abmil", "mg"]
ABL = ["mg", "mg_noles", "mg_nodesc", "mg_noqual", "mg_noaug", "mg_nocons", "mg_notok", "mg_adapter"]
CORR = [f"{k}_{l}" for k in ["blur", "jpeg", "dark", "motion", "noise", "glare"] for l in [1, 2, 3]]

def sub(d, idx): return {k: v[idx] for k, v in d.items()}


def conformal_block(method, seed, C, D, qd):
    tk, tn, trk = ci["test_known"], ci["test_nls"], ci["train_known"]
    psi = 1 - qd[:, 5]; P = C.probs(D["logits"]); yk = ycds[tk]
    is_mg = method.startswith("mg")
    feats = ({"rl": "rl", "z": "z", "rg": "rg", "ev": "ev"} if method == "mg" else {"rl": "rl"}) if is_mg else {"z": "z"}
    mass = D["mass"] if "mass" in D else None
    mu0 = np.quantile(C.cal["mass"][C.y_cal > 0], 0.05) if "mass" in C.cal else None
    res = {}; ood = {}
    def record(name, Sk, Sn):
        r = stats(Sk, yk); r["nls_accept"] = float((Sn.sum(1) > 0).mean()); r["nls_flag"] = float((Sn.sum(1) == 0).mean())
        if mass is not None and mu0 is not None:
            r["nls_unknown"] = float(((Sn.sum(1) == 0) & (mass[tn] >= mu0)).mean())
            r["known_reimage"] = float(((Sk.sum(1) == 0) & (mass[tk] < mu0)).mean())
        res[name] = r
    # ----- source calibration (no target labels) -----
    Pc = C.probs(C.cal["logits"])
    CF = Conformal(C, mondrian=False)
    record("src_lac", CF.lac_sets(sub(D, tk), 0.10), CF.lac_sets(sub(D, tn), 0.10))
    record("src_aps", CF.aps_sets(sub(D, tk), 0.10), CF.aps_sets(sub(D, tn), 0.10))
    for fn, key in feats.items():
        Gs = fit_gauss(C.val[key], C.y_val, range(4))
        dual = Dual(Pc, C.y_cal, C.psi_cal, maha(Gs, C.cal[key]))
        E = np.zeros((len(D[key]), 4)); E[tk] = maha(Gs, D[key][tk]); E[tn] = maha(Gs, D[key][tn])
        Sk, plk, pek = dual.sets(P[tk], psi[tk], E[tk], 0.05, 0.05); Sn, pln, pen = dual.sets(P[tn], psi[tn], E[tn], 0.05, 0.05)
        record(f"src_dual_{fn}", Sk, Sn)
        ood[f"src_U_{fn}"] = ood_metrics(1 - np.minimum(plk, pek).max(1), 1 - np.minimum(pln, pen).max(1))
    # ----- few-label target calibration: CD&S training folder (known classes only) -----
    rng = np.random.default_rng(seed); perm = rng.permutation(trk); half = len(perm) // 2
    fit_i, cal_i = perm[:half], perm[half:]
    def target_eval(fit_i, cal_i, tag, full=True):
        out_local = {}
        Pcal = P[cal_i]; ycal = ycds[cal_i]
        for fn, key in feats.items():
            G = fit_gauss(C.val[key], C.y_val, [0, 3]); G.update(fit_gauss(D[key][fit_i], ycds[fit_i], [1, 2]))
            E = np.zeros((len(D[key]), 4)); need = np.concatenate([cal_i, tk, tn]); E[need] = maha(G, D[key][need])
            Esrc = maha(G, C.cal[key]); fb = {k: Esrc[C.y_cal == k, k] for k in (0, 3)}
            dual = Dual(Pcal, ycal, psi[cal_i], E[cal_i], fallback=fb)
            Sk, plk, pek = dual.sets(P[tk], psi[tk], E[tk], 0.05, 0.05); Sn, pln, pen = dual.sets(P[tn], psi[tn], E[tn], 0.05, 0.05)
            if full:
                record(f"tgt_dual_{fn}", Sk, Sn)
                ood[f"tgt_maha_{fn}"] = ood_metrics(E[tk][:, [1, 2]].min(1), E[tn][:, [1, 2]].min(1))
                ood[f"tgt_U_{fn}"] = ood_metrics(1 - np.minimum(plk, pek).max(1), 1 - np.minimum(pln, pen).max(1))
                if fn == "rl":
                    dual_ns = Dual(Pcal, ycal, psi[cal_i], E[cal_i], strata=False, fallback=fb)
                    a, _, _ = dual_ns.sets(P[tk], psi[tk], E[tk], 0.05, 0.05); b, _, _ = dual_ns.sets(P[tn], psi[tn], E[tn], 0.05, 0.05)
                    record("tgt_dual_rl_nostrata", a, b)
                    for al, be in [(0.10, 0.0), (0.0, 0.10), (0.025, 0.075), (0.075, 0.025), (0.05, 0.05)]:
                        a, _, _ = dual.sets(P[tk], psi[tk], E[tk], al, be) if be > 0 else dual.sets(P[tk], psi[tk], E[tk], al, 0)
                        if al == 0: a = (dual.p_ev(E[tk]) > be); b = (dual.p_ev(E[tn]) > be)
                        else: b, _, _ = dual.sets(P[tn], psi[tn], E[tn], al, be)
                        record(f"tgt_ab_{al}_{be}", a, b)
            else:
                r = stats(Sk, yk); r["nls_flag"] = float((Sn.sum(1) == 0).mean()); out_local[fn] = r
        if full:
            # label-only baselines calibrated on the same target images
            CFt = Conformal(C, cal=sub(D, cal_i), y_cal=ycds[cal_i], psi_cal=psi[cal_i], mondrian=False)
            record("tgt_lac", CFt.lac_sets(sub(D, tk), 0.10), CFt.lac_sets(sub(D, tn), 0.10))
            record("tgt_aps", CFt.aps_sets(sub(D, tk), 0.10), CFt.aps_sets(sub(D, tn), 0.10))
            record("tgt_raps", CFt.aps_sets(sub(D, tk), 0.10, lam=0.1, kreg=1), CFt.aps_sets(sub(D, tn), 0.10, lam=0.1, kreg=1))
            record("tgt_classwise", CFt.lac_sets(sub(D, tk), 0.10, True), CFt.lac_sets(sub(D, tn), 0.10, True))
        return out_local
    target_eval(fit_i, cal_i, "full")
    sweep = {}
    for n in ([20, 40, 80, 160, 320] if method == "mg" else []):
        rs = []
        for rep in range(5):
            r2 = np.random.default_rng(100 * seed + rep).permutation(trk)[:n]
            rs.append(target_eval(r2[:n // 2], r2[n // 2:], f"n{n}", full=False))
        sweep[n] = {fn: {m: float(np.mean([x[fn][m] for x in rs])) for m in rs[0][fn]} for fn in rs[0]}
    return res, ood, sweep

def per_seed(method, seed, zs_cds=None):
    f = f"runs/{method}_s{seed}_pred.npz"
    if not os.path.exists(f): return None
    sp = seed_split(rows, seed); pr = lambda s: load_pred(f, s)
    mg = lambda s: load_pred(f"runs/mg_s{seed}_pred.npz", s)
    C = Calibrated(pr("val"), pr("cal"), pr("cal_aug"), Y[sp["val"]], Y[sp["cal"]], mg("val")["q"], mg("cal")["q"], mg("cal_aug")["q"])
    out = {}
    for s, labs in [("test_pv", [0, 1, 2, 3]), ("test_pd", [1, 2, 3])]:
        out[s] = cls_metrics(pr(s)["logits"], Y[sp[s]], labs, T=C.T)
    fc = f"runs/{method}_s{seed}_cds.npz"
    if os.path.exists(fc):
        D = load_pred(fc); qd = load_pred(f"runs/mg_s{seed}_cds.npz")["q"]
        tk, tn = ci["test_known"], ci["test_nls"]
        out["cds"] = cls_metrics(D["logits"][tk], ycds[tk], [1, 2], T=C.T)
        # ---------- open set (E2) ----------
        sc_in = open_scores(C, sub(D, tk), None if zs_cds is None else zs_cds[tk])
        sc_out = open_scores(C, sub(D, tn), None if zs_cds is None else zs_cds[tn])
        out["ood"] = {k: ood_metrics(sc_in[k], sc_out[k]) for k in sc_in}
        # ---------- conformal (E3) and few-label target novelty ----------
        out["conformal"], out["ood_target"], out["nsweep"] = conformal_block(method, seed, C, D, qd)
        # ---------- severity on CD&S expert grades & lesion boxes (E5) ----------
        if "sev" in D:
            sv = ci["sev"]; g = np.array([int(cds[i]["sev"]) for i in sv])
            out["sev_cds"] = dict(spearman=spearmanr(D["sev"][sv], g)[0], kendall=kendalltau(D["sev"][sv], g)[0],
                                  means=[float(D["sev"][sv][g == k].mean()) for k in range(1, 6)])
    # PV pseudo-severity
    if "sev" in pr("test_pv"):
        st = np.load("feats/pv_token_targets.npz")["sev"][sp["test_pv"]]; ok = ~np.isnan(st)
        p = pr("test_pv")["sev"]
        out["sev_pv"] = dict(mae=float(np.abs(p[ok] - st[ok]).mean()), spearman=spearmanr(p[ok], st[ok])[0])
    # ---------- corruption robustness (E4) ----------
    rob = {}
    n_pv = len(sp["test_pv"])
    for c in CORR:
        fcorr = f"runs/{method}_s{seed}_test_{c}.npz"
        if not os.path.exists(fcorr): continue
        D = load_pred(fcorr); yt = np.r_[Y[sp["test_pv"]], Y[sp["test_pd"]]]
        r = cls_metrics(D["logits"], yt, [0, 1, 2, 3], T=C.T)
        r["f1_pd"] = cls_metrics(D["logits"][n_pv:], yt[n_pv:], [1, 2, 3], T=C.T)["f1"]
        if method.startswith("mg"):
            qd2 = load_pred(f"runs/mg_s{seed}_test_{c}.npz")["q"]; psi_t = 1 - qd2[:, 5]
            Gs = fit_gauss(C.val["rl"], C.y_val, range(4)); dual = Dual(C.probs(C.cal["logits"]), C.y_cal, C.psi_cal, maha(Gs, C.cal["rl"]))
            S, _, _ = dual.sets(C.probs(D["logits"]), psi_t, maha(Gs, D["rl"]), 0.05, 0.05); r["dual_cov"] = stats(S, yt)["cov"]; r["dual_size"] = stats(S, yt)["size"]
            dual0 = Dual(C.probs(C.cal["logits"]), C.y_cal, C.psi_cal, maha(Gs, C.cal["rl"]), strata=False)
            S2, _, _ = dual0.sets(C.probs(D["logits"]), psi_t, maha(Gs, D["rl"]), 0.05, 0.05); r["dual_cov_nostrata"] = stats(S2, yt)["cov"]; r["dual_size_nostrata"] = stats(S2, yt)["size"]
        CFl = Conformal(C, mondrian=False); S3 = CFl.lac_sets(D, 0.10); r["lac_cov"] = set_stats(S3, yt)["cov"]; r["lac_size"] = S3.sum(1).mean()
        rob[c] = r
    out["robust"] = rob
    Dc = {k: np.concatenate([pr("test_pv")[k], pr("test_pd")[k]]) for k in pr("test_pv")}
    yt = np.r_[Y[sp["test_pv"]], Y[sp["test_pd"]]]
    rc = cls_metrics(Dc["logits"], yt, [0, 1, 2, 3], T=C.T); rc["f1_pd"] = out["test_pd"]["f1"]
    if method.startswith("mg"):
        psi_t = 1 - np.concatenate([mg("test_pv")["q"], mg("test_pd")["q"]])[:, 5]
        Gs = fit_gauss(C.val["rl"], C.y_val, range(4)); dual = Dual(C.probs(C.cal["logits"]), C.y_cal, C.psi_cal, maha(Gs, C.cal["rl"]))
        S, _, _ = dual.sets(C.probs(Dc["logits"]), psi_t, maha(Gs, Dc["rl"]), 0.05, 0.05); rc["dual_cov"] = stats(S, yt)["cov"]; rc["dual_size"] = stats(S, yt)["size"]
        dual0 = Dual(C.probs(C.cal["logits"]), C.y_cal, C.psi_cal, maha(Gs, C.cal["rl"]), strata=False)
        S2, _, _ = dual0.sets(C.probs(Dc["logits"]), psi_t, maha(Gs, Dc["rl"]), 0.05, 0.05); rc["dual_cov_nostrata"] = stats(S2, yt)["cov"]; rc["dual_size_nostrata"] = stats(S2, yt)["size"]
    CFl = Conformal(C, mondrian=False); S3 = CFl.lac_sets(Dc, 0.10); rc["lac_cov"] = set_stats(S3, yt)["cov"]; rc["lac_size"] = S3.sum(1).mean()
    out["robust_clean"] = rc
    ck = __import__("torch").load(f"runs/{method}_s{seed}.pt", weights_only=False)
    out["train_time"] = ck["train_time"]; out["params"] = int(sum(v.numel() for k, v in ck["state"].items() if k not in ("P", "Qp", "D0", "log_prior", "W", "V")))
    return out

if __name__ == "__main__":
    import sys
    res = {}
    seeds_run = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else SEEDS
    for seed in seeds_run:
        zs_cds = None
        fz = f"runs/zs_s{seed}_cds.npz"
        if os.path.exists(fz): zs_cds = load_pred(fz)["logits"]
        mlist = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(set(MAIN + ABL))
        for m in mlist:
            r = per_seed(m, seed, zs_cds)
            if r is not None: res[f"{m}|{seed}"] = r
        os.makedirs("results", exist_ok=True)
        tag = "_abl" if all(m.startswith("mg_") for m in mlist) else ""
        json.dump({k: v for k, v in res.items() if k.endswith(f"|{seed}")}, open(f"results/vision{tag}_s{seed}.json", "w"), default=float, indent=0)
        print("seed", seed, "done", flush=True)
