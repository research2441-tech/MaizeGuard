"""Collect every number quoted in the manuscript into results/numbers.json (single source of truth)."""
import json, glob, numpy as np
V = {}
for f in sorted(glob.glob("results/vision_s*.json")): V.update(json.load(open(f)))
for f in sorted(glob.glob("results/vision_abl_s*.json")): V.update(json.load(open(f)))
def agg(m, path):
    vals = []
    for k, v in V.items():
        if k.split("|")[0] != m: continue
        x = v
        try:
            for p in path: x = x[p]
            if x is not None and not (isinstance(x, float) and np.isnan(x)): vals.append(float(x))
        except (KeyError, TypeError): pass
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)
N = {}
MAIN = ["zs", "zsdesc", "lp", "clipad", "tipf", "abmil", "mg"]
ABL = ["mg", "mg_noles", "mg_nodesc", "mg_noqual", "mg_noaug", "mg_nocons", "mg_notok", "mg_adapter"]
for m in MAIN + ABL[1:]:
    for s in ["test_pv", "test_pd", "cds"]:
        for q in ["f1", "bacc", "mcc", "acc", "ece", "nll", "brier"]:
            N[f"{m}.{s}.{q}"] = agg(m, [s, q])
    for q in ["msp", "maxlogit", "energy", "mahalanobis", "knn", "mcm"]:
        for z in ["auroc", "fpr95", "aupr"]: N[f"{m}.ood.{q}.{z}"] = agg(m, ["ood", q, z])
    for q in ["tgt_maha_rl", "tgt_maha_z", "tgt_maha_rg", "tgt_maha_ev", "tgt_U_rl", "tgt_U_z", "src_U_rl", "src_U_z"]:
        for z in ["auroc", "fpr95"]: N[f"{m}.oodt.{q}.{z}"] = agg(m, ["ood_target", q, z])
    for k in ["src_lac", "src_aps", "src_dual_rl", "src_dual_z", "tgt_lac", "tgt_aps", "tgt_raps", "tgt_classwise", "tgt_dual_z", "tgt_dual_rg",
              "tgt_dual_ev", "tgt_dual_rl", "tgt_dual_rl_nostrata", "tgt_ab_0.025_0.075", "tgt_ab_0.075_0.025", "tgt_ab_0.1_0.0", "tgt_ab_0.0_0.1"]:
        for q in ["cov", "size", "single", "single_acc", "nls_flag", "nls_unknown", "known_reimage", "nls_accept"]:
            N[f"{m}.conf.{k}.{q}"] = agg(m, ["conformal", k, q])
    for q in ["spearman", "kendall"]: N[f"{m}.sev_cds.{q}"] = agg(m, ["sev_cds", q])
    for k in range(5): N[f"{m}.sev_cds.mean{k+1}"] = (agg(m, ["sev_cds", "means"])[0],)  # placeholder
    for q in ["mae", "spearman"]: N[f"{m}.sev_pv.{q}"] = agg(m, ["sev_pv", q])
    for c in [f"{k}_{l}" for k in ["blur", "jpeg", "dark", "motion", "noise", "glare"] for l in [1, 2, 3]] + ["clean"]:
        path = ["robust_clean"] if c == "clean" else ["robust", c]
        for q in ["f1", "f1_pd", "ece", "dual_cov", "dual_cov_nostrata", "dual_size", "dual_size_nostrata", "lac_cov", "lac_size"]:
            N[f"{m}.rob.{c}.{q}"] = agg(m, path + [q])
    N[f"{m}.params"] = agg(m, ["params"]); N[f"{m}.train_time"] = agg(m, ["train_time"])
    for n in [20, 40, 80, 160, 320]:
        for fn in ["rl", "z", "rg", "ev"]:
            for q in ["cov", "nls_flag", "size"]: N[f"{m}.nsweep.{n}.{fn}.{q}"] = agg(m, ["nsweep", str(n), fn, q])
# severity means per grade (mg)
means = [V[k]["sev_cds"]["means"] for k in V if k.split("|")[0] == "mg" and "sev_cds" in V[k]]
N["mg.sev_cds.grade_means"] = list(np.mean(means, 0)) if means else []
N["sevbase"] = json.load(open("results/sev_baselines.json"))
N["loc"] = json.load(open("results/localize.json"))
N["naming"] = json.load(open("results/naming.json"))
N["stats"] = json.load(open("results/stats.json"))
T = {}
for f in sorted(glob.glob("runs/temporal_s[0-9].json")):
    for k, v in json.load(open(f)).items(): T.setdefault(k, []).append(v)
N["temporal"] = {k: {m: (float(np.mean([x[m] for x in v])), float(np.std([x[m] for x in v]))) for m in v[0]} for k, v in T.items()}
for f in ["results/efficiency.json"]:
    try: N["eff"] = json.load(open(f))
    except FileNotFoundError: pass
json.dump(N, open("results/numbers.json", "w"), indent=0, default=float)
print(len(N))
