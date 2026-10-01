"""Progression forecasting experiment (E6). Usage: python run_temporal.py <seed>"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
import progression as P, temporal as TM
from analysis_core import load_pred, Calibrated, Conformal
from splits import load_index, seed_split
from scipy.special import softmax

CORR = [f"{k}_{l}" for k in ["blur", "jpeg", "dark", "motion", "noise", "glare"] for l in [1, 3]]

def vision_bank(seed, rows, sp):
    y = np.array([["healthy", "gls", "nlb", "rust"].index(r["label"]) for r in rows])
    pr = lambda s: load_pred(f"runs/mg_s{seed}_pred.npz", s)
    C = Calibrated(pr("val"), pr("cal"), pr("cal_aug"), y[sp["val"]], y[sp["cal"]], pr("val")["q"], pr("cal")["q"], pr("cal_aug")["q"])
    CF = Conformal(C)
    n = len(sp["test_pv"]); vis = {}
    for c in ["clean"] + [c for c in CORR if os.path.exists(f"runs/mg_s{seed}_test_{c}.npz")]:
        d = pr("test_pv") if c == "clean" else {k: v[:n] for k, v in load_pred(f"runs/mg_s{seed}_test_{c}.npz").items()}
        psi = 1 - d["q"][:, 5]
        S, pl, pe = CF.dual_sets(d, psi, 0.05, 0.05)
        vis[c] = dict(sev=np.clip(d["sev"], 1e-3, 0.97), p=C.probs(d["logits"]), degr=psi, abst=(S.sum(1) != 1).astype(float))
    true_sev = np.load("feats/pv_token_targets.npz")["sev"][sp["test_pv"]]
    true_sev = np.where(np.isnan(true_sev), 0, true_sev)
    return vis, y[sp["test_pv"]], true_sev

def evaluate(pred_fn, seqs, horizons=(7, 14)):
    """pred_fn(seq_batch) -> dict h -> (median, q10, q90, pexceed) arrays over (B,T) forecast origins"""
    obs, wf, y = TM.batchify(seqs); out = pred_fn(obs, wf); res = {}
    for h in horizons:
        med, lo, hi, pex = out[h]; T = y.shape[1]
        origins = np.zeros((len(seqs), T), bool)
        for i, s in enumerate(seqs):
            v = s["visits"]; v = v[(v >= 14) & (v + h < T)]; origins[i, v] = True
        tgt = np.zeros_like(med); tgt[:, :T - h] = y[:, h:].numpy()
        res[h] = dict(med=med[origins], lo=lo[origins], hi=hi[origins], pex=pex[origins], y=tgt[origins])
    return res

def filter_pred(model):
    G = TM.GRID.numpy(); yg = 1 / (1 + np.exp(-G))
    def f(obs, wf):
        with torch.no_grad():
            filt = model(obs, wf); B, T, K = filt.shape; out = {}
            for h in (7, 14):
                med = np.zeros((B, T)); lo = np.zeros((B, T)); hi = np.zeros((B, T)); pex = np.zeros((B, T))
                for t in range(T - h):
                    lp = model.forecast(filt[:, t], wf, t, h, model._beliefs[:, t]).exp().numpy()
                    cdf = np.cumsum(lp, 1)
                    med[:, t] = yg[np.argmax(cdf >= 0.5, 1)]; lo[:, t] = yg[np.argmax(cdf >= 0.1, 1)]; hi[:, t] = yg[np.argmax(cdf >= 0.9, 1)]
                    pex[:, t] = lp[:, yg > 0.10].sum(1)
                out[h] = (med, lo, hi, pex)
        return out
    return f

def gru_pred(model):
    from scipy.stats import norm
    def f(obs, wf):
        with torch.no_grad():
            o = model(obs, wf).numpy(); out = {}
            for k, h in enumerate((7, 14)):
                m, s = o[..., k, 0], np.exp(np.clip(o[..., k, 1], -4, 3)); sg = lambda e: 1 / (1 + np.exp(-e))
                out[h] = (sg(m), sg(m - 1.2816 * s), sg(m + 1.2816 * s), 1 - norm.cdf((np.log(0.1 / 0.9) - m) / s))
        return out
    return f

def naive_pred(kind):
    def f(obs, wf):
        o = obs.numpy(); B, T, _ = o.shape; out = {}
        for h in (7, 14):
            med = np.zeros((B, T)); lo = np.zeros((B, T)); hi = np.zeros((B, T)); pex = np.zeros((B, T))
            for i in range(B):
                vis_t = np.where(o[i, :, 0] > 0)[0]
                for t in range(T):
                    past = vis_t[vis_t <= t]
                    if len(past) == 0: continue
                    l1 = o[i, past[-1], 1]
                    if kind == "persist" or len(past) < 2: e = l1
                    else:
                        l0 = o[i, past[-2], 1]; dt = max(1, past[-1] - past[-2]); e = l1 + (l1 - l0) / dt * (t + h - past[-1])
                    yv = 1 / (1 + np.exp(-e)); med[i, t] = yv; lo[i, t] = 1 / (1 + np.exp(-(e - 1.0))); hi[i, t] = 1 / (1 + np.exp(-(e + 1.0)))
                    pex[i, t] = float(yv > 0.1)
            out[h] = (med, lo, hi, pex)
        return out
    return f

def metrics(r, cal=None):
    from sklearn.metrics import roc_auc_score
    y, med, lo, hi = r["y"], r["med"], r["lo"], r["hi"]
    out = dict(mae=100 * np.abs(med - y).mean(), cov_raw=((y >= lo) & (y <= hi)).mean(), width_raw=100 * (hi - lo).mean())
    yb = y > 0.10
    out["auroc_exceed"] = roc_auc_score(yb, r["pex"]) if 0 < yb.mean() < 1 else np.nan
    if cal is not None:
        E = np.maximum(cal["lo"] - cal["y"], cal["y"] - cal["hi"]); n = len(E)
        Q = np.sort(E)[min(n - 1, int(np.ceil((n + 1) * 0.8)) - 1)]
        out["cov_cqr"] = ((y >= lo - Q) & (y <= hi + Q)).mean(); out["width_cqr"] = 100 * (hi - lo + 2 * Q).mean()
    return out

if __name__ == "__main__":
    seed = int(sys.argv[1]); t0 = time.time()
    rows = load_index(); sp = seed_split(rows, seed)
    vis, labels, true_sev = vision_bank(seed, rows, sp)
    rng = np.random.default_rng(1000 + seed)
    bank = TM.ObsBank(vis, labels, true_sev, rng)
    tr = P.build("champion", range(1982, 2006), 20, seed, start=160, length=100)
    ca = P.build("champion", range(2006, 2011), 40, seed + 50, start=160, length=100)
    te = P.build("champion", range(2011, 2019), 40, seed + 100, start=160, length=100)
    hy = P.build("hyderabad", range(2000, 2011), 30, seed + 150, start=185, length=100)
    S = {k: TM.make_sequences(v, bank, rng) for k, v in dict(tr=tr, ca=ca, te=te, hy=hy).items()}
    # oracle-observation variant: true lesion fraction observed without vision noise
    models = {}
    for name, kw in [("filter", {}), ("filter_noenv", dict(use_env=False)), ("filter_noqual", dict(use_quality=False))]:
        m = TM.EnvFilter(**kw); TM.train_filter(m, S["tr"], epochs=40, lr=0.02, seed=seed); models[name] = filter_pred(m)
        print(name, f"{time.time()-t0:.0f}s", flush=True)
    def strip(seqs):
        out = []
        for s_ in seqs:
            s2 = dict(s_); o = s_["obs"].copy(); o[:, 2:4] = 0; s2["obs"] = o; out.append(s2)
        return out
    for name, kw, tf in [("gru", {}, None), ("gru_noenv", dict(use_env=False), None), ("gru_noqual", {}, strip)]:
        m = TM.GRUForecaster(**kw); TM.train_gru(m, tf(S["tr"]) if tf else S["tr"], epochs=60, seed=seed)
        g = gru_pred(m); models[name] = (lambda g, tf: (lambda o, w: g(torch.tensor(o.numpy().copy()).index_fill_(2, torch.tensor([2, 3]), 0.) if tf else o, w)))(g, tf)
        print(name, f"{time.time()-t0:.0f}s", flush=True)
    models["persist"] = naive_pred("persist"); models["linear"] = naive_pred("linear")
    res = {}
    for name, f in models.items():
        cal = evaluate(f, S["ca"])
        for split in ["te", "hy"]:
            r = evaluate(f, S[split])
            for h in (7, 14):
                res[f"{name}|{split}|{h}"] = metrics(r[h], cal[h])
    # example trajectories for the figure (test split, first 6 diseased plots)
    ex = [i for i, s in enumerate(S["te"]) if s["disease"] != "healthy"][:6]
    obs, wf, y = TM.batchify([S["te"][i] for i in ex])
    fp = models["filter"](obs, wf)
    np.savez(f"runs/temporal_examples_s{seed}.npz", y=y.numpy(), obs=obs.numpy(), med7=fp[7][0], lo7=fp[7][1], hi7=fp[7][2],
             med14=fp[14][0], lo14=fp[14][1], hi14=fp[14][2])
    json.dump(res, open(f"runs/temporal_s{seed}.json", "w"), indent=1, default=float)
    print("done", f"{time.time()-t0:.0f}s")
