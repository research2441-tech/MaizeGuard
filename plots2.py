"""Severity, temporal and qualitative figures."""
import json, csv, numpy as np, sys, glob
sys.path.insert(0, ".")
from plots import *
from plots import _bar
from scipy.stats import spearmanr

def fig_severity():
    cds = list(csv.DictReader(open(f"{MG}/index_cds.csv"))); sv = [i for i, r in enumerate(cds) if r["ds"] == "cds_sev"]
    g = np.array([int(cds[i]["sev"]) for i in sv])
    preds = [np.load(f"{MG}/runs/mg_s{s}_cds.npz")["sev"][sv] for s in range(5) if os.path.exists(f"{MG}/runs/mg_s{s}_cds.npz")]
    p = np.mean(preds, 0)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]
    data = [100 * p[g == k] for k in range(1, 6)]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.0),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for b in bp["boxes"]: b.set_facecolor("0.85")
    rng = np.random.default_rng(0)
    for k, d in enumerate(data):
        ax.scatter(k + 1 + rng.uniform(-0.18, 0.18, len(d)), d, s=1.2, color="0.35", alpha=0.5, lw=0, zorder=3)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([f"{k}\n(n={len(d)})" for k, d in zip(range(1, 6), data)], fontsize=6)
    ax.set_xlabel("Expert NLS severity grade (CD&S)"); ax.set_ylabel("Predicted lesion area $\\hat{s}$ (%)")
    ax.set_title("(a) Severity read-out on an unseen disease", fontsize=7); ax.grid(axis="y", lw=0.3, color="0.88")
    ax = axs[1]
    B = json.load(open(f"{MG}/results/sev_baselines.json"))
    labs = ["Colour rule", "Zero-shot tokens", "Ridge on $\\mathbf{r}_g$", "No token loss", "MaizeGuard"]
    ms = [B["colour_rule"]["spearman"], B["zeroshot_tokens"]["spearman"], B["ridge_global"]["spearman"],
          get("mg_notok", ["sev_cds", "spearman"])[0], get("mg", ["sev_cds", "spearman"])[0]]
    sd = [np.nan, np.nan, B["ridge_global"]["spearman_sd"], get("mg_notok", ["sev_cds", "spearman"])[1], get("mg", ["sev_cds", "spearman"])[1]]
    _bar(ax, labs, ms, [0 if np.isnan(s) else s for s in sd], ylabel="Spearman $\\rho$ with expert grade", ylim=(0, 0.85))
    ax.set_title("(b) Rank agreement with expert grades", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_severity.png", dpi=400); plt.close(fig)

def fig_temporal():
    T = {}
    for f in sorted(glob.glob(f"{MG}/runs/temporal_s[0-9].json")):
        for k, v in json.load(open(f)).items(): T.setdefault(k, []).append(v)
    E = np.load(f"{MG}/runs/temporal_examples_s0.npz")
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, i in zip(axs[:2], [0, 1]):
        y = 100 * E["y"][i]; t = np.arange(len(y)); o = E["obs"][i]; v = o[:, 0] > 0
        ax.plot(t, y, color="black", lw=1.2, label="latent lesion area")
        med = 100 * E["med7"][i]; lo = 100 * E["lo7"][i]; hi = 100 * E["hi7"][i]
        tt = t[14:len(y) - 7]
        ax.plot(tt + 7, med[14:len(y) - 7], color="0.35", ls="--", lw=0.9, label="7-day forecast (median)")
        ax.fill_between(tt + 7, lo[14:len(y) - 7], hi[14:len(y) - 7], color="0.85", lw=0, label="80% interval")
        sev = 100 / (1 + np.exp(-o[v, 1])); deg = o[v, 2] > 0.5
        ax.scatter(t[v][~deg], sev[~deg], s=9, marker="o", facecolor="white", edgecolor="black", lw=0.6, zorder=4, label="visit, good image")
        ax.scatter(t[v][deg], sev[deg], s=10, marker="x", color="black", lw=0.7, zorder=4, label="visit, degraded image")
        ax.set_xlabel("Day of season"); ax.set_ylabel("Lesion area (%)"); ax.grid(lw=0.3, color="0.9")
        ax.set_title(f"({'ab'[i]}) Test plot {i+1}", fontsize=7)
    axs[0].legend(fontsize=5.2, loc="upper left")
    ax = axs[2]
    ms = [("persist", "Persistence"), ("linear", "Logit-linear"), ("gru_noenv", "GRU, no weather"), ("gru_noqual", "GRU, no quality"),
          ("gru", "GRU"), ("filter_noqual", "Filter, no quality"), ("filter_noenv", "Filter, no weather"), ("filter", "Filter (proposed)")]
    x = np.arange(len(ms)); w = 0.38
    for j, (h, fc) in enumerate([(7, "0.8"), (14, "0.3")]):
        vals = [np.mean([r["mae"] for r in T.get(f"{m}|te|{h}", [{"mae": np.nan}])]) for m, _ in ms]
        sds = [np.std([r["mae"] for r in T.get(f"{m}|te|{h}", [{"mae": np.nan}])]) for m, _ in ms]
        ax.bar(x + (j - 0.5) * w, vals, w, yerr=sds, color=fc, edgecolor="black", lw=0.5, capsize=1.2, error_kw=dict(lw=0.4), label=f"{h}-day horizon")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in ms], rotation=35, ha="right", fontsize=5.8)
    ax.set_ylabel("MAE (lesion-area points)"); ax.legend(fontsize=5.6, loc="upper right"); ax.grid(axis="y", lw=0.3, color="0.88")
    ax.set_title("(c) Forecast error, Champion 2011–2018", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_temporal.png", dpi=400); plt.close(fig)

def fig_maps():
    L = np.load(f"{MG}/results/localize_maps.npz"); idx = L["idx"]; gt = L["gt"]; mg = L["mg"].reshape(len(idx), 14, 14)
    cds = list(csv.DictReader(open(f"{MG}/index_cds.csv")))
    pick = []
    for lab in ["nlb", "gls", "nls"]:
        cand = [j for j, i in enumerate(idx) if cds[i]["label"] == lab and 0.08 < (gt[j] >= 0.25).mean() < 0.4]
        pick += cand[3:5]
    fig, axs = plt.subplots(2, len(pick), figsize=(7.0, 2.6))
    for c, j in enumerate(pick):
        im = Image.open(cds[idx[j]]["path"]).convert("L").resize((280, 280))
        for r in range(2):
            ax = axs[r, c]; ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
        txt = os.path.splitext(cds[idx[j]]["path"])[0] + ".txt"
        for l in open(txt):
            p = l.split()
            if len(p) < 5: continue
            _, cx, cy, w, h = map(float, p[:5])
            axs[0, c].add_patch(plt.Rectangle(((cx - w / 2) * 280, (cy - h / 2) * 280), w * 280, h * 280, fill=False, ec="white", lw=0.6))
            axs[0, c].add_patch(plt.Rectangle(((cx - w / 2) * 280, (cy - h / 2) * 280), w * 280, h * 280, fill=False, ec="black", lw=0.3, ls="--"))
        hm = (mg[j] - mg[j].min()) / (mg[j].max() - mg[j].min() + 1e-9)
        up = np.kron(hm, np.ones((20, 20)))
        axs[1, c].imshow(up, cmap="gray_r", alpha=0.6, vmin=0, vmax=1)
        axs[1, c].contour(up, levels=[0.6], colors="white", linewidths=1.2)
        axs[1, c].contour(up, levels=[0.6], colors="black", linewidths=0.5)
        axs[0, c].set_title({"nlb": "NLB", "gls": "GLS", "nls": "NLS (unseen)"}[cds[idx[j]]["label"]], fontsize=6.5)
    axs[0, 0].set_ylabel("expert boxes", fontsize=6.5); axs[1, 0].set_ylabel("lesion tokens $\\pi^{\\ell}$ (scaled)", fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.3); fig.savefig(f"{OUT}/fig_maps.png", dpi=400); plt.close(fig)

if __name__ == "__main__":
    for f in sys.argv[1:]: globals()[f"fig_{f}"]()
