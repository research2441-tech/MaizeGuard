"""Result figures (grayscale, no overlapping legends). Reads results/*.json."""
import json, os, sys, numpy as np, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                     "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
MG = "."; OUT = "."
import glob
V = {}
for _f in sorted(glob.glob(f"{MG}/results/vision*_s*.json")): V.update(json.load(open(_f)))
NAMES = {"zs": "Zero-shot CLIP", "zsdesc": "Zero-shot descr.", "lp": "Linear probe", "clipad": "CLIP-Adapter",
         "tipf": "Tip-Adapter-F", "abmil": "ABMIL tokens", "mg": "MaizeGuard"}
STY = {"zs": dict(color="0.65", ls=":", marker="v"), "zsdesc": dict(color="0.55", ls=":", marker="^"),
       "lp": dict(color="0.35", ls="--", marker="s"), "clipad": dict(color="0.45", ls="-.", marker="D"),
       "tipf": dict(color="0.55", ls="--", marker="o"), "abmil": dict(color="0.25", ls=(0, (1, 1)), marker="x"),
       "mg": dict(color="0.0", ls="-", marker="o", lw=1.6)}
def seeds(m): return [k for k in V if k.split("|")[0] == m]
def get(m, path, agg=np.mean):
    vals = []
    for k in seeds(m):
        x = V[k]
        try:
            for p in path: x = x[p]
            vals.append(float(x))
        except (KeyError, TypeError): pass
    return (agg(vals), np.std(vals), len(vals)) if vals else (np.nan, np.nan, 0)

def fig_robust():
    kinds = ["blur", "jpeg", "dark", "motion", "noise", "glare"]
    titles = {"blur": "Gaussian blur", "jpeg": "JPEG", "dark": "Low light", "motion": "Motion blur (held out)",
              "noise": "Sensor noise (held out)", "glare": "Glare (held out)"}
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.6), sharey=True)
    ms = ["lp", "clipad", "tipf", "abmil", "mg"]
    for ax, k in zip(axs.ravel(), kinds):
        for m in ms:
            xs, ys, es = [0], [get(m, ["test_pv", "f1"])[0] * 0.0], []
            # level 0 = clean fixed test (PV+PD pooled macro-F1 recomputed in analysis as robust["clean"])
            lv = [l for l in [1, 3] if not np.isnan(get(m, ["robust", f"{k}_{l}", "f1"])[0])]
            c0 = get(m, ["robust_clean", "f1"])[0]
            x = ([0] if not np.isnan(c0) else []) + lv
            y = ([c0] if not np.isnan(c0) else []) + [get(m, ["robust", f"{k}_{l}", "f1"])[0] for l in lv]
            e = ([get(m, ["robust_clean", "f1"])[1]] if not np.isnan(c0) else []) + [get(m, ["robust", f"{k}_{l}", "f1"])[1] for l in lv]
            st = STY[m]
            ax.errorbar(x, y, yerr=e, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.2, lw=st.get("lw", 1.0),
                        capsize=1.5, elinewidth=0.5, label=NAMES[m])
        ax.set_title(titles[k], fontsize=7); ax.set_xticks([0, 1, 3]); ax.set_xticklabels(["clean", "mild (1)", "strong (3)"])
        ax.grid(axis="y", lw=0.3, color="0.85")
    for ax in axs[:, 0]: ax.set_ylabel("Macro-F1 (PV+PD test)")
    for ax in axs[1]: ax.set_xlabel("Corruption strength")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/fig_robust.png", dpi=400); plt.close(fig)


def _bar(ax, labels, means, sds, hatches=None, ylabel="", ylim=None, fmt="{:.3f}"):
    x = np.arange(len(labels)); greys = ["0.85", "0.65", "0.45", "0.25", "0.10", "0.55", "0.75", "0.35"]
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.bar(x[i], m, 0.68, yerr=s, color=greys[i % len(greys)], edgecolor="black", lw=0.5, capsize=1.8,
               error_kw=dict(lw=0.5), hatch=(hatches[i] if hatches else None))
        ax.text(x[i], m + (s if not np.isnan(s) else 0) + 0.012, fmt.format(m), ha="center", va="bottom", fontsize=5.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis="y", lw=0.3, color="0.85"); ax.set_axisbelow(True)

def fig_ood():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw=dict(width_ratios=[1.35, 1]))
    rows = [("lp", "maxlogit", "LP MaxLogit"), ("clipad", "maxlogit", "CLIP-Ad. MaxLogit"), ("tipf", "energy", "Tip-F energy"),
            ("zs", "mcm", "MCM (zero-shot)"), ("lp", "knn", "LP kNN"), ("lp", "mahalanobis", "LP Mahalanobis"),
            ("mg", "maxlogit", "MaizeGuard MaxLogit"), ("mg", "mahalanobis", "MaizeGuard Mahal.")]
    m_ = [get(a, ["ood", b, "auroc"]) for a, b, _ in rows]
    _bar(axs[0], [r[2] for r in rows], [x[0] for x in m_], [x[1] for x in m_], ylabel="AUROC (NLS vs. known)", ylim=(0, 1.08))
    axs[0].set_title("(a) No target labels", fontsize=7)
    rows2 = [("lp", "tgt_maha_z", "LP $\\mathbf{r}_g$"), ("mg", "tgt_maha_rg", "MG $\\mathbf{r}_g$"), ("mg", "tgt_maha_ev", "MG descr. profile"),
             ("mg", "tgt_maha_z", "MG $\\mathbf{z}$"), ("mg", "tgt_maha_rl", "MG $\\mathbf{r}_{\\ell}$")]
    m2 = [get(a, ["ood_target", b, "auroc"]) for a, b, _ in rows2]
    _bar(axs[1], [r[2] for r in rows2], [x[0] for x in m2], [x[1] for x in m2], ylabel="AUROC (NLS vs. known)", ylim=(0, 1.08))
    axs[1].set_title("(b) Gaussians refitted on 255 labelled field images", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ood.png", dpi=400); plt.close(fig)

def fig_conformal():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axs[0]
    pts = [("lp", "src_lac", "LP LAC (source cal.)", "s"), ("mg", "src_lac", "MG LAC (source cal.)", "o"),
           ("mg", "src_dual_rl", "MG dual (source cal.)", "D"), ("lp", "tgt_lac", "LP LAC (target cal.)", "s"),
           ("lp", "tgt_aps", "LP APS (target cal.)", "^"), ("mg", "tgt_lac", "MG LAC (target cal.)", "o"),
           ("lp", "tgt_dual_z", "LP + evidence test", "P"), ("mg", "tgt_dual_rg", "MG dual, global evid.", "X"),
           ("mg", "tgt_dual_rl", "MG dual (proposed)", "*")]
    fills = ["white", "white", "white", "0.6", "0.6", "0.6", "0.3", "0.3", "black"]
    for (m, k, lab, mk), fc in zip(pts, fills):
        c = get(m, ["conformal", k, "cov"]); f = get(m, ["conformal", k, "nls_flag"])
        ax.errorbar(c[0], f[0], xerr=c[1], yerr=f[1], fmt=mk, ms=5 if mk != "*" else 8, mfc=fc, mec="black", ecolor="0.4",
                    elinewidth=0.5, capsize=1.5, label=lab, mew=0.6)
    ax.axvline(0.90, color="0.5", lw=0.6, ls="--"); ax.text(0.897, 0.74, "nominal 0.90", fontsize=5.5, ha="right", va="top", color="0.3")
    ax.set_xlabel("Coverage of known diseases (CD&S test)"); ax.set_ylabel("NLS images flagged (empty set)")
    ax.set_title("(a) Coverage vs. unknown-disease rejection", fontsize=7); ax.grid(lw=0.3, color="0.88")
    ax.legend(fontsize=5.2, loc="upper left", bbox_to_anchor=(1.0, 1.02), handletextpad=0.3, labelspacing=0.35)
    ax = axs[1]; ns = [20, 40, 80, 160, 320]
    for off, (fn, lab, st) in zip([0.93, 1.0, 1.07], [("rl", "MG $\\mathbf{r}_{\\ell}$ evidence", dict(color="black", ls="-", marker="o")),
                        ("z", "MG $\\mathbf{z}$ evidence", dict(color="0.4", ls="--", marker="s")),
                        ("rg", "MG $\\mathbf{r}_{g}$ evidence", dict(color="0.6", ls=":", marker="^"))]):
        f = [get("mg", ["nsweep", str(n), fn, "nls_flag"]) for n in ns]; c = [get("mg", ["nsweep", str(n), fn, "cov"]) for n in ns]
        ax.errorbar([n * off for n in ns], [x[0] for x in f], yerr=[x[1] for x in f], capsize=1.5, lw=1, ms=3.5, label=lab + " (flag)", **st)
        ax.plot([n * off for n in ns], [x[0] for x in c], color=st["color"], ls=st["ls"], lw=0.6, marker=st["marker"], ms=2.5, mfc="white")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Labelled field images of known diseases"); ax.set_ylabel("Rate")
    ax.text(21, 0.88, "coverage (open markers)", fontsize=5.5, ha="left", va="top")
    ax.set_title("(b) Few-label target calibration", fontsize=7); ax.grid(lw=0.3, color="0.88"); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=5.4, loc="center left", bbox_to_anchor=(0.0, 0.55))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_conformal.png", dpi=400); plt.close(fig)

if __name__ == "__main__":
    for f in sys.argv[1:]: globals()[f"fig_{f}"]()
