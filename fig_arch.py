"""Fig. 1: MaizeGuard architecture (grayscale). Boxes and orthogonal arrows are placed on an explicit grid and
checked automatically for box overlap, label overflow and arrows passing through unrelated boxes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import itertools, sys
import matplotlib.patheffects as pe, json
EQ = json.load(open("eqnums.json"))
def E(a, b=None): return f"Eq. ({EQ[a]})" if b is None else f"Eqs. ({EQ[a]})–({EQ[b]})"

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"})
W, H = 7.16, 4.55
fig = plt.figure(figsize=(W, H), dpi=400)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
FS = 6.3
B = {}
STY = {"frozen": dict(fc="#d9d9d9", ec="black", lw=0.8, hatch=None),
       "train": dict(fc="white", ec="black", lw=1.1, hatch=None),
       "data": dict(fc="#f2f2f2", ec="black", lw=0.6, hatch=None),
       "out": dict(fc="#404040", ec="black", lw=0.8, hatch=None),
       "loss": dict(fc="white", ec="black", lw=0.9, hatch=None)}

def box(name, x, y, w, h, text, kind="train", fs=FS, dashed=False):
    s = STY[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.05", fc=s["fc"], ec=s["ec"], lw=s["lw"],
                       ls="--" if dashed else "-", zorder=3)
    ax.add_patch(p)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
                color="white" if kind == "out" else "black", linespacing=1.15)
    B[name] = dict(x=x, y=y, w=w, h=h, t=t)

ARROWS = []
def arrow(pts, dashed=False, src=None, dst=None, lw=0.8, head=True):
    """orthogonal polyline through pts drawn as ONE path (clean joints); arrow head at the last point"""
    ls = (0, (3, 2)) if dashed else "-"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if head:  # stop the line just short of the tip; the head covers the rest
        (ax0, ay0), (ax1, ay1) = pts[-2], pts[-1]
        dx, dy = ax1 - ax0, ay1 - ay0; n = (dx * dx + dy * dy) ** 0.5
        xs[-1] = ax1 - 0.05 * dx / n; ys[-1] = ay1 - 0.05 * dy / n
    ax.plot(xs, ys, color="black", lw=lw, ls=ls, zorder=2, solid_capstyle="butt", solid_joinstyle="miter",
            dash_joinstyle="miter", path_effects=[pe.Stroke(linewidth=lw + 2.6, foreground="white"), pe.Normal()])
    if head:
        ax.add_patch(FancyArrowPatch((xs[-1] - 0.03 * dx / n, ys[-1] - 0.03 * dy / n), pts[-1], arrowstyle="-|>",
                                     mutation_scale=6.5, color="black", lw=0.6, zorder=2.5, shrinkA=0, shrinkB=0))
    ARROWS.append((pts, src, dst))

def L(n): b = B[n]; return (b["x"], b["y"] + b["h"] / 2)
def R(n): b = B[n]; return (b["x"] + b["w"], b["y"] + b["h"] / 2)
def T(n, f=0.5): b = B[n]; return (b["x"] + f * b["w"], b["y"] + b["h"])
def Bo(n, f=0.5): b = B[n]; return (b["x"] + f * b["w"], b["y"])
def Ly(n, y): return (B[n]["x"], y)
def Ry(n, y): return (B[n]["x"] + B[n]["w"], y)

# ---------------- layout ----------------
X = [0.10, 1.55, 3.00, 4.45, 5.90]; WC = [1.20, 1.20, 1.20, 1.20, 1.16]
def col(c): return X[c], WC[c]
box("prompts", X[0], 3.86, WC[0], 0.50, "Symptom descriptors,\ntoken & quality prompts", "data")
box("tenc", X[1], 3.86, WC[1], 0.50, "Frozen CLIP\ntext encoder", "frozen")
box("bank", X[2], 3.86, WC[2], 0.50, "Text embeddings:\n$\\mathbf{d}_{km}$, $\\mathbf{P}$, $\\mathbf{Q}$", "data")
box("align", X[4], 3.86, WC[4], 0.50, "Descriptor alignment\n$s_{k}$, "+E("eq:align"), "train")
box("img", X[0], 2.88, WC[0], 0.50, "Field image $x$\n(any quality)", "data")
box("venc", X[1], 1.90, WC[1], 1.48, "Frozen CLIP\nViT-B/32\nimage encoder\n(no gradient)", "frozen")
box("gv", X[2], 2.88, WC[2], 0.50, "View embeddings\n$\\mathbf{g}_{v}$, $\\mathbf{r}_{g}$", "data")
box("qual", X[3], 3.02, WC[3], 0.40, "Quality probe\n$\\mathbf{q},\\psi$, "+E("eq:qual"), "train")
box("fuse", X[4], 2.62, WC[4], 0.80, "Quality-gated\nfusion $\\mathbf{z}$\n"+E("eq:gate","eq:fuse"), "train")
box("views", X[0], 1.90, WC[0], 0.58, "Multi-view input:\nglobal 224 + four\n224 tiles of 448", "data")
box("tj", X[2], 1.90, WC[2], 0.55, "Value-path tokens\n$\\hat{\\mathbf{t}}_{j}$, "+E("eq:tok"), "data")
box("type", X[3], 1.90, WC[3], 0.55, "Token typing\n$\\boldsymbol{\\pi}_{j}$ (uses $\\mathbf{P}$), "+E("eq:type"), "train")
box("ev", X[4], 1.90, WC[4], 0.55, "Lesion-evidence\nGaussians (fitted),\n"+E("eq:ev"), "data")
box("wx", X[1], 1.04, WC[1], 0.42, "Daily weather $\\mathbf{w}_{t}$", "data")
box("sev", X[2], 1.04, WC[2], 0.42, "Severity read-out\n$\\hat{s}$, "+E("eq:sev"), "train")
box("pool", X[3], 1.04, WC[3], 0.42, "Lesion pooling\n$\\mathbf{r}_{\\ell}$, "+E("eq:pool"), "train")
box("loss", X[0], 0.18, WC[0], 0.58, "Training loss $\\mathcal{L}$\n"+E("eq:ltok","eq:loss")+"\non $s_{k}$, $\\hat{s}$, $\\boldsymbol{\\pi}_{j}$", "loss", dashed=True)
box("filt", X[1], 0.18, WC[1], 0.58, "Weather-conditioned\nfilter, "+E("eq:trans","eq:cqr")+"\n$\\to$ forecast $C$", "train")
box("cal", X[2], 0.18, WC[2], 0.58, "Calibration set\n(strata by $\\psi$,\nclasses)", "data")
box("conf", X[3], 0.18, WC[3], 0.58, "Dual conformal\ntest $\\Gamma(x)$\n"+E("eq:ncs","eq:set"), "train")
box("dec", X[4], 0.18, WC[4], 0.58, "Decision $D(x)$\n"+E("eq:dec"), "out")

# ---------------- arrows (data flow) ----------------
arrow([R("prompts"), L("tenc")], src="prompts", dst="tenc")
arrow([R("tenc"), L("bank")], src="tenc", dst="bank")
arrow([R("bank"), L("align")], src="bank", dst="align")
arrow([Bo("img"), T("views")], src="img", dst="views")
arrow([R("views"), Ly("venc", 2.19)], src="views", dst="venc")
arrow([Ry("venc", 3.13), Ly("gv", 3.13)], src="venc", dst="gv")
arrow([Ry("venc", 2.17), Ly("tj", 2.17)], src="venc", dst="tj")
arrow([Ry("gv", 3.22), Ly("qual", 3.22)], src="gv", dst="qual")
arrow([Ry("gv", 2.96), Ly("fuse", 2.96)], src="gv", dst="fuse")
arrow([R("tj"), L("type")], src="tj", dst="type")
# probes P (typing) and Q (quality) from the text bank
arrow([Bo("bank", 0.80), (3.96, 3.60), (5.05, 3.60), T("qual")], src="bank", dst="qual")
# quality -> fusion
arrow([Ry("qual", 3.32), Ly("fuse", 3.32)], src="qual", dst="fuse")
# typing -> evidence, pooling, severity
arrow([Bo("type", 0.50), T("pool")], src="type", dst="pool")
arrow([Bo("type", 0.15), (4.63, 1.70), (3.60, 1.70), T("sev")], src="type", dst="sev")
# pooling -> fusion (lane between columns 4 and 5; hops over the typing->evidence line)
arrow([R("pool"), (5.78, 1.25), (5.78, 2.74), Ly("fuse", 2.74)], src="pool", dst="fuse")
arrow([(5.78, 2.10), Ly("ev", 2.10)], src="pool", dst="ev")
ax.plot([5.78], [2.10], "o", color="black", ms=2.2, zorder=5)
# fusion -> alignment, fusion -> evidence
arrow([T("fuse"), Bo("align")], src="fuse", dst="align")
# alignment -> conformal (right lane), evidence -> conformal
arrow([R("align"), (7.11, 4.11), (7.11, 0.86), (5.40, 0.86), T("conf", 0.79)], src="align", dst="conf")
arrow([Bo("ev", 0.35), (6.31, 0.95), (5.15, 0.95), (5.15, 0.76)], src="ev", dst="conf")
arrow([R("cal"), L("conf")], src="cal", dst="conf")
arrow([R("conf"), L("dec")], src="conf", dst="dec")
# severity, abstention and weather -> progression filter
arrow([L("sev"), (2.84, 1.25), (2.84, 0.60), Ry("filt", 0.60)], src="sev", dst="filt")
arrow([Bo("dec"), (6.48, 0.07), (2.92, 0.07), (2.92, 0.32), Ry("filt", 0.32)], src="dec", dst="filt")
arrow([Bo("wx"), T("filt")], src="wx", dst="filt")
# ---------------- reverse-mode gradient (dashed) ----------------
arrow([T("loss", 0.75), (1.00, 1.56), (4.33, 1.56), (4.33, 2.02), Ly("type", 2.02)], dashed=True, src="loss", dst="type")
ax.text(1.06, 1.60, "reverse-mode gradient $\\nabla\\mathcal{L}$ (trainable heads only)", fontsize=5.4, style="italic",
        ha="left", va="bottom", zorder=4)
# legend (bottom-left band)
lx, ly = 0.12, 1.40
for i, (k, lab) in enumerate([("frozen", "frozen"), ("train", "trainable head"), ("data", "data / tensor")]):
    s_ = STY[k]; yy = ly - i * 0.15
    ax.add_patch(FancyBboxPatch((lx, yy), 0.16, 0.10, boxstyle="round,pad=0,rounding_size=0.02", fc=s_["fc"], ec="black", lw=0.6, zorder=3))
    ax.text(lx + 0.22, yy + 0.05, lab, fontsize=5.4, va="center", zorder=4)
ax.plot([lx, lx + 0.16], [ly - 0.40, ly - 0.40], color="black", lw=0.8, ls=(0, (3, 2)))
ax.text(lx + 0.22, ly - 0.40, "gradient", fontsize=5.4, va="center")
LEG = dict(x=0.10, y=0.95, w=0.86, h=0.57)

# ---------------- automated checks ----------------
fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
for a, b in itertools.combinations(B, 2):
    A, C = B[a], B[b]
    if A["x"] < C["x"] + C["w"] and C["x"] < A["x"] + A["w"] and A["y"] < C["y"] + C["h"] and C["y"] < A["y"] + A["h"]:
        bad.append(f"overlap {a}-{b}")
inv = ax.transData.inverted()
for n, b in B.items():
    bb = b["t"].get_window_extent(r); (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    if x0 < b["x"] + 0.02 or x1 > b["x"] + b["w"] - 0.02 or y0 < b["y"] + 0.01 or y1 > b["y"] + b["h"] - 0.01:
        bad.append(f"text overflow {n}")
def seg_hits(p, q, b, pad=0.005):
    x0, x1 = sorted([p[0], q[0]]); y0, y1 = sorted([p[1], q[1]])
    return x0 < b["x"] + b["w"] - pad and x1 > b["x"] + pad and y0 < b["y"] + b["h"] - pad and y1 > b["y"] + pad
for pts, s, d in ARROWS:
    for p, q in zip(pts[:-1], pts[1:]):
        for n, b in B.items():
            if n in (s, d): continue
            if seg_hits(p, q, b): bad.append(f"arrow {s}->{d} crosses {n}")
for n, b in B.items():
    if seg_hits((LEG["x"], LEG["y"]), (LEG["x"] + LEG["w"], LEG["y"] + LEG["h"]), b): bad.append(f"legend overlaps {n}")
print("CHECKS:", "OK" if not bad else bad)
out = sys.argv[1] if len(sys.argv) > 1 else "fig_arch.png"
fig.savefig(out, dpi=400)
