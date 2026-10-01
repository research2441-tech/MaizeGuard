import os, glob, csv, numpy as np
from PIL import Image
PV = "./data/PlantVillage-Dataset/raw"
PD = "./data/PlantDoc-Dataset"
pvmap = {"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "gls", "Corn_(maize)___Common_rust_": "rust",
         "Corn_(maize)___Northern_Leaf_Blight": "nlb", "Corn_(maize)___healthy": "healthy"}
pdmap = {"Corn Gray leaf spot": "gls", "Corn rust leaf": "rust", "Corn leaf blight": "nlb"}
def phash(p):
    im = Image.open(p).convert("L").resize((32, 32), Image.BILINEAR)
    a = np.asarray(im, np.float32)
    from scipy.fft import dctn
    d = dctn(a, norm="ortho")[:8, :8]; return (d > np.median(d)).flatten()
rows = []
for d, lab in pvmap.items():
    for f in sorted(glob.glob(f"{PV}/color/{d}/*")):
        seg = f.replace("/color/", "/segmented/"); base = os.path.splitext(seg)[0]
        segc = [x for x in glob.glob(base + "*")]
        rows.append(dict(ds="pv", path=f, seg=segc[0] if segc else "", label=lab, split=""))
for sp in ["train", "test"]:
    for d, lab in pdmap.items():
        for f in sorted(glob.glob(f"{PD}/{sp}/{d}/*")):
            rows.append(dict(ds="pd", path=f, seg="", label=lab, split=sp))
H = np.stack([phash(r["path"]) for r in rows])
# near-duplicate grouping (hamming <= 6) by union-find
n = len(rows); par = list(range(n))
def f(i):
    while par[i] != i: par[i] = par[par[i]]; i = par[i]
    return i
Hi = H.astype(np.int16)
for i in range(n):
    dist = (Hi[i + 1:] != Hi[i]).sum(1)
    for j in np.nonzero(dist <= 6)[0]:
        a, b = f(i), f(i + 1 + j)
        if a != b: par[a] = b
for i, r in enumerate(rows): r["group"] = f(i)
cross = sum(1 for g in set(r["group"] for r in rows) if len({r["ds"] for r in rows if r["group"] == g}) > 1)
print("rows", n, "groups", len(set(r['group'] for r in rows)), "cross-dataset groups", cross)
with open("index_pv_pd.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
