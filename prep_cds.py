import os, glob, csv, numpy as np
from PIL import Image
from scipy.fft import dctn
R = "./data/cds/cds"
def phash(p):
    a = np.asarray(Image.open(p).convert("L").resize((32, 32), Image.BILINEAR), np.float32)
    d = dctn(a, norm="ortho")[:8, :8]; return (d > np.median(d)).flatten()
rows = []
for f in sorted(glob.glob(f"{R}/orig/*/*/*")):
    sp, lab = f.split("/")[-3], f.split("/")[-2]
    rows.append(dict(ds="cds", path=f, label=lab, split=sp, sev=""))
sevmap = dict(one=1, two=2, three=3, four=4, five=5)
for f in sorted(glob.glob(f"{R}/sev/*/*")):
    rows.append(dict(ds="cds_sev", path=f, label="", split="sev", sev=sevmap[f.split("/")[-2]]))
for f in sorted(glob.glob(f"{R}/ann/*/*.jpg")) + sorted(glob.glob(f"{R}/ann/*/*.JPG")):
    rows.append(dict(ds="cds_ann", path=f, label=f.split("/")[-2], split="ann", sev=""))
H = np.stack([phash(r["path"]) for r in rows]).astype(np.int16)
# duplicate check across subsets
def dups(a, b, thr=6):
    ia = [i for i, r in enumerate(rows) if a(r)]; ib = [i for i, r in enumerate(rows) if b(r)]
    if not ia or not ib: return 0, []
    Hb = H[ib]; c = 0; pairs = []
    for i in ia:
        d = (Hb != H[i]).sum(1); j = np.argmin(d)
        if d[j] <= thr: c += 1; pairs.append((i, ib[j]))
    return c, pairs
print("n", len(rows))
c, _ = dups(lambda r: r["split"] == "test", lambda r: r["split"] == "train"); print("orig test~train dups", c)
c, p = dups(lambda r: r["ds"] == "cds_sev", lambda r: r["ds"] == "cds"); print("sev~orig dups", c)
import collections
print(collections.Counter(rows[j]["label"] for i, j in p))
c, p = dups(lambda r: r["ds"] == "cds_ann", lambda r: r["ds"] == "cds"); print("ann~orig dups", c, collections.Counter(rows[j]["split"] for i, j in p))
np.save("feats/cds_phash.npy", H)
with open("index_cds.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
