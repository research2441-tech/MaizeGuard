"""Token-level pseudo targets from PlantVillage segmented leaves: leaf mask and colour-defined lesion mask,
mapped onto the same 5 views (global 224 crop + 2x2 tiles of 448 crop) as the CLIP features, pooled to 7x7."""
import numpy as np, csv
from PIL import Image
def rc_arr(a, s):
    im = Image.fromarray(a); w, h = im.size; r = s / min(w, h)
    im = im.resize((max(s, round(w * r)), max(s, round(h * r))), Image.BILINEAR)
    w, h = im.size; l = (w - s) // 2; t = (h - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)), np.float32) / 255.
def masks(color_path, seg_path):
    seg = np.asarray(Image.open(seg_path).convert("RGB"), np.float32) / 255.
    leaf = (seg.max(-1) > 0.08)
    hsv = np.asarray(Image.open(color_path).convert("HSV"), np.float32) / 255.
    H, S, V = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    green = (H >= 50) & (H <= 170) & (S >= 0.10)
    specular = (S < 0.14) & (V > 0.70)
    lesion = leaf & ~green & ~specular & (V > 0.12)
    return leaf.astype(np.uint8) * 255, lesion.astype(np.uint8) * 255
def pool7(m):  # 224x224 -> 7x7 fraction
    return m.reshape(7, 32, 7, 32).mean((1, 3))
def views_targets(color_path, seg_path):
    leaf, les = masks(color_path, seg_path)
    out = []
    for a in (leaf, les):
        g = rc_arr(a, 224); b = rc_arr(a, 448)
        vs = [g, b[:224, :224], b[:224, 224:], b[224:, :224], b[224:, 224:]]
        out.append(np.stack([pool7(v) for v in vs]))
    leafp, lesp = out
    sev = les.sum() / max(1, leaf.sum())
    return leafp.astype(np.float32), lesp.astype(np.float32), float(sev)
if __name__ == "__main__":
    rows = list(csv.DictReader(open("index_pv_pd.csv")))
    import glob, os
    segs = {}
    for f in glob.glob("./data/PlantVillage-Dataset/raw/segmented/*/*"):
        k = os.path.basename(f).split("___")[-1].replace("_final_masked", "").rsplit(".", 1)[0]
        segs[(os.path.basename(os.path.dirname(f)), k)] = f
    for r in rows:
        if r["ds"] == "pv":
            k = os.path.basename(r["path"]).split("___")[-1].rsplit(".", 1)[0]
            r["seg"] = segs.get((os.path.basename(os.path.dirname(r["path"])), k), "")
    L = np.zeros((len(rows), 5, 7, 7), np.float32); E = np.zeros_like(L); S = np.full(len(rows), np.nan, np.float32)
    for i, r in enumerate(rows):
        if r["ds"] == "pv" and r["seg"]:
            L[i], E[i], S[i] = views_targets(r["path"], r["seg"])
    np.savez("feats/pv_token_targets.npz", leaf=L, lesion=E, sev=S)
    import collections
    d = collections.defaultdict(list)
    for r, s in zip(rows, S):
        if r["ds"] == "pv": d[r["label"]].append(s)
    for k, v in d.items():
        v = np.array(v); print(k, len(v), np.nanmean(v).round(3), np.nanpercentile(v, [5, 50, 95]).round(3))
