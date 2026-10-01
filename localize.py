"""Lesion-token localisation against CD&S lesion boxes (E5b) and token maps for the qualitative figure."""
import os, csv, json, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from predict_ext import load_model
import run_main as R

def box_grid(txt, n=14):
    """fraction of each of the n x n cells (over the central square = whole square image) covered by lesion boxes"""
    m = np.zeros((280, 280), bool)
    for l in open(txt):
        p = l.split()
        if len(p) < 5: continue
        _, cx, cy, w, h = map(float, p[:5])
        x0, x1 = int((cx - w / 2) * 280), int(np.ceil((cx + w / 2) * 280)); y0, y1 = int((cy - h / 2) * 280), int(np.ceil((cy + h / 2) * 280))
        m[max(0, y0):y1, max(0, x0):x1] = True
    return m.reshape(n, 280 // n, n, 280 // n).mean((1, 3))

def tiles_to_grid(v):  # v: (196,) ordered tile0..3 each 7x7 row-major -> 14x14
    t = v.reshape(4, 7, 7); g = np.zeros((14, 14))
    g[:7, :7], g[:7, 7:], g[7:, :7], g[7:, 7:] = t
    return g

if __name__ == "__main__":
    cds = list(csv.DictReader(open("index_cds.csv")))
    ann = [i for i, r in enumerate(cds) if r["ds"] == "cds_ann" and os.path.exists(os.path.splitext(r["path"])[0] + ".txt")]
    z = np.load("feats/cds_clean.npz"); G = torch.from_numpy(z["G"][ann]); D = torch.from_numpy(z["D"][ann].reshape(len(ann), 245, 512))
    gt = np.stack([box_grid(os.path.splitext(cds[i]["path"])[0] + ".txt") for i in ann])
    lab = (gt >= 0.25).reshape(len(ann), -1)
    TEXT = R.TEXT; P = torch.tensor(TEXT["tok"], dtype=torch.float32)
    t = F.normalize(D.float(), dim=-1)
    out = {}
    zs = (30 * t @ P.T).softmax(-1)[..., 0][:, 49:].numpy()
    def score(name, s):
        g = np.stack([tiles_to_grid(x) for x in s]).reshape(len(ann), -1)
        out[name] = dict(auroc=roc_auc_score(lab.ravel(), g.ravel()), ap=average_precision_score(lab.ravel(), g.ravel()),
                         img_auroc=float(np.nanmean([roc_auc_score(l, x) for l, x in zip(lab, g) if 0 < l.mean() < 1])))
        return g
    maps = {"zeroshot": score("zeroshot", zs)}
    res = {}
    for seed in range(5):
        for m in ["mg", "abmil", "mg_notok"]:
            p = f"runs/{m}_s{seed}.pt"
            if not os.path.exists(p): continue
            model = load_model(p); model.eval()
            with torch.no_grad():
                if m == "abmil":
                    o = model(G.float(), D); s = o["att"][:, 49:].numpy()
                else:
                    o = model(G.float(), D); s = o["pi"][..., 0][:, 49:].numpy()
            g = score(f"{m}|{seed}", s)
            if seed == 0: maps[m] = g
        res = out
    json.dump(res, open("results/localize.json", "w"), indent=1, default=float)
    np.savez("results/localize_maps.npz", idx=np.array(ann), gt=gt, **{k: v for k, v in maps.items()})
    print(json.dumps({k: v for k, v in out.items() if "|0" in k or k == "zeroshot"}, indent=1, default=float))
