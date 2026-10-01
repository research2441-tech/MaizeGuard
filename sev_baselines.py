"""Severity baselines on CD&S expert grades: colour rule, zero-shot token typing, global-embedding ridge."""
import csv, json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import Ridge
from splits import load_index, seed_split
import run_main as R

cds = list(csv.DictReader(open("index_cds.csv"))); sv = [i for i, r in enumerate(cds) if r["ds"] == "cds_sev"]
g = np.array([int(cds[i]["sev"]) for i in sv])
def colour_rule(path):
    hsv = np.asarray(Image.open(path).convert("HSV").resize((224, 224)), np.float32) / 255.
    H, S, V = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    green = (H >= 50) & (H <= 170) & (S >= 0.10); spec = (S < 0.14) & (V > 0.70)
    les = ~green & ~spec & (V > 0.12) & (S > 0.12) & ((H < 50) | (H > 330))  # tan/brown hues only (excludes soil-free sky)
    return les.sum() / max(1, (les | green).sum())
res = {}
cr = np.array([colour_rule(cds[i]["path"]) for i in sv])
res["colour_rule"] = dict(spearman=spearmanr(cr, g)[0], kendall=kendalltau(cr, g)[0], means=[float(cr[g == k].mean()) for k in range(1, 6)])
z = np.load("feats/cds_clean.npz"); D = torch.from_numpy(z["D"][sv].reshape(len(sv), 245, 512)); G = z["G"][sv]
t = F.normalize(D.float(), dim=-1); P = torch.tensor(R.TEXT["tok"], dtype=torch.float32)
pi = (30 * t @ P.T).softmax(-1)[:, 49:]
zs = (pi[..., 0].sum(1) / (pi[..., 0].sum(1) + pi[..., 1].sum(1))).numpy()
res["zeroshot_tokens"] = dict(spearman=spearmanr(zs, g)[0], kendall=kendalltau(zs, g)[0], means=[float(zs[g == k].mean()) for k in range(1, 6)])
rows = load_index(); pv = np.load("feats/pvpd_clean.npz")["G"]; st = np.load("feats/pv_token_targets.npz")["sev"]
nrm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
rg = nrm(nrm(pv).mean(1)); rgc = nrm(nrm(G).mean(1))
out = []
for seed in range(5):
    tr = seed_split(rows, seed)["train"]; tr = tr[~np.isnan(st[tr])]
    m = Ridge(alpha=1.0).fit(rg[tr], st[tr]); p = m.predict(rgc); out.append((spearmanr(p, g)[0], kendalltau(p, g)[0]))
res["ridge_global"] = dict(spearman=float(np.mean([o[0] for o in out])), spearman_sd=float(np.std([o[0] for o in out])),
                           kendall=float(np.mean([o[1] for o in out])))
json.dump(res, open("results/sev_baselines.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
