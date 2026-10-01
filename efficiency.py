"""E7: parameters, head latency, encoder latency (single image, CPU, 1 thread)."""
import os, time, json, glob, numpy as np, torch
os.environ["OMP_NUM_THREADS"] = "1"; torch.set_num_threads(1)
from predict_ext import load_model
from clipfeat import CLIP
out = {}
z = np.load("feats/cds_clean.npz"); G = torch.from_numpy(z["G"][:1]); D = torch.from_numpy(z["D"][:1].reshape(1, 245, 512))
for m in ["lp", "clipad", "tipf", "abmil", "mg"]:
    model = load_model(f"runs/{m}_s0.pt")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        for _ in range(5): model(G.float(), D)
        t = time.perf_counter()
        for _ in range(200): model(G.float(), D)
    out[m] = dict(params=int(n), head_ms=1000 * (time.perf_counter() - t) / 200)
c = CLIP(threads=1); x = np.random.rand(5, 224, 224, 3).astype(np.float32)
c.encode_batch(x[:1]); t = time.perf_counter()
for _ in range(5): c.encode_batch(x)
out["encoder_ms_5views"] = 1000 * (time.perf_counter() - t) / 5
t = time.perf_counter()
for _ in range(5): c.encode_batch(x[:1])
out["encoder_ms_1view"] = 1000 * (time.perf_counter() - t) / 5
json.dump(out, open("results/efficiency.json", "w"), indent=1); print(out)
