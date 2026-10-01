"""Sequential, resumable feature-extraction jobs (one process; 2 BLAS threads)."""
import os, sys, time, zlib
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from clipfeat import CLIP, load_rgb, views
from corrupt import apply_corruption
TRAIN_KINDS = ["blur", "dark", "jpeg"]
def run(paths, out, corr_fn, c):
    if os.path.exists(out): print("skip", out, flush=True); return
    N = len(paths); G = np.zeros((N, 5, 512), np.float32); D = np.zeros((N, 5, 49, 512), np.float16)
    meta = []; t0 = time.time(); B = 8
    for s in range(0, N, B):
        ims = []
        for p in paths[s:s + B]:
            im = load_rgb(p); kind, lev = corr_fn(p)
            if kind: im = apply_corruption(im, kind, lev, seed=zlib.crc32(p.encode()))
            meta.append(f"{kind}:{lev}"); ims.append(views(im))
        g, d = c.encode_batch(np.concatenate(ims)); n = len(ims)
        G[s:s + n] = g.reshape(n, 5, 512); D[s:s + n] = d.reshape(n, 5, 49, 512).astype(np.float16)
        if (s // B) % 50 == 0: print(out, s + n, N, f"{time.time()-t0:.0f}s", flush=True)
    np.savez(out + ".tmp.npz", G=G, D=D, paths=np.array(paths), meta=np.array(meta)); os.replace(out + ".tmp.npz", out)
    print("done", out, f"{time.time()-t0:.0f}s", flush=True)
if __name__ == "__main__":
    c = CLIP(threads=2)
    jobs = sys.argv[1:]
    for job in jobs:
        if job == "nontest_aug":
            paths = [l.strip() for l in open("list_nontest.txt") if l.strip()]
            def fn(p):
                r = np.random.default_rng(zlib.crc32(p.encode()))
                return TRAIN_KINDS[r.integers(3)], int(r.integers(1, 4))
            run(paths, "feats/nontest_aug.npz", fn, c)
        elif job.startswith("test_"):  # test_<kind>_<level>
            _, kind, lev = job.split("_")
            paths = [l.strip() for l in open("list_test.txt") if l.strip()]
            run(paths, f"feats/test_{kind}_{lev}.npz", lambda p: (kind, int(lev)), c)
        elif job.startswith("list:"):  # list:<listfile>:<out>[:kind:level]
            parts = job.split(":"); paths = [l.strip() for l in open(parts[1]) if l.strip()]
            kind = parts[3] if len(parts) > 3 else None; lev = int(parts[4]) if len(parts) > 4 else 0
            run(paths, parts[2], lambda p: (kind, lev), c)
