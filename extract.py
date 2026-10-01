"""Extract frozen CLIP features for an image list. Usage: python extract.py <list.txt> <out.npz>
Each line of list: path. Stores global embeddings for 5 views and dense tokens (fp16)."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from clipfeat import CLIP, load_rgb, views
from corrupt import apply_corruption

def main(lst, out, corr=None, level=0):
    paths = [l.strip() for l in open(lst) if l.strip()]
    c = CLIP(threads=2)
    N = len(paths)
    G = np.zeros((N, 5, 512), np.float32); D = np.zeros((N, 5, 49, 512), np.float16)
    t0 = time.time(); B = 8
    for s in range(0, N, B):
        ims = []
        for p in paths[s:s + B]:
            im = load_rgb(p)
            if corr: im = apply_corruption(im, corr, level, seed=hash(p) % (2**31))
            ims.append(views(im))
        X = np.concatenate(ims)
        g, d = c.encode_batch(X)
        n = len(ims)
        G[s:s + n] = g.reshape(n, 5, 512); D[s:s + n] = d.reshape(n, 5, 49, 512).astype(np.float16)
        if (s // B) % 25 == 0:
            print(f"{out} {s+n}/{N} {time.time()-t0:.0f}s", flush=True)
    np.savez(out, G=G, D=D, paths=np.array(paths))
    print("done", out, time.time() - t0, flush=True)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3] if len(a) > 3 else None, int(a[4]) if len(a) > 4 else 0)
