"""Run saved models on external/corrupted feature files. Usage: python predict_ext.py <feat.npz> <tag> [methods] [seeds]"""
import os, sys, glob
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
import run_main as R
from featstore import Store
def load_model(path, S=None):
    ck = torch.load(path, weights_only=False)
    m = ck["method"]
    if m == "tipf":
        st = ck["state"]; model = R.TipAdapterF(R.TEXT["zs"], st["keys"].numpy(), st["V"].numpy())
    else:
        model = R.make(m, S, None)
    model.load_state_dict(ck["state"]); model.eval(); return model
if __name__ == "__main__":
    feat, tag = sys.argv[1], sys.argv[2]
    z = np.load(feat); G = torch.from_numpy(z["G"]); D = torch.from_numpy(z["D"].reshape(len(G), 245, 512))
    ck_paths = sorted(glob.glob("runs/*_s*.pt"))
    if len(sys.argv) > 3 and sys.argv[3] != "all":
        ms = sys.argv[3].split(","); ck_paths = [p for p in ck_paths if os.path.basename(p).rsplit("_s", 1)[0] in ms]
    for p in ck_paths:
        out = p.replace(".pt", f"_{tag}.npz")
        if os.path.exists(out): continue
        model = load_model(p)
        pr = R.predict(model, G, D)
        np.savez_compressed(out, **pr)
    print("done", tag, len(ck_paths))
