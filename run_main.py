"""Train MaizeGuard + baselines for one seed; save weights and predictions on in-memory evaluation sets."""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
from featstore import Store
from splits import seed_split
from mgmodel import MaizeGuard, LinearProbe, ClipAdapter, TipAdapterF, ABMIL
from sklearn.metrics import f1_score
TEXT = dict(np.load("feats/text.npz", allow_pickle=True))

def make(method, S, tr):
    if method == "mg": return MaizeGuard(TEXT)
    if method == "mg_noles": return MaizeGuard(TEXT, use_lesion=False)
    if method == "mg_nodesc": return MaizeGuard(TEXT, use_desc=False)
    if method == "mg_noqual": return MaizeGuard(TEXT, use_quality=False)
    if method == "mg_adapter": return MaizeGuard(TEXT, adapter=True)
    if method in ("lp", "zs", "zsdesc"): return LinearProbe()
    if method == "lpzs": return LinearProbe(init=TEXT["zs"])
    if method == "clipad": return ClipAdapter(TEXT["zs"])
    if method == "tipf":
        rg = F.normalize(F.normalize(S.G[tr].float(), dim=-1).mean(1), dim=-1).numpy()
        return TipAdapterF(TEXT["zs"], rg, np.eye(4)[S.y[tr].numpy()])
    if method == "abmil": return ABMIL()
    raise ValueError(method)

def predict(model, G, D, bs=256):
    model.eval(); out = {}
    with torch.no_grad():
        for s in range(0, len(G), bs):
            o = model(G[s:s + bs].float(), D[s:s + bs])
            for k, v in o.items():
                if k in ("pi", "att"): continue
                out.setdefault(k, []).append(v.float())
    return {k: torch.cat(v).numpy() for k, v in out.items()}

def fixed_linear(W):
    m = LinearProbe(); 
    with torch.no_grad(): m.fc.weight.copy_(torch.tensor(W, dtype=torch.float32)); m.fc.bias.zero_()
    return m

def train_lp(S, sp, seed, sel="pd"):
    from sklearn.linear_model import LogisticRegression
    tr = torch.tensor(sp["train"]); va = torch.tensor(sp["val"])
    G0, _ = S.batch(tr, need_D=False); G1, _ = S.batch(tr, torch.ones(len(tr), dtype=torch.bool), need_D=False)
    f = lambda G: F.normalize(F.normalize(G, dim=-1).mean(1), dim=-1).numpy()
    X = np.concatenate([f(G0), f(G1)]); y = np.concatenate([S.y[tr].numpy()] * 2)
    if sel == "pd": va = torch.tensor([i for i in sp["val"] if S.ds[i] == "pd"])
    Gv, _ = S.batch(va, need_D=False); Xv = f(Gv); yv = S.y[va].numpy()
    best = (-1, None)
    for C in [0.1, 0.3, 1, 3, 10, 30, 100]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)
        f1 = f1_score(yv, clf.predict(Xv), labels=sorted(set(yv.tolist())), average="macro")
        if f1 > best[0]: best = (f1, clf)
    m = LinearProbe()
    with torch.no_grad():
        m.fc.weight.copy_(torch.tensor(best[1].coef_, dtype=torch.float32)); m.fc.bias.copy_(torch.tensor(best[1].intercept_, dtype=torch.float32))
    return m, best[0]

def train(method, S, sp, seed, epochs=40, bs=64, aug=True, cons=True, tok=True, sel="pd"):
    torch.manual_seed(seed); np.random.seed(seed)
    if method == "lp": return train_lp(S, sp, seed, sel)
    if method in ("zs", "zsdesc"):
        W = TEXT["zs"] if method == "zs" else F.normalize(torch.tensor(TEXT["desc"]).mean(1), dim=-1).numpy()
        m = fixed_linear(100 * W); va = torch.tensor([i for i in sp["val"] if S.ds[i] == "pd"]); Gv, Dv = S.batch(va)
        yv = S.y[va].numpy(); return m, f1_score(yv, predict(m, Gv, Dv)["logits"].argmax(1), labels=sorted(set(yv.tolist())), average="macro")
    tr = torch.tensor(sp["train"]); va = torch.tensor(sp["val"])
    model = make(method, S, tr.numpy())
    is_mg = method.startswith("mg")
    if is_mg:
        cnt = torch.bincount(S.y[tr], minlength=4).float(); model.log_prior.copy_(torch.log(cnt / cnt.sum()))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3 if is_mg else 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = -1, None, 0
    if sel == "pd": va = torch.tensor([i for i in sp["val"] if S.ds[i] == "pd"])
    Gv, Dv = S.batch(va); yv = S.y[va].numpy()
    for ep in range(epochs):
        model.train(); perm = tr[torch.randperm(len(tr))]
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]; y = S.y[idx]
            cm = (torch.rand(len(idx)) < 0.5) if aug else None
            G, D = S.batch(idx, cm)
            o = model(G, D)
            lg = o["logits"] + (model.log_prior if is_mg else 0)
            loss = F.cross_entropy(lg, y)
            if is_mg:
                if tok:
                    ht = S.has_tok[idx]
                    if ht.any():
                        tt = S.tok_t[idx][ht]
                        loss = loss + 1.0 * -(tt * torch.log(o["pi"][ht] + 1e-6)).sum(-1).mean()
                        loss = loss + 2.0 * F.mse_loss(o["sev"][ht], S.sev[idx][ht])
                loss = loss + 1e-2 * model.delta.pow(2).sum(-1).mean()
                if cons and aug:
                    with torch.no_grad():
                        Gc, Dc = S.batch(idx, ~cm)  # the opposite view of each image
                    oc = model(Gc, Dc)
                    loss = loss + 0.5 * F.kl_div(F.log_softmax(oc["logits"], -1), F.softmax(o["logits"].detach(), -1), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        pv = predict(model, Gv, Dv)["logits"].argmax(1)
        f1 = f1_score(yv, pv, labels=sorted(set(yv.tolist())), average="macro")
        if f1 > best + 1e-4: best, best_state, bad = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10: break
    model.load_state_dict(best_state)
    return model, best

if __name__ == "__main__":
    seed = int(sys.argv[1]); methods = sys.argv[2].split(",")
    os.makedirs("runs", exist_ok=True)
    S = Store(); sp = seed_split(S.rows, seed)
    sets = {k: sp[k] for k in ["val", "cal", "test_pv", "test_pd", "cal_pv", "cal_pd"]}
    for m in methods:
        out = f"runs/{m}_s{seed}.pt"
        if os.path.exists(out): print("skip", out); continue
        t0 = time.time()
        kw = {}
        if m == "mg_noaug": kw = dict(aug=False); mm = "mg"
        elif m == "mg_nocons": kw = dict(cons=False); mm = "mg"
        elif m == "mg_notok": kw = dict(tok=False); mm = "mg"
        else: mm = m
        model, best = train(mm, S, sp, seed, **kw)
        torch.save(dict(state=model.state_dict(), method=("lp" if mm in ("zs","zsdesc") else mm), variant=m, seed=seed, val_f1=best,
                        train_time=time.time() - t0), out)
        preds = {}
        for name, idx in sets.items():
            G, D = S.batch(torch.tensor(idx)); preds[name] = predict(model, G, D)
        cm = np.ones(len(sp["cal"]), bool)
        G, D = S.batch(torch.tensor(sp["cal"]), torch.tensor(cm)); preds["cal_aug"] = predict(model, G, D)
        np.savez_compressed(f"runs/{m}_s{seed}_pred.npz", **{f"{n}__{k}": v for n, d in preds.items() for k, v in d.items()})
        print(m, seed, f"val_f1={best:.4f}", f"{time.time()-t0:.0f}s", flush=True)
