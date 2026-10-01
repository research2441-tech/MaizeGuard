"""Group-aware partitions. Fixed test partition (seed 2026) for PV and PD; per-seed train/val/cal for the rest."""
import csv, numpy as np
def load_index(path="index_pv_pd.csv"):
    return list(csv.DictReader(open(path)))
def _group_split(rows, idx, fracs, seed):
    rng = np.random.default_rng(seed)
    out = [[] for _ in fracs]
    labs = sorted({rows[i]["label"] for i in idx})
    for lab in labs:
        ii = [i for i in idx if rows[i]["label"] == lab]
        groups = sorted({rows[i]["group"] for i in ii}); rng.shuffle(groups)
        cuts = np.cumsum([0] + [int(round(f * len(groups))) for f in fracs[:-1]] + [len(groups)])
        cuts[-1] = len(groups)
        for k in range(len(fracs)):
            gs = set(groups[cuts[k]:cuts[k + 1]])
            out[k] += [i for i in ii if rows[i]["group"] in gs]
    return [np.array(sorted(o)) for o in out]
def fixed_test(rows):
    pv = [i for i, r in enumerate(rows) if r["ds"] == "pv"]
    pd = [i for i, r in enumerate(rows) if r["ds"] == "pd"]
    pv_rest, pv_test = _group_split(rows, pv, [0.85, 0.15], 2026)
    pd_rest, pd_test = _group_split(rows, pd, [0.75, 0.25], 2026)
    return pv_rest, pv_test, pd_rest, pd_test
def seed_split(rows, seed):
    pv_rest, pv_test, pd_rest, pd_test = fixed_test(rows)
    a = _group_split(rows, list(pv_rest), [0.70, 0.12, 0.18], 100 + seed)
    b = _group_split(rows, list(pd_rest), [0.60, 0.13, 0.27], 200 + seed)
    return dict(train=np.concatenate([a[0], b[0]]), val=np.concatenate([a[1], b[1]]),
                cal=np.concatenate([a[2], b[2]]), test_pv=pv_test, test_pd=pd_test,
                train_pv=a[0], cal_pv=a[2], cal_pd=b[2])
if __name__ == "__main__":
    rows = load_index()
    s = seed_split(rows, 0)
    for k, v in s.items():
        import collections
        print(k, len(v), dict(collections.Counter(rows[i]["label"] for i in v)))
    pv_rest, pv_test, pd_rest, pd_test = fixed_test(rows)
    nontest = np.concatenate([pv_rest, pd_rest]); test = np.concatenate([pv_test, pd_test])
    open("list_nontest.txt", "w").write("\n".join(rows[i]["path"] for i in nontest))
    open("list_test.txt", "w").write("\n".join(rows[i]["path"] for i in test))
    np.save("feats/nontest_idx.npy", nontest); np.save("feats/test_idx.npy", test)
