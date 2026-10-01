import numpy as np, csv, json
from analysis_core import load_pred
T=dict(np.load('feats/text.npz',allow_pickle=True)); names=list(T['lex_names']); own=T['lex_owner']; L=T['lex_flat']; Lc=T['lex_cname']
cds=list(csv.DictReader(open('index_cds.csv')))
tn=[i for i,r in enumerate(cds) if r['ds']=='cds' and r['split']=='test' and r['label']=='nls']
nrm=lambda x:x/np.linalg.norm(x,axis=-1,keepdims=True)
res={}
for s in range(5):
    D=load_pred(f'runs/mg_s{s}_cds.npz')
    for feat in ['rl','rg','z']:
        F=nrm(D[feat][tn]); sim=F@L.T
        sc=np.stack([sim[:,own==c].mean(1) for c in range(len(names))],1)
        rank=(-sc).argsort(1); pos=np.array([list(r).index(0) for r in rank])  # nls index 0
        res.setdefault(f'desc_{feat}',[]).append(((pos==0).mean(), (pos<3).mean()))
    F=nrm(D['rg'][tn]); sc=F@Lc.T; rank=(-sc).argsort(1); pos=np.array([list(r).index(0) for r in rank])
    res.setdefault('classname_rg',[]).append(((pos==0).mean(), (pos<3).mean()))
out={k:dict(top1=float(np.mean([a for a,b in v])), top3=float(np.mean([b for a,b in v])), sd1=float(np.std([a for a,b in v]))) for k,v in res.items()}
json.dump(out,open('results/naming.json','w'),indent=1); print(json.dumps(out,indent=1)); print(names)
