import os, sys, numpy as np, torch, csv
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
import run_main as R
from featstore import Store
from splits import seed_split
from mgmodel import MaizeGuard
from sklearn.metrics import f1_score
from scipy.special import logsumexp
S=Store(); cds=list(csv.DictReader(open('index_cds.csv')))
CL=["healthy","gls","nlb","rust"]
dk=[i for i,r in enumerate(cds) if r['ds']=='cds' and r['split']=='train' and r['label'] in ('gls','nlb')]
dn=[i for i,r in enumerate(cds) if r['ds']=='cds' and r['split']=='train' and r['label']=='nls']
yd=np.array([CL.index(cds[i]['label']) for i in dk])
z=np.load('feats/cds_clean.npz'); Gd=torch.from_numpy(z['G'][dk+dn]); Dd=torch.from_numpy(z['D'][dk+dn].reshape(-1,245,512))
from sklearn.metrics import roc_auc_score
def run(name, seed, mk, **kw):
    orig=R.make
    R.make=lambda m,S_,tr: mk() if m=='mg' else orig(m,S_,tr)
    model,best=R.train('mg',S,seed_split(S.rows,seed),seed,**kw); R.make=orig
    p=R.predict(model,Gd,Dd); lg=p['logits']; n=len(dk)
    f1=f1_score(yd, lg[:n].argmax(1), labels=[1,2], average='macro')
    ml=-lg.max(1); au=roc_auc_score(np.r_[np.zeros(n),np.ones(len(dn))], ml)
    print(name, seed, 'devF1 %.3f maxlogitAUROC %.3f'%(f1,au), flush=True)
T=R.TEXT
import sys
for seed in [int(a) for a in sys.argv[1].split(',')]:
    run('base_pdsel',seed,lambda: MaizeGuard(T), sel='pd')
    run('noad_pdsel',seed,lambda: MaizeGuard(T, adapter=False), sel='pd')
