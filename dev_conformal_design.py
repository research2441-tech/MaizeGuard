import numpy as np, csv
from analysis_core import *
from metrics import pvalues
from splits import load_index, seed_split
CL=["healthy","gls","nlb","rust"]; rows=load_index(); Y=np.array([CL.index(r["label"]) for r in rows])
cds=list(csv.DictReader(open('index_cds.csv'))); ycds=np.array([CL.index(r['label']) if r['label'] in CL else -1 for r in cds])
I=lambda f:np.array([i for i,r in enumerate(cds) if f(r)])
dk=I(lambda r:r['ds']=='cds' and r['split']=='train' and r['label'] in('gls','nlb')); dn=I(lambda r:r['ds']=='cds' and r['split']=='train' and r['label']=='nls')
def classcond_p(cal_scores, cal_y, test_scores):  # test_scores (n,K)
    out=np.ones_like(test_scores)
    for k in range(test_scores.shape[1]):
        ref=cal_scores[cal_y==k]
        if len(ref): out[:,k]=pvalues(ref,test_scores[:,k])
    return out
res={}
for seed in range(5):
    sp=seed_split(rows,seed); pr=lambda s: load_pred(f'runs/mg_s{seed}_pred.npz',s)
    C=Calibrated(pr('val'),pr('cal'),pr('cal_aug'),Y[sp['val']],Y[sp['cal']],pr('val')['q'],pr('cal')['q'],pr('cal_aug')['q'])
    D=load_pred(f'runs/mg_s{seed}_cds.npz'); psi=1-D['q'][:,5]
    rng=np.random.default_rng(seed); perm=rng.permutation(len(dk)); ca, de = dk[perm[:len(dk)//2]], dk[perm[len(dk)//2:]]
    for calname in ['source','target']:
        if calname=='source':
            cal=C.cal; ycal=C.y_cal; psical=C.psi_cal
        else:
            cal={k:v[ca] for k,v in D.items()}; ycal=ycds[ca]; psical=psi[ca]
        A_lab_cal=1-C.probs(cal['logits'])[np.arange(len(ycal)),ycal]
        A_m_cal=-cal['logits'][np.arange(len(ycal)),ycal]
        A_e_cal=C.ev_score(cal['ev'])[np.arange(len(ycal)),ycal]
        def sets(idx, design, a):
            x={k:v[idx] for k,v in D.items()}
            plab=pvalues(A_lab_cal, 1-C.probs(x['logits']))  # marginal
            pm=classcond_p(A_m_cal,ycal,-x['logits']); pe=classcond_p(A_e_cal,ycal,C.ev_score(x['ev']))
            if design=='lab': return plab>a
            if design=='lab+ev': return (plab>a/2)&(pe>a/2)
            if design=='lab+match': return (plab>a/2)&(pm>a/2)
            if design=='lab+match+ev': return (plab>a/3)&(pm>a/3)&(pe>a/3)
            if design=='match': return pm>a
        for design in ['lab','lab+ev','lab+match','lab+match+ev','match']:
            Sk=sets(de,design,0.10); Sn=sets(dn,design,0.10)
            r=dict(cov=Sk[np.arange(len(de)),ycds[de]].mean(), size=Sk.sum(1).mean(), single=(Sk.sum(1)==1).mean(), nls_flag=(Sn.sum(1)==0).mean())
            res.setdefault((calname,design),[]).append(r)
for k,v in res.items():
    print(k, {m: round(np.mean([x[m] for x in v]),3) for m in v[0]})
