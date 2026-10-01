import numpy as np, csv
exec(open('dev_conformal.py').read().split('res={}')[0])
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
res={}
for seed in range(5):
    D=load_pred(f'runs/mg_s{seed}_cds.npz'); L=load_pred(f'runs/lp_s{seed}_cds.npz')
    rng=np.random.default_rng(seed); perm=rng.permutation(len(dk)); fa, ca, de = dk[perm[:170]], dk[perm[170:340]], dk[perm[340:]]
    for name,feat in [('mg_z',D['z']),('mg_rl',D['rl']),('mg_rg',D['rg']),('mg_ev',D['ev']),('lp_rg',L['z'])]:
        sc=np.full((len(feat),2),0.)
        for j,k in enumerate([1,2]):
            E=feat[fa][ycds[fa]==k]; lw=LedoitWolf().fit(E); P=np.linalg.inv(lw.covariance_+1e-4*np.eye(E.shape[1])); d=feat-lw.location_
            sc[:,j]=np.einsum('ij,jk,ik->i',d,P,d)
        s=sc.min(1); res.setdefault(name,[]).append(roc_auc_score(np.r_[np.zeros(len(de)),np.ones(len(dn))], np.r_[s[de],s[dn]]))
    for name,feat in [('mg_z',D['z']),('lp_rg',L['z'])]:
        f=feat/np.linalg.norm(feat,axis=1,keepdims=True); s=-np.sort(f@f[fa].T,1)[:,-5]
        res.setdefault('knn_'+name,[]).append(roc_auc_score(np.r_[np.zeros(len(de)),np.ones(len(dn))], np.r_[s[de],s[dn]]))
for k,v in res.items(): print(k, np.round(v,3), round(np.mean(v),3), flush=True)
