import sys, numpy as np, torch, time, json
import run_temporal as RT, temporal as TM, progression as P
from splits import load_index, seed_split
seed=0; rows=load_index(); sp=seed_split(rows,seed)
vis, labels, true_sev = RT.vision_bank(seed, rows, sp)
rng=np.random.default_rng(1000+seed); bank=TM.ObsBank(vis,labels,true_sev,rng)
tr=P.build("champion", range(1982,2006),40,seed,start=160,length=100); ca=P.build("champion", range(2006,2011),40,seed+50,start=160,length=100)
te=P.build("champion", range(2011,2019),40,seed+100,start=160,length=100); hy=P.build("hyderabad", range(2000,2011),30,seed+150,start=185,length=100)
S={k:TM.make_sequences(v,bank,rng) for k,v in dict(tr=tr,ca=ca,te=te,hy=hy).items()}
t0=time.time()
for ep,lr in [(40,0.02)]:
    m=TM.EnvFilter(); TM.train_filter(m,S['tr'],epochs=ep,lr=lr,seed=seed); f=RT.filter_pred(m)
    for sp_ in ['te','hy']:
        r=RT.evaluate(f,S[sp_]); print(ep,lr,sp_,{h:round(RT.metrics(r[h])['mae'],3) for h in (7,14)}, time.time()-t0, flush=True)
    print({n:float(p.detach().mean()) for n,p in m.named_parameters() if p.numel()==1})
# GRU without quality columns
class NoQ: pass
def strip(seqs):
    out=[]
    for s in seqs:
        s2=dict(s); o=s['obs'].copy(); o[:,2:4]=0; s2['obs']=o; out.append(s2)
    return out
m=TM.GRUForecaster(); TM.train_gru(m,strip(S['tr']),epochs=60,seed=seed); f=RT.gru_pred(m)
for sp_ in ['te','hy']:
    r=RT.evaluate(f,strip(S[sp_])); print('gru_noqual',sp_,{h:round(RT.metrics(r[h])['mae'],3) for h in (7,14)}, flush=True)
