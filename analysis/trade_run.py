import sys, json, time, numpy as np
sys.path.insert(0,'code'); sys.path.insert(0,'analysis')
from connectome_analysis import res_load, res_save
from directed import fit_directed, predictive_directed
from experiments2 import GEOS
from inference import logloss, auc
d=np.load('data/trade/trade_dominance.npz'); Y=d['Y']; N=len(Y); iu=np.triu_indices(N,1); off=~np.eye(N,dtype=bool)
rng=np.random.default_rng(int(__import__('os').environ.get('SPLIT',0))+800); hold=rng.uniform(size=len(iu[0]))<0.1
mask=np.ones((N,N)); mask[iu[0][hold],iu[1][hold]]=0; mask[iu[1][hold],iu[0][hold]]=0; H=(mask==0)&off; tr=(mask==1)&off
for f in sys.argv[1:]:
    R=res_load(); rec=R.get('trade_directed', {}).get('s'+__import__('os').environ.get('SPLIT','0'), dict(N=N, fits={}))
    if f in rec['fits']: continue
    g=f.replace('_sym',''); lg=not f.endswith('_sym'); t0=time.time(); best=None
    for g0 in ((0.0,0.4) if (lg and g in ('nil','sl2')) else (0.0,)):
        m_,v_=fit_directed(GEOS[g],Y,mask,learn_gamma=lg,n_iter=400,S=6,seed=0,gamma0=g0); P_=predictive_directed(m_,v_,n=60); trl=float(logloss(P_[tr],Y[tr]))
        if best is None or trl<best[0]: best=(trl,m_,v_,P_)
    trl,model,vi,Pf=best
    one=((Y+Y.T)==1)&(mask==0)&np.triu(np.ones((N,N),bool),1); ii,jj=np.where(one); sc=np.log(Pf[ii,jj]+1e-9)-np.log(Pf[jj,ii]+1e-9); acc=float(np.mean((sc>0)==(Y[ii,jj]==1)))
    rec['fits'][f]=dict(logloss=float(logloss(Pf[H],Y[H])), train_logloss=trl, direction_acc=acc, n_asym_pairs=int(len(ii)), gamma_hat=float(model.gamma), time=time.time()-t0)
    R=res_load(); R.setdefault('trade_directed',{})['s'+__import__('os').environ.get('SPLIT','0')]=rec; res_save(R); print(f, {k:(round(v,4) if isinstance(v,float) else v) for k,v in rec['fits'][f].items()}, flush=True)
