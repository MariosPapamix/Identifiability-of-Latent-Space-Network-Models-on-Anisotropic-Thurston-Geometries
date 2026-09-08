import sys, os, time, numpy as np
sys.path.insert(0,'code'); sys.path.insert(0,'analysis')
from connectome_analysis import res_load, res_save
from directed import fit_control, predictive_directed
from experiments2 import GEOS
from inference import logloss
def run(name, combos=(('euc','skew'),('h3','skew'),('euc','dc'),('h3','dc')), nsplit=3, frac=0.1):
    d=np.load(f'data/games/{name}_dominance.npz'); Y=d['Y']; N=len(Y); iu=np.triu_indices(N,1); off=~np.eye(N,dtype=bool)
    for split in range(nsplit):
        key=f'{name}_s{split}'; R=res_load(); rec=R['games_directed'].get(key, dict(N=N, fits={}))
        rng=np.random.default_rng(900+split); hold=rng.uniform(size=len(iu[0]))<frac
        mask=np.ones((N,N)); mask[iu[0][hold],iu[1][hold]]=0; mask[iu[1][hold],iu[0][hold]]=0; H=(mask==0)&off; tr=(mask==1)&off
        for g,kind in combos:
            f=f'{g}_{kind}'
            if f in rec['fits']: continue
            t0=time.time(); m,vi=fit_control(GEOS[g],Y,mask,kind=kind,n_iter=500,S=6,seed=split); P=predictive_directed(m,vi,n=60)
            one=((Y+Y.T)==1)&(mask==0)&np.triu(np.ones((N,N),bool),1); ii,jj=np.where(one); sc=np.log(P[ii,jj]+1e-9)-np.log(P[jj,ii]+1e-9); acc=float(np.mean((sc>0)==(Y[ii,jj]==1))) if len(ii) else float('nan')
            rec['fits'][f]=dict(logloss=float(logloss(P[H],Y[H])), train_logloss=float(logloss(P[tr],Y[tr])), direction_acc=acc, time=time.time()-t0)
            R=res_load(); R['games_directed'][key]=rec; res_save(R); print(key, f, {k:(round(v,4) if isinstance(v,float) else v) for k,v in rec['fits'][f].items()}, flush=True)
if __name__=='__main__':
    for n in sys.argv[1:]: run(n)
