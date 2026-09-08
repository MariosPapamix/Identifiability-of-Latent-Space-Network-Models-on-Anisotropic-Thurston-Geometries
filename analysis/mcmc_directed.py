"""Metropolis-within-Gibbs for the directed circulation model on a real counter network: Algorithm 2 for positions,
alpha, centre and scales on the gauge slice, plus a Gaussian random-walk Metropolis step for gamma (prior N(0,1)),
alternated in blocks; held-out predictive from the retained draws, compared with BBVI on the same split."""
import sys, json, time, numpy as np
sys.path.insert(0,'code'); sys.path.insert(0,'analysis')
from inference import mcmc, initialise, choose_anchors, sigmoid, logloss, auc
from experiments2 import GEOS
from directed import DirectedLSM, fit_directed, predictive_directed
from connectome_analysis import res_load, res_save

def run(name='pokemon', geo='nil', split=0, blocks=int(sys.argv[3]) if len(sys.argv) > 3 else 150, inner=10, seed=0):
    d=np.load(f'data/games/{name}_dominance.npz'); Y=d['Y']; N=len(Y); iu=np.triu_indices(N,1); off=~np.eye(N,dtype=bool)
    rng=np.random.default_rng(900+split); hold=rng.uniform(size=len(iu[0]))<0.1
    mask=np.ones((N,N)); mask[iu[0][hold],iu[1][hold]]=0; mask[iu[1][hold],iu[0][hold]]=0; H=(mask==0)&off
    anchors=choose_anchors(Y+Y.T, geo)
    # warm start at the variational solution (positions, alpha, gamma), then Metropolis-within-Gibbs
    mv, vi = fit_directed(GEOS[geo], Y, mask, learn_gamma=True, n_iter=500, S=6, seed=seed, gamma0=0.4)
    S0=vi.S; vi.S=40; Zs=vi.sample()[0]; vi.S=S0; Zmean=GEOS[geo].chart(np.mean([GEOS[geo].chart_inv(z) if hasattr(GEOS[geo],'chart_inv') else z for z in Zs],0)) if False else Zs[0]
    model=DirectedLSM(GEOS[geo], Y, anchors, gamma=float(mv.gamma), mask=mask)
    Z=GEOS[geo].gauge(Zmean, anchors); a=float(vi.am); sig=None
    # the gauge may apply the reflection, which reverses every v_ij: keep the sign of gamma with the higher likelihood
    l1=model.loglik(Z,a); model.gamma=-model.gamma; l2=model.loglik(Z,a)
    if l1>=l2: model.gamma=-model.gamma
    print('warm start: loglik +gamma %.1f, -gamma %.1f, chosen gamma %.3f' % (l1, l2, model.gamma), flush=True)
    Pv=predictive_directed(mv, vi, n=80); vi_ll=float(logloss(Pv[H],Y[H]))
    dz=np.array([GEOS[geo].chart_inv(z) if hasattr(GEOS[geo],'chart_inv') else z for z in Zmean]); print('VI configuration spread: sd of coordinates', np.round(dz.std(0),2), flush=True)
    rs=np.random.default_rng(seed); gam=[]; keepZ=[]; keepA=[]; keepG=[]; acc=0; t0=time.time()
    for b in range(blocks):
        o=mcmc(model, Z, a, n_iter=inner, step_z=float(__import__('os').environ.get('STEPZ',0.45)), step_alpha=float(__import__('os').environ.get('STEPA',0.2)), thin=inner, burn=0, seed=seed*1000+b, fix_sigma=bool(int(__import__('os').environ.get('FIXSIG','0'))), sig0=(sig if not int(__import__('os').environ.get('FIXSIG','0')) else (float(__import__('os').environ.get('SIGH',1.5)), float(__import__('os').environ.get('SIGV',1.5)))))
        Z=o['Z'][-1]; a=float(o['alpha'][-1]); sig=tuple(o['sig'][-1])
        # Metropolis step for gamma
        g0=model.gamma; ll0=model.loglik(Z,a); g1=g0+float(__import__('os').environ.get('STEPG',0.1))*rs.normal(); model.gamma=g1; ll1=model.loglik(Z,a)
        if np.log(rs.uniform()) < (ll1-ll0)-0.5*(g1**2-g0**2): acc+=1
        else: model.gamma=g0
        if b>=blocks//3: keepZ.append(Z.copy()); keepA.append(a); keepG.append(model.gamma)
        if b%50==0: print('block', b, 'train loglik %.1f' % model.loglik(Z,a), 'alpha %.2f gamma %.3f sig %s' % (a, model.gamma, np.round(sig,2)), flush=True)
    P=np.mean([sigmoid(a_ - GEOS[geo].pdist(Z_) + g_ * __import__('directed').vertical_offsets(GEOS[geo], Z_)) for Z_,a_,g_ in zip(keepZ,keepA,keepG)],0)
    ll=float(logloss(P[H],Y[H])); one=((Y+Y.T)==1)&(mask==0)&np.triu(np.ones((N,N),bool),1); ii,jj=np.where(one); sc=np.log(P[ii,jj]+1e-9)-np.log(P[jj,ii]+1e-9); accd=float(np.mean((sc>0)==(Y[ii,jj]==1)))
    res=dict(N=N, blocks=blocks, inner=inner, sweeps=blocks*inner, vi_logloss=vi_ll, vi_gamma=float(mv.gamma), gamma_mean=float(np.mean(keepG)), gamma_sd=float(np.std(keepG)), gamma_q=[float(np.quantile(keepG,q)) for q in (0.025,0.5,0.975)], gamma_accept=acc/blocks, alpha_mean=float(np.mean(keepA)), logloss=ll, direction_acc=accd, time=time.time()-t0)
    R=res_load(); R.setdefault('games_mcmc',{})[f'{name}_{geo}_s{split}']=res; res_save(R); print(name, geo, split, {k:(round(v,4) if isinstance(v,float) else v) for k,v in res.items()}, flush=True)

if __name__=='__main__':
    run(sys.argv[1], sys.argv[2])
