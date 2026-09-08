"""Metropolis-within-Gibbs for the directed circulation model: random-walk proposals for each position (directed
likelihood of the node's row and column), for alpha and for gamma; scales by conjugate Gibbs steps under
InvGamma(2,2); centre point-fixed at the chart mean; unanchored. Held-out predictive from retained draws."""
import sys, json, time, os, numpy as np
sys.path.insert(0,'code'); sys.path.insert(0,'analysis')
from inference import bern_ll, sigmoid, logloss
from experiments2 import GEOS
from directed import vertical_offsets, fit_directed, predictive_directed
from connectome_analysis import res_load, res_save

def run(name, geo='nil', split=0, sweeps=int(os.environ.get('SWEEPS',3000)), seed=0, step=0.15):
    d=np.load(f'data/games/{name}_dominance.npz'); Y=d['Y']; N=len(Y); iu=np.triu_indices(N,1); off=~np.eye(N,dtype=bool); G=GEOS[geo]
    rng=np.random.default_rng(900+split); hold=rng.uniform(size=len(iu[0]))<0.1
    mask=np.ones((N,N)); mask[iu[0][hold],iu[1][hold]]=0; mask[iu[1][hold],iu[0][hold]]=0; H=(mask==0)&off; M=mask.copy(); np.fill_diagonal(M,0)
    mv, vi = fit_directed(G, Y, mask, learn_gamma=True, n_iter=500, S=6, seed=seed, gamma0=0.4)
    S0=vi.S; vi.S=20; Z=vi.sample()[0][0].copy(); vi.S=S0; a=float(vi.am); g=float(mv.gamma)
    Pv=predictive_directed(mv, vi, n=80); vi_ll=float(logloss(Pv[H],Y[H]))
    D=G.pdist(Z); V=vertical_offsets(G,Z)
    def ll_full(D,V,a,g): E=a-D+g*V; return float(np.sum((bern_ll(Y,E)*M)))
    def ll_node(i,Di,Vi,a,g):   # row i and column i (V antisymmetric: V[j,i] = -Vi[j])
        Er=a-Di+g*Vi; Ec=a-Di-g*Vi; return float(np.sum(bern_ll(Y[i],Er)*M[i]) + np.sum(bern_ll(Y[:,i],Ec)*M[:,i]))
    ll=ll_full(D,V,a,g); rs=np.random.default_rng(seed); acc=np.zeros(3); keep=[]
    sig=np.array([1.5,1.5]); a0,b0=2.0,2.0
    chart_inv = (lambda z: G.chart_inv(z)) if hasattr(G,'chart_inv') else (lambda z: z)
    for t in range(sweeps):
        for i in rs.permutation(N):
            znew=G.translate(Z[i], G.chart(step*rs.normal(size=3))); Zi_old=Z[i].copy()
            Di=G.dist_to_all(znew, Z); Di[i]=0.0
            Ztmp=Z.copy(); Ztmp[i]=znew; Vi=vertical_offsets(G, Ztmp)[i]
            lo=ll_node(i,D[i],V[i],a,g); ln=ll_node(i,Di,Vi,a,g)
            # prior: wrapped Normal about the chart mean (centre fixed at current mean), diag scales
            mu=Z.mean(0); do=Zi_old-mu; dn=znew-mu
            lp_o=-0.5*(do[0]**2/sig[0]**2+do[1]**2/sig[0]**2+do[2]**2/sig[1]**2); lp_n=-0.5*(dn[0]**2/sig[0]**2+dn[1]**2/sig[0]**2+dn[2]**2/sig[1]**2)
            if np.log(rs.uniform()) < (ln-lo)+(lp_n-lp_o):
                Z[i]=znew; D[i,:]=Di; D[:,i]=Di; V[i,:]=Vi; V[:,i]=-Vi; ll+= (ln-lo); acc[0]+=1
        a1=a+0.05*rs.normal(); l1=ll_full(D,V,a1,g)
        if np.log(rs.uniform()) < (l1-ll)-0.5*((a1/3)**2-(a/3)**2): a,ll=a1,l1; acc[1]+=1
        g1=g+0.05*rs.normal(); l1=ll_full(D,V,a,g1)
        if np.log(rs.uniform()) < (l1-ll)-0.5*(g1**2-g**2): g,ll=g1,l1; acc[2]+=1
        dz=Z-Z.mean(0); sig[0]=np.sqrt(1/rs.gamma(a0+N, 1/(b0+0.5*np.sum(dz[:,:2]**2)))); sig[1]=np.sqrt(1/rs.gamma(a0+N/2, 1/(b0+0.5*np.sum(dz[:,2]**2))))
        if t%10==0 and t>=sweeps//3: keep.append((Z.copy(),a,g))
        if t%500==0: print('sweep',t,'loglik %.1f alpha %.2f gamma %.3f sig %s acc %s' % (ll,a,g,np.round(sig,2),np.round(acc/((t+1)*np.array([N,1,1])),2)), flush=True)
    P=np.mean([sigmoid(a_-G.pdist(Z_)+g_*vertical_offsets(G,Z_)) for Z_,a_,g_ in keep],0)
    llh=float(logloss(P[H],Y[H])); gs=np.array([k[2] for k in keep])
    one=((Y+Y.T)==1)&(mask==0)&np.triu(np.ones((N,N),bool),1); ii,jj=np.where(one); sc=np.log(P[ii,jj]+1e-9)-np.log(P[jj,ii]+1e-9); accd=float(np.mean((sc>0)==(Y[ii,jj]==1)))
    res=dict(N=N, sweeps=sweeps, vi_logloss=vi_ll, vi_gamma=float(mv.gamma), gamma_mean=float(gs.mean()), gamma_q=[float(np.quantile(gs,q)) for q in (0.025,0.5,0.975)], alpha_mean=float(np.mean([k[1] for k in keep])), logloss=llh, direction_acc=accd, final_train_loglik=ll)
    R=res_load(); R.setdefault('games_mcmc',{})[f'{name}_{geo}_s{split}_directed']=res; res_save(R); print(name, {k:(round(v,4) if isinstance(v,float) else v) for k,v in res.items()}, flush=True)

if __name__=='__main__': run(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else 'nil')
