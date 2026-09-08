"""Matchup networks that are intransitive by design: Dota 2 hero matchups (OpenDota) and Pokemon Showdown checks and counters (Smogon)."""
import json, glob, os, sys, time, numpy as np
sys.path.insert(0,'code'); sys.path.insert(0,'analysis')
from connectome_analysis import rotational_fraction, res_load, res_save
from directed import fit_directed, predictive_directed
from experiments2 import GEOS
from inference import logloss, auc

def build_dota(min_games=30):
    heroes=json.load(open('data/games/dota/heroes.json')); ids=[h['id'] for h in heroes]; idx={h:i for i,h in enumerate(ids)}; N=len(ids)
    W=np.full((N,N),np.nan); G=np.zeros((N,N))
    for h in ids:
        f=f'data/games/dota/matchups_{h}.json'
        if not os.path.exists(f) or os.path.getsize(f)<50: continue
        for m in json.load(open(f)):
            if m['hero_id'] in idx and m['games_played']>0:
                W[idx[h],idx[m['hero_id']]]=m['wins']/m['games_played']; G[idx[h],idx[m['hero_id']]]=m['games_played']
    Y=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i!=j and G[i,j]>=min_games and not np.isnan(W[i,j]) and W[i,j]>0.5: Y[i,j]=1
    keep=np.where((Y+Y.T).sum(1)>=10)[0]; return Y[np.ix_(keep,keep)], [heroes[i]['localized_name'] for i in keep]

def build_pokemon(min_usage=0.01, thr=0.5, tier='gen9ou', fname=None):
    d=json.load(open(fname or f'data/games/{tier}-1500_json'))['data']
    names=[n for n,p in d.items() if p.get('usage',0)>=min_usage]; idx={n:i for i,n in enumerate(names)}; N=len(names); Y=np.zeros((N,N))
    for n,p in d.items():
        if n not in idx: continue
        for c,(cnt,pr,sd) in p.get('Checks and Counters',{}).items():
            if c in idx and c!=n and pr-4*sd>thr: Y[idx[c],idx[n]]=1     # c checks/counters n
    keep=np.where((Y+Y.T).sum(1)>=3)[0]; return Y[np.ix_(keep,keep)], [names[i] for i in keep]

NETS={'dota': build_dota, 'pokemon': build_pokemon, 'pokemon_uu': (lambda: build_pokemon(tier='gen9uu')), 'pokemon_ubers': (lambda: build_pokemon(tier='gen9ubers')), 'pokemon_doubles': (lambda: build_pokemon(tier='gen9doublesou'))}
FILES={'ou_feb':'gen9ou-1500_feb_json','ou_mar':'gen9ou-1500_mar_json','ou_apr':'gen9ou-1500_april_json','ou_may':'gen9ou-1500_may_json','ou_jun':'gen9ou-1500_jun_json',
       'ou_1695':'gen9ou-1695_json','ou_1825':'gen9ou-1825_json','gen3ou':'gen3ou-1500_json','gen4ou':'gen4ou-1500_json','gen5ou':'gen5ou-1500_json','gen6ou':'gen6ou-1500_json',
       'gen7ou':'gen7ou-1500_json','gen8ou':'gen8ou-1500_json','lc':'gen9lc-1500_json','pu':'gen9pu-1500_json','nu':'gen9nu-1500_json','ru':'gen9ru-1500_json','monotype':'gen9monotype-1500_json','natdex':'gen9nationaldex-1500_json'}
for k,v in FILES.items():
    NETS['pk_'+k]=(lambda v=v: build_pokemon(fname='data/games/'+v))
NETS['pk_ou_mutual']=(lambda: build_pokemon(thr=0.3, min_usage=0.01))
for thr in (0.3,0.5,0.7):
    for mu in (0.005,0.01,0.02):
        NETS[f'pk_ou_thr{thr}_use{mu}']=(lambda thr=thr,mu=mu: build_pokemon(min_usage=mu, thr=thr))

def screen(only=None):
    R=res_load(); out=R.get('games_rotfrac',{})
    for name,b in NETS.items():
        if only and name not in only: continue
        Y,labels=b(); N=len(Y)
        if N<10:
            print(name,'too few nodes',N, flush=True); continue
        S=Y+Y.T; iu=np.triu_indices(N,1); rf,asym=rotational_fraction(Y)
        rng=np.random.default_rng(0); vals=[]; one=S[iu]==1; both=S[iu]==2
        for _ in range(10):
            Yn=np.zeros_like(Y); Yn[iu[0][both],iu[1][both]]=1; Yn[iu[1][both],iu[0][both]]=1
            flip=rng.uniform(size=one.sum())<0.5; a,b_=iu[0][one],iu[1][one]; Yn[a[flip],b_[flip]]=1; Yn[b_[~flip],a[~flip]]=1; vals.append(rotational_fraction(Yn)[0])
        cyc=dec=0; rng2=np.random.default_rng(1)
        for _ in range(100000):
            i,j,k=rng2.choice(N,3,replace=False); s=[Y[a,b_]-Y[b_,a] for a,b_ in ((i,j),(j,k),(k,i))]
            if all(abs(v)==1 for v in s): dec+=1; cyc+=(abs(sum(s))==3)
        out[name]=dict(N=N, decided_pairs=int((S[iu]>0).sum()), pairs=int(len(iu[0])), reciprocal_pairs=int((S[iu]==2).sum()), rotfrac=rf, rotfrac_random=float(np.mean(vals)), cyclic_triads=cyc/max(dec,1))
        print(name, {k:(round(v,3) if isinstance(v,float) else v) for k,v in out[name].items()}, flush=True)
        np.savez(f'data/games/{name}_dominance.npz', Y=Y, names=np.array(labels))
    R['games_rotfrac']=out; res_save(R)

def run_directed(name, fits, nsplit=int(os.environ.get('NSPLIT',2)), frac=0.1):
    tag=os.environ.get('TAG',''); d=np.load(f'data/games/{name}_dominance.npz'); Y=d['Y']; N=len(Y); iu=np.triu_indices(N,1); off=~np.eye(N,dtype=bool)
    for split in range(nsplit):
        key=f'{name}_s{split}'; R=res_load(); rec=R.get('games_directed',{}).get(key, dict(N=N, fits={}))
        rng=np.random.default_rng(900+split); hold=rng.uniform(size=len(iu[0]))<frac
        mask=np.ones((N,N)); mask[iu[0][hold],iu[1][hold]]=0; mask[iu[1][hold],iu[0][hold]]=0; H=(mask==0)&off; tr=(mask==1)&off
        for f in fits:
            fkey=f+tag
            if fkey in rec['fits']: continue
            g=f.replace('_sym',''); lg=not f.endswith('_sym'); t0=time.time(); best=None
            for g0 in ((0.0,0.4) if (lg and g in ('nil','sl2')) else (0.0,)):
                m_,v_=fit_directed(GEOS[g],Y,mask,learn_gamma=lg,n_iter=int(os.environ.get('NITER',600)),S=int(os.environ.get('NS',8)),seed=split,gamma0=g0); P_=predictive_directed(m_,v_,n=80); trl=float(logloss(P_[tr],Y[tr]))
                if best is None or trl<best[0]: best=(trl,m_,v_,P_)
            trl,model,vi,Pf=best
            one=((Y+Y.T)==1)&(mask==0)&np.triu(np.ones((N,N),bool),1); ii,jj=np.where(one); sc=np.log(Pf[ii,jj]+1e-9)-np.log(Pf[jj,ii]+1e-9); acc=float(np.mean((sc>0)==(Y[ii,jj]==1))) if len(ii) else float('nan')
            rec['fits'][fkey]=dict(logloss=float(logloss(Pf[H],Y[H])), train_logloss=trl, direction_acc=acc, n_asym_pairs=int(len(ii)), gamma_hat=float(model.gamma), time=time.time()-t0)
            R=res_load(); R.setdefault('games_directed',{})[key]=rec; res_save(R); print(key, fkey, {k:(round(v,4) if isinstance(v,float) else v) for k,v in rec['fits'][fkey].items()}, flush=True)

if __name__=='__main__':
    a=sys.argv[1:]
    if a[0]=='screen': screen(a[1:] or None)
    elif a[0]=='directed': run_directed(a[1], a[2:])
