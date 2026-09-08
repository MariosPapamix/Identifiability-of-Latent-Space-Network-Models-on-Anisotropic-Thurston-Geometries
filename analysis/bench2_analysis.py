"""Undirected benchmark networks (GitHub mirrors): BFS balls of 400 nodes, structural diagnostics, and six-geometry fits."""
import sys, os, json, time, collections, pickle, numpy as np
sys.path.insert(0,'code'); sys.path.insert(0,'analysis')
from connectome_analysis import res_load, res_save
from experiments2 import GEOS, vi_fit
from inference import logloss, auc
B='data/benchmarks_undirected'
def edges_csv(path, skip=1):
    E=[]
    for k,l in enumerate(open(path)):
        if k<skip or not l.strip(): continue
        a,b=l.strip().split(',')[:2]; E.append((int(a),int(b)))
    return E
def edges_roget():
    import re
    E=[]
    for l in open(f'{B}/graph_roget_dat.txt', errors='ignore'):
        m=re.match(r'\s*(\d+)[A-Za-z\-\' ,]*:(.*)', l)
        if not m: continue
        cur=int(m.group(1)); E+=[(cur,int(x)) for x in m.group(2).split() if x.isdigit()]
    return E
def edges_planetoid(name):
    d=pickle.load(open(f'{B}/data_ind.{name}.graph','rb'), encoding='latin1')
    return [(a,b) for a,nb in d.items() for b in nb if a!=b]
SRC={'roget': edges_roget, 'citeseer': lambda: edges_planetoid('citeseer'), 'pubmed': lambda: edges_planetoid('pubmed'), 'facebook': lambda: edges_csv(f'{B}/facebook_edges.csv'), 'lastfm': lambda: edges_csv(f'{B}/lastfm_edges.csv'), 'twitch': lambda: edges_csv(f'{B}/twitch_edges.csv')}
def ball(E, n=400):
    adj=collections.defaultdict(set)
    for a,b in E:
        if a!=b: adj[a].add(b); adj[b].add(a)
    # BFS from a node of median-high degree (not the hub) to avoid a star
    degs=sorted(adj, key=lambda v: -len(adj[v])); start=degs[len(degs)//20]
    seen=[start]; S=set(seen); q=[start]
    while q and len(seen)<n:
        v=q.pop(0)
        for w in sorted(adj[v], key=lambda w: -len(adj[w])):
            if w not in S: S.add(w); seen.append(w); q.append(w)
            if len(seen)>=n: break
    idx={v:i for i,v in enumerate(seen)}; N=len(seen); Y=np.zeros((N,N))
    for v in seen:
        for w in adj[v]:
            if w in idx: Y[idx[v],idx[w]]=1; Y[idx[w],idx[v]]=1
    return Y
def diagnostics(Y):
    N=len(Y); deg=Y.sum(1); tri=np.einsum('ij,jk,ki->i',Y,Y,Y)/2; cl=np.where(deg>1, 2*tri/np.maximum(deg*(deg-1),1),0)
    ii,jj=np.where(np.triu(Y,1)>0); t_e=(Y[ii]*Y[jj]).sum(1); forman=4-deg[ii]-deg[jj]+3*t_e   # augmented Forman-Ricci
    return dict(N=N, density=float(Y[np.triu_indices(N,1)].mean()), deg_cv=float(deg.std()/deg.mean()), clustering=float(cl.mean()), forman_mean=float(forman.mean()), forman_pos=float((forman>0).mean()), max_deg=int(deg.max()))
def build(names):
    R=res_load(); out=R.get('bench2_diag',{})
    for n in names:
        Y=ball(SRC[n]()); np.savez(f'{B}/{n}_ball.npz', Y=Y); out[n]=diagnostics(Y); print(n, {k:(round(v,3) if isinstance(v,float) else v) for k,v in out[n].items()}, flush=True)
    R['bench2_diag']=out; res_save(R)
def fit(names, geos=('nil','sol','sl2','euc','h3','s3','h2r'), frac=0.1):
    for n in names:
        Y=np.load(f'{B}/{n}_ball.npz')['Y']; N=len(Y); iu=np.triu_indices(N,1); rng=np.random.default_rng(300); hold=rng.uniform(size=len(iu[0]))<frac
        mask=np.ones((N,N)); mask[iu[0][hold],iu[1][hold]]=0; mask[iu[1][hold],iu[0][hold]]=0
        for g in geos:
            R=res_load(); rec=R.get('bench2_fits',{}).get(n, dict(N=N, fits={}))
            if g in rec['fits']: continue
            t0=time.time(); m,vi=vi_fit(GEOS[g], Y, mask=mask, n_iter=400, S=6, seed=0, M=20000); P=vi.predictive(60)
            rec['fits'][g]=dict(logloss=float(logloss(P[iu][hold],Y[iu][hold])), auc=float(auc(P[iu][hold],Y[iu][hold])), time=time.time()-t0)
            R=res_load(); R.setdefault('bench2_fits',{})[n]=rec; res_save(R); print(n, g, {k:round(v,4) for k,v in rec['fits'][g].items()}, flush=True)
if __name__=='__main__':
    a=sys.argv[1:]
    if a[0]=='build': build(a[1:])
    elif a[0]=='fit': fit(a[1:])
