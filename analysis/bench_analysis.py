"""Benchmark directed networks: KONECT chess (game results), SNAP wiki-Vote and email-Eu-core. Builds the dominance /
adjacency networks on the most active nodes, screens them, and fits all geometries."""
import sys, os, json, time, collections, numpy as np
sys.path.insert(0,'code'); sys.path.insert(0,'analysis')
from connectome_analysis import rotational_fraction, res_load, res_save
from games_analysis import run_directed
from directed import fit_directed, predictive_directed
from experiments2 import GEOS

def build_chess(top=150):
    wins=collections.Counter(); games=collections.Counter()
    for l in open('data/benchmarks/out.chess'):
        if l.startswith('%'): continue
        p=l.split(); a,b,r=int(p[0]),int(p[1]),int(p[2]); games[a]+=1; games[b]+=1
        if r==1: wins[(a,b)]+=1
        elif r==-1: wins[(b,a)]+=1
    keep=[k for k,_ in games.most_common(top)]; idx={k:i for i,k in enumerate(keep)}; N=len(keep); Y=np.zeros((N,N))
    for (a,b),w in wins.items():
        if a in idx and b in idx and w>wins[(b,a)]: Y[idx[a],idx[b]]=1
    return Y, [str(k) for k in keep]

def build_edgelist(path, top=150, skip='#'):
    E=[]; deg=collections.Counter()
    for l in open(path):
        if l.startswith(skip) or not l.strip(): continue
        a,b=map(int,l.split()[:2]); E.append((a,b)); deg[a]+=1; deg[b]+=1
    keep=[k for k,_ in deg.most_common(top)]; idx={k:i for i,k in enumerate(keep)}; N=len(keep); Y=np.zeros((N,N))
    for a,b in E:
        if a in idx and b in idx and a!=b: Y[idx[a],idx[b]]=1
    return Y, [str(k) for k in keep]

NETS={'chess150': (lambda: build_chess(150)), 'chess300': (lambda: build_chess(300)), 'email400': (lambda: build_edgelist('data/benchmarks/email.txt',400)), 'email250': (lambda: build_edgelist('data/benchmarks/email.txt',250)), 'wikivote150': (lambda: build_edgelist('data/benchmarks/wiki.txt',150)), 'email150': (lambda: build_edgelist('data/benchmarks/email.txt',150))}

def screen(names):
    R=res_load(); out=R.get('games_rotfrac',{})
    for name in names:
        Y,labels=NETS[name](); N=len(Y); S=Y+Y.T; iu=np.triu_indices(N,1); rf,asym=rotational_fraction(Y)
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

if __name__=='__main__':
    a=sys.argv[1:]
    if a[0]=='screen': screen(a[1:])
    elif a[0]=='directed': run_directed(a[1], a[2:])
