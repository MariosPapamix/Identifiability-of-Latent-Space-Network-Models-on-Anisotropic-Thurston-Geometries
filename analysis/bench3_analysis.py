import sys, os, re, collections, numpy as np
sys.path.insert(0,'code'); sys.path.insert(0,'analysis')
from bench_analysis import screen as bscreen
import bench_analysis as BA
from games_analysis import build_pokemon
B='data/benchmarks'
def cit(path, n=200):
    E=[]; deg=collections.Counter()
    for l in open(path):
        if l.startswith('#'): continue
        a,b=map(int,l.split()[:2]); E.append((a,b)); deg[a]+=1; deg[b]+=1
    adj=collections.defaultdict(set)
    for a,b in E: adj[a].add(b); adj[b].add(a)
    degs=sorted(adj,key=lambda v:-len(adj[v])); start=degs[len(degs)//50]
    seen=[start]; S={start}; q=[start]
    while q and len(seen)<n:
        v=q.pop(0)
        for w in sorted(adj[v],key=lambda w:-len(adj[w])):
            if w not in S: S.add(w); seen.append(w); q.append(w)
            if len(seen)>=n: break
    idx={v:i for i,v in enumerate(seen)}; N=len(seen); Y=np.zeros((N,N))
    for a,b in E:
        if a in idx and b in idx and a!=b: Y[idx[a],idx[b]]=1
    return Y, [str(v) for v in seen]
def polblogs(top=150):
    g=open(f'{B}/polblogs.gml').read(); E=re.findall(r'source\s+(\d+)\s+target\s+(\d+)', g); E=[(int(a),int(b)) for a,b in E]
    deg=collections.Counter()
    for a,b in E: deg[a]+=1; deg[b]+=1
    keep=[k for k,_ in deg.most_common(top)]; idx={k:i for i,k in enumerate(keep)}; N=len(keep); Y=np.zeros((N,N))
    for a,b in E:
        if a in idx and b in idx and a!=b: Y[idx[a],idx[b]]=1
    return Y, [str(k) for k in keep]
BA.NETS.update({'hepth200': lambda: cit(f'{B}/cit-HepTh.txt',200), 'hepph200': lambda: cit(f'{B}/cit-HepPh.txt',200), 'polblogs150': lambda: polblogs(150),
                'natdex_apr': lambda: build_pokemon(fname='data/games/gen9nationaldex-1500_april_json'), 'natdex_jul': lambda: build_pokemon(fname='data/games/gen9nationaldex-1500_july_json')})
if __name__=='__main__':
    a=sys.argv[1:]
    if a[0]=='screen': bscreen(a[1:])
    elif a[0]=='directed': BA.run_directed(a[1], a[2:])
