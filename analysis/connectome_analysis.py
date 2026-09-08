"""
Real connectomes: (1) rotational fraction of the asymmetry, (2) circulation model against gradient models with
held-out dyads, (3) undirected geometry comparison on the Allard-Serrano networks used by Celinska-Kopczynska and
Kopczynski, with their reported results extracted from brains.csv.
"""
import numpy as np, json, os, sys, time, csv, collections
sys.path.insert(0, 'code')
from inference import sigmoid, logloss, auc
from experiments2 import GEOS, vi_fit
from directed import DirectedLSM, fit_directed, predictive_directed
import directed as D

NAV = 'data/connectomes/allard_serrano_2020/maps'
NS = 'data/connectomes/netzschleuder'
RESF = 'results/results_real_networks.json'


def load_allard(name):
    E = [tuple(l.split()[:2]) for l in open(f'{NAV}/{name}/{name}.edge') if l.strip()]
    nodes = sorted({u for e in E for u in e}, key=lambda s: (0, int(s)) if s.isdigit() else (1, s)); idx = {u: i for i, u in enumerate(nodes)}
    N = len(nodes); Y = np.zeros((N, N))
    for a, b in E:
        if a != b:
            Y[idx[a], idx[b]] = 1
    S = Y + Y.T; pairs = (np.triu(S, 1) > 0).sum(); one = (np.triu(S, 1) == 1).sum()
    directed = 0.02 < one / max(pairs, 1) < 0.98
    return (Y if directed else (S > 0).astype(float)), directed, one / max(pairs, 1)


def load_ns(folder, neurons_only=True):
    edges = [l.strip().split(',') for l in open(f'{NS}/{folder}/edges.csv') if l.strip() and not l.startswith('#')]
    nodes = [l.strip().split(',') for l in open(f'{NS}/{folder}/nodes.csv') if l.strip() and not l.startswith('#')]
    keep = set()
    for row in nodes:
        idx = int(row[0]); ntype = row[1] if len(row) > 1 else ''
        if (not neurons_only) or ntype not in ('BODYWALL MUSCLES', 'OTHER END ORGANS'):
            keep.add(idx)
    ids = sorted(keep); pos = {v: i for i, v in enumerate(ids)}; N = len(ids); Y = np.zeros((N, N))
    for e in edges:
        a, b = int(e[0]), int(e[1])
        if a in pos and b in pos and a != b:
            Y[pos[a], pos[b]] = 1
    return Y


def rotational_fraction(Y):
    F = Y - Y.T; r = F.sum(1); N = len(Y)
    G = (r[:, None] - r[None, :]) / N            # gradient (coboundary) component
    P = F - G                                    # cycle-space component
    iu = np.triu_indices(N, 1)
    return float(np.sum(P[iu] ** 2) / max(np.sum(F[iu] ** 2), 1e-12)), float((F[iu] != 0).mean())


NETWORKS = {}
for name in sorted(os.listdir(NAV)):
    if os.path.exists(f'{NAV}/{name}/{name}.edge'):
        NETWORKS[name] = ('allard', name)
NETWORKS['CElegans2019_herm'] = ('ns', 'hermaphrodite_chemical_corrected_csv')
NETWORKS['CElegans2019_male'] = ('ns', 'male_chemical_corrected_csv')


def load(name):
    kind, key = NETWORKS[name]
    if kind == 'allard':
        return load_allard(key)
    Y = load_ns(key); return Y, True, float((np.triu(Y + Y.T, 1) == 1).sum() / max((np.triu(Y + Y.T, 1) > 0).sum(), 1))


def res_load():
    return json.load(open(RESF)) if os.path.exists(RESF) else {}


def res_save(R):
    json.dump(R, open(RESF, 'w'), indent=1)


def run_rotfrac():
    R = res_load(); out = R.get('rotfrac', {})
    for name in NETWORKS:
        Y, directed, frac_one = load(name); N = len(Y); dens = float(Y.sum() / (N * (N - 1)))
        rf, asym = rotational_fraction(Y) if directed else (0.0, 0.0)
        null = float('nan')
        if directed:
            rng = np.random.default_rng(0); vals = []
            for _ in range(20):
                Yn = np.maximum(Y, Y.T) * 0; S = Y + Y.T; iu = np.triu_indices(N, 1)
                one = (S[iu] == 1); both = (S[iu] == 2); flip = rng.uniform(size=one.sum()) < 0.5
                Yn[iu[0][both], iu[1][both]] = 1; Yn[iu[1][both], iu[0][both]] = 1
                a, b = iu[0][one], iu[1][one]; Yn[a[flip], b[flip]] = 1; Yn[b[~flip], a[~flip]] = 1
                vals.append(rotational_fraction(Yn)[0])
            null = float(np.mean(vals))
        out[name] = dict(N=N, density=dens, directed=bool(directed), frac_one_directional=frac_one, rotfrac=rf, rotfrac_random_directions=null, frac_asym_pairs=asym)
        print(name, out[name], flush=True)
    R['rotfrac'] = out; res_save(R)


def run_directed(names, nsplit=int(os.environ.get('NSPLIT', 2)), frac=0.1, fits=('nil', 'nil_sym', 'sl2', 'h2r', 'euc')):
    for name in names:
        Y, directed, _ = load(name); N = len(Y)
        if not directed:
            continue
        iu = np.triu_indices(N, 1); off = ~np.eye(N, dtype=bool)
        big = N > 300
        for split in range(nsplit):
            R = res_load(); rec = R.get('directed', {}).get(f'{name}_s{split}', dict(N=N, density=float(Y.sum() / (N * (N - 1))), fits={}))
            rng = np.random.default_rng(200 + split); hold = rng.uniform(size=len(iu[0])) < frac
            mask = np.ones((N, N)); mask[iu[0][hold], iu[1][hold]] = 0; mask[iu[1][hold], iu[0][hold]] = 0
            H = (mask == 0) & off
            for f in fits:
                if f in rec['fits']:
                    continue
                g = f.replace('_sym', ''); lg = not f.endswith('_sym')
                t0 = time.time(); best = None
                tr = (mask == 1) & off
                for g0 in ((0.0, 0.4) if (lg and g in ('nil', 'sl2')) else (0.0,)):   # restarts, chosen by training log-loss
                    model_, vi_ = fit_directed(GEOS[g], Y, mask, learn_gamma=lg, n_iter=(400 if big else 600), S=(6 if big else 8), seed=split, gamma0=g0)
                    Pf_ = predictive_directed(model_, vi_, n=60 if big else 100); trl = float(logloss(Pf_[tr], Y[tr]))
                    if best is None or trl < best[0]:
                        best = (trl, model_, vi_, Pf_)
                trl, model, vi, Pf = best
                one = ((Y + Y.T) == 1) & (mask == 0) & np.triu(np.ones((N, N), bool), 1)
                ii, jj = np.where(one); sc = np.log(Pf[ii, jj] + 1e-9) - np.log(Pf[jj, ii] + 1e-9)
                acc = float(np.mean((sc > 0) == (Y[ii, jj] == 1))) if len(ii) else float('nan')
                rec['fits'][f] = dict(logloss=float(logloss(Pf[H], Y[H])), train_logloss=trl, auc=float(auc(Pf[H], Y[H])), direction_acc=acc, n_asym_pairs=int(len(ii)), gamma_hat=float(model.gamma), time=time.time() - t0)
                R = res_load(); R.setdefault('directed', {})[f'{name}_s{split}'] = rec; res_save(R)
                print(name, split, f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec['fits'][f].items()}, flush=True)


def run_undirected(names, geos=('nil', 'sol', 'sl2', 'euc', 'h3', 'h2r'), frac=0.1):
    for name in names:
        Y, directed, _ = load(name); Y = ((Y + Y.T) > 0).astype(float); N = len(Y); iu = np.triu_indices(N, 1)
        rng = np.random.default_rng(300); hold = rng.uniform(size=len(iu[0])) < frac
        mask = np.ones((N, N)); mask[iu[0][hold], iu[1][hold]] = 0; mask[iu[1][hold], iu[0][hold]] = 0
        big = N > 300
        for g in geos:
            R = res_load(); rec = R.get('undirected', {}).get(name, dict(N=N, density=float(Y[iu].mean()), fits={}))
            if g in rec['fits']:
                continue
            t0 = time.time()
            m, vi = vi_fit(GEOS[g], Y, mask=mask, n_iter=(400 if big else 600), S=(6 if big else 8), seed=0, M=(20000 if big else None))
            P = vi.predictive(60 if big else 100)
            rec['fits'][g] = dict(logloss=float(logloss(P[iu][hold], Y[iu][hold])), auc=float(auc(P[iu][hold], Y[iu][hold])), time=time.time() - t0)
            R = res_load(); R.setdefault('undirected', {})[name] = rec; res_save(R)
            print(name, g, {k: round(v, 4) for k, v in rec['fits'][g].items()}, flush=True)


def neighbourhood_scores(Dm, Y):
    """Mean average precision and mean rank of true neighbours, ranking the other nodes of each node by fitted distance
    (the criteria used in the connectome embedding literature; Nickel and Kiela 2017, Celinska-Kopczynska and Kopczynski 2024)."""
    N = len(Y); aps = []; ranks = []
    for i in range(N):
        nb = np.where(Y[i] > 0)[0]
        if len(nb) == 0:
            continue
        d = Dm[i].copy(); d[i] = np.inf; order = np.argsort(d); rank = np.empty(N); rank[order] = np.arange(1, N + 1)
        r = np.sort(rank[nb]); prec = np.arange(1, len(r) + 1) / r
        aps.append(prec.mean()); ranks.extend(r.tolist())
    return float(np.mean(aps)), float(np.mean(ranks))


def run_map(names, geos=('nil', 'sol', 'sl2', 'euc', 'h3', 'h2r'), beta=float(os.environ.get('BETA', 1.0))):
    """Full-graph fits (no holdout, as in the embedding literature) and neighbourhood criteria from the posterior mean distances."""
    for name in names:
        Y, directed, _ = load(name); Y = ((Y + Y.T) > 0).astype(float); N = len(Y); iu = np.triu_indices(N, 1); big = N > 300
        for g in geos:
            key = 'map' if beta == 1.0 else f'map_b{beta:g}'
            R = res_load(); rec = R.get(key, {}).get(name, dict(N=N, density=float(Y[iu].mean()), fits={}))
            if g in rec['fits']:
                continue
            t0 = time.time()
            m, vi = vi_fit(GEOS[g], Y, mask=None, n_iter=(400 if big else 600), S=(6 if big else 8), seed=0, M=(20000 if big else None), beta=beta)
            S0 = vi.S; vi.S = 60; Zq = vi.sample()[0]; vi.S = S0
            Dm = np.mean([GEOS[g].pdist(Zq[k]) for k in range(len(Zq))], 0)
            P = vi.predictive(60)
            mAP, mr = neighbourhood_scores(Dm, Y)
            rec['fits'][g] = dict(mAP=mAP, mean_rank=mr, train_logloss=float(logloss(P[iu], Y[iu])), time=time.time() - t0)
            R = res_load(); R.setdefault(key, {})[name] = rec; res_save(R)
            print(name, g, {k: round(v, 4) for k, v in rec['fits'][g].items()}, flush=True)


def run_anneal_map(names, geos=('nil', 'sol', 'sl2', 'euc', 'h3', 'h2r'), n_sweeps=int(os.environ.get('SWEEPS', 80))):
    """Annealed maximum likelihood with learned (R,T) on the full graph (the protocol of the embedding literature),
    neighbourhood criteria from the point estimate."""
    from experiments3 import anneal_fit
    for name in names:
        Y, directed, _ = load(name); Y = ((Y + Y.T) > 0).astype(float); N = len(Y); iu = np.triu_indices(N, 1)
        for g in geos:
            R = res_load(); rec = R.get('anneal_map', {}).get(name, dict(N=N, density=float(Y[iu].mean()), fits={}))
            if g in rec['fits']:
                continue
            t0 = time.time(); best = None
            for seed in range(3):
                Z, alpha, beta = anneal_fit(GEOS[g], Y, np.ones((N, N)), n_sweeps=n_sweeps, seed=seed)
                Dm = GEOS[g].pdist(Z); P = sigmoid(alpha - beta * Dm); ll = float(logloss(P[iu], Y[iu]))
                if best is None or ll < best[0]:
                    best = (ll, Dm, alpha, beta)
            ll, Dm, alpha, beta = best; mAP, mr = neighbourhood_scores(Dm, Y)
            rec['fits'][g] = dict(mAP=mAP, mean_rank=mr, train_logloss=ll, alpha=float(alpha), beta=float(beta), time=time.time() - t0)
            R = res_load(); R.setdefault('anneal_map', {})[name] = rec; res_save(R)
            print(name, g, {k: round(v, 4) for k, v in rec['fits'][g].items()}, flush=True)


CKK_CLASS = {'hyperbolic3': ('g435', 'g435b2', 'g435ch', 'g435d', 'h3m', 'BIG-g435', 'BIG-g435b2'), 'hyperbolic2': ('g711', 'g711ch', 'g711d', 'h2m'),
             'product': ('h2r', 'h2ra', 'BIG-h2r'), 'euclid': ('e3b', 't3'), 'sphere': ('s3',), 'nil': ('nil', 'subnil'),
             'solv': ('sol3', 'sol3lie', 'subsol3', 'subsol3lie', 'BIG-sol3'), 'twist': ('twist', 'BIG-twist')}


def run_ckk():
    rows = list(csv.DictReader(open('data/connectomes/ckk_results/brains.csv'), delimiter=';'))
    code2class = {c: k for k, v in CKK_CLASS.items() for c in v}
    best = collections.defaultdict(dict)
    for r in rows:
        k = code2class.get(r['geom']);
        if k is None:
            continue
        who = r['who']; ll = float(r['loglik']); mp = float(r['mAP'])
        b = best[who].get(k)
        if b is None or mp > b['mAP']:
            best[who][k] = dict(mAP=mp, loglik=ll, opt_r=float(r['opt_r']), opt_t=float(r['opt_t']), geom=r['geom'], n=int(r['n']), m=int(r['m']))
    R = res_load(); R['ckk'] = best; res_save(R)
    for who in sorted(best):
        rk = sorted(best[who], key=lambda k: -best[who][k]['mAP'])
        print(who, 'mAP ranking:', [(k, round(best[who][k]['mAP'], 3)) for k in rk])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'rotfrac': run_rotfrac()
    elif a[0] == 'directed': run_directed(a[1:])
    elif a[0] == 'undirected': run_undirected(a[1:])
    elif a[0] == 'ckk': run_ckk()
    elif a[0] == 'map': run_map(a[1:])
    elif a[0] == 'anneal_map': run_anneal_map(a[1:])
