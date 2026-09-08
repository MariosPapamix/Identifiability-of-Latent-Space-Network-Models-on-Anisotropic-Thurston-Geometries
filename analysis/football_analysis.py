"""Football win networks from openfootball (CC0): nodes are clubs, Y_ij = 1 if i won more of its matches against j than
it lost over the seasons available (dominance), and the same screen and fits as for the connectomes."""
import json, glob, collections, sys, os, time, numpy as np
sys.path.insert(0, 'code'); sys.path.insert(0, 'analysis')
from connectome_analysis import rotational_fraction, res_load, res_save
from directed import fit_directed, predictive_directed
from experiments2 import GEOS
from inference import logloss, auc

def build(leagues, variant='dominance'):
    wins = collections.Counter(); teams = set()
    for f in sorted(glob.glob('data/football/20*.json')):
        if not any(f.endswith(f'.{l}.json') for l in leagues):
            continue
        try: d = json.load(open(f))
        except Exception: continue
        for m in d.get('matches', []):
            sc = m.get('score', {}).get('ft')
            if not sc: continue
            t1, t2 = m['team1'], m['team2']; teams.update([t1, t2])
            if sc[0] > sc[1]: wins[(t1, t2)] += 1
            elif sc[1] > sc[0]: wins[(t2, t1)] += 1
    teams = sorted(teams); idx = {t: i for i, t in enumerate(teams)}; N = len(teams); Y = np.zeros((N, N))
    for (a, b), w in wins.items():
        i, j = idx[a], idx[b]
        if variant == 'dominance':
            if w > wins[(b, a)]: Y[i, j] = 1
        else:
            Y[i, j] = 1
    return Y, teams

NETS = {'england': ['en.1'], 'germany': ['de.1'], 'spain': ['es.1'], 'italy': ['it.1'], 'france': ['fr.1'], 'big5': ['en.1', 'de.1', 'es.1', 'it.1', 'fr.1']}

def run_rotfrac():
    R = res_load(); out = R.get('football_rotfrac', {})
    for name, lg in NETS.items():
        for variant in ('dominance', 'anywin'):
            Y, teams = build(lg, variant); N = len(Y); rf, asym = rotational_fraction(Y)
            rng = np.random.default_rng(0); vals = []
            S = Y + Y.T; iu = np.triu_indices(N, 1); one = S[iu] == 1; both = S[iu] == 2
            for _ in range(20):
                Yn = np.zeros((N, N)); Yn[iu[0][both], iu[1][both]] = 1; Yn[iu[1][both], iu[0][both]] = 1
                flip = rng.uniform(size=one.sum()) < 0.5; a, b = iu[0][one], iu[1][one]; Yn[a[flip], b[flip]] = 1; Yn[b[~flip], a[~flip]] = 1
                vals.append(rotational_fraction(Yn)[0])
            out[f'{name}_{variant}'] = dict(N=N, density=float(Y.sum() / (N * (N - 1))), rotfrac=rf, rotfrac_random_directions=float(np.mean(vals)), frac_asym_pairs=asym)
            print(name, variant, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[f'{name}_{variant}'].items()}, flush=True)
    R['football_rotfrac'] = out; res_save(R)

def run_directed(names, variant='dominance', nsplit=int(os.environ.get('NSPLIT', 3)), frac=0.1, fits=('nil', 'sl2', 'h2r', 'euc', 'nil_sym')):
    for name in names:
        Y, teams = build(NETS[name], variant); N = len(Y); iu = np.triu_indices(N, 1); off = ~np.eye(N, dtype=bool); big = N > 150
        for split in range(nsplit):
            key = f'{name}_{variant}_s{split}'
            R = res_load(); rec = R.get('football_directed', {}).get(key, dict(N=N, density=float(Y.sum() / (N * (N - 1))), fits={}))
            rng = np.random.default_rng(400 + split); hold = rng.uniform(size=len(iu[0])) < frac
            mask = np.ones((N, N)); mask[iu[0][hold], iu[1][hold]] = 0; mask[iu[1][hold], iu[0][hold]] = 0
            H = (mask == 0) & off; tr = (mask == 1) & off
            for f in fits:
                if f in rec['fits']: continue
                g = f.replace('_sym', ''); lg = not f.endswith('_sym'); t0 = time.time(); best = None
                for g0 in ((0.0, 0.4) if (lg and g in ('nil', 'sl2')) else (0.0,)):
                    m_, v_ = fit_directed(GEOS[g], Y, mask, learn_gamma=lg, n_iter=(400 if big else 600), S=(6 if big else 8), seed=split, gamma0=g0)
                    P_ = predictive_directed(m_, v_, n=60 if big else 100); trl = float(logloss(P_[tr], Y[tr]))
                    if best is None or trl < best[0]: best = (trl, m_, v_, P_)
                trl, model, vi, Pf = best
                one = ((Y + Y.T) == 1) & (mask == 0) & np.triu(np.ones((N, N), bool), 1); ii, jj = np.where(one)
                sc = np.log(Pf[ii, jj] + 1e-9) - np.log(Pf[jj, ii] + 1e-9); acc = float(np.mean((sc > 0) == (Y[ii, jj] == 1))) if len(ii) else float('nan')
                rec['fits'][f] = dict(logloss=float(logloss(Pf[H], Y[H])), train_logloss=trl, auc=float(auc(Pf[H], Y[H])), direction_acc=acc, n_asym_pairs=int(len(ii)), gamma_hat=float(model.gamma), time=time.time() - t0)
                R = res_load(); R.setdefault('football_directed', {})[key] = rec; res_save(R)
                print(key, f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec['fits'][f].items()}, flush=True)

if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'rotfrac': run_rotfrac()
    elif a[0] == 'directed': run_directed(a[2:], variant=a[1])
