"""Competition tournaments from Soliveres et al. (Dryad doi:10.5061/dryad.bh41r): Y_ij = 1 if species i outcompetes j."""
import json, sys, os, time, numpy as np
sys.path.insert(0, 'code'); sys.path.insert(0, 'analysis')
from connectome_analysis import rotational_fraction, res_load, res_save
from directed import fit_directed, predictive_directed
from experiments2 import GEOS
from inference import logloss, auc
MATS = json.load(open('data/tournaments/matrices_binomial.json'))
NAMES = {'moss:moss.control:1:0': 'moss_control', 'moss:moss.fert:1:11': 'moss_fertilised', 'protists:protists:0:0': 'protists', 'EU.fungi:EU.fungi:0:0': 'fungi_EU',
         'vascular.plants:vascular.plants:0:0': 'vascular_plants', 'bacteria:poor.soil:1:0': 'bacteria', 'US.fungi:US.fungi:0:0': 'fungi_US'}

def load(name):
    key = [k for k, v in NAMES.items() if v == name][0]; M = np.array(MATS[key]['M'], dtype=float); np.fill_diagonal(M, 0); return M

def cyclic_triads(Y):
    N = len(Y); cyc = dec = 0
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                s = [(Y[a, b] - Y[b, a]) for a, b in ((i, j), (j, k), (k, i))]
                if all(abs(v) == 1 for v in s):
                    dec += 1; cyc += (abs(sum(s)) == 3)
    return cyc / max(dec, 1), dec

def run_screen():
    R = res_load(); out = R.get('intrans_rotfrac', {})
    for name in NAMES.values():
        Y = load(name); N = len(Y); rf, asym = rotational_fraction(Y); cf, dec = cyclic_triads(Y)
        rng = np.random.default_rng(0); vals = []; cyc0 = []
        S = Y + Y.T; iu = np.triu_indices(N, 1); one = S[iu] == 1; both = S[iu] == 2
        for _ in range(50):
            Yn = np.zeros((N, N)); Yn[iu[0][both], iu[1][both]] = 1; Yn[iu[1][both], iu[0][both]] = 1
            flip = rng.uniform(size=one.sum()) < 0.5; a, b = iu[0][one], iu[1][one]; Yn[a[flip], b[flip]] = 1; Yn[b[~flip], a[~flip]] = 1
            vals.append(rotational_fraction(Yn)[0]); cyc0.append(cyclic_triads(Yn)[0])
        out[name] = dict(N=N, density=float(Y.sum() / (N * (N - 1))), frac_asym_pairs=asym, rotfrac=rf, rotfrac_random=float(np.mean(vals)), cyclic_triads=cf, cyclic_random=float(np.mean(cyc0)), decided_triads=int(dec))
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
    R['intrans_rotfrac'] = out; res_save(R)

def run_directed(names, nsplit=int(os.environ.get('NSPLIT',5)), frac=0.15, fits=tuple(os.environ.get('FITS','nil,sl2,h2r,euc,nil_sym').split(','))):
    for name in names:
        Y = load(name); N = len(Y); iu = np.triu_indices(N, 1); off = ~np.eye(N, dtype=bool)
        for split in range(nsplit):
            key = f'{name}_s{split}'; R = res_load(); rec = R.get('intrans_directed', {}).get(key, dict(N=N, fits={}))
            rng = np.random.default_rng(500 + split); hold = rng.uniform(size=len(iu[0])) < frac
            mask = np.ones((N, N)); mask[iu[0][hold], iu[1][hold]] = 0; mask[iu[1][hold], iu[0][hold]] = 0
            H = (mask == 0) & off; tr = (mask == 1) & off
            for f in fits:
                if f in rec['fits']: continue
                g = f.replace('_sym', ''); lg = not f.endswith('_sym'); t0 = time.time(); best = None
                for g0 in ((0.0, 0.4, 1.0) if (lg and g in ('nil', 'sl2')) else (0.0, 0.4)):
                    m_, v_ = fit_directed(GEOS[g], Y, mask, learn_gamma=lg, n_iter=800, S=8, seed=split, gamma0=g0)
                    P_ = predictive_directed(m_, v_, n=100); trl = float(logloss(P_[tr], Y[tr]))
                    if best is None or trl < best[0]: best = (trl, m_, v_, P_)
                trl, model, vi, Pf = best
                one = ((Y + Y.T) == 1) & (mask == 0) & np.triu(np.ones((N, N), bool), 1); ii, jj = np.where(one)
                sc = np.log(Pf[ii, jj] + 1e-9) - np.log(Pf[jj, ii] + 1e-9); acc = float(np.mean((sc > 0) == (Y[ii, jj] == 1))) if len(ii) else float('nan')
                rec['fits'][f] = dict(logloss=float(logloss(Pf[H], Y[H])), train_logloss=trl, direction_acc=acc, n_asym_pairs=int(len(ii)), gamma_hat=float(model.gamma), time=time.time() - t0)
                R = res_load(); R.setdefault('intrans_directed', {})[key] = rec; res_save(R)
                print(key, f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec['fits'][f].items()}, flush=True)

if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'screen': run_screen()
    elif a[0] == 'directed': run_directed(a[1:])
