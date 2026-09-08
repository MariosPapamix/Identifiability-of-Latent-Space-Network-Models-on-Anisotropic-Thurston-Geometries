import numpy as np, pickle, json, time, sys, os
import networkx as nx
from scipy.optimize import least_squares
from geometry import *
from inference import *
from inference import predictive_mcmc
from competitors import *
from competitors import _logistic
from experiments2 import GEOS, gen_network, load_real, bvp_gradient, _shoot_sol, _shoot_sl2, vi_fit

os.makedirs('results3', exist_ok=True)
RES = json.load(open('results3/results.json')) if os.path.exists('results3/results.json') else {}


def save():
    json.dump(RES, open('results3/results.json', 'w'), indent=1, default=float)


# ===========================================================================
#  (1) Augmented rank certificates [1, -J] and singular-value distributions
# ===========================================================================
def jacobian_nil_exact(Z, h=1e-5):
    N = Z.shape[0]; iu = np.triu_indices(N, 1); g = Nil(None)
    J = np.zeros((len(iu[0]), 3 * N))
    for k in range(3 * N):
        Zp = Z.copy(); Zp.flat[k] += h; Zm = Z.copy(); Zm.flat[k] -= h
        J[:, k] = (g.pdist(Zp)[iu] - g.pdist(Zm)[iu]) / (2 * h)
    return J


def slice_basis(gname, Z):
    """Orthonormal basis of the tangent space of the gauge slice (anchors 0,1) in coordinates."""
    N = Z.shape[0]; cols = []
    for i in range(N):
        for k in range(3):
            if i == 0:
                continue
            if i == 1 and gname != 'sol' and k == 1:
                continue
            e = np.zeros(3 * N); e[3 * i + k] = 1; cols.append(e)
    return np.stack(cols, 1)


def run_rigidity_nil(Ns=(6, 7, 8, 10), reps=100, seed=0):
    key = 'rig_nil'
    if key in RES:
        return
    rng = np.random.default_rng(seed); out = {}
    for N in Ns:
        sk = []; sk_aug = []; ndy = N * (N - 1) // 2
        for r in range(reps):
            Z = Nil(None).gauge(Nil(None).chart(rng.normal(size=(N, 3)) * 1.0), [0, 1])
            J = jacobian_nil_exact(Z) @ slice_basis('nil', Z)           # ndy x (3N-4)
            s = np.linalg.svd(J, compute_uv=False); sk.append(s[-1] if J.shape[0] >= J.shape[1] else 0.0)
            A = np.column_stack([np.ones(ndy), -J])                       # ndy x (3N-3)
            sa = np.linalg.svd(A, compute_uv=False); sk_aug.append(sa[-1] if A.shape[0] >= A.shape[1] else 0.0)
        out[str(N)] = dict(N=N, ndyads=ndy, k=3 * N - 4, s_min_slice=dict(min=float(np.min(sk)), q05=float(np.quantile(sk, .05)), median=float(np.median(sk)), max=float(np.max(sk))),
                           s_min_aug=dict(min=float(np.min(sk_aug)), q05=float(np.quantile(sk_aug, .05)), median=float(np.median(sk_aug))), frac_rank_deficient=float(np.mean(np.array(sk) < 1e-8)),
                           frac_aug_deficient=float(np.mean(np.array(sk_aug) < 1e-8)))
        print('nil', N, out[str(N)], flush=True)
    RES[key] = out; save()


def run_rigidity_bvp_aug(gname, N, seed=0):
    key = f'rig_{gname}_N{N}'
    if key in RES:
        return
    geo = GEOS[gname]; rng = np.random.default_rng(seed); iu = np.triu_indices(N, 1)
    Z = geo.gauge(geo.chart(rng.normal(size=(N, 3)) * 0.9), [0, 1])
    if not geo.anchor2_ok(Z[1]):
        Z[1] = np.abs(Z[1]) + 0.05
    J = np.zeros((len(iu[0]), 3 * N)); h = 1e-6; ok = True
    for row, (i, j) in enumerate(zip(*iu)):
        for (p_idx, q_idx) in ((i, j), (j, i)):
            g = geo.translate_inv(Z[p_idx], Z[q_idx])
            d, grad = bvp_gradient(gname, g, rng=rng, dtab=float(geo.dist_e(g[None])[0]))
            if grad is None:
                ok = False; continue
            Jt = np.zeros((3, 3))
            for k in range(3):
                qp = Z[q_idx].copy(); qp[k] += h; qm = Z[q_idx].copy(); qm[k] -= h
                Jt[:, k] = (geo.translate_inv(Z[p_idx], qp) - geo.translate_inv(Z[p_idx], qm)) / (2 * h)
            J[row, 3 * q_idx:3 * q_idx + 3] = Jt.T @ grad
    Js = J @ slice_basis(gname, Z); ndy = len(iu[0])
    s = np.linalg.svd(Js, compute_uv=False); A = np.column_stack([np.ones(ndy), -Js]); sa = np.linalg.svd(A, compute_uv=False)
    RES[key] = dict(N=N, complete=ok, ndyads=ndy, slice_dim=Js.shape[1], sv_slice=s.tolist(), sv_aug=sa.tolist(),
                    rank_slice=int(np.sum(s > 1e-7 * s[0])), rank_aug=int(np.sum(sa > 1e-7 * sa[0])))
    save(); print(key, 'dyads', ndy, 'slice dim', Js.shape[1], 'rank', RES[key]['rank_slice'], 'aug rank', RES[key]['rank_aug'], 'smallest', np.round(s[-2:], 4), np.round(sa[-2:], 4), flush=True)


# ===========================================================================
#  (2) Representation deficit: additive-constant l2 distortion into a competitor geometry
# ===========================================================================
def deficit(Dtrue, geo_t, N, rng, nstart=3, w=None, return_fit=False):
    """Fisher deficit: inf over Z' in T^N and c of (1/2) sum w_ij (d_ij - d'_ij - c)^2, w_ij = p_ij(1-p_ij)."""
    iu = np.triu_indices(N, 1); dt = Dtrue[iu]; best = np.inf
    sw = np.sqrt(0.5 * (w if w is not None else np.ones_like(dt)))
    for s in range(nstart):
        X0 = classical_mds(Dtrue, 3); X0 = X0 / np.std(X0) * 1.0 + 0.05 * rng.normal(size=X0.shape) if s == 0 else rng.normal(size=(N, 3))
        x0 = np.concatenate([geo_t.chart(X0).ravel() if geo_t.name != 'euc' else X0.ravel(), [0.0]])
        def res(x):
            Z = x[:-1].reshape(N, 3)
            if geo_t.name in ('h3',):
                Z = np.clip(Z, -0.999, 0.999)
            if geo_t.name in ('h2r',):
                nrm = np.hypot(Z[:, 0], Z[:, 1]); f = np.where(nrm > 0.999, 0.999 / np.maximum(nrm, 1e-12), 1.0); Z = Z * np.stack([f, f, np.ones(N)], 1)
            return sw * (geo_t.pdist(Z)[iu] - dt - x[-1])
        r = least_squares(res, x0, max_nfev=300)
        val = float(np.sum(r.fun ** 2))
        if val < best:
            best = val; bestx = r.x.copy()
    if return_fit:
        Z = bestx[:-1].reshape(N, 3)
        if geo_t.name in ('h3',):
            Z = np.clip(Z, -0.999, 0.999)
        if geo_t.name in ('h2r',):
            nrm = np.hypot(Z[:, 0], Z[:, 1]); f = np.where(nrm > 0.999, 0.999 / np.maximum(nrm, 1e-12), 1.0); Z = Z * np.stack([f, f, np.ones(N)], 1)
        return best, Z, float(bestx[-1])
    return best


def run_deficit(seed=21):
    """Delta_T(Z)^2 for the win-map configurations (same seeds as experiments2.run_winmap)."""
    key = 'deficit'
    if key in RES and RES[key].get('complete'):
        return
    settings = {'nil': dict(vals=(0.5, 1.0, 2.0, 3.0), sigv=1.5, targets=('euc', 'h3', 'h2r'), Ns=(40, 100)),
                'sl2': dict(vals=(0.5, 1.0, 2.0), sigv=2.0, targets=('h3', 'h2r', 'euc'), Ns=(60,)),
                'sol': dict(vals=(0.3, 0.8, 1.4, 2.0), sigh=1.5, targets=('euc', 'h3', 'h2r'), Ns=(60,))}
    out = RES.get(key, {})
    for truth, st in settings.items():
        gt = GEOS[truth]; rng = np.random.default_rng(seed)
        for N in st['Ns']:
            for val in st['vals']:
                sig = (val, st['sigv']) if truth != 'sol' else (st['sigh'], val)
                Zt, P, alpha, Y, Yrep = gen_network(gt, N, sig, None, rng, density=0.15)
                if f'{truth}_N{N}_{val}' in out:
                    continue
                D = gt.pdist(Zt); iu = np.triu_indices(N, 1); ndy = len(iu[0])
                eta = alpha - D[iu]; pp = sigmoid(eta); w = pp * (1 - pp)
                r = dict(N=N, sig=sig, alpha=alpha, ndyads=ndy, mean_w=float(w.mean()), deficits={})
                for tname in st['targets']:
                    d2 = deficit(D, GEOS[tname], N, np.random.default_rng(1), w=w, nstart=1 if N >= 100 else 2)
                    r['deficits'][tname] = dict(fisher_deficit=d2, per_dyad=d2 / ndy, per_parameter=d2 / (0.5 * (3 * N + 1)))
                out[f'{truth}_N{N}_{val}'] = r; RES[key] = out; save()
                print(truth, N, val, {t: (round(v['fisher_deficit'], 2), round(v['per_dyad'], 5), round(v['per_parameter'], 2)) for t, v in r['deficits'].items()}, flush=True)
    out['complete'] = True; RES[key] = out; save()


# ===========================================================================
#  (3) Replicated coverage study
# ===========================================================================
def bern_kl(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12); q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def deficit_kl(Dtrue, P, alpha, geo_t, N, rng, x_inits, max_nfev=300):
    """Exact dyad KL minimised over (Z', alpha') by least squares on deviance residuals
    r_ij = sign(p - q) sqrt(2 KL(p||q)), whose squared sum is 2 * sum KL. Returns (KL_min, Z', alpha')."""
    iu = np.triu_indices(N, 1); pp = P[iu]; best = (np.inf, None, None)

    def proj(Z):
        if geo_t.name == 'h3':
            Z = np.clip(Z, -0.999, 0.999)
        if geo_t.name == 'h2r':
            nrm = np.hypot(Z[:, 0], Z[:, 1]); f = np.where(nrm > 0.999, 0.999 / np.maximum(nrm, 1e-12), 1.0); Z = Z * np.stack([f, f, np.ones(N)], 1)
        return Z

    def res(x):
        Z = proj(x[:-1].reshape(N, 3)); q = sigmoid(x[-1] - geo_t.pdist(Z)[iu])
        kl = bern_kl(pp, q)
        return np.sign(pp - q) * np.sqrt(2 * np.maximum(kl, 0))

    for x0 in x_inits:
        r = least_squares(res, x0, max_nfev=max_nfev)
        val = 0.5 * float(np.sum(r.fun ** 2))
        if val < best[0]:
            best = (val, proj(r.x[:-1].reshape(N, 3)), float(r.x[-1]))
    return best


def run_deficit_kl(seed=21, only_N=None):
    key = 'deficit_kl'
    settings = {'nil': dict(vals=(0.5, 1.0, 2.0, 3.0), sigv=1.5, targets=('euc', 'h3', 'h2r'), Ns=(40, 100)),
                'sl2': dict(vals=(0.5, 1.0, 2.0), sigv=2.0, targets=('h3', 'h2r', 'euc'), Ns=(60,)),
                'sol': dict(vals=(0.3, 0.8, 1.4, 2.0), sigh=1.5, targets=('euc', 'h3', 'h2r'), Ns=(60,))}
    out = RES.get(key, {})
    for truth, st in settings.items():
        gt = GEOS[truth]; rng = np.random.default_rng(seed)
        for N in st['Ns']:
            for val in st['vals']:
                sig = (val, st['sigv']) if truth != 'sol' else (st['sigh'], val)
                Zt, P, alpha, Y, Yrep = gen_network(gt, N, sig, None, rng, density=0.15)
                k = f'{truth}_N{N}_{val}'
                if k in out or (only_N is not None and N != only_N):
                    continue
                D = gt.pdist(Zt); iu = np.triu_indices(N, 1); pp = P[iu]; w = pp * (1 - pp)
                r = dict(N=N, sig=sig, alpha=alpha, ndyads=len(iu[0]), targets={})
                for tname in st['targets']:
                    geo_t = GEOS[tname]
                    d2, Zf, c = deficit(D, geo_t, N, np.random.default_rng(1), w=w, nstart=1 if N >= 100 else 2, return_fit=True)
                    X0 = classical_mds(D, 3); X0 = X0 / np.std(X0)
                    inits = [np.concatenate([Zf.ravel(), [alpha - c]]), np.concatenate([(geo_t.chart(X0) if geo_t.name != 'euc' else X0).ravel(), [alpha]])]
                    klmin, Zk, ak = deficit_kl(D, P, alpha, geo_t, N, rng, inits, max_nfev=120 if N >= 100 else 300)
                    q = sigmoid(ak - geo_t.pdist(Zk)[iu]); kl_check = float(np.sum(bern_kl(pp, q)))
                    r['targets'][tname] = dict(fisher_deficit=d2, kl_min=klmin, tv_bound=float(min(1, np.sqrt(klmin / 2))), power_bound=float(min(1, 0.05 + np.sqrt(klmin / 2))),
                                               dim_quotient=3 * N + 1 - (6 if tname in ('euc', 'h3') else 4))
                out[k] = r; RES[key] = out; save()
                print(k, {t: (round(v['fisher_deficit'], 2), round(v['kl_min'], 2), round(v['power_bound'], 2)) for t, v in r['targets'].items()}, flush=True)


def run_lecam(seed=21):
    """Two-point (Le Cam) undetectability bound: exact dyad KL between the true law and the competitor at the
    Fisher-deficit minimiser (alpha' = alpha - c), and the bound power <= level + sqrt(KL/2)."""
    key = 'lecam'
    settings = {'nil': dict(vals=(0.5, 1.0, 2.0, 3.0), sigv=1.5, targets=('euc', 'h3', 'h2r'), Ns=(40,)),
                'sl2': dict(vals=(0.5, 1.0, 2.0), sigv=2.0, targets=('h3', 'h2r', 'euc'), Ns=(60,)),
                'sol': dict(vals=(0.3, 0.8, 1.4, 2.0), sigh=1.5, targets=('euc', 'h3', 'h2r'), Ns=(60,))}
    out = RES.get(key, {})
    for truth, st in settings.items():
        gt = GEOS[truth]; rng = np.random.default_rng(seed)
        for N in st['Ns']:
            for val in st['vals']:
                sig = (val, st['sigv']) if truth != 'sol' else (st['sigh'], val)
                Zt, P, alpha, Y, Yrep = gen_network(gt, N, sig, None, rng, density=0.15)
                k = f'{truth}_N{N}_{val}'
                if k in out:
                    continue
                D = gt.pdist(Zt); iu = np.triu_indices(N, 1); pp = P[iu]; w = pp * (1 - pp)
                r = dict(N=N, sig=sig, alpha=alpha, ndyads=len(iu[0]), targets={})
                for tname in st['targets']:
                    d2, Zp, c = deficit(D, GEOS[tname], N, np.random.default_rng(1), w=w, nstart=2, return_fit=True)
                    Dp = GEOS[tname].pdist(Zp)[iu]; q = sigmoid(alpha - c - Dp)
                    kl = float(np.sum(bern_kl(pp, q))); tv = float(min(1.0, np.sqrt(kl / 2)))
                    r['targets'][tname] = dict(fisher_deficit=d2, kl_exact=kl, tv_bound=tv, power_bound_at_5pct=min(1.0, 0.05 + tv))
                out[k] = r; RES[key] = out; save()
                print(k, {t: (round(v['fisher_deficit'], 2), round(v['kl_exact'], 2), round(v['power_bound_at_5pct'], 2)) for t, v in r['targets'].items()}, flush=True)


def run_coverage_rep(gname, reps=10, N=30, n_iter=4000, burn=1200, fix=False, seed0=500, truesig=False, ig=(2.0, 2.0), tag=''):
    key = f'covrep_{gname}_' + ('true' if truesig else ('fix' if fix else 'ig')) + tag + ('' if N == 30 else f'_N{N}')
    done = RES.get(key, {}).get('reps', [])
    geo = GEOS[gname]
    sig = {'nil': (2.0, 2.0), 'sol': (1.8, 1.4), 'sl2': (1.5, 2.0), 'euc': (2.0, 2.0), 'h3': (1.5, 1.5)}[gname]
    for r in range(len(done), reps):
        rng = np.random.default_rng(seed0 + r)
        Zt, P, alpha, Y, _ = gen_network(geo, N, sig, 2.5, rng)
        anchors = choose_anchors(Y, gname) if gname in ('nil', 'sol', 'sl2') else []
        model = LSM(geo, Y, anchors); Z0, a0 = initialise(model)
        o = mcmc(model, Z0, a0, n_iter=n_iter, step_z=0.45, step_alpha=0.2, thin=10, burn=burn, seed=r, fix_sigma=(fix or truesig), sig0=(sig if truesig else None), ig_prior=ig)
        iu = np.triu_indices(N, 1); Dt = geo.pdist(Zt)[iu]
        Ds = np.array([geo.pdist(Z)[iu] for Z in o['Z']])
        lo, hi = np.quantile(Ds, [0.05, 0.95], axis=0); c90 = float(np.mean((Dt >= lo) & (Dt <= hi)))
        lo2, hi2 = np.quantile(Ds, [0.25, 0.75], axis=0); c50 = float(np.mean((Dt >= lo2) & (Dt <= hi2)))
        Ps = sigmoid(o['alpha'][:, None] - Ds); plo, phi = np.quantile(Ps, [0.05, 0.95], axis=0); pc = float(np.mean((P[iu] >= plo) & (P[iu] <= phi)))
        acov = bool(np.quantile(o['alpha'], .05) <= alpha <= np.quantile(o['alpha'], .95))
        rmse = float(np.sqrt(np.mean((Ds.mean(0) - Dt) ** 2))); bias = float(np.mean(Ds.mean(0) - Dt))
        done.append(dict(cov90=c90, cov50=c50, cov90_prob=pc, alpha_cov=acov, alpha_mean=float(o['alpha'].mean()), sig=o['sig'].mean(0).tolist(), rmse=rmse, bias=bias))
        RES[key] = dict(N=N, sig_true=list(sig), n_iter=n_iter, reps=done); save()
        print(key, r, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in done[-1].items()}, flush=True)


# ===========================================================================
#  (4) Corrected baselines on the real networks (same splits as experiments2)
# ===========================================================================
def run_baselines(dname, nsplit=10, frac=0.1, seed0=100):
    Y = load_real(dname); N = Y.shape[0]; iu = np.triu_indices(N, 1)
    for split in range(nsplit):
        rng = np.random.default_rng(seed0 + split)
        hold = rng.uniform(size=len(iu[0])) < frac
        mask = np.ones((N, N)); mask[iu[0][hold], iu[1][hold]] = 0; mask[iu[1][hold], iu[0][hold]] = 0
        y = Y[iu][hold]
        for name, f in (('const', constant_predict), ('beta', beta_model_predict), ('cl', chung_lu_predict), ('rdpg', rdpg_logistic_predict), ('eigen', eigenmodel_predict), ('dcsbm', lambda Y, m: dcsbm_predict(Y, m)[0])):
            key = f'cmp3_{dname}_{name}_{split}'
            if key in RES:
                continue
            P = f(Y, mask); p = P[iu][hold]
            RES[key] = dict(logloss=float(logloss(p, y)), auc=float(auc(p, y)), n=int(hold.sum()))
        save()
    print('baselines done', dname, {name: round(np.mean([RES[f'cmp3_{dname}_{name}_{s}']['logloss'] for s in range(nsplit)]), 4) for name in ('const', 'beta', 'cl', 'rdpg', 'eigen', 'dcsbm')}, flush=True)


# ===========================================================================
#  (5) Learned-slope comparisons (variational, unanchored, q(mu), q(sigma), q(beta))
# ===========================================================================
def run_compare_beta(dname, method, nsplit=5, frac=0.1, seed0=100, lb_sd=1.0, tag=''):
    Y = load_real(dname); N = Y.shape[0]; iu = np.triu_indices(N, 1)
    geo = GEOS[method] if method != 'sl2p' else SL2(pickle.load(open('tables.pkl', 'rb'))['sl2'], period=2 * np.pi)
    for split in range(nsplit):
        key = f'cmpb{tag}_{dname}_{method}_{split}'
        if key in RES:
            continue
        rng = np.random.default_rng(seed0 + split)
        hold = rng.uniform(size=len(iu[0])) < frac
        mask = np.ones((N, N)); mask[iu[0][hold], iu[1][hold]] = 0; mask[iu[1][hold], iu[0][hold]] = 0
        model = LSM(geo, Y, [], mask=mask); model.logbeta_sd = lb_sd; Z0, a0 = initialise(model, scale=1.2, seed=split)
        t0 = time.time()
        vi = BBVI(model, Z0, a0, S=10, lr=0.02, seed=split, learn_sigma=True, learn_mu=True, learn_beta=True).run(800)
        p = vi.predictive(100)[iu][hold]; y = Y[iu][hold]
        RES[key] = dict(logloss=float(logloss(p, y)), auc=float(auc(p, y)), beta=float(np.exp(vi.bm)), alpha=float(vi.am), time=time.time() - t0)
        save(); print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in RES[key].items()}, flush=True)


# ===========================================================================
#  (6) Replicated win maps (independent networks per cell)
# ===========================================================================
def run_winmap_rep(truth, reps=(22, 23, 24, 25)):
    settings = {'nil': dict(vals=(0.5, 1.0, 2.0, 3.0), sigv=1.5, fits=('nil', 'euc', 'h3', 'h2r'), Ns=(40,)),
                'sl2': dict(vals=(0.5, 1.0, 2.0), sigv=2.0, fits=('sl2', 'h3', 'h2r', 'nil'), Ns=(60,)),
                'sol': dict(vals=(0.3, 0.8, 1.4, 2.0), sigh=1.5, fits=('sol', 'euc', 'h3', 'h2r'), Ns=(60,))}[truth]
    gt = GEOS[truth]
    for seed in reps:
        rng = np.random.default_rng(seed)
        for N in settings['Ns']:
            for val in settings['vals']:
                key = f'winrep_{truth}_N{N}_{val}_s{seed}'
                sig = (val, settings['sigv']) if truth != 'sol' else (settings['sigh'], val)
                Zt, P, alpha, Y, Yrep = gen_network(gt, N, sig, None, rng, density=0.15)
                if key in RES:
                    continue
                iu = np.triu_indices(N, 1); r = dict(N=N, sig=sig, density=float(Y[iu].mean()), oracle=float(logloss(P[iu], Yrep[iu])), fits={})
                for g in settings['fits']:
                    m, v = vi_fit(GEOS[g], Y, n_iter=600, S=8, seed=seed); pv = v.predictive(120)[iu]
                    r['fits'][g] = dict(logloss_rep=float(logloss(pv, Yrep[iu])), auc_rep=float(auc(pv, Yrep[iu])))
                RES[key] = r; save(); print(key, {g: round(x['logloss_rep'], 3) for g, x in r['fits'].items()}, flush=True)


def run_winmap_rep_big(truth='nil', reps=(22, 23), N=100, val=3.0):
    settings = dict(sigv=1.5, fits=('nil', 'euc', 'h3', 'h2r'))
    gt = GEOS[truth]
    for seed in reps:
        rng = np.random.default_rng(seed)
        key = f'winrep_{truth}_N{N}_{val}_s{seed}'
        Zt, P, alpha, Y, Yrep = gen_network(gt, N, (val, settings['sigv']), None, rng, density=0.15)
        r = RES.get(key, dict(N=N, sig=(val, settings['sigv']), density=float(Y[np.triu_indices(N, 1)].mean()), oracle=float(logloss(P[np.triu_indices(N, 1)], Yrep[np.triu_indices(N, 1)])), fits={}))
        iu = np.triu_indices(N, 1)
        for g in settings['fits']:
            if g in r['fits']:
                continue
            m, v = vi_fit(GEOS[g], Y, n_iter=600, S=8, seed=seed); pv = v.predictive(100)[iu]
            r['fits'][g] = dict(logloss_rep=float(logloss(pv, Yrep[iu])), auc_rep=float(auc(pv, Yrep[iu])))
            RES[key] = r; save(); print(key, g, round(r['fits'][g]['logloss_rep'], 4), flush=True)


# ===========================================================================
#  (7) Annealed maximum likelihood with learned (R, T), in the style of CKK (2024)
# ===========================================================================
def anneal_fit(geo, Y, mask, n_sweeps=60, T0=1.0, T1=0.02, seed=0):
    """Point estimate of positions by simulated annealing on the log-likelihood, with (alpha, beta) re-fitted
    by logistic regression on the current distances after every sweep (the link p = 1/(1+exp((d-R)/T)) is
    alpha = R/T, beta = 1/T)."""
    rng = np.random.default_rng(seed); N = Y.shape[0]; iu = np.triu_indices(N, 1)
    model = LSM(geo, Y, [], mask=mask); Z, a0 = initialise(model, scale=1.2, seed=seed)
    alpha, beta = a0, 1.0
    D = geo.pdist(Z)
    def ll_rows_of(D, alpha, beta):
        return (bern_ll(Y, alpha - beta * D) * mask).sum(1)
    rows = ll_rows_of(D, alpha, beta)
    for sw in range(n_sweeps):
        temp = T0 * (T1 / T0) ** (sw / max(n_sweeps - 1, 1))
        step = 0.5 * np.sqrt(temp / T0) + 0.05
        for i in rng.permutation(N):
            znew = geo.translate(Z[i], geo.chart(step * rng.normal(size=3)))
            d_new = geo.dist_to_all(znew, Z); d_new[i] = 0.0
            ln = bern_ll(Y[i], alpha - beta * d_new) * mask[i]; lo = bern_ll(Y[i], alpha - beta * D[i]) * mask[i]
            delta = ln.sum() - lo.sum()
            if delta > 0 or rng.uniform() < np.exp(delta / temp):
                Z[i] = znew; D[i, :] = d_new; D[:, i] = d_new
        # refit (alpha, beta) by Newton on the observed dyads
        obs = mask[iu] > 0; d = D[iu][obs]; y = Y[iu][obs]
        th = _logistic(-d[:, None], y)   # logit p = a + b*(-d)
        alpha, beta = float(th[0]), float(max(th[1], 1e-3))
    return Z, alpha, beta


def run_anneal(dname, method, nsplit=3, frac=0.1, seed0=100):
    Y = load_real(dname); N = Y.shape[0]; iu = np.triu_indices(N, 1); geo = GEOS[method]
    for split in range(nsplit):
        key = f'anneal_{dname}_{method}_{split}'
        if key in RES:
            continue
        rng = np.random.default_rng(seed0 + split)
        hold = rng.uniform(size=len(iu[0])) < frac
        mask = np.ones((N, N)); mask[iu[0][hold], iu[1][hold]] = 0; mask[iu[1][hold], iu[0][hold]] = 0
        t0 = time.time(); Z, alpha, beta = anneal_fit(geo, Y, mask, seed=split)
        p = sigmoid(alpha - beta * geo.pdist(Z)[iu][hold]); y = Y[iu][hold]
        RES[key] = dict(logloss=float(logloss(p, y)), auc=float(auc(p, y)), alpha=alpha, beta=beta, time=time.time() - t0)
        save(); print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in RES[key].items()}, flush=True)


# ===========================================================================
#  (8) MCMC diagnostics: dispersed chains, R-hat and ESS on invariant quantities
# ===========================================================================
def rhat_ess(chains):
    """Gelman-Rubin R-hat (split) and effective sample size (Geyer initial positive sequence) for chains (m, n)."""
    m, n = chains.shape; h = n // 2
    x = np.concatenate([chains[:, :h], chains[:, h:2 * h]], 0); m2 = x.shape[0]
    W = x.var(1, ddof=1).mean(); B = h * x.mean(1).var(ddof=1)
    var_hat = (h - 1) / h * W + B / h; rhat = float(np.sqrt(var_hat / W))
    # ESS
    y = x - x.mean(1, keepdims=True); tau = 0.0
    for lag in range(1, h - 1):
        rho = np.mean([np.sum(y[c, :-lag] * y[c, lag:]) / (h * W) for c in range(m2)])
        if rho < 0.0:
            break
        tau += 2 * rho
    ess = float(m2 * h / (1 + tau))
    return rhat, ess


def run_diagnostics(gname='nil', N=40, n_chains=4, n_iter=6000, burn=2000, seed=11):
    key = f'diag_{gname}'
    if key in RES:
        return
    geo = GEOS[gname]; rng = np.random.default_rng(seed)
    sig = {'nil': (2.0, 2.0), 'sol': (1.8, 1.4), 'sl2': (1.5, 2.0)}[gname]
    Zt, P, alpha, Y, _ = gen_network(geo, N, sig, 2.5, rng)
    anchors = choose_anchors(Y, gname); model = LSM(geo, Y, anchors); Z0, a0 = initialise(model)
    iu = np.triu_indices(N, 1); pairs = [(iu[0][k], iu[1][k]) for k in np.random.default_rng(0).choice(len(iu[0]), 3, replace=False)]
    chains = dict(alpha=[], ll=[], d1=[], d2=[], d3=[], sigh=[])
    for c in range(n_chains):
        crng = np.random.default_rng(100 + c)
        Zc = geo.translate(Z0, geo.chart(0.7 * crng.normal(size=Z0.shape))) if c > 0 else Z0.copy()   # dispersed starts
        Zc = geo.gauge(Zc, anchors)
        if not geo.anchor2_ok(Zc[anchors[1]]):
            Zc = Z0.copy()
        ac = a0 + (0.0 if c == 0 else crng.normal() * 1.0)
        o = mcmc(model, Zc, ac, n_iter=n_iter, step_z=0.45, step_alpha=0.2, thin=1, burn=burn, seed=1000 + c, fix_sigma=False)
        chains['alpha'].append(o['alpha']); chains['ll'].append(o['ll']); chains['sigh'].append(o['sig'][:, 0])
        for k, (i, j) in enumerate(pairs):
            chains[f'd{k+1}'].append(np.array([geo.dist(Z[i], Z[j]) for Z in o['Z']]))
    out = {}
    for q, ch in chains.items():
        arr = np.array(ch); rh, ess = rhat_ess(arr)
        out[q] = dict(rhat=rh, ess=ess, mean=float(arr.mean()), chain_means=arr.mean(1).tolist())
        print(q, 'Rhat %.3f ESS %.0f (of %d) chain means %s' % (rh, ess, arr.size, np.round(arr.mean(1), 3)), flush=True)
    RES[key] = dict(N=N, n_chains=n_chains, n_iter=n_iter, burn=burn, stats=out); save()


# ===========================================================================
#  (9) Replicated simulations: MCMC vs BBVI over independent (positions, network) pairs
# ===========================================================================
def run_sim_rep(gname, reps=10, N=40, n_iter=4000, burn=1200, seed0=700):
    key = f'simrep_{gname}'
    out = RES.get(key, {'reps': []}); done = len(out['reps'])
    geo = GEOS[gname]; sig = {'nil': (2.0, 2.0), 'sol': (1.8, 1.4), 'sl2': (1.5, 2.0)}[gname]
    for r in range(done, reps):
        rng = np.random.default_rng(seed0 + r)
        Zt, P, alpha, Y, Yrep = gen_network(geo, N, sig, 2.5, rng)
        iu = np.triu_indices(N, 1); Dt = geo.pdist(Zt)[iu]
        anchors = choose_anchors(Y, gname); model = LSM(geo, Y, anchors); Z0, a0 = initialise(model)
        t0 = time.time(); o = mcmc(model, Z0, a0, n_iter=n_iter, step_z=0.45, step_alpha=0.2, thin=10, burn=burn, seed=r, fix_sigma=False); tm = time.time() - t0
        Dm = np.mean([geo.pdist(Z) for Z in o['Z'][-200:]], 0)[iu]; pm = predictive_mcmc(model, o, 200)[iu]
        t0 = time.time(); m, vi = vi_fit(geo, Y, n_iter=600, S=8, seed=r); tv = time.time() - t0
        S0 = vi.S; vi.S = 200; Zq = vi.sample()[0]; vi.S = S0; Dv = np.mean([geo.pdist(Zq[k]) for k in range(200)], 0)[iu]; pv = vi.predictive(200)[iu]
        rec = dict(oracle=float(logloss(P[iu], Yrep[iu])), mcmc_alpha=float(np.mean(o['alpha'])), vi_alpha=float(vi.am), alpha=alpha,
                   mcmc_Dcorr=float(np.corrcoef(Dt, Dm)[0, 1]), vi_Dcorr=float(np.corrcoef(Dt, Dv)[0, 1]), mcmc_vi_Dcorr=float(np.corrcoef(Dm, Dv)[0, 1]),
                   mcmc_ll_rep=float(logloss(pm, Yrep[iu])), vi_ll_rep=float(logloss(pv, Yrep[iu])), mcmc_auc_rep=float(auc(pm, Yrep[iu])), vi_auc_rep=float(auc(pv, Yrep[iu])), t_mcmc=tm, t_vi=tv)
        out['reps'].append(rec); RES[key] = out; save()
        print(key, r, {k: round(v, 3) for k, v in rec.items() if k not in ('t_mcmc', 't_vi')}, flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'rig_nil': run_rigidity_nil()
    elif a[0] == 'rig_bvp': run_rigidity_bvp_aug(a[1], int(a[2]))
    elif a[0] == 'deficit': run_deficit()
    elif a[0] == 'lecam': run_lecam()
    elif a[0] == 'deficit_kl': run_deficit_kl(only_N=(int(a[1]) if len(a) > 1 else None))
    elif a[0] == 'winrep': run_winmap_rep(a[1])
    elif a[0] == 'winrep_big': run_winmap_rep_big(a[1])
    elif a[0] == 'diag': run_diagnostics(a[1])
    elif a[0] == 'simrep': run_sim_rep(a[1], reps=int(a[2]))
    elif a[0] == 'anneal': run_anneal(a[1], a[2], nsplit=int(a[3]) if len(a) > 3 else 3)
    elif a[0] == 'covrep': run_coverage_rep(a[1], reps=int(a[2]), fix=(len(a) > 3 and a[3] == 'fix'), truesig=(len(a) > 3 and a[3] == 'true'))
    elif a[0] == 'covrep_matched': run_coverage_rep(a[1], reps=int(a[2]), ig=(3.0, 8.0), tag='_matched', N=int(a[3]) if len(a) > 3 else 30)
    elif a[0] == 'covrep_trueN': run_coverage_rep(a[1], reps=int(a[2]), truesig=True, N=int(a[3]))
    elif a[0] == 'baselines': run_baselines(a[1], nsplit=int(a[2]) if len(a) > 2 else 10)
    elif a[0] == 'compare_beta': run_compare_beta(a[1], a[2], nsplit=int(a[3]) if len(a) > 3 else 5)
    elif a[0] == 'compare_beta_tight': run_compare_beta(a[1], a[2], nsplit=int(a[3]) if len(a) > 3 else 5, lb_sd=0.25, tag='T')
