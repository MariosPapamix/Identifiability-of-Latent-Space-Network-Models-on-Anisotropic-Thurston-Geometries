"""
Directed latent space model with an asymmetric term driven by the vertical offset:
    logit p_{i->j} = alpha - d_M(z_i, z_j) + gamma * v_ij,
where v_ij is the vertical coordinate of z_i^{-1} z_j (Nil: zeta_j - zeta_i - A(w_i, w_j), with the area cocycle A)
and, for the gradient-type competitors (R^3, H^3, H^2 x R), v_ij = zeta_j - zeta_i (a coboundary).
Around a triangle the asymmetries sum to -2 gamma Area (Nil) and to 0 (competitors): circulation.
"""
import numpy as np, json, time, sys
from inference import LSM, BBVI, bern_ll, sigmoid, initialise, logloss, auc
from experiments2 import GEOS, load_real

RESF = 'results3/results.json'


def vertical_offsets(geo, Z):
    """Antisymmetric matrix V with V_ij = vertical offset of z_j seen from z_i."""
    N = len(Z)
    if geo.name == 'nil':
        x, y, z = Z[:, 0], Z[:, 1], Z[:, 2]
        return (z[None, :] - z[:, None]) - 0.5 * (x[:, None] * y[None, :] - y[:, None] * x[None, :])
    if geo.name == 'sl2':
        w = Z[:, 0] + 1j * Z[:, 1]; zeta = Z[:, 2]
        A = -2 * np.angle(1 - np.conj(w)[:, None] * w[None, :])
        return (zeta[None, :] - zeta[:, None]) - A
    return Z[None, :, 2] - Z[:, None, 2]


class DirectedLSM(LSM):
    """Directed model; the mask hides unordered pairs (both directions)."""
    def __init__(self, geo, Y, anchors, gamma=0.0, mask=None, **kw):
        super().__init__(geo, Y, anchors, mask=mask, **kw)
        self.gamma = gamma
        self.off = ~np.eye(self.N, dtype=bool)

    def eta(self, Z, alpha, D=None):
        if D is None:
            D = self.geo.pdist(Z)
        return alpha - self.beta * D + self.gamma * vertical_offsets(self.geo, Z)

    def loglik(self, Z, alpha):
        E = self.eta(Z, alpha)
        return float(np.sum((bern_ll(self.Y, E) * self.mask)[self.off]))

    def loglik_node(self, i, Z, alpha):
        E = self.eta(Z, alpha)
        L = bern_ll(self.Y, E) * self.mask; np.fill_diagonal(L, 0.0)
        return L[i].sum() + L[:, i].sum()

    def loglik_rows(self, Z, alpha, sub=None):
        D = self.geo.pdist(Z); E = self.eta(Z, alpha, D)
        L = bern_ll(self.Y, E) * self.mask; np.fill_diagonal(L, 0.0)
        if sub is None:
            return L.sum(1) + L.sum(0), D
        ii, jj = sub
        l = L[ii, jj] + L[jj, ii]; l = l * (len(self.iu[0]) / len(ii))
        rows = np.zeros(self.N); np.add.at(rows, ii, l); np.add.at(rows, jj, l)
        return rows, None

    def gamma_newton(self, Z, alpha, n_steps=3):
        """Newton steps for gamma given positions (profile M-step)."""
        D = self.geo.pdist(Z); V = vertical_offsets(self.geo, Z)
        for _ in range(n_steps):
            E = alpha - self.beta * D + self.gamma * V; P = sigmoid(E)
            g = np.sum(((self.Y - P) * V * self.mask)[self.off]); prec = float(__import__('os').environ.get('GPREC', 1.0)); h = np.sum((P * (1 - P) * V * V * self.mask)[self.off]) + prec   # N(0, 1/prec) prior on gamma
            self.gamma = float(self.gamma + (g - prec * self.gamma) / h)


def gen_directed(geo, N, sig, alpha, gamma, rng, density=None):
    from experiments2 import gen_network
    Zt = geo.translate(np.zeros((N, 3)), geo.chart(rng.normal(size=(N, 3)) * np.array([sig[0], sig[0], sig[1]])))
    D = geo.pdist(Zt); V = vertical_offsets(geo, Zt)
    if alpha is None:
        from scipy.optimize import brentq
        alpha = brentq(lambda a: sigmoid(a - D)[~np.eye(N, dtype=bool)].mean() - density, -20, 20)
    P = sigmoid(alpha - D + gamma * V); np.fill_diagonal(P, 0)
    Y = (rng.uniform(size=(N, N)) < P).astype(float); np.fill_diagonal(Y, 0)
    Yrep = (rng.uniform(size=(N, N)) < P).astype(float); np.fill_diagonal(Yrep, 0)
    return Zt, P, alpha, Y, Yrep


def fit_directed(geo, Y, mask, learn_gamma=True, n_iter=600, S=8, seed=0, gamma0=0.0):
    model = DirectedLSM(geo, Y, [], gamma=gamma0, mask=mask)
    Z0, a0 = initialise(model, scale=1.2, seed=seed)
    vi = BBVI(model, Z0, a0, S=S, seed=seed)
    for it in range(n_iter):
        vi.step(it)
        if learn_gamma and it % 10 == 9:
            Zs = vi.sample()[0][0]; model.gamma_newton(Zs, vi.am)
    return model, vi


def predictive_directed(model, vi, n=100):
    S0 = vi.S; vi.S = n; Zs = vi.sample()[0]; vi.S = S0
    P = np.mean([sigmoid(model.eta(Zs[k], vi.am)) for k in range(n)], 0); np.fill_diagonal(P, 0.5)
    return P


def run_directed(gammas=(0.0, 0.25, 0.5, 1.0), reps=3, N=60, sig=(2.0, 1.5), density=0.15, seed0=900):
    R = json.load(open(RESF)); out = R.get('directed', {})
    fits = [('nil', True), ('nil', False), ('h2r', True), ('euc', True), ('h3', True)]
    for gamma in gammas:
        for r in range(reps):
            key = f'g{gamma}_r{r}'
            rng = np.random.default_rng(seed0 + 100 * r + int(10 * gamma))
            Zt, P, alpha, Y, Yrep = gen_directed(GEOS['nil'], N, sig, None, gamma, rng, density)
            rec = out.get(key, dict(gamma=gamma, N=N, alpha=alpha, density=float(Y.mean() * N / (N - 1)), fits={}))
            off = ~np.eye(N, dtype=bool)
            rec['oracle'] = float(logloss(P[off], Yrep[off]))
            # circulation content of the truth: mean squared triangle sum of the asymmetry
            for g, lg in fits:
                name = g + ('' if lg else '_sym')
                if name in rec['fits']:
                    continue
                t0 = time.time(); model, vi = fit_directed(GEOS[g], Y, np.ones((N, N)), learn_gamma=lg, seed=r)
                Pf = predictive_directed(model, vi)
                # asymmetry accuracy on replicate: among pairs with exactly one direction present, predict which
                one = (Yrep + Yrep.T == 1) & np.triu(np.ones((N, N), bool), 1)
                ii, jj = np.where(one); dirn = Yrep[ii, jj]; sc = np.log(Pf[ii, jj] + 1e-9) - np.log(Pf[jj, ii] + 1e-9)
                acc = float(np.mean((sc > 0) == (dirn == 1))) if len(ii) else np.nan
                rec['fits'][name] = dict(logloss_rep=float(logloss(Pf[off], Yrep[off])), auc_rep=float(auc(Pf[off], Yrep[off])),
                                         direction_acc=acc, gamma_hat=float(model.gamma), time=time.time() - t0)
                out[key] = rec; R = json.load(open(RESF)); R['directed'] = out; json.dump(R, open(RESF, 'w'), indent=1)
                print(key, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec['fits'][name].items()}, flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'sim':
        run_directed(gammas=tuple(float(x) for x in a[1:]) if len(a) > 1 else (0.0, 0.25, 0.5, 1.0))


# ---------------------------------------------------------------------------------------------
# Two control models for the counter networks: (i) a ranking on any geometry with a free rank-two
# antisymmetric term s_ij = u_i^T J u_j (not tied to enclosed area), (ii) a degree-corrected ranking
# with sender and receiver effects a_i + b_j. Node-level extra parameters are point estimates updated
# by gradient steps between BBVI steps, with N(0,1) priors; gamma is fixed at zero for these models.
# ---------------------------------------------------------------------------------------------
class ControlLSM(DirectedLSM):
    def __init__(self, geo, Y, anchors, kind='skew', mask=None):
        super().__init__(geo, Y, anchors, gamma=0.0, mask=mask)
        self.kind = kind; N = len(Y)
        self.U = 0.3 * np.random.default_rng(0).normal(size=(N, 2)); self.a = np.zeros(N); self.b = np.zeros(N)

    def extra(self):
        if self.kind == 'skew':
            return np.outer(self.U[:, 0], self.U[:, 1]) - np.outer(self.U[:, 1], self.U[:, 0])
        return self.a[:, None] + self.b[None, :]

    def eta(self, Z, alpha, D=None):
        if D is None:
            D = self.geo.pdist(Z)
        return alpha - self.beta * D + self.extra()

    def extra_step(self, Z, alpha, n_steps=10, lr=0.05):
        D = self.geo.pdist(Z); M = self.mask.copy(); np.fill_diagonal(M, 0)
        for _ in range(n_steps):
            E = alpha - self.beta * D + self.extra(); R = (self.Y - sigmoid(E)) * M      # residuals on observed ordered pairs
            if self.kind == 'skew':
                # d/dU_i1 : sum_j R_ij U_j2 - sum_j R_ji U_j2 ;  d/dU_i2 : -sum_j R_ij U_j1 + sum_j R_ji U_j1
                g1 = R @ self.U[:, 1] - R.T @ self.U[:, 1]; g2 = -R @ self.U[:, 0] + R.T @ self.U[:, 0]
                G = np.stack([g1, g2], 1) - self.U
                self.U += 0.02 * G / (1.0 + np.abs(G).mean())
            else:
                ga = R.sum(1) - self.a; gb = R.sum(0) - self.b
                W = (sigmoid(E) * (1 - sigmoid(E)) * M)
                self.a = np.clip(self.a + 0.5 * ga / (W.sum(1) + 1.0), -4, 4); self.b = np.clip(self.b + 0.5 * gb / (W.sum(0) + 1.0), -4, 4)


def fit_control(geo, Y, mask, kind='skew', n_iter=600, S=8, seed=0):
    model = ControlLSM(geo, Y, [], kind=kind, mask=mask)
    Z0, a0 = initialise(model, scale=1.2, seed=seed)
    vi = BBVI(model, Z0, a0, S=S, seed=seed)
    for it in range(n_iter):
        vi.step(it)
        if it % 5 == 4:
            Zs = vi.sample()[0][0]; model.extra_step(Zs, vi.am)
    return model, vi


# ---------------------------------------------------------------------------------------------
# Decisive control: the same directed term as Nil, gamma (zeta_j - zeta_i - (x_i y_j - y_i x_j)/2), on the SAME three
# coordinates that set a Euclidean or H2xR symmetric distance. Also an additive-and-multiplicative-effects model
# (Hoff 2005 / AME) without any distance: alpha + a_i + b_j + u_i . v_j (rank two), point estimates for node effects.
# ---------------------------------------------------------------------------------------------
from experiments2 import GEOS as _GEOS_ALL
GEOS_NIL = _GEOS_ALL['nil']


class SharedCouplingLSM(DirectedLSM):
    """Symmetric distance in geo (euc or h2r), directed term with Nil's formula on the same chart coordinates."""
    def eta(self, Z, alpha, D=None):
        if D is None:
            D = self.geo.pdist(Z)
        return alpha - self.beta * D + self.gamma * vertical_offsets(GEOS_NIL, Z)

    def gamma_newton(self, Z, alpha, n_steps=3):
        D = self.geo.pdist(Z); V = vertical_offsets(GEOS_NIL, Z); M = self.mask.copy(); np.fill_diagonal(M, 0)
        for _ in range(n_steps):
            E = alpha - self.beta * D + self.gamma * V; P = sigmoid(E)
            g = np.sum(((self.Y - P) * V * M)[self.off]); h = np.sum((P * (1 - P) * V * V * M)[self.off]) + 1.0
            self.gamma = float(self.gamma + (g - self.gamma) / h)


class AMEModel(ControlLSM):
    """alpha + a_i + b_j + u_i^T v_j, rank two, no distance term (Hoff's additive and multiplicative effects)."""
    def __init__(self, geo, Y, anchors, mask=None):
        super().__init__(geo, Y, anchors, kind='ame', mask=mask); N = len(Y); self.beta = 0.0
        rng = np.random.default_rng(1); self.U = 0.3 * rng.normal(size=(N, 2)); self.V = 0.3 * rng.normal(size=(N, 2))

    def extra(self):
        return self.a[:, None] + self.b[None, :] + self.U @ self.V.T

    def extra_step(self, Z, alpha, n_steps=10, lr=0.02):
        M = self.mask.copy(); np.fill_diagonal(M, 0)
        for _ in range(n_steps):
            E = alpha + self.extra(); P = sigmoid(E); R = (self.Y - P) * M; W = P * (1 - P) * M
            self.a = np.clip(self.a + 0.5 * (R.sum(1) - self.a) / (W.sum(1) + 1.0), -4, 4)
            self.b = np.clip(self.b + 0.5 * (R.sum(0) - self.b) / (W.sum(0) + 1.0), -4, 4)
            GU = R @ self.V - self.U; GV = R.T @ self.U - self.V
            self.U += lr * GU / (1.0 + np.abs(GU).mean()); self.V += lr * GV / (1.0 + np.abs(GV).mean())


def fit_shared(geo, Y, mask, n_iter=600, S=8, seed=0, gamma0=0.4):
    model = SharedCouplingLSM(geo, Y, [], gamma=gamma0, mask=mask)
    Z0, a0 = initialise(model, scale=1.2, seed=seed); vi = BBVI(model, Z0, a0, S=S, seed=seed)
    for it in range(n_iter):
        vi.step(it)
        if it % 10 == 9:
            Zs = vi.sample()[0][0]; model.gamma_newton(Zs, vi.am)
    return model, vi


def fit_ame(Y, mask, n_iter=600, S=8, seed=0):
    from experiments2 import GEOS as _G
    model = AMEModel(_G['euc'], Y, [], mask=mask)
    Z0, a0 = initialise(model, scale=1.2, seed=seed); vi = BBVI(model, Z0, a0, S=S, seed=seed)
    for it in range(n_iter):
        vi.step(it)
        if it % 5 == 4:
            model.extra_step(None, vi.am)
    return model, vi


class NilDCModel(ControlLSM):
    """Nil (or shared-coupling Euclidean) circulation model with sender and receiver effects: alpha + a_i + b_j - d + gamma v_ij."""
    def __init__(self, geo, Y, anchors, mask=None, gamma=0.4, coupling_geo=None):
        super().__init__(geo, Y, anchors, kind='dc', mask=mask); self.gamma = gamma; self.cg = coupling_geo or geo

    def eta(self, Z, alpha, D=None):
        if D is None:
            D = self.geo.pdist(Z)
        return alpha - self.beta * D + self.extra() + self.gamma * vertical_offsets(self.cg, Z)

    def gamma_newton(self, Z, alpha, n_steps=3):
        D = self.geo.pdist(Z); V = vertical_offsets(self.cg, Z); M = self.mask.copy(); np.fill_diagonal(M, 0); X = self.extra()
        for _ in range(n_steps):
            E = alpha - self.beta * D + X + self.gamma * V; P = sigmoid(E)
            g = np.sum(((self.Y - P) * V * M)[self.off]); h = np.sum((P * (1 - P) * V * V * M)[self.off]) + 1.0
            self.gamma = float(self.gamma + (g - self.gamma) / h)

    def extra_step(self, Z, alpha, n_steps=10, lr=0.05):
        D = self.geo.pdist(Z); V = vertical_offsets(self.cg, Z); M = self.mask.copy(); np.fill_diagonal(M, 0)
        for _ in range(n_steps):
            E = alpha - self.beta * D + self.extra() + self.gamma * V; P = sigmoid(E); R = (self.Y - P) * M; W = P * (1 - P) * M
            self.a = np.clip(self.a + 0.5 * (R.sum(1) - self.a) / (W.sum(1) + 1.0), -4, 4); self.b = np.clip(self.b + 0.5 * (R.sum(0) - self.b) / (W.sum(0) + 1.0), -4, 4)


def fit_nil_dc(geo, Y, mask, n_iter=600, S=8, seed=0, gamma0=0.4, coupling_geo=None):
    model = NilDCModel(geo, Y, [], mask=mask, gamma=gamma0, coupling_geo=coupling_geo)
    Z0, a0 = initialise(model, scale=1.2, seed=seed); vi = BBVI(model, Z0, a0, S=S, seed=seed)
    for it in range(n_iter):
        vi.step(it)
        if it % 5 == 4:
            Zs = vi.sample()[0][0]; model.extra_step(Zs, vi.am)
        if it % 10 == 9:
            Zs = vi.sample()[0][0]; model.gamma_newton(Zs, vi.am)
    return model, vi
