import numpy as np
from scipy.optimize import brentq
from geometry import nil_dist_e

def nil_log(q):
    """Riemannian logarithm at the identity of Nil: u with |u| = d(e,q) and exp_e(u) = q (q inside the cut locus)."""
    x, y, z = q; r = np.hypot(x, y); zeta = abs(z); sgn = 1 if z >= 0 else -1
    d = nil_dist_e(np.array(q, dtype=float))
    if r < 1e-14:
        return np.array([0.0, 0.0, z])
    if zeta < 1e-14:
        return np.array([x, y, 0.0])
    Phi = lambda s: s + r * r * (s - np.sin(s)) / (8 * np.sin(s / 2) ** 2) - zeta
    s = brentq(Phi, 1e-12, 2 * np.pi - 1e-12)
    c = s / d; a = np.sqrt(max(1 - c * c, 0)); theta = np.arctan2(y, x)
    phi0 = theta - sgn * s / 2
    return d * np.array([a * np.cos(phi0), a * np.sin(phi0), sgn * c])

def nil_exp(u):
    """Closed-form geodesic from the identity with initial velocity u, at time one."""
    L = np.linalg.norm(u)
    if L < 1e-15:
        return np.zeros(3)
    v = u / L; a = np.hypot(v[0], v[1]); c = v[2]; t = L
    if abs(c) < 1e-12:
        return np.array([v[0] * t, v[1] * t, 0.0])
    phi = np.arctan2(v[1], v[0]); s = c * t
    x = (a / c) * (np.sin(phi + s) - np.sin(phi)); y = (a / c) * (np.cos(phi) - np.cos(phi + s))
    z = c * t + (a * a / (2 * c * c)) * (s - np.sin(s))
    return np.array([x, y, z])
