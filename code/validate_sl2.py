import numpy as np, pickle, time
from scipy.optimize import least_squares
from geometry import *
T = pickle.load(open('tables.pkl','rb')); geo = SL2(T['sl2']); nil = Nil(T['nil'])
rng=np.random.default_rng(5)
# random points with base radius up to ~3
def rand_pts(n, geo):
    d = rng.normal(size=(n,3))*np.array([1.2,1.2,1.5]); return geo.chart(d)
for g in (geo, nil):
    Q1=rand_pts(500,g); Q2=rand_pts(500,g); P=rand_pts(500,g)
    d12 = g.dist(Q1,Q2); d21 = g.dist(Q2,Q1)
    tQ1 = g.translate(P,Q1); tQ2 = g.translate(P,Q2); dT = g.dist(tQ1,tQ2)
    # translate then translate back
    back = g.translate_inv(P, tQ1)
    print(g.name, "symmetry max err %.4f, translation-invariance max err %.4f, inverse-consistency %.2e" % (np.abs(d12-d21).max(), np.abs(d12-dT).max(), np.abs(back-Q1).max()))
    # random walk symmetry: delta' = chart_inv(T_{z'}^{-1} z) should equal -delta
    dl = rng.normal(size=(500,3)); zp = g.translate(Q1, g.chart(dl)); dl2 = g.chart_inv(g.translate_inv(zp, Q1))
    print("   reverse displacement = -delta: max err %.2e" % np.abs(dl2+dl).max())
# cocycle identity in SL2: sum of 2 arg(1 - conj(w_i) w_j) around a triangle vs hyperbolic area (angle defect)
def harea(w1,w2,w3):
    # angle defect: pi - sum of interior angles; compute angles via hyperbolic law of cosines
    a=h2_dist(w2,w3); b=h2_dist(w1,w3); c=h2_dist(w1,w2)
    A=np.arccos((np.cosh(b)*np.cosh(c)-np.cosh(a))/(np.sinh(b)*np.sinh(c)))
    B=np.arccos((np.cosh(a)*np.cosh(c)-np.cosh(b))/(np.sinh(a)*np.sinh(c)))
    C=np.arccos((np.cosh(a)*np.cosh(b)-np.cosh(c))/(np.sinh(a)*np.sinh(b)))
    return np.pi-A-B-C
for k in range(3):
    w = rng.uniform(-0.8,0.8,size=(3,2)); w = w[:,0]+1j*w[:,1]
    cyc = 2*(np.angle(1-np.conj(w[0])*w[1])+np.angle(1-np.conj(w[1])*w[2])+np.angle(1-np.conj(w[2])*w[0]))
    orient = np.sign(np.imag(np.conj(w[1]-w[0])*(w[2]-w[0])))
    print("SL2 cocycle sum %.6f   signed area %.6f" % (cyc, orient*harea(*w)))
# BVP check for SL2 on a few generic points (hyperboloid magnetic geodesics)
def shoot(c, T, n=600):
    a=np.sqrt(max(1-c*c,0)); X=np.array([0,0,1.]); V=np.array([a,0,0.]); ze=0.0; dt=T/n; eta=np.array([1,1,-1.])
    def f(X,V,ze):
        acc=a*a*X + c*eta*np.cross(X,V); return V, acc, c+(X[0]*V[1]-X[1]*V[0])/(X[2]+1)
    for _ in range(n):
        k1=f(X,V,ze); k2=f(X+.5*dt*k1[0],V+.5*dt*k1[1],ze+.5*dt*k1[2]); k3=f(X+.5*dt*k2[0],V+.5*dt*k2[1],ze+.5*dt*k2[2]); k4=f(X+dt*k3[0],V+dt*k3[1],ze+dt*k3[2])
        X=X+dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); V=V+dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]); ze=ze+dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
    return np.array([np.arccosh(max(X[2],1.0)), ze])
def bvp(rho, zeta, nstart=16):
    best=np.inf
    for s in range(nstart):
        c0=rng.uniform(0,0.999); lt=np.log(np.hypot(rho,zeta)+0.2)
        res=lambda p: shoot(np.tanh(p[0]), np.exp(p[1]))-np.array([rho,zeta])
        try:
            r=least_squares(res,[np.arctanh(c0),lt],xtol=1e-10,ftol=1e-12,max_nfev=150)
            if np.linalg.norm(r.fun)<1e-5: best=min(best,np.exp(r.x[1]))
        except Exception: pass
    return best
t0=time.time()
for (rho,ze) in [(1.0,1.0),(2.0,3.0),(0.5,5.0),(3.0,8.0),(1.5,0.7),(4.0,2.0)]:
    print("SL2 (rho,zeta)=(%.1f,%.1f): table %.4f  bvp %.4f" % (rho,ze,T['sl2'](np.array([rho]),np.array([ze]))[0], bvp(rho,ze)))
print("time", round(time.time()-t0))
