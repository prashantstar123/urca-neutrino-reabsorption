"""RUNG 6d: FSU-type NONLINEAR RMF family (sigma^3, sigma^4, omega^4 and omega^2 rho^2 couplings) with Deltas.  Lagrangian convention of
Todd-Rutel & Piekarewicz PRL 95, 122501 (2005) / Fattoyev-Horowitz-Piekarewicz-Shen PRC 82, 055803 (2010):
  L_int = -kappa/3! Phi^3 - lambda/4! Phi^4 + zeta/4! W^4 + Lambda_v W^2 B^2,   Phi = g_s sigma, W = g_v omega_0, B = g_rho rho_0,
  nucleon vertex g_rho/2 tau.rho  ->  isospin vertex I_3 (= t3/2 in the species table, t3 = 2 I_3).
Mean fields (densities in fm^-3, fields in MeV):
  (m_s^2/g_s^2 Phi + kappa/2 Phi^2 + lambda/6 Phi^3)/HC^3 = sum_i x_si rho_si,     m*_i = m_i - x_si Phi
  (m_v^2/g_v^2 W + zeta/6 W^3 + 2 Lambda_v B^2 W)/HC^3 = sum_i x_wi n_i
  (m_r^2/g_r^2 B + 2 Lambda_v W^2 B)/HC^3            = sum_i x_ri I_3i n_i
  mu_i = sqrt(k_i^2 + m*_i^2) + x_wi W + x_ri I_3i B
  eps = sum eps_kin + [1/2 m_s^2/g_s^2 Phi^2 + kappa/6 Phi^3 + lambda/24 Phi^4 + 1/2 m_v^2/g_v^2 W^2 + zeta/8 W^4 + 1/2 m_r^2/g_r^2 B^2 + 3 Lambda_v W^2 B^2]/HC^3
Parameter sets (Table I of the papers quoted; VERIFY with saturation_check before use, see results/fsu_saturation.txt):
  NL3   : Lalazissis 1997 in this convention (Fattoyev 2010 Table I)  -> cross-check against rmf_nl3_delta.py
  FSUGold: Todd-Rutel & Piekarewicz 2005
  IUFSU : Fattoyev et al. 2010 (L = 47)
  BigApple: Fattoyev, Horowitz, Piekarewicz, Reed PRC 102, 065805 (2020) (L = 40, M_max 2.6)
  FSU2R : Tolos, Centelles, Ramos ApJ 834, 3 (2017) (L = 47, M_max 2.05)"""
import numpy as np, sys, os; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from scipy.optimize import fsolve, brentq
from rmf_dd2_delta import ns_of, eps_kin, p_kin, kf, baryons, MN, MP, ME, MMU, HC
SETS = {  # m_s, m_v, m_r, g_s^2, g_v^2, g_r^2, kappa [MeV], lambda, zeta, Lambda_v, n0_lit
    'NL3':      (508.194, 782.501, 763.0, 104.3871, 165.5854,  79.6000, 3.8599, -0.015905, 0.000, 0.000, 0.148),
    'FSUGold':  (491.500, 782.500, 763.0, 112.1996, 204.5469, 138.4701, 1.4203, +0.023762, 0.060, 0.030, 0.148),
    'IUFSU':    (491.500, 782.500, 763.0,  99.4266, 169.8349, 184.6877, 3.3808, +0.000296, 0.030, 0.046, 0.155),
    'BigApple': (492.730, 782.500, 763.0,  93.5074, 151.6839, 200.5562, 5.20326, -0.021739, 0.00070, 0.047471, 0.155),
    'FSU2R':    (497.479, 782.500, 763.0, 107.5751, 182.3949, 206.4260, 3.0911, -0.001680, 0.024, 0.045, 0.1505)}
LIT = {'NL3': (0.148, -16.24, 271.5, 37.4, 118.5, 0.60), 'FSUGold': (0.148, -16.30, 230.0, 32.6, 60.5, 0.61), 'IUFSU': (0.155, -16.40, 231.2, 31.3, 47.2, 0.61),
       'BigApple': (0.155, -16.34, 227.0, 31.3, 39.8, 0.61), 'FSU2R': (0.1505, -16.28, 238.0, 30.7, 46.9, 0.593)}
P = {}
def use(name):
    global P
    ms, mv, mr, gs2, gv2, gr2, kap, lam, zet, Lv, n0 = SETS[name]; P = dict(name=name, ms=ms, mv=mv, mr=mr, gs2=gs2, gv2=gv2, gr2=gr2, kap=kap, lam=lam, zet=zet, Lv=Lv, N0=n0); _C.clear(); return P
def I3(t3): return 0.5 * t3
def solve_fields(nb_list, B, guess=None):
    ms2, mv2, mr2 = P['ms'] ** 2 / P['gs2'], P['mv'] ** 2 / P['gv2'], P['mr'] ** 2 / P['gr2']; kap, lam, zet, Lv = P['kap'], P['lam'], P['zet'], P['Lv']
    def F(ph):
        rhs = sum(xs * ns_of(kf(nb, g), m - xs * ph, g) for nb, (nm, m, t3, q, g, xs, xw, xr, *rest) in zip(nb_list, B))
        return (ms2 * ph + 0.5 * kap * ph ** 2 + lam / 6 * ph ** 3) / HC ** 3 - rhs
    phmax = min(m / xs for (nm, m, t3, q, g, xs, xw, xr, *rest) in B) * 0.999
    ph = brentq(F, 0.0, phmax, xtol=1e-13) if F(phmax) > 0 else phmax
    nw = sum(xw * nb for nb, (nm, m, t3, q, g, xs, xw, xr, *rest) in zip(nb_list, B)) * HC ** 3; nr = sum(xr * I3(t3) * nb for nb, (nm, m, t3, q, g, xs, xw, xr, *rest) in zip(nb_list, B)) * HC ** 3
    Bof = lambda W: nr / (mr2 + 2 * Lv * W ** 2)
    G = lambda W: mv2 * W + zet / 6 * W ** 3 + 2 * Lv * Bof(W) ** 2 * W - nw
    Wlin = nw / mv2
    W = brentq(G, 0.0, Wlin * 1.0000001, xtol=1e-13) if nw > 0 else 0.0
    return ph, W, Bof(W)
def thermo(nb_list, B, nl):
    ph, w, r = solve_fields(nb_list, B); ms2, mv2, mr2 = P['ms'] ** 2 / P['gs2'], P['mv'] ** 2 / P['gv2'], P['mr'] ** 2 / P['gr2']; kap, lam, zet, Lv = P['kap'], P['lam'], P['zet'], P['Lv']
    eps = (0.5 * ms2 * ph ** 2 + kap / 6 * ph ** 3 + lam / 24 * ph ** 4 + 0.5 * mv2 * w ** 2 + zet / 8 * w ** 4 + 0.5 * mr2 * r ** 2 + 3 * Lv * w ** 2 * r ** 2) / HC ** 3; mu = {}; mst = {}
    for nb, (nm, m, t3, q, g, xs, xw, xr, *rest) in zip(nb_list, B):
        ms = m - xs * ph; k = kf(nb, g); eps += eps_kin(k, ms, g); mu[nm] = np.sqrt(k ** 2 + ms ** 2) + xw * w + xr * I3(t3) * r; mst[nm] = ms
    p = sum(nb * mu[nm] for nb, (nm, *_ ) in zip(nb_list, B)) - eps
    for nl_, ml in zip(nl, (ME, MMU)):
        k = kf(nl_, 2); eps += eps_kin(k, ml, 2); p += p_kin(k, ml, 2)
    return eps, p, mu, mst, ph, w, r
def saturation_check(verbose=True):
    B = baryons()
    def EA(n): e, *_ = thermo([n / 2, n / 2, 0, 0, 0, 0], B, (0, 0)); return e / n - (MN + MP) / 2
    n0 = brentq(lambda n: EA(n + 1e-5) - EA(n - 1e-5), 0.12, 0.19); d = 1e-3; K = 9 * n0 ** 2 * (EA(n0 + d) - 2 * EA(n0) + EA(n0 - d)) / d ** 2
    def Esym(n, a=0.02):
        ep, *_ = thermo([n * (1 + a) / 2, n * (1 - a) / 2, 0, 0, 0, 0], B, (0, 0)); em, *_ = thermo([n * (1 - a) / 2, n * (1 + a) / 2, 0, 0, 0, 0], B, (0, 0)); return ((ep + em) / (2 * n) - EA(n) - (MN + MP) / 2) / a ** 2
    J = Esym(n0); L = 3 * n0 * (Esym(n0 + d) - Esym(n0 - d)) / (2 * d); e, p, mu, mst, *_ = thermo([n0 / 2, n0 / 2, 0, 0, 0, 0], B, (0, 0))
    # thermodynamic consistency: p = n de/dn - e at fixed composition (symmetric at 2 n0 and pure neutron at 3 n0)
    def tc(nn, npp):
        n = nn + npp; e, p, *_ = thermo([nn, npp, 0, 0, 0, 0], B, (0, 0)); h = 1e-4
        ep_, *_ = thermo([nn * (1 + h), npp * (1 + h), 0, 0, 0, 0], B, (0, 0)); em_, *_ = thermo([nn * (1 - h), npp * (1 - h), 0, 0, 0, 0], B, (0, 0)); return p, n * (ep_ - em_) / (2 * h * n) - e
    t1 = tc(n0, n0); t2 = tc(3 * n0, 0.0)
    lit = LIT[P['name']]; res = (n0, EA(n0), K, J, L, mst['n'] / MN)
    ok = abs(n0 - lit[0]) < 0.002 and abs(res[1] - lit[1]) < 0.12 and abs(K - lit[2]) < 4 and abs(J - lit[3]) < 0.6 and abs(L - lit[4]) < 2.5 and abs(res[5] - lit[5]) < 0.02
    line = (f"{P['name']:9s} n0 {n0:.4f} ({lit[0]})  E/A {res[1]:.2f} ({lit[1]})  K {K:.1f} ({lit[2]})  J {J:.2f} ({lit[3]})  L {L:.1f} ({lit[4]})  m*/m {res[5]:.3f} ({lit[5]})"
            f"  | p = n de/dn - e: sym 2n0 {t1[0]:.3f} vs {t1[1]:.3f}, PNM 3n0 {t2[0]:.3f} vs {t2[1]:.3f} | {'PASS' if ok else 'FAIL'}")
    if verbose: print(line, flush=True)
    return ok, line, res
def beta_eq(n, B, guess=None):
    names = [b[0] for b in B]; N0 = P['N0']
    def dens(v):
        ph, w, r, mun, mue = v; nb = []
        for (nm, m, t3, q, g, xs, xw, xr, *rest) in B:
            ms = m - xs * ph; estar = mun - q * mue - xw * w - xr * I3(t3) * r
            k = np.sqrt(estar ** 2 - ms ** 2) if estar > ms else 0.0; nb.append(g * k ** 3 / (6 * np.pi ** 2) / HC ** 3)
        ne = 2 * (mue ** 2 - ME ** 2) ** 1.5 / (6 * np.pi ** 2) / HC ** 3 if mue > ME else 0.0; nmu = 2 * (mue ** 2 - MMU ** 2) ** 1.5 / (6 * np.pi ** 2) / HC ** 3 if mue > MMU else 0.0
        return nb, ne, nmu
    def eqs(v):
        ph, w, r, mun, mue = v; nb, ne, nmu = dens(v); s2, w2, r2 = solve_fields(nb, B)
        return [(ph - s2) / 10, (w - w2) / 10, (r - r2) / 10, (sum(nb) - n) * 100, (sum(q * ni for ni, (nm, m, t3, q, g, xs, xw, xr, *rest) in zip(nb, B)) - ne - nmu) * 100]
    if guess is None: guess = np.array([300.0 * (n / N0) ** 0.5, P['gv2'] * n * HC ** 3 / P['mv'] ** 2, -P['gr2'] * 0.2 * n * HC ** 3 / P['mr'] ** 2, MN + 60 * n / N0, 100.0])
    sol, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-11, maxfev=5000); nb, ne, nmu = dens(sol); e, p, mu, mst, *_ = thermo(nb, B, (ne, nmu))
    return dict(zip(names, nb)), ne, nmu, e, p, mu, mst, ier, sol
_C = {}
def beta_eq_cont(n, B, dx=0.1):
    N0 = P['N0']; key = (P['name'],) + tuple((b[0],) + tuple(b[5:9]) for b in B); cache = _C.setdefault(key, {}); x = n / N0; start = max([v for v in cache if v <= x], default=None)
    sol = cache[start] if start is not None else None; xcur = start if start is not None else 0.3
    if start is None: sol = beta_eq(0.3 * N0, B)[-1]; cache[0.3] = sol
    while xcur < x - 1e-9: xcur = min(xcur + dx, x); sol = beta_eq(xcur * N0, B, sol)[-1]; cache[round(xcur, 6)] = sol
    return beta_eq(n, B, sol)
def state(n, YL, B, guess=None):
    """Partial equilibrium at fixed n and lepton fraction (mu_mu = mu_e), Deltas in strong equilibrium; unknowns Phi, W, B, mu_n, mu_p."""
    from partial_eq import mue_of_nL, lepton_density
    mue = mue_of_nL(YL * n); ne, nmu = lepton_density(mue)
    def mu_of(nm, mun, mup): return {'n': mun, 'p': mup, 'D-': 2 * mun - mup, 'D0': mun, 'D+': mup, 'D++': 2 * mup - mun}[nm]
    def dens(v):
        ph, w, r, mun, mup = v; nb = []
        for (nm, m, t3, q, g, xs, xw, xr, *rest) in B:
            ms = m - xs * ph; estar = mu_of(nm, mun, mup) - xw * w - xr * I3(t3) * r
            k = np.sqrt(estar ** 2 - ms ** 2) if estar > ms else 0.0; nb.append(g * k ** 3 / (6 * np.pi ** 2) / HC ** 3)
        return nb
    def eqs(v):
        ph, w, r, mun, mup = v; nb = dens(v); s2, w2, r2 = solve_fields(nb, B)
        return [(ph - s2) / 10, (w - w2) / 10, (r - r2) / 10, (sum(nb) - n) * 100, (sum(q * ni for ni, (nm, m, t3, q, g, xs, xw, xr, *rest) in zip(nb, B)) - ne - nmu) * 100]
    if guess is None:
        c, ne0, nmu0, e0, p0, mu0, mst0, ier0, sol0 = beta_eq_cont(n, B); guess = np.array([sol0[0], sol0[1], sol0[2], sol0[3], sol0[3] - sol0[4]])
    sol, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-12, maxfev=4000); nb = dens(sol); e, p, *_ = thermo(nb, B, (ne, nmu))
    return dict(zip([b[0] for b in B], nb)), sol[3] - sol[4] - mue, sol, ier, p
def susceptibilities(n, B):
    c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B); YL = (ne + nmu) / n; st0 = state(n, YL, B); g0 = st0[2]; d = 2e-3
    Bq = (state(n, YL * (1 + d), B, g0)[1] - state(n, YL * (1 - d), B, g0)[1]) / (2 * d * YL * n)
    Cq = n * (state(n * (1 + d), YL, B, g0)[1] - state(n * (1 - d), YL, B, g0)[1]) / (2 * d * n)
    dn = 2e-3 * n; dYL = ((lambda s: (s[1] + s[2]) / (n + dn))(beta_eq_cont(n + dn, B)) - (lambda s: (s[1] + s[2]) / (n - dn))(beta_eq_cont(n - dn, B))) / (2 * dn)
    kS = n * (state(n * (1 + d), YL, B, g0)[4] - state(n * (1 - d), YL, B, g0)[4]) / (2 * d * n)
    return Bq, Cq, -n ** 2 * Bq * dYL, YL, st0[1], c, ne, nmu, mst, kS
if __name__ == '__main__':
    lines = []
    for name in SETS:
        use(name); ok, line, res = saturation_check(); lines.append(line)
    open('results/fsu_saturation.txt', 'w').write("\n".join(lines) + "\n")
