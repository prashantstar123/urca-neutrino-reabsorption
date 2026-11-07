"""Finite-temperature RMF for the susceptibilities (delta_urca_bulkvisc, ladder rung 2).  Fermi-Dirac integrals (numerical) for each species:
n(mu*, m*, T), scalar density, kinetic energy, pressure, entropy; antiparticles neglected (T << m*).  Partial-equilibrium state at fixed
(n, Y_e, Y_mu, T) with the Deltas in strong equilibrium; beta equilibrium at T; susceptibilities isothermal (fixed T) and ADIABATIC (fixed s/n:
T re-solved along each perturbation).  Checks: T -> 0 limit reproduces channels.susc2; entropy -> 0 as T -> 0."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from scipy.integrate import quad
from scipy.optimize import fsolve, brentq
from rmf_dd2_delta import couplings, baryons, beta_eq_cont, ME, MMU, MS, MW, MR, HC, N0
def fd_integrals(mu, m, T, g):
    """Returns n [fm^-3], n_s [fm^-3], eps_kin [MeV/fm^3], p_kin [MeV/fm^3], s [1/fm^3] for one species with effective chemical potential mu (= E at k=0 -> m), mass m, T (MeV)."""
    if T < 1e-3:
        if mu <= m: return 0.0, 0.0, 0.0, 0.0, 0.0
        k = np.sqrt(mu ** 2 - m ** 2); E = mu
        n = g * k ** 3 / (6 * np.pi ** 2); ns = g / (4 * np.pi ** 2) * m * (k * E - m ** 2 * np.log((k + E) / m))
        e = g / (16 * np.pi ** 2) * (k * E * (2 * k ** 2 + m ** 2) - m ** 4 * np.log((k + E) / m)); p = g / (48 * np.pi ** 2) * (k * E * (2 * k ** 2 - 3 * m ** 2) + 3 * m ** 4 * np.log((k + E) / m))
        return n / HC ** 3, ns / HC ** 3, e / HC ** 3, p / HC ** 3, 0.0
    if not np.isfinite(mu) or abs(mu) > 1e5: return (0.0, 0.0, 0.0, 0.0, 0.0) if mu < m else (1e3, 1e3, 1e6, 1e6, 1e3)   # guard against solver excursions
    kmax = np.sqrt(max((mu + 40 * T) ** 2 - m ** 2, 0.0)) + 40 * T
    def f(k): E = np.sqrt(k * k + m * m); x = (E - mu) / T; return 1.0 / (np.exp(min(x, 700.0)) + 1.0)
    def sfun(k):
        E = np.sqrt(k * k + m * m); x = (E - mu) / T
        if x > 700: return 0.0
        ff = 1.0 / (np.exp(x) + 1.0); return -(ff * np.log(max(ff, 1e-300)) + (1 - ff) * np.log(max(1 - ff, 1e-300)))
    pref = g / (2 * np.pi ** 2); pts = [np.sqrt(max(mu ** 2 - m ** 2, 0.0))] if mu > m else None
    n = pref * quad(lambda k: k * k * f(k), 0, kmax, points=pts, limit=200)[0]
    ns = pref * quad(lambda k: k * k * m / np.sqrt(k * k + m * m) * f(k), 0, kmax, points=pts, limit=200)[0]
    e = pref * quad(lambda k: k * k * np.sqrt(k * k + m * m) * f(k), 0, kmax, points=pts, limit=200)[0]
    p = pref / 3 * quad(lambda k: k ** 4 / np.sqrt(k * k + m * m) * f(k), 0, kmax, points=pts, limit=200)[0]
    s = pref * quad(lambda k: k * k * sfun(k), 0, kmax, points=pts, limit=200)[0]
    return n / HC ** 3, ns / HC ** 3, e / HC ** 3, p / HC ** 3, s / HC ** 3
def mu_lepton(nl, m, T):
    if nl <= 0: return m - 50 * T
    return brentq(lambda mu: fd_integrals(mu, m, T, 2)[0] - nl, m - 60 * T - 1, 3000.0)
def stateT(n, Ye, Ymu, T, B, guess=None):
    """Fixed n, Y_e, Y_mu, T; baryons in strong equilibrium.  Returns dict with composition, dmu_e, dmu_mu, p, e, s (per fm^3), sol."""
    ne, nmu = Ye * n, Ymu * n; mue, mumu = mu_lepton(ne, ME, T), mu_lepton(nmu, MMU, T); gs, gw, gr, dgs, dgw, dgr = couplings(n)
    def mu_of(nm, mun, mup): return {'n': mun, 'p': mup, 'D-': 2 * mun - mup, 'D0': mun, 'D+': mup, 'D++': 2 * mup - mun}[nm]
    def dens(v):
        s, w, r, mun, mup = v; SigR = 0.0; out = None
        for it in range(80):
            nb = []; ns_tot = 0.0; nw = 0.0; nr = 0.0; ek = 0.0; pk = 0.0; sk = 0.0
            for (nm, m, t3, q, g, xs, xw, xr, *rest) in B:
                ms = m - xs * gs * s; mustar = mu_of(nm, mun, mup) - xw * gw * w - xr * gr * t3 * r - SigR
                ni, nsi, ei, pi_, si = fd_integrals(mustar, ms, T, g); nb.append(ni); ns_tot += xs * nsi; nw += xw * ni; nr += xr * t3 * ni; ek += ei; pk += pi_; sk += si
            SigR_new = dgw * w * nw - dgs * s * ns_tot + dgr * r * nr
            if abs(SigR_new - SigR) < 1e-8: SigR = SigR_new; break
            SigR = 0.5 * SigR + 0.5 * SigR_new
        return nb, ns_tot, nw, nr, ek, pk, sk, SigR
    def eqs(v):
        if not np.all(np.isfinite(v)): return [1e6] * 5
        s, w, r, mun, mup = v; nb, ns_tot, nw, nr, *_ = dens(v)
        return [MS ** 2 * s / HC ** 3 - gs * ns_tot, MW ** 2 * w / HC ** 3 - gw * nw, MR ** 2 * r / HC ** 3 - gr * nr, (sum(nb) - n) * 100,
                (sum(q * ni for ni, (nm, m, t3, q, g, xs, xw, xr, *rest) in zip(nb, B)) - ne - nmu) * 100]
    if guess is None:
        c, ne0, nmu0, e0, p0, mu0, mst0, ier0, sol0 = beta_eq_cont(n, B); guess = np.array([sol0[0], sol0[1], sol0[2], sol0[-2], sol0[-2] - sol0[-1]])
    sol, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-11, maxfev=3000); s_, w_, r_, mun, mup = sol
    nb, ns_tot, nw, nr, ek, pk, sk, SigR = dens(sol); names = [b[0] for b in B]
    efield = 0.5 * (MS ** 2 * s_ ** 2 + MW ** 2 * w_ ** 2 + MR ** 2 * r_ ** 2) / HC ** 3
    le = [fd_integrals(mue, ME, T, 2), fd_integrals(mumu, MMU, T, 2)]
    e = ek + efield + le[0][2] + le[1][2]; p = pk - efield + (MW ** 2 * w_ ** 2 + MR ** 2 * r_ ** 2) / HC ** 3 + n * SigR + le[0][3] + le[1][3]   # RMF pressure with rearrangement
    s = sk + le[0][4] + le[1][4]
    return dict(comp=dict(zip(names, nb)), mue=mue, mumu=mumu, dmu_e=mun - mup - mue, dmu_mu=mun - mup - mumu, p=p, e=e, s=s, sol=sol, ier=ier, mun=mun, mup=mup)
def beta_eqT(n, T, B, Ye0, Ymu0):
    """Beta equilibrium at T: solve dmu_e = dmu_mu = 0 for (Y_e, Y_mu) starting from the T=0 values."""
    g = [None]
    def F(v):
        st = stateT(n, abs(v[0]), abs(v[1]), T, B, g[0]); g[0] = st['sol']; return [st['dmu_e'], st['dmu_mu']]
    sol = fsolve(F, [Ye0, Ymu0], xtol=1e-10); return abs(sol[0]), abs(sol[1])
def suscT(n, T, B, Ye, Ymu, adiabatic=False, d=2e-3):
    st0 = stateT(n, Ye, Ymu, T, B); g0 = st0['sol']; s_n0 = st0['s'] / n
    def dm(nn, ye, ym):
        if not adiabatic or T < 1e-3: st = stateT(nn, ye, ym, T, B, g0); return np.array([st['dmu_e'], st['dmu_mu']])
        Tad = brentq(lambda TT: stateT(nn, ye, ym, TT, B, g0)['s'] / nn - s_n0, 0.3 * T, 3.0 * T); st = stateT(nn, ye, ym, Tad, B, g0); return np.array([st['dmu_e'], st['dmu_mu']])
    A = np.zeros((2, 2)); A[:, 0] = (dm(n, Ye * (1 + d), Ymu) - dm(n, Ye * (1 - d), Ymu)) / (2 * d * Ye * n)
    A[:, 1] = (dm(n, Ye, Ymu * (1 + d)) - dm(n, Ye, Ymu * (1 - d))) / (2 * d * Ymu * n) if Ymu > 1e-6 else 0.0
    C = n * (dm(n * (1 + d), Ye, Ymu) - dm(n * (1 - d), Ye, Ymu)) / (2 * d * n); return A, C, st0
if __name__ == '__main__':
    import time; t0 = time.time()
    for name, B in (('DD2+D1.1', baryons(xs=1.1, xw=1.1)),):
        for x in (2.0, 2.5, 3.0):
            n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B); Ye0, Ymu0 = ne / n, nmu / n
            A0, C0, st = suscT(n, 0.0, B, Ye0, Ymu0); print(f"{name} n={x}: T=0  C_e {C0[0]:7.1f} C_mu {C0[1]:7.1f}  Y_D- {st['comp'].get('D-',0)/n:.3f}  (T=0 reference)", flush=True)
            for T in (2.0, 5.0, 10.0):
                Ye, Ymu = beta_eqT(n, T, B, Ye0, Ymu0); Ai, Ci, st = suscT(n, T, B, Ye, Ymu, adiabatic=False)
                print(f"{name} n={x}: T={T:4.1f} MeV  Y_e {Ye:.4f} Y_mu {Ymu:.4f} Y_D- {st['comp'].get('D-',0)/n:.3f} s/n {st['s']/n:.3f}  isothermal C_e {Ci[0]:7.1f} C_mu {Ci[1]:7.1f}  A_ee {Ai[0,0]:8.0f}  [{time.time()-t0:.0f}s]", flush=True)
