"""DD2-type density-dependent RMF (Typel et al. PRC 81, 015803 (2010) parameters) with the Delta(1232) quartet, beta-equilibrated npe(mu)Delta matter.
Self-contained (delta_urca_bulkvisc project).  Units: MeV, fm.  Mean fields sigma, omega, rho with density-dependent couplings Gamma_i(n_B) and the
rearrangement term.  Delta couplings: x_sD = Gamma_sD/Gamma_sN etc. (defaults 1.1, 1.1, 1.0).
Outputs: saturation check of DD2 (n0, E/A, K, J, L, m*/m), beta-equilibrium composition vs n_B, Delta onset, and the direct-Urca triangle
conditions for n <-> p e, Delta^- <-> n e, Delta^0 <-> p e."""
import numpy as np
from scipy.optimize import brentq, fsolve
HC = 197.3269804
# Parameter sets: DD2 (Typel 2010) and DD-ME2 (Lalazissis 2005); selected by env RMFSET (default DD2)
import os
SETS = {'DD2': dict(MS=546.212459, MW=783.0, MR=763.0, N0=0.149065, GS0=10.686681, GW0=13.342362, GR0=3.626940, AS=1.357630, BS=0.634442, CS=1.005358, DS=0.575810, AW=1.369718, BW=0.496475, CW=0.817753, DW=0.638452, AR=0.518903),
        'DDME2': dict(MS=550.1238, MW=783.0, MR=763.0, N0=0.152, GS0=10.5396, GW0=13.0189, GR0=3.6836, AS=1.3881, BS=1.0943, CS=1.7057, DS=0.4421, AW=1.3892, BW=0.9240, CW=1.4620, DW=0.4775, AR=0.5647)}
RMFSET = os.environ.get('RMFSET', 'DD2'); _P = SETS[RMFSET]
MS, MW, MR, N0 = _P['MS'], _P['MW'], _P['MR'], _P['N0']; GS0, GW0, GR0 = _P['GS0'], _P['GW0'], _P['GR0']
AS, BS, CS, DS = _P['AS'], _P['BS'], _P['CS'], _P['DS']; AW, BW, CW, DW = _P['AW'], _P['BW'], _P['CW'], _P['DW']
AR = _P['AR']   # DD2 a_rho = 0.518903 (Typel 2010; cross-checked: arXiv:2407.04403 lists 0.5189, L0 = 55.12; my earlier 0.983955 was wrong); DD-ME2 a_rho = 0.5647
MN, MP, ME, MMU, MD = 939.56536, 938.27203, 0.510999, 105.6584, 1232.0
def h(x, a, b, c, d): return a * (1 + b * (x + d) ** 2) / (1 + c * (x + d) ** 2)
def dh(x, a, b, c, d): return a * (2 * b * (x + d) * (1 + c * (x + d) ** 2) - 2 * c * (x + d) * (1 + b * (x + d) ** 2)) / (1 + c * (x + d) ** 2) ** 2
def couplings(n):
    x = n / N0
    gs, gw, gr = GS0 * h(x, AS, BS, CS, DS), GW0 * h(x, AW, BW, CW, DW), GR0 * np.exp(-AR * (x - 1))
    dgs, dgw, dgr = GS0 * dh(x, AS, BS, CS, DS) / N0, GW0 * dh(x, AW, BW, CW, DW) / N0, -AR / N0 * gr
    return gs, gw, gr, dgs, dgw, dgr
# baryons: (name, mass, tau_3 = 2 I3 (DD2 rho vertex Gamma_rho tau_3; nucleons +-1), charge, degeneracy, x_sigma, x_omega, x_rho)
MPHI = 1019.461; MLAM, MSIG, MXI = 1115.683, 1193.0, 1318.0
SQ2 = np.sqrt(2.0)
def baryons(xs=1.1, xw=1.1, xr=1.0, hyperons=False, xphi=None):
    """(name, mass, tau_3 = 2 I3, charge, degeneracy, x_sigma, x_omega, x_rho[, x_phi, strangeness]).  Hyperon couplings = Raduta CompOSE
    R(DD2YDelta) sheet: x_sL 0.6154, x_sX 0.3259, x_sS 0.4740; x_w 2/3, 1/3, 2/3; x_r 0, 1, 2 (tau_3 of Xi^- = -1, Sigma^- = -2 in this convention);
    x_phi (SU(6)): Lambda, Sigma -sqrt2/3, Xi -2 sqrt2/3 (relative to g_omega N; the phi couples to hyperons only)."""
    L = [('n', MN, -1.0, 0, 2, 1, 1, 1, 0.0, 0), ('p', MP, 1.0, 1, 2, 1, 1, 1, 0.0, 0),
         ('D-', MD, -3.0, -1, 4, xs, xw, xr, 0.0, 0), ('D0', MD, -1.0, 0, 4, xs, xw, xr, 0.0, 0), ('D+', MD, 1.0, 1, 4, xs, xw, xr, 0.0, 0), ('D++', MD, 3.0, 2, 4, xs, xw, xr, 0.0, 0)]
    if hyperons:
        xpL, xpS, xpX = (xphi if xphi else (-SQ2 / 3, -SQ2 / 3, -2 * SQ2 / 3))
        L += [('L', MLAM, 0.0, 0, 2, 0.6154, 2 / 3, 0.0, xpL, -1), ('S-', MSIG, -2.0, -1, 2, 0.4740, 2 / 3, 2.0, xpS, -1), ('S0', MSIG, 0.0, 0, 2, 0.4740, 2 / 3, 2.0, xpS, -1), ('S+', MSIG, 2.0, 1, 2, 0.4740, 2 / 3, 2.0, xpS, -1),
              ('X-', MXI, -1.0, -1, 2, 0.3259, 1 / 3, 1.0, xpX, -2), ('X0', MXI, 1.0, 0, 2, 0.3259, 1 / 3, 1.0, xpX, -2)]
    return L
def kf(n, g): return (6 * np.pi ** 2 * n / g) ** (1 / 3) * HC if n > 0 else 0.0          # MeV
def ns_of(kF, ms, g):
    """scalar density (fm^-3) for degeneracy g"""
    if kF <= 0: return 0.0
    ef = np.sqrt(kF ** 2 + ms ** 2); return g / (4 * np.pi ** 2) * ms * (kF * ef - ms ** 2 * np.log((kF + ef) / ms)) / HC ** 3
def eps_kin(kF, ms, g):
    if kF <= 0: return 0.0
    ef = np.sqrt(kF ** 2 + ms ** 2); return g / (16 * np.pi ** 2) * (kF * ef * (2 * kF ** 2 + ms ** 2) - ms ** 4 * np.log((kF + ef) / ms)) / HC ** 3
def p_kin(kF, ms, g):
    if kF <= 0: return 0.0
    ef = np.sqrt(kF ** 2 + ms ** 2); return g / (48 * np.pi ** 2) * (kF * ef * (2 * kF ** 2 - 3 * ms ** 2) + 3 * ms ** 4 * np.log((kF + ef) / ms)) / HC ** 3
def solve_fields(nb_list, B, sigma0=None):
    """Given baryon densities (fm^-3) per species, solve for sigma (MeV); omega, rho are algebraic."""
    n = sum(nb_list); gs, gw, gr, dgs, dgw, dgr = couplings(n)
    def F(s):
        rhs = 0.0
        for nb, (nm, m, i3, q, g, xs, xw, xr, xp, S) in zip(nb_list, B):
            ms = m - xs * gs * s; rhs += xs * ns_of(kf(nb, g), ms, g)
        return MS ** 2 * s / HC ** 3 - gs * rhs
    smax = min(m / (xs * gs) for (nm, m, i3, q, g, xs, xw, xr, xp, S) in B) * 0.999
    s = brentq(F, 0.0, smax) if F(smax) > 0 else smax
    w = gw * sum(xw * nb for nb, (nm, m, i3, q, g, xs, xw, xr, xp, S) in zip(nb_list, B)) * HC ** 3 / MW ** 2
    r = gr * sum(xr * i3 * nb for nb, (nm, m, i3, q, g, xs, xw, xr, xp, S) in zip(nb_list, B)) * HC ** 3 / MR ** 2
    ph = gw * sum(xp * nb for nb, (nm, m, i3, q, g, xs, xw, xr, xp, S) in zip(nb_list, B)) * HC ** 3 / MPHI ** 2
    return s, w, r, ph, (gs, gw, gr, dgs, dgw, dgr)
def thermo(nb_list, B, nl):
    """Energy density, pressure (MeV/fm^3), baryon chemical potentials (MeV), effective masses; nl = (n_e, n_mu).  Includes the phi meson."""
    n = sum(nb_list); s, w, r, ph, (gs, gw, gr, dgs, dgw, dgr) = solve_fields(nb_list, B)
    ns_tot = sum(xs * ns_of(kf(nb, g), m - xs * gs * s, g) for nb, (nm, m, i3, q, g, xs, xw, xr, xp, S) in zip(nb_list, B))
    nw = sum(xw * nb for nb, (nm, m, i3, q, g, xs, xw, xr, xp, S) in zip(nb_list, B)); nr = sum(xr * i3 * nb for nb, (nm, m, i3, q, g, xs, xw, xr, xp, S) in zip(nb_list, B)); nph = sum(xp * nb for nb, (nm, m, i3, q, g, xs, xw, xr, xp, S) in zip(nb_list, B))
    SigR = (dgw * w * nw - dgs * s * ns_tot + dgr * r * nr + dgw * ph * nph)           # rearrangement self-energy (MeV); the phi coupling scales with g_omega(n)
    eps = 0.5 * MS ** 2 * s ** 2 / HC ** 3 + 0.5 * MW ** 2 * w ** 2 / HC ** 3 + 0.5 * MR ** 2 * r ** 2 / HC ** 3 + 0.5 * MPHI ** 2 * ph ** 2 / HC ** 3; mu = {}; mst = {}
    for nb, (nm, m, i3, q, g, xs, xw, xr, xp, S) in zip(nb_list, B):
        ms = m - xs * gs * s; k = kf(nb, g); eps += eps_kin(k, ms, g); mu[nm] = np.sqrt(k ** 2 + ms ** 2) + xw * gw * w + xr * gr * i3 * r + xp * gw * ph + SigR; mst[nm] = ms
    p = sum(nb * mu[nm] for nb, (nm, *_ ) in zip(nb_list, B)) - eps
    for nl_, ml in zip(nl, (ME, MMU)):
        k = kf(nl_, 2); eps += eps_kin(k, ml, 2); p += p_kin(k, ml, 2)
    return eps, p, mu, mst, s, w, r
def saturation_check():
    B = baryons()[:2]
    def EA(n): e, p, mu, mst, *_ = thermo([n / 2, n / 2, 0, 0, 0, 0], baryons(), (0, 0)); return e / n - (MN + MP) / 2
    n0 = brentq(lambda n: (EA(n + 1e-5) - EA(n - 1e-5)), 0.12, 0.18); e0 = EA(n0); d = 1e-3
    K = 9 * n0 ** 2 * (EA(n0 + d) - 2 * EA(n0) + EA(n0 - d)) / d ** 2
    def Esym(n, a=0.02):   # [(E/A)(+delta) + (E/A)(-delta)]/2 - (E/A)(0) ~ J delta^2  (the +- average removes the n-p mass difference term)
        ep, *_ = thermo([n * (1 + a) / 2, n * (1 - a) / 2, 0, 0, 0, 0], baryons(), (0, 0)); em, *_ = thermo([n * (1 - a) / 2, n * (1 + a) / 2, 0, 0, 0, 0], baryons(), (0, 0))
        return ((ep + em) / (2 * n) - EA(n) - (MN + MP) / 2) / a ** 2
    J = Esym(n0); L = 3 * n0 * (Esym(n0 + d) - Esym(n0 - d)) / (2 * d)
    e, p, mu, mst, *_ = thermo([n0 / 2, n0 / 2, 0, 0, 0, 0], baryons(), (0, 0))
    print(f"{RMFSET} saturation: n0 {n0:.4f} fm^-3 (lit 0.1491), E/A {e0:.2f} MeV (lit -16.02), K {K:.1f} (lit 242.7), J {J:.2f} (lit 31.67), L {L:.1f} (lit 55.0), m*/m {mst['n']/MN:.4f} (lit 0.5625), p(n0) {p:.3f} MeV/fm^3 (should be ~0)")
def beta_eq(n, B, guess=None):
    """Beta equilibrium + charge neutrality (full chemical equilibrium incl. strangeness).  Unknowns: sigma, omega, rho, phi (MeV), mu_n, mu_e.
    mu_i = mu_n - q_i mu_e; k_F,i from mu_i - vector self-energies - Sigma_R = sqrt(k^2 + m*^2) (zero below threshold)."""
    names = [b[0] for b in B]
    def dens(v):
        s, w, r, ph, mun, mue = v; gs, gw, gr, dgs, dgw, dgr = couplings(n); SigR = 0.0
        for it in range(60):
            nb = []; ns_tot = 0.0; nw = 0.0; nr = 0.0; nph = 0.0
            for (nm, m, t3, q, g, xs, xw, xr, xp, S) in B:
                ms = m - xs * gs * s; mui = mun - q * mue; estar = mui - xw * gw * w - xr * gr * t3 * r - xp * gw * ph - SigR
                k = np.sqrt(estar ** 2 - ms ** 2) if estar > ms else 0.0; ni = g * k ** 3 / (6 * np.pi ** 2) / HC ** 3; nb.append(ni)
                ns_tot += xs * ns_of(k, ms, g); nw += xw * ni; nr += xr * t3 * ni; nph += xp * ni
            SigR_new = dgw * w * nw - dgs * s * ns_tot + dgr * r * nr + dgw * ph * nph
            if abs(SigR_new - SigR) < 1e-9: SigR = SigR_new; break
            SigR = 0.5 * SigR + 0.5 * SigR_new
        ne = 2 * (mue ** 2 - ME ** 2) ** 1.5 / (6 * np.pi ** 2) / HC ** 3 if mue > ME else 0.0; nmu = 2 * (mue ** 2 - MMU ** 2) ** 1.5 / (6 * np.pi ** 2) / HC ** 3 if mue > MMU else 0.0
        return nb, ne, nmu, ns_tot, nw, nr, nph, gs, gw, gr
    def eqs(v):
        s, w, r, ph, mun, mue = v; nb, ne, nmu, ns_tot, nw, nr, nph, gs, gw, gr = dens(v)
        return [MS ** 2 * s / HC ** 3 - gs * ns_tot, MW ** 2 * w / HC ** 3 - gw * nw, MR ** 2 * r / HC ** 3 - gr * nr, MPHI ** 2 * ph / HC ** 3 - gw * nph, (sum(nb) - n) * 100, (sum(q * ni for ni, (nm, m, t3, q, g, xs, xw, xr, xp, S) in zip(nb, B)) - ne - nmu) * 100]
    if guess is None:
        gs, gw, gr, *_ = couplings(n); s0 = 0.5 * MN / gs * 0.6 * (n / N0) ** 0.5; guess = np.array([min(s0, 0.6 * MN / gs), gw * n * HC ** 3 / MW ** 2, -gr * 0.8 * n * HC ** 3 / MR ** 2, 0.0, MN + 60 * n / N0, 100.0])
    sol, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-11, maxfev=5000)
    nb, ne, nmu, *_ = dens(sol); e, p, mu, mst, *_ = thermo(nb, B, (ne, nmu))
    return dict(zip(names, nb)), ne, nmu, e, p, mu, mst, ier, sol
_CONT_CACHE = {}
def beta_eq_cont(n, B, key=None, dx=0.1):
    """Beta equilibrium by density continuation from 0.3 n0 (the direct default guess fails above ~0.7 n0).  key = hashable id of B."""
    key = key if key is not None else tuple((b[0],) + tuple(b[5:9]) for b in B); cache = _CONT_CACHE.setdefault(key, {}); xs = sorted(cache)
    x = n / N0; start = max([v for v in xs if v <= x], default=None); sol = cache[start] if start is not None else None; xcur = start if start is not None else 0.3
    if start is None: sol = beta_eq(0.3 * N0, B)[-1]; cache[0.3] = sol; xcur = 0.3
    while xcur < x - 1e-9:
        xcur = min(xcur + dx, x); sol = beta_eq(xcur * N0, B, sol)[-1]; cache[round(xcur, 6)] = sol
    return beta_eq(n, B, sol)
if __name__ == '__main__':
    saturation_check()
    for xs in (1.1, 1.2):
        B = baryons(xs=xs); print(f"\n--- beta-equilibrated npe mu Delta matter, x_sD = {xs}, x_wD = 1.1, x_rD = 1.0")
        print("  n/n0   Y_p     Y_D-    Y_D0    Y_e     Y_mu    p[MeV/fm3]  e[MeV/fm3]   dUrca n->p e   D- -> n e   D0 -> p e")
        guess = None; onset = None
        for x in np.arange(0.5, 5.01, 0.25):
            n = x * N0; comp, ne, nmu, e, p, mu, mst, ier, sol = beta_eq(n, B, guess); guess = sol
            kp, kn, ke, kdm, kd0 = kf(comp['p'], 2), kf(comp['n'], 2), kf(ne, 2), kf(comp['D-'], 4), kf(comp['D0'], 4)
            du_np = kp + ke >= kn; du_dm = (comp['D-'] > 1e-6) and (kdm + ke >= kn); du_d0 = (comp['D0'] > 1e-6) and (kd0 + ke >= kp)
            if onset is None and comp['D-'] / n > 1e-4: onset = x
            print(f"  {x:4.2f}  {comp['p']/n:.4f}  {comp['D-']/n:.4f}  {comp['D0']/n:.4f}  {ne/n:.4f}  {nmu/n:.4f}  {p:9.3f}   {e:9.2f}      {str(du_np):5s}         {str(du_dm):5s}       {str(du_d0):5s}   {'' if ier==1 else 'NOCONV'}")
        print(f"  Delta^- onset ~ {onset} n0")
