# NOTE 2026-10-16 20:45: gamma_nd/lambda_fullT below implement AHS Eq. (10) (nonrelativistic, with antineutrino blocking = trapped form). SUPERSEDED for the physics tables by urca_rel.py (two Landau masses, transparent). Kept for the equal-mass cross-check.
"""RUNG 4: Urca rates at finite temperature beyond the Fermi-surface approximation (Alford-Harutyunyan-Sedrakian PRD 100, 103021 (2019),
Eqs. 10, 12, 20, 21; neutrino-transparent: mu_nu = 0, alpha_nu = 0, no incoming neutrinos).  Nonrelativistic nucleons with effective mass m*,
effective chemical potentials mu_j* (alpha_j = mu_j*/T, measured from the effective mass, i.e. alpha_n = (mu_n - U_n - m_n*)/T etc.).
  Gamma_nd(muD) = G~^2 m*^2 T^6/(8 pi^5) int_{-ae}^{inf} dy g(y - muD/T) int_0^{y+ae} dz ln[(1+e^{-y0})/(1+e^{-y0-y+muD/T})] int_{xmin}^{xmax} dx x (y+ae-x) f(x-y)[1-f(x)]
  Gamma_ec(muD) = G~^2 m*^2 T^6/(8 pi^5) int_{-inf}^{inf} dy [1+g(y - muD/T)] int_{|y+ae|}^{inf} dz ln[...] int_{xbar_min}^{inf} dx x (y+ae+x) f(x+y)[1-f(x)]
  y0 = m*/(2 T z^2) (an - ap + y - z^2 T/(2m*) - muD/T)^2 - ap,   xmin/max = (y + ae -+ z)/2,  xbar_min = (z - y - ae)/2,  G~^2 = G_F^2 cos^2(theta_c)(1 + 3 g_A^2)
  lambda = d(Gamma_nd - Gamma_ec)/d muD at muD = 0  (detailed balance: Gamma_nd(0) = Gamma_ec(0)).
Units: natural (MeV); converted with NAT_TO_PHYS of urca_rates.  Verification: FS limit lambda_dU (17/240 pi) G~^2 m_n m_p p_Fe T^4 theta at T = 1 MeV."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from numpy.polynomial.legendre import leggauss
from urca_rates import G2, GA, NAT_TO_PHYS
GT2 = G2 * (1 + 3 * GA ** 2)
def fF(x): return 1.0 / (1.0 + np.exp(np.clip(x, -700, 700)))
def gB(x): x = np.where(np.abs(x) < 1e-12, 1e-12, x); return 1.0 / (np.expm1(np.clip(x, -700, 700)))
def _ln(y0, y, md): return np.log1p(np.exp(-np.clip(y0, -700, 700))) - np.log1p(np.exp(-np.clip(y0 + y - md, -700, 700)))
def gamma_nd(T, mstar, an, ap, ae, muD, ny=120, nz=80, nx=48, ymax=40.0):
    """neutron decay rate (natural units MeV^4) ; an, ap, ae = alpha_j = mu_j*/T (kinetic chemical potentials / T)."""
    md = muD / T; yg, yw = leggauss(ny); y = -ae + (ymax + ae) * (yg + 1) / 2; wy = yw * (ymax + ae) / 2; tot = 0.0
    xg, xw = leggauss(nx); zg, zw = leggauss(nz)
    for yi, wyi in zip(y, wy):
        zmax = yi + ae
        if zmax <= 0: continue
        z = zmax * (zg + 1) / 2; wz = zw * zmax / 2
        y0 = mstar / (2 * T * z ** 2) * (an - ap + yi - z ** 2 * T / (2 * mstar) - md) ** 2 - ap; L = _ln(y0, yi, md)
        xmin = (yi + ae - z) / 2; xmax = (yi + ae + z) / 2; xm = (xmax - xmin)[:, None] * (xg[None, :] + 1) / 2 + xmin[:, None]; wx = (xmax - xmin)[:, None] * xw[None, :] / 2
        inner = np.sum(xm * (yi + ae - xm) * fF(xm - yi) * (1 - fF(xm)) * wx, axis=1)
        tot += wyi * gB(yi - md) * np.sum(L * inner * wz)
    return GT2 * mstar ** 2 * T ** 6 / (8 * np.pi ** 5) * tot
def gamma_ec(T, mstar, an, ap, ae, muD, ny=160, nz=80, nx=48, ymax=40.0, zmax_extra=40.0):
    md = muD / T; yg, yw = leggauss(ny); y = -ymax + 2 * ymax * (yg + 1) / 2; wy = yw * ymax; tot = 0.0; zg, zw = leggauss(nz); xg, xw = leggauss(nx)
    for yi, wyi in zip(y, wy):
        zlo = abs(yi + ae); zhi = zlo + zmax_extra; z = zlo + (zhi - zlo) * (zg + 1) / 2; wz = zw * (zhi - zlo) / 2
        y0 = mstar / (2 * T * z ** 2) * (an - ap + yi - z ** 2 * T / (2 * mstar) - md) ** 2 - ap; L = _ln(y0, yi, md)
        xbmin = (z - yi - ae) / 2; xbmax = xbmin + 40.0; xm = (xbmax - xbmin)[:, None] * (xg[None, :] + 1) / 2 + xbmin[:, None]; wx = (xbmax - xbmin)[:, None] * xw[None, :] / 2
        inner = np.sum(xm * (yi + ae + xm) * fF(xm + yi) * (1 - fF(xm)) * wx, axis=1)
        tot += wyi * (1 + gB(yi - md)) * np.sum(L * inner * wz)
    return GT2 * mstar ** 2 * T ** 6 / (8 * np.pi ** 5) * tot
def lambda_fullT(T, mstar, mu_n_kin, mu_p_kin, mu_e, dmu=None):
    """lambda = d(Gamma_nd - Gamma_ec)/d muD at 0, in 1/(MeV s fm^3).  mu_j_kin = kinetic (effective) chemical potentials = sqrt(k_F^2 + m*^2) - m* (nonrel: k_F^2/2m*)."""
    an, ap, ae = mu_n_kin / T, mu_p_kin / T, mu_e / T; d = dmu if dmu else 0.05 * T
    # time-reversal symmetry with mu_nu = 0: Gamma_ec(muD) = Gamma_nd(-muD)  ->  lambda = d[Gamma_nd(muD) - Gamma_nd(-muD)]/d muD = 2 dGamma_nd/dmuD
    Gp = gamma_nd(T, mstar, an, ap, ae, d); Gm = gamma_nd(T, mstar, an, ap, ae, -d)
    return (Gp - Gm) / d * NAT_TO_PHYS, gamma_nd(T, mstar, an, ap, ae, 0.0) * NAT_TO_PHYS, gamma_ec(T, mstar, an, ap, ae, 0.0) * NAT_TO_PHYS
if __name__ == '__main__':
    import time; from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0; from urca_rates import lam_dU, lam_mU, theta_dU
    # (a) Eq. 10 above threshold vs the Fermi-surface direct-Urca formula: artificial composition pFn = 300, pFp = 200, pFe = 150 MeV, m* = 600 MeV
    ms = 600.0; pn, pp, pe = 300.0, 200.0, 150.0; mun_k, mup_k = pn ** 2 / (2 * ms), pp ** 2 / (2 * ms)
    for T in (0.5, 1.0, 2.0):
        lam, gnd, gec = lambda_fullT(T, ms, mun_k, mup_k, pe); fs = lam_dU(ms + mun_k, ms + mup_k, pe, T, 1.0)
        print(f"ABOVE threshold (pFn-pFp-pFe = {pn-pp-pe:+.0f}): T={T}: lambda_fullT {lam:.4e} vs FS lambda_dU {fs:.4e}  ratio {lam/fs:.3f}; Gamma_nd(0) {gnd:.3e} Gamma_ec_eq21(0) {gec:.3e}", flush=True)
    pn = 380.0; mun_k = pn ** 2 / (2 * ms)
    for T in (1.0, 3.0, 5.0):
        lam, gnd, gec = lambda_fullT(T, ms, mun_k, mup_k, pe); print(f"BELOW threshold (pFn-pFp-pFe = {pn-pp-pe:+.0f}): T={T}: lambda_fullT {lam:.4e} (FS: 0); Gamma_nd(0) {gnd:.3e}", flush=True)
    B = baryons()[:2]; t0 = time.time()
    print("Full-T Urca rates (AHS 2019 Eqs. 10/21) vs Fermi-surface direct Urca; DD2 npemu.  Columns: T, Gamma_nd(0), Gamma_ec(0) [detailed balance], lambda_fullT, lambda_dU(FS, with theta), lambda_mU(FS)")
    for x in (1.0, 2.0, 3.0, 4.0, 5.0):
        n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B); pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2)
        ms = 0.5 * (mst['n'] + mst['p']); mun_k = np.sqrt(pn ** 2 + mst['n'] ** 2) - mst['n']; mup_k = np.sqrt(pp ** 2 + mst['p'] ** 2) - mst['p']; En, Ep = np.sqrt(pn ** 2 + mst['n'] ** 2), np.sqrt(pp ** 2 + mst['p'] ** 2)
        print(f"n={x} n0: pFn {pn:.0f} pFp {pp:.0f} pFe {pe:.0f} MeV, m* {ms:.0f}, dUrca(FS) {'open' if theta_dU(pn,pp,pe) else 'closed'} (pFn-pFp-pFe = {pn-pp-pe:+.0f} MeV)")
        for T in (1.0, 3.0, 5.0, 10.0):
            lam, gnd, gec = lambda_fullT(T, ms, mun_k, mup_k, pe)
            print(f"    T={T:4.1f}: Gamma_nd {gnd:.3e} Gamma_ec {gec:.3e} (ratio {gnd/gec if gec else float('nan'):.3f}) | lambda_fullT {lam:.3e} | FS: lambda_dU {lam_dU(En, Ep, pe, T, theta_dU(pn,pp,pe)):.3e} lambda_mU {lam_mU(En, Ep, pn, pp, pe, T):.3e}  [{time.time()-t0:.0f}s]", flush=True)
# ---------------------------------------------------------------------------------------------------------------------------------
# 2026-10-16 20:20 FIX: Eq. (10) is NONRELATIVISTIC in the nucleons (dispersion p^2/2m*).  Feeding it relativistic kinetic chemical
# potentials sqrt(pF^2+m*^2)-m* makes its internal Fermi momenta sqrt(2 m* mu*) smaller than the true ones (threshold gap +12 instead
# of +49 MeV at 2 n0 in DD2), i.e. far too much thermal unblocking.  Consistent use for relativistic quasiparticles: expand around the
# Fermi surface with the LANDAU mass m_L = sqrt(pF^2 + m*^2): mu* = pF^2/(2 m_L) reproduces the exact pF, and v_F = pF/m_L the exact Fermi
# velocity.  Eq. (10) has a single mass: we take the proton Landau mass (the proton is the cheapest particle to excite below threshold,
# v_p < v_n < v_e = 1, so it sets the unblocking energy), and restore the density-of-states prefactor m_n m_p -> E*_Fn E*_Fp.
# Residual approximation: the neutron Fermi velocity is pFn/E*_Fp instead of pFn/E*_Fn (over-suppresses the subdominant neutron-hole path).
def _landau_inputs(T, mstn, mstp, pn, pp, pe):
    EFn, EFp = np.sqrt(pn ** 2 + mstn ** 2), np.sqrt(pp ** 2 + mstp ** 2); mL = EFp
    return mL, pn ** 2 / (2 * mL * T), pp ** 2 / (2 * mL * T), pe / T, EFn / EFp
def gamma_nd_rel(T, mstn, mstp, pn, pp, pe, muD, **kw):
    mL, an, ap, ae, pref = _landau_inputs(T, mstn, mstp, pn, pp, pe); return pref * gamma_nd(T, mL, an, ap, ae, muD, **kw)
def lambda_fullT_rel(T, mstn, mstp, pn, pp, pe, dmu=None):
    """lambda = 2 dGamma_nd/dmuD at 0 in 1/(MeV s fm^3) (time reversal: Gamma_ec(muD) = Gamma_nd(-muD)); also Gamma_nd(0) in 1/(s fm^3)."""
    d = dmu if dmu else 0.05 * T; Gp = gamma_nd_rel(T, mstn, mstp, pn, pp, pe, d); Gm = gamma_nd_rel(T, mstn, mstp, pn, pp, pe, -d)
    return (Gp - Gm) / d * NAT_TO_PHYS, gamma_nd_rel(T, mstn, mstp, pn, pp, pe, 0.0) * NAT_TO_PHYS
