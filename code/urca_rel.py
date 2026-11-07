"""Direct-Urca rates beyond the Fermi-surface approximation for RELATIVISTIC quasiparticles, neutrino-transparent regime.
Own derivation (2026-10-16 20:30), same phase-space reduction as Alford-Harutyunyan-Sedrakian PRD 100, 103021 Eq. (10), generalised to
two different nucleon dispersions: each nucleon j is expanded around its Fermi surface with its LANDAU mass m_j = sqrt(pF_j^2 + m*_j^2):
  eps_j(p) = p^2/(2 m_j) - mu_j*,   mu_j* = pF_j^2/(2 m_j)     (exact pF_j, exact Fermi velocity v_j = pF_j/m_j; curvature approximate)
Energy bookkeeping uses the TRUE chemical potentials: omega = eps_n - eps_p = E_e + E_nu - mu_e - dmu, dmu = mu_n - mu_p - mu_e.
Nucleon structure function (angular integral analytic, energy integral analytic):
  S(q, omega) = m_n m_p T/(4 pi^2 q) g(c) [Lam(x_lo) - Lam(x_hi)],  c = omega/T,  Lam(x) = ln[(1+e^-x)/(1+e^-(x-c))],  x = eps_n/T,
  p in [p_lo, p_hi] from |cos theta*| <= 1 with cos theta* = m_p [omega + mu_n* - mu_p* + q^2/(2 m_p) + p^2 (1/(2m_p) - 1/(2m_n))]/(p q).
Rate (per volume, natural units MeV^4):
  Gamma_nd(dmu) = G~^2 m_n m_p T^6/(8 pi^5) int dx_e dx_nu dz  x_e x_nu  g(c) [Lam_lo - Lam_hi] [1 - f(x_e - a_e)] [1 - f(x_nu) if trapped(mu_nu=0)]
with z = q/T in [|x_e - x_nu|, x_e + x_nu].  Equal masses + 'trapped' blocking reproduces AHS Eq. (10) (urca_fullT.gamma_nd) exactly.
Electron capture by time reversal (mu_nu = 0): Gamma_ec(dmu) = Gamma_nd(-dmu);  lambda = 2 dGamma_nd/ddmu."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from urca_fullT import GT2, fF, gB
from urca_rates import NAT_TO_PHYS
def _lam(x, c): return np.logaddexp(0.0, -x) - np.logaddexp(0.0, -x + c)
def _prange(q, omega, mn, mp, mun, mup):
    """allowed neutron momentum range [p_lo, p_hi] (MeV) for momentum transfer q and energy transfer omega (arrays broadcast)."""
    D = 1.0 / (2 * mp) - 1.0 / (2 * mn); K = omega + mun - mup + q ** 2 / (2 * mp); b = q / mp
    if abs(D) < 1e-12:
        plo = np.abs(K) / b; phi = np.full_like(plo, np.inf); return plo, phi
    if D > 0:
        disc1 = b ** 2 - 4 * D * K; ok = disc1 >= 0; s1 = np.sqrt(np.where(ok, disc1, 0.0))
        p1m = (b - s1) / (2 * D); p1p = (b + s1) / (2 * D); p2 = np.where(K < 0, (-b + s1) / (2 * D), 0.0)
        plo = np.maximum(np.maximum(p1m, p2), 0.0); phi = np.where(ok, p1p, -1.0); return plo, phi
    A = -D; disc = b ** 2 + 4 * A * K; ok = disc >= 0; s = np.sqrt(np.where(ok, disc, 0.0))
    pa = np.where(K > 0, (-b + s) / (2 * A), 0.0); plo = np.maximum(np.maximum(pa, (b - s) / (2 * A)), 0.0); phi = np.where(ok, (b + s) / (2 * A), -1.0); return plo, phi
def gamma_nd_2m(T, mstn, mstp, pFn, pFp, mue, dmu, trapped=False, ne=120, nn=80, nz=48, xmax=40.0):
    mn, mp = np.sqrt(pFn ** 2 + mstn ** 2), np.sqrt(pFp ** 2 + mstp ** 2); mun, mup = pFn ** 2 / (2 * mn), pFp ** 2 / (2 * mp); ae = mue / T
    xe_g, xe_w = leggauss(ne); xn_g, xn_w = leggauss(nn); z_g, z_w = leggauss(nz)
    e_lo, e_hi = max(0.0, ae - xmax), ae + xmax; xe = e_lo + (e_hi - e_lo) * (xe_g + 1) / 2; we = xe_w * (e_hi - e_lo) / 2
    xn = xmax * (xn_g + 1) / 2; wn = xn_w * xmax / 2
    XE, XN = np.meshgrid(xe, xn, indexing='ij'); WE, WN = np.meshgrid(we, wn, indexing='ij')
    c = XE + XN - ae - dmu / T; omega = c * T; csafe = np.where(np.abs(c) < 1e-9, 1e-9, c); g = gB(csafe)
    block = (1 - fF(XE - ae)) * ((1 - fF(XN)) if trapped else 1.0)
    zlo, zhi = np.abs(XE - XN), XE + XN; tot = np.zeros_like(XE)
    for zg, zw in zip(z_g, z_w):
        z = zlo + (zhi - zlo) * (zg + 1) / 2; wz = zw * (zhi - zlo) / 2; q = z * T
        plo, phi = _prange(q, omega, mn, mp, mun, mup); valid = phi > plo
        xlo = (plo ** 2 / (2 * mn) - mun) / T; xhi = np.where(np.isfinite(phi), (np.maximum(phi, 0.0) ** 2 / (2 * mn) - mun) / T, 1e6)
        L = np.where(valid, _lam(xhi, c) - _lam(xlo, c), 0.0); tot += wz * L   # integral of f(x-c)-f(x) from x_lo to x_hi
    val = np.sum(WE * WN * XE * XN * g * block * tot)
    return GT2 * mn * mp * T ** 6 / (8 * np.pi ** 5) * val
def lambda_2m(T, mstn, mstp, pFn, pFp, mue, dmu=None):
    """lambda = d(Gamma_nd - Gamma_ec)/ddmu at 0 = 2 dGamma_nd/ddmu, in 1/(MeV s fm^3); and Gamma_nd(0) in 1/(s fm^3)."""
    d = dmu if dmu else 0.05 * T; Gp = gamma_nd_2m(T, mstn, mstp, pFn, pFp, mue, d); Gm = gamma_nd_2m(T, mstn, mstp, pFn, pFp, mue, -d)
    return (Gp - Gm) / d * NAT_TO_PHYS, gamma_nd_2m(T, mstn, mstp, pFn, pFp, mue, 0.0) * NAT_TO_PHYS
if __name__ == '__main__':
    import sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from urca_fullT import gamma_nd; from urca_rates import lam_dU
    print("(1) EQUAL-MASS CHECK vs AHS Eq. (10) transcription (trapped blocking on, same NR inputs): must agree")
    m = 400.0; pn, pp, pe = 300.0, 200.0, 150.0
    for T in (1.0, 3.0):
        for dmu in (0.0, 2.0):
            an, ap, ae = pn ** 2 / (2 * m * T), pp ** 2 / (2 * m * T), pe / T; g_old = gamma_nd(T, m, an, ap, ae, dmu); mst = np.sqrt(m ** 2 - pn ** 2)   # Landau mass = m for n requires m*n = sqrt(m^2 - pn^2)
            g_new = gamma_nd_2m(T, np.sqrt(m ** 2 - pn ** 2), np.sqrt(m ** 2 - pp ** 2), pn, pp, pe, dmu, trapped=True)
            print(f"   T={T} dmu={dmu}: old {g_old:.6e} new {g_new:.6e} ratio {g_new/g_old:.6f}", flush=True)
    print("(2) FERMI-SURFACE LIMIT above threshold (transparent): lambda_2m / lambda_FS; relativistic composition m* = 350, pFp 200, pFe 150, pFn 300 (gap -50) and 340 (gap -10)")
    ms = 350.0
    for pn in (300.0, 340.0):
        En, Ep = np.sqrt(pn ** 2 + ms ** 2), np.sqrt(pp ** 2 + ms ** 2)
        for T in (0.5, 1.0, 2.0, 5.0):
            lam, g0 = lambda_2m(T, ms, ms, pn, pp, pe); print(f"   pFn {pn:.0f} T={T}: ratio {lam/lam_dU(En, Ep, pe, T, 1.0):.4f}", flush=True)
    print("(3) BELOW threshold, DD2 2 n0 composition (gap +49): lambda_2m/lambda_FS(open) vs exp(-E_min/T), E_min = v_p gap + gap^2/(2 m_Lp)")
    pn, pp, pe, mn_, mp_ = 395.0, 183.5, 162.5, 355.5, 354.2; mLp = np.sqrt(pp ** 2 + mp_ ** 2); vp = pp / mLp; Emin = vp * 49.0 + 49.0 ** 2 / (2 * mLp)
    En, Ep = np.sqrt(pn ** 2 + mn_ ** 2), np.sqrt(pp ** 2 + mp_ ** 2)
    for T in (2.0, 3.0, 5.0, 8.0, 10.0):
        lam, g0 = lambda_2m(T, mn_, mp_, pn, pp, pe); print(f"   T={T:4.1f}: {lam/lam_dU(En, Ep, pe, T, 1.0):9.3e}   exp(-E_min/T) = {np.exp(-Emin/T):9.3e}  (E_min {Emin:.1f} MeV)", flush=True)
    print("(4) quadrature convergence (gap +49, T = 3): default vs doubled")
    print(f"   {gamma_nd_2m(3.0, mn_, mp_, pn, pp, pe, 0.0):.6e} vs {gamma_nd_2m(3.0, mn_, mp_, pn, pp, pe, 0.0, ne=240, nn=160, nz=96):.6e}")
# ---------------------------------------------------------------------------------------------------------------------------------
# 2026-10-16 22:50 ELECTRON CAPTURE computed independently (Astra finding 1: capture is NOT the time reverse of decay in transparent matter).
# p + e -> n + nu:  momentum transfer to the nucleons q = p_e - p_nu, energy transfer omega' = eps_n - eps_p = E_e - E_nu - mu_e - dmu.
# Structure function for p -> n at the same kinematics: S_pn(q, omega') = exp(omega'/T) S_np(q, omega') (f_p(1-f_n) = e^{(eps_n-eps_p)/T} f_n(1-f_p)).
# Lepton factor: f_e(E_e) (electron in the initial state), neutrino free.  Same reduction as gamma_nd_2m:
#   Gamma_ec = G~^2 m_n m_p T^6/(8 pi^5) int dx_e dx_nu dz  x_e x_nu  f(x_e - a_e) e^{c'} g(c') [Lam(x_hi,c') - Lam(x_lo,c')],  c' = x_e - x_nu - a_e - dmu/T,
#   z = q/T in [|x_e - x_nu|, x_e + x_nu].
# Absorption of a thermal nu_e bath (mu_nu = 0), nu + n -> p + e (inverse of capture): same kinematics (q, omega'), occupation f_n(1-f_p) = S_np,
#   lepton factor (1 - f_e) f_nu(x_nu):  Gamma_abs = ... int x_e x_nu (1 - f(x_e - a_e)) f(x_nu) g(c') [Lam_hi - Lam_lo].  Detailed balance at dmu = 0: Gamma_abs = Gamma_ec.
# Absorption of a thermal nubar_e bath: we evaluate p + e^- + nubar_e -> n, the 3-body inverse of neutron decay (positron-production absorption
#   nubar_e + p -> n + e^+ is negligible in degenerate matter and is not evaluated); lepton factor f_e f_nubar with decay kinematics
#   (the electron is in the INITIAL state of p + e + nubar -> n, so it carries f_e, not 1 - f_e; f_e e^{c} S_np = (1 - f_e) e^{(E_nu - dmu)/T} S_np,
#   i.e. exactly the detailed-balance transform of the decay spectrum, Letter Eq. (1) with dmu -> -dmu).
#   CORRECTED 2026-11-07 (full re-audit): until then the factor (1 - f_e) e^{c} was used, which violated detailed balance for the nubar channel
#   (immaterial where inverse modified Urca dominates the nubar absorption: < 0.3 % in tau at the Table I states from n0 upward, -5.8 % at 0.5 n0, 3 MeV;
#   the previous kernel is kept in urca_rel_before_nubar_fix_20261107.bak).
def _lepton_grid(T, mue, ne, nn, xmax):
    ae = mue / T; xe_g, xe_w = leggauss(ne); xn_g, xn_w = leggauss(nn)
    e_lo, e_hi = max(0.0, ae - xmax), ae + xmax; xe = e_lo + (e_hi - e_lo) * (xe_g + 1) / 2; we = xe_w * (e_hi - e_lo) / 2
    xn = xmax * (xn_g + 1) / 2; wn = xn_w * xmax / 2
    XE, XN = np.meshgrid(xe, xn, indexing='ij'); WE, WN = np.meshgrid(we, wn, indexing='ij'); return ae, XE, XN, WE, WN
def _rate_generic(T, mstn, mstp, pFn, pFp, mue, dmu, sign_nu, lepton_factor, ne=120, nn=80, nz=48, xmax=40.0):
    """sign_nu = +1: neutrino in the final state with the electron (decay kinematics, omega = E_e + E_nu - mu_e - dmu);
       sign_nu = -1: capture/absorption kinematics (omega' = E_e - E_nu - mu_e - dmu).  lepton_factor(XE, XN, ae, c) -> weight."""
    mn, mp = np.sqrt(pFn ** 2 + mstn ** 2), np.sqrt(pFp ** 2 + mstp ** 2); mun, mup = pFn ** 2 / (2 * mn), pFp ** 2 / (2 * mp)
    ae, XE, XN, WE, WN = _lepton_grid(T, mue, ne, nn, xmax); z_g, z_w = leggauss(nz)
    c = XE + sign_nu * XN - ae - dmu / T; omega = c * T; csafe = np.where(np.abs(c) < 1e-9, 1e-9, c); g = gB(csafe); block = lepton_factor(XE, XN, ae, c)
    zlo, zhi = np.abs(XE - XN), XE + XN; tot = np.zeros_like(XE)
    for zg, zw in zip(z_g, z_w):
        z = zlo + (zhi - zlo) * (zg + 1) / 2; wz = zw * (zhi - zlo) / 2; q = z * T
        plo, phi = _prange(q, omega, mn, mp, mun, mup); valid = phi > plo
        xlo = (plo ** 2 / (2 * mn) - mun) / T; xhi = np.where(np.isfinite(phi), (np.maximum(phi, 0.0) ** 2 / (2 * mn) - mun) / T, 1e6)
        L = np.where(valid, _lam(xhi, c) - _lam(xlo, c), 0.0); tot += wz * L
    return GT2 * mn * mp * T ** 6 / (8 * np.pi ** 5) * np.sum(WE * WN * XE * XN * g * block * tot)
def gamma_nd_2m_b(T, mstn, mstp, pFn, pFp, mue, dmu, **kw):
    """neutron decay via the generic kernel (must equal gamma_nd_2m)."""
    return _rate_generic(T, mstn, mstp, pFn, pFp, mue, dmu, +1, lambda XE, XN, ae, c: 1 - fF(XE - ae), **kw)
def gamma_ec_2m(T, mstn, mstp, pFn, pFp, mue, dmu, **kw):
    """electron capture p + e -> n + nu (transparent): f_e e^{c'} S_np."""
    return _rate_generic(T, mstn, mstp, pFn, pFp, mue, dmu, -1, lambda XE, XN, ae, c: fF(XE - ae) * np.exp(np.clip(c, -700, 700)), **kw)
def gamma_abs_nu_2m(T, mstn, mstp, pFn, pFp, mue, dmu, **kw):
    """absorption rate per volume of a thermal nu_e bath (mu_nu = 0): nu + n -> p + e."""
    return _rate_generic(T, mstn, mstp, pFn, pFp, mue, dmu, -1, lambda XE, XN, ae, c: (1 - fF(XE - ae)) * fF(XN), **kw)
def gamma_abs_nubar_2m(T, mstn, mstp, pFn, pFp, mue, dmu, **kw):
    """absorption rate per volume of a thermal nubar_e bath via the 3-body inverse of decay (p + e + nubar -> n): f_e f_nubar S_pn = f_e f_nubar e^{c} S_np
    (occupation f_p f_e f_nubar (1 - f_n); electron in the initial state carries f_e; corrected 2026-11-07, see the comment block above)."""
    return _rate_generic(T, mstn, mstp, pFn, pFp, mue, dmu, +1, lambda XE, XN, ae, c: fF(XE - ae) * fF(XN) * np.exp(np.clip(c, -700, 700)), **kw)
def nu_mfp_km(T, Gamma):
    """mean free path [km] of a thermal (mu = 0) neutrino bath with number density n_nu = 3 zeta(3)/(4 pi^2) T^3 against a total absorption rate Gamma [1/(s fm^3)]."""
    HC = 197.3269804; nnu = 3 * 1.2020569 / (4 * np.pi ** 2) * T ** 3 / HC ** 3; return nnu * 2.99792458e23 / max(Gamma, 1e-300) * 1e-18
# ---------------------------------------------------------------------------------------------------------------------------------
# 2026-10-16 23:20 ENERGY-RESOLVED absorption and emission spectra (Astra finding 2 settling test, first part).
#   W_abs(E) [MeV]: absorption rate per neutrino of energy E = T x_nu (nu + n -> p + e), = kernel at a single x_nu node / neutrino phase-space density
#     thermal-bath rate Gamma_abs = int dn_nu(E) W(E), dn_nu = T^3 x^2 f(x) dx/(2 pi^2)  ->  W(x) = G~^2 m_n m_p T^3/(4 pi^3 x) int dx_e dz x_e (1-f_e) g Lam
#   mean free path lambda(E) = hbar c / W(E).  Emission spectra: dGamma_ec/dx_nu (capture nu_e), dGamma_nd/dx_nu (decay nubar_e), and the 3-body
#   nubar absorption W_abs,nubar(E) (p + e + nubar -> n) with decay kinematics.
def _kernel_fixed_xnu(T, mstn, mstp, pFn, pFp, mue, dmu, sign_nu, lepton_factor_e, xnu, ne=160, nz=64, xmax=40.0):
    """int dx_e dz x_e lepton_factor_e(x_e) g(c) [Lam_hi - Lam_lo] at fixed x_nu (no x_nu weight)."""
    mn, mp = np.sqrt(pFn ** 2 + mstn ** 2), np.sqrt(pFp ** 2 + mstp ** 2); mun, mup = pFn ** 2 / (2 * mn), pFp ** 2 / (2 * mp); ae = mue / T
    xe_g, xe_w = leggauss(ne); e_lo, e_hi = max(0.0, ae - xmax), ae + xmax; xe = e_lo + (e_hi - e_lo) * (xe_g + 1) / 2; we = xe_w * (e_hi - e_lo) / 2
    z_g, z_w = leggauss(nz); c = xe + sign_nu * xnu - ae - dmu / T; omega = c * T; csafe = np.where(np.abs(c) < 1e-9, 1e-9, c); g = gB(csafe); block = lepton_factor_e(xe, ae, c)
    zlo, zhi = np.abs(xe - xnu), xe + xnu; tot = np.zeros_like(xe)
    for zg, zw in zip(z_g, z_w):
        z = zlo + (zhi - zlo) * (zg + 1) / 2; wz = zw * (zhi - zlo) / 2; q = z * T
        plo, phi = _prange(q, omega, mn, mp, mun, mup); valid = phi > plo
        xlo = (plo ** 2 / (2 * mn) - mun) / T; xhi = np.where(np.isfinite(phi), (np.maximum(phi, 0.0) ** 2 / (2 * mn) - mun) / T, 1e6)
        tot += wz * np.where(valid, _lam(xhi, c) - _lam(xlo, c), 0.0)
    return float(np.sum(we * xe * g * block * tot)), mn, mp
def spectra(T, mstn, mstp, pFn, pFp, mue, dmu, xnus):
    """returns dict of arrays over xnus: W_abs_nu [MeV], W_abs_nubar [MeV], dGec_dx, dGnd_dx [MeV^4 per unit x] ."""
    Wn, Wb, Sec, Snd = [], [], [], []
    for x in xnus:
        k_abs, mn, mp = _kernel_fixed_xnu(T, mstn, mstp, pFn, pFp, mue, dmu, -1, lambda xe, ae, c: 1 - fF(xe - ae), x)
        k_ec, _, _ = _kernel_fixed_xnu(T, mstn, mstp, pFn, pFp, mue, dmu, -1, lambda xe, ae, c: fF(xe - ae) * np.exp(np.clip(c, -700, 700)), x)
        k_nd, _, _ = _kernel_fixed_xnu(T, mstn, mstp, pFn, pFp, mue, dmu, +1, lambda xe, ae, c: 1 - fF(xe - ae), x)
        k_abb, _, _ = _kernel_fixed_xnu(T, mstn, mstp, pFn, pFp, mue, dmu, +1, lambda xe, ae, c: fF(xe - ae) * np.exp(np.clip(c, -700, 700)), x)
        pref = GT2 * mn * mp * T ** 6 / (8 * np.pi ** 5)
        Wn.append(GT2 * mn * mp * T ** 3 / (4 * np.pi ** 3 * x) * k_abs); Wb.append(GT2 * mn * mp * T ** 3 / (4 * np.pi ** 3 * x) * k_abb)
        Sec.append(pref * x * k_ec); Snd.append(pref * x * k_nd)
    return dict(W_abs_nu=np.array(Wn), W_abs_nubar=np.array(Wb), dGec=np.array(Sec), dGnd=np.array(Snd))
HBARC = 197.3269804
def mfp_of_W_km(W): return HBARC / np.maximum(W, 1e-300) * 1e-18
