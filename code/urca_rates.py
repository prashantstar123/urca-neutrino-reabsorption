"""Urca rate coefficients lambda = d(Gamma_{n->p} - Gamma_{p->n})/d(delta mu) at delta mu -> 0 (degenerate, subthermal), after Alford & Harris PRC 100,
035803 (2019) Eqs. 27-30 with RMF effective masses (Landau masses E_F* used for the density of states), natural units (MeV) converted to
1/(MeV s fm^3).  Direct Urca: lambda_dU = (17/240 pi) G^2 (1+3 g_A^2) m_n* m_p* p_Fe T^4 Theta_dU.  Modified Urca (n, p spectators) Eqs. 29-30.
Delta channels (Delta^- <-> n e nu, Delta^0 <-> p e nu): same form as dUrca with (f1^2 + 3 g1^2) and the Delta/nucleon masses and threshold.
Bulk viscosity (Alford-Harris Eq. 24): zeta = (C^2/B) gamma/(omega^2 + gamma^2), gamma = B lambda; B = -d(dmu)/d n_p (> 0), C = n d(dmu)/d n|_YL.
Units: zeta in MeV s/fm^3; 1 MeV s fm^-3 = 1.6022e33 g cm^-1 s^-1."""
import numpy as np
HC = 197.3269804; HBAR = 6.582119569e-22          # MeV fm, MeV s
GF = 1.1663788e-11                                # MeV^-2
COSC, GA, FPI, MPI = 0.97370, 1.26, 1.0, 139.57
KB = 8.617333e-11                                 # MeV/K
G2 = (GF * COSC) ** 2
NAT_TO_PHYS = 1.0 / (HBAR * HC ** 3)              # MeV^3 (natural rate coefficient) -> 1/(MeV s fm^3)
def theta_dU(pn, pp, pe): return 1.0 if pn < pp + pe else 0.0
def theta_n(pn, pp, pe): return 1.0 if pn > pp + pe else 1.0 - 3.0 / 8.0 * (pp + pe - pn) ** 2 / (pp * pe)
def theta_p(pn, pp, pe):
    if pn > 3 * pp + pe: return 0.0
    if pn > 3 * pp - pe: return (3 * pp + pe - pn) ** 2 / (pn * pe)
    if pn > pp + pe: return 4 * (3 * pp - pn) / pn
    return 2 + 3 * (2 * pp - pn) / pe - 3 * (pp - pe) ** 2 / (pn * pe)
def lam_dU(mn, mp, pe, T, theta=1.0):
    """MeV^3 natural -> physical units; T in MeV; masses = Landau effective masses (MeV); pe = electron Fermi momentum (MeV)."""
    return 17.0 / (240 * np.pi) * G2 * (1 + 3 * GA ** 2) * mn * mp * pe * T ** 4 * theta * NAT_TO_PHYS
def lam_mU(mn, mp, pn, pp, pe, T):
    ln = 367.0 / (1152 * np.pi ** 3) * G2 * GA ** 2 * FPI ** 4 * mn ** 3 * mp / MPI ** 4 * pn ** 4 * pp / (pn ** 2 + MPI ** 2) ** 2 * theta_n(pn, pp, pe) * T ** 6
    lp = 367.0 / (8064 * np.pi ** 3) * G2 * GA ** 2 * FPI ** 4 * mn * mp ** 3 / MPI ** 4 * pn * (pn - pp) ** 4 / ((pn - pp) ** 2 + MPI ** 2) ** 2 * theta_p(pn, pp, pe) * T ** 6
    return (ln + lp) * NAT_TO_PHYS
def lam_delta(mD, mB, pD, pB, pe, T, f1g1=None):
    """Delta <-> baryon + e + nu channel in dUrca form; f1g1 = f1^2 + 3 g1^2 (set when the couplings are fixed); triangle threshold."""
    if f1g1 is None: return 0.0
    th = 1.0 if (abs(pD - pB) < pe < pD + pB) else 0.0
    return 17.0 / (240 * np.pi) * G2 * f1g1 * mD * mB * pe * T ** 4 * th * NAT_TO_PHYS
def zeta(C, Bpos, lam, omega):
    """Alford-Harris Eq. 24; C [MeV], Bpos = -d(dmu)/dn_p [MeV fm^3] > 0, lam [1/(MeV s fm^3)], omega [1/s]  ->  zeta [MeV s / fm^3]."""
    gam = Bpos * lam; return (C ** 2 / Bpos) * gam / (omega ** 2 + gam ** 2)
def zeta_cgs(z): return z * 1.6021766e33
if __name__ == '__main__':
    # Sanity check against the Sawyer/HLY modified-Urca benchmark used in the r-mode literature: zeta_mU ~ 6e-59 rho^2 omega^-2 T^6 (cgs) for npe matter
    import sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from rmf_dd2_delta import baryons, beta_eq, beta_eq_cont, kf, N0, MN, MP
    from partial_eq import susceptibilities
    B = baryons(); sol = None
    for x in (1.0, 2.0, 3.0):
        n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B); Bq, Cq, YL, st0 = susceptibilities(n, B, (ne + nmu) / n, sol)
        pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); En = np.sqrt(pn ** 2 + mst['n'] ** 2); Ep = np.sqrt(pp ** 2 + mst['p'] ** 2)   # Landau masses
        T = 1e9 * KB; omega = 2 * np.pi * 500 * 2 / 3 * 2   # ~ r-mode: omega = 4/3 Omega for nu_spin = 500 Hz -> Omega = 2 pi 500
        lm = lam_mU(En, Ep, pn, pp, pe, T); ld = lam_dU(En, Ep, pe, T, theta_dU(pn, pp, pe)); rho = e * 1.7827e12   # MeV/fm^3 -> g/cm^3
        zm = zeta_cgs(zeta(Cq, -Bq, lm, omega)); sawyer = 6e-59 * rho ** 2 * omega ** -2 * 1e9 ** 6
        print(f"n={x} n0: pFn {pn:.0f} pFp {pp:.0f} pFe {pe:.0f} MeV, dUrca open: {theta_dU(pn,pp,pe)}; T=1e9 K, omega={omega:.0f}/s: lambda_mU {lm:.3e}, lambda_dU {ld:.3e} [1/(MeV s fm^3)]; gamma_mU = B lambda = {-Bq*lm:.3e}/s; zeta_mU {zm:.3e} g/cm/s vs Sawyer 6e-59 rho^2 T^6/omega^2 = {sawyer:.3e}; ratio {zm/sawyer:.2f}; (C {Cq:.1f} MeV, B {-Bq:.0f} MeV fm^3)")
