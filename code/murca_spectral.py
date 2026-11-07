"""Modified-Urca neutrino emission spectra and the inverse (absorption) mean free paths in the Fermi-surface approximation.
Five degenerate fermions (2 in, 3 out) + neutrino of energy E:  J5(E) = int prod d eps  f f (1-f)(1-f)(1-f) delta(sum - E) = P(E)/(24 (1 + e^{E/T})),
P(E) = E^4 + 10 pi^2 T^2 E^2 + 9 pi^4 T^4  (checked below by Monte Carlo).
Decay  n N -> p N e nubar : dGamma_dec/dE = A E^2 J5(E - dmu);   capture e p N -> n N nu : dGamma_cap/dE = A E^2 J5(E + dmu),  dmu = mu_n - mu_p - mu_e.
Normalisation A from the AH2019 FS rate coefficient lam_mU = d(Gamma_dec - Gamma_cap)/d dmu |_0 = 4 A T^6 int x J5(x) dx  (urca_rates.lam_mU, n+p branches).
Inverse absorption per mode (detailed balance, mu_nu = 0 bath not needed):  1/lambda_nu(E) = 2 pi^2 A P(E+dmu) / (24 (1 + e^{-(E+dmu)/T}))  [nu + n N -> p e N],
                                                                            1/lambda_nubar(E) = 2 pi^2 A P(E-dmu) / (24 (1 + e^{-(E-dmu)/T}))  [nubar + p e N -> n N].
Cross-check of the absolute scale: Yakovlev+2001 Eq. (140) Q_Mn = 8.1e21 (m*n/mn)^3 (m*p/mp) (np/n0)^{1/3} T9^8 alpha_n beta_n erg/cm^3/s (Friman-Maxwell)."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from urca_rates import lam_mU, NAT_TO_PHYS, HC, HBAR
from numpy.polynomial.legendre import leggauss
def P5(E, T): return E ** 4 + 10 * np.pi ** 2 * T ** 2 * E ** 2 + 9 * np.pi ** 4 * T ** 4
def J5(E, T): return P5(E, T) / (24 * (1 + np.exp(np.clip(E / T, -700, 700))))
_xg, _xw = leggauss(96); XN = 60.0 * (_xg + 1) / 2; WN = _xw * 30.0
def A_coeff(mn, mp, pn, pp, pe, T):
    """A in physical units [1/(MeV^7 s fm^3)]: lam_mU [1/(MeV s fm^3)] = 4 A T^6 int x J5(x,1) dx."""
    I1 = np.sum(WN * XN * J5(XN, 1.0)); return lam_mU(mn, mp, pn, pp, pe, T) / (4 * T ** 6 * I1)
def rates(mn, mp, pn, pp, pe, T, dmu=0.0):
    """Gamma_dec, Gamma_cap [1/(s fm^3)] at dmu (FS approximation, both branches)."""
    A = A_coeff(mn, mp, pn, pp, pe, T); E = XN * T; w = WN * T
    return A * np.sum(w * E ** 2 * J5(E - dmu, T)), A * np.sum(w * E ** 2 * J5(E + dmu, T))
def spectra_mU(mn, mp, pn, pp, pe, T, E, dmu=0.0):
    """dGamma_dec/dE, dGamma_cap/dE [1/(s fm^3 MeV)]; 1/lambda_nu, 1/lambda_nubar [1/m] at energies E."""
    A = A_coeff(mn, mp, pn, pp, pe, T); Ap = A / NAT_TO_PHYS            # natural: MeV^-7 * MeV^3 ... -> rate per mode in MeV when multiplied by MeV^4
    dec = A * E ** 2 * J5(E - dmu, T); cap = A * E ** 2 * J5(E + dmu, T)
    inv_nu = 2 * np.pi ** 2 * Ap * P5(E + dmu, T) / (24 * (1 + np.exp(-np.clip((E + dmu) / T, -700, 700))))     # MeV
    inv_nb = 2 * np.pi ** 2 * Ap * P5(E - dmu, T) / (24 * (1 + np.exp(-np.clip((E - dmu) / T, -700, 700))))
    return dec, cap, inv_nu / HC * 1e15, inv_nb / HC * 1e15
if __name__ == '__main__':
    out = []
    def say(s): print(s, flush=True); out.append(s)
    # (1) Monte-Carlo check of J5 at T = 1 for E = 0, 2, 5, 10: sample eps1..eps4 from the Fermi factors' shapes is awkward; use direct importance-free MC on a box
    rng = np.random.default_rng(1); N = 4_000_000; L = 40.0
    for E in (0.0, 2.0, 5.0, 10.0):
        e = rng.uniform(-L, L, size=(N, 4)); e5 = e[:, 0] + e[:, 1] - e[:, 2] - e[:, 3] - E       # delta solved for eps5
        f = lambda x: 1 / (1 + np.exp(np.clip(x, -700, 700)))
        w = f(e[:, 0]) * f(e[:, 1]) * (1 - f(e[:, 2])) * (1 - f(e[:, 3])) * (1 - f(e5))
        val = w.mean() * (2 * L) ** 4; err = w.std() / np.sqrt(N) * (2 * L) ** 4
        say(f"J5 MC check T=1 E={E:4.1f}: MC {val:.5f} +- {err:.5f}   formula {J5(E, 1.0):.5f}   ratio {val / J5(E, 1.0):.4f}")
    # (2) absolute scale vs Yakovlev+2001 Eq. 140 (n-branch, Friman-Maxwell) at n0 for a generic point: use FM's own alpha_n beta_n = 1.13*0.68 and compare to
    #     the AH pion-exchange factor pn^4/(pn^2+mpi^2)^2 with beta = 1 -> expect agreement up to the (alpha beta) convention; report both.
    from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0, MN, MP
    from urca_rates import MPI
    B = baryons()[:2]; n = N0; c, ne, nmu, e_, p_, mu, mst, ier, sol = beta_eq_cont(n, B)
    pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2)
    T = 3.0; A = A_coeff(mn, mp, pn, pp, pe, T); E = XN * T; w = WN * T
    Q_mine_n_plus_p = np.sum(w * E ** 3 * (A * E ** 0 * J5(E, T))) * 2       # emissivity, both nu and nubar (decay+capture), [MeV/(s fm^3)]
    Q_cgs = Q_mine_n_plus_p * 1.602176634e-6 / 1e-39                          # erg cm^-3 s^-1
    T9 = T / 8.617333e-2; alpha_FM = pn ** 4 / (pn ** 2 + MPI ** 2) ** 2
    Q_Y_n = 8.1e21 * (mst['n'] / MN) ** 3 * (mst['p'] / MP) * (c['p'] / 0.16) ** (1 / 3) * T9 ** 8 * 1.13 * 0.68
    Q_Y_n_AHfactor = 8.1e21 * (mn / MN) ** 3 * (mp / MP) * (c['p'] / 0.16) ** (1 / 3) * T9 ** 8 * alpha_FM * 1.0
    # split my n and p branches
    from urca_rates import G2, GA, FPI, theta_n, theta_p
    ln = 367.0 / (1152 * np.pi ** 3) * G2 * GA ** 2 * FPI ** 4 * mn ** 3 * mp / MPI ** 4 * pn ** 4 * pp / (pn ** 2 + MPI ** 2) ** 2 * theta_n(pn, pp, pe) * T ** 6 * NAT_TO_PHYS
    frac_n = ln / lam_mU(mn, mp, pn, pp, pe, T)
    say(f"mUrca scale at DD2 n0, T=3 MeV: my emissivity (n+p branches, nu+nubar) {Q_cgs:.3e} erg/cm3/s, n-branch share {frac_n:.3f} -> n-branch {Q_cgs*frac_n:.3e};"
        f" Yakovlev Eq.140 with FM alpha*beta=1.13*0.68 and Dirac masses {Q_Y_n:.3e}; with Landau masses and the bare pion factor {alpha_FM:.3f} {Q_Y_n_AHfactor:.3e}")
    Gd, Gc = rates(mn, mp, pn, pp, pe, T); say(f"  rates at dmu=0: Gamma_dec = Gamma_cap = {Gd:.4e} / {Gc:.4e} fm^-3 s^-1; lam_mU {lam_mU(mn, mp, pn, pp, pe, T):.4e}; check 4 A T^6 int xJ5 = lam: {4*A*T**6*np.sum(WN*XN*J5(XN,1.0))/lam_mU(mn, mp, pn, pp, pe, T):.6f}")
    for Tt in (1.0, 2.0, 3.0, 5.0):
        Ev = np.array([1.0, 3.0, 10.0, 20.0, 30.0]); dec, cap, inv_nu, inv_nb = spectra_mU(mn, mp, pn, pp, pe, Tt, Ev)
        Gd, Gc = rates(mn, mp, pn, pp, pe, Tt); Emean = np.sum(WN * Tt * (XN * Tt) ** 3 * J5(XN * Tt, Tt)) / np.sum(WN * Tt * (XN * Tt) ** 2 * J5(XN * Tt, Tt))
        say(f"  T={Tt:3.0f}: Gamma_mU(dec) {Gd:.3e} fm^-3 s^-1, <E_nu> {Emean:5.1f} MeV; inverse-mUrca lambda_nu(E=1,3,10,20,30 MeV) [km] = " + " ".join(f"{1e-3/x:.3g}" for x in inv_nu))
    open('results/murca_spectral_checks.txt', 'w').write("\n".join(out) + "\n")
