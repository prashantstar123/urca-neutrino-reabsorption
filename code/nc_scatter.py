"""Neutral-current neutrino-nucleon scattering.
PRODUCTION: inv_lambda_nc_full (one nucleon species) and inv_lambda_nc_np (neutrons + protons) give the total scattering rate from the full
nonrelativistic free-gas response at temperature T (half charges hV, hA; constant angle-averaged vertex; Landau masses; unblocked total rate for
the isotropic elastic collision operator). These are the functions used by rate_state.py and every result of the Letter and its Supplement.
LEGACY: inv_lambda_nc is the degenerate-neutrino formula of Reddy, Prakash & Lattimer (1998) Eq. 27 in the pre-correction convention. It is kept
only for explicit historical comparisons (and as the degenerate-limit reference in the checks), and the direct-execution block below uses it."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from urca_rates import GF, GA, HC
SW2 = 0.231
def inv_lambda_nc(E, T, mst_landau, cV, cA, mu_nu=0.0):
    """1/lambda in 1/m for one species."""
    x = (E - mu_nu) / T
    W = GF ** 2 * (cV ** 2 + 3 * cA ** 2) * mst_landau ** 2 * T ** 2 * E * (np.pi ** 2 + x ** 2) / (16 * np.pi ** 3 * (1 + np.exp(-x)))   # MeV
    return W / HC * 1e15
if __name__ == '__main__':
    out = ["NC scattering mean free path (RPL98 Eq. 27, degenerate), DD2 beta-equilibrated npe(mu) matter, Landau masses. n: neutrons only (as in AH2018 Fig. 8) and n+p.",
           "n/n0  m_n^L  m_p^L |  E=T:  T=3 (n | n+p)   T=5 (n | n+p)   T=7 (n | n+p)   [km]   ||  T=3 MeV, E = 10, 20, 30 MeV (n+p) [m]"]
    B = baryons()[:2]
    for x in (1.0, 2.0, 3.0, 4.0, 6.0):
        n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B)
        mn = np.sqrt(kf(c['n'], 2) ** 2 + mst['n'] ** 2); mp = np.sqrt(kf(c['p'], 2) ** 2 + mst['p'] ** 2)
        def lam(E, T, both):
            il = inv_lambda_nc(E, T, mn, -0.5, -GA / 2) + (inv_lambda_nc(E, T, mp, 0.5 - 2 * SW2, GA / 2) if both else 0.0); return 1.0 / il
        s = f"{x:4.1f} {mn:6.0f} {mp:6.0f} | "
        for T in (3.0, 5.0, 7.0): s += f"{lam(T, T, False)/1e3:6.2f} | {lam(T, T, True)/1e3:5.2f}    "
        s += " || " + "  ".join(f"{lam(E, 3.0, True):7.1f}" for E in (10.0, 20.0, 30.0))
        out.append(s); print(s, flush=True)
    open('results/nc_scatter_dd2.txt', 'w').write("\n".join(out) + "\n")
# =================================================================================================================================
# 2026-11-08 FIX ROUND: full nonrelativistic neutral-current response (replaces RPL98 Eq. 27 in production).
# Audit findings (six-agent audit + Astra, 2026-11-08): (i) Eq. 27 of RPL98 is written with the UNHALVED couplings of their Table II and 16 pi^3;
# inv_lambda_nc above used half charges with 16 pi^3 (factor 4 too small); (ii) Eq. 27 is a degenerate-NEUTRINO reduction (q0 -> 0 in the q limits),
# not valid for mu_nu ~ 0 and E ~ T.  Production now uses the full free-gas response of each nucleon species (RPL98 Eqs. 15, 21-24 in the
# nonrelativistic limit) with its Landau mass and its density-consistent finite-T reference mu^T (urca_rel.occ_ref):
#   d Gamma/(dE' dOmega') = G_F^2/(8 pi^3) E'^2 V S(q0, q) [x (1 - f_nu(E')) if blocked],  q0 = E - E',  V = hV^2 + 3 hA^2 (constant vertex, production)
#   or V = hV^2 (1 + cos) + hA^2 (3 - cos) (vertex='angle', sensitivity),
#   S(q0, q) = 2 int d^3p/(2pi)^3 f(e_p) (1 - f(e_{p+q})) 2 pi delta(q0 + e_p - e_{p+q})
#            = (m^2 T/(pi q)) [L(a) - L(a + z)]/(1 - e^{-z}),  L(y) = ln(1 + e^{-y}),  a = e_min/T,  z = q0/T,  e_min = (m/(2q^2))(q0 - q^2/(2m))^2 - mu.
# Half charges: hV_n = -1/2, hA_n = -gA/2, hV_p = 1/2 - 2 sin^2 thW, hA_p = gA/2  (static dilute limit: sigma = G_F^2 E^2 (hV^2 + 3 hA^2)/pi).
# The isotropic elastic angular operator of the kinetic model takes the UNBLOCKED total collision frequency (final-state blocking cancels between
# gain and loss in elastic scattering); blocked=True is kept for checks.
from numpy.polynomial.legendre import leggauss as _lg
HV_N, HA_N = -0.5, -GA / 2
HV_P, HA_P = 0.5 - 2 * SW2, GA / 2
def S_free(q0, q, T, m, mu):
    """free nonrelativistic two-spin dynamic structure factor [MeV^2]; int dq0/(2 pi) S -> n (MeV^3) in the static nondegenerate limit"""
    z = q0 / T; a = (m / (2 * q * q)) * (q0 - q * q / (2 * m)) ** 2 / T - mu / T
    small = np.abs(z) < 1e-7
    num = np.logaddexp(0.0, -a) - np.logaddexp(0.0, -a - z); den = -np.expm1(-np.where(small, 1.0, z))
    ratio = np.where(small, 1.0 / (1.0 + np.exp(np.clip(a, -700, 700))), num / den)
    return m * m * T / (np.pi * q) * ratio
def inv_lambda_nc_full(E, T, m, mu, hV, hA, blocked=False, mu_nu=0.0, nq0=96, nc=64, qup=35.0, vertex='const'):
    """total NC scattering rate 1/lambda [1/m] of neutrinos with energies E [MeV] (array) on one nucleon species (Landau mass m, reference mu).
    vertex='const': angle-averaged NR vertex hV^2 + 3 hA^2 (the constant-vertex convention of the charged-current rates; reduces to the corrected
    RPL98 Eq. 27 in the degenerate-neutrino limit); vertex='angle': hV^2 (1 + cos) + hA^2 (3 - cos) (sensitivity variant)."""
    E = np.atleast_1d(np.asarray(E, dtype=float)); g1, w1 = _lg(nq0); gc, wc = _lg(nc)
    tot = np.zeros_like(E)
    for lo_f, hi_f in ((lambda e: -qup * T + 0.0 * e, lambda e: 0.0 * e), (lambda e: 0.0 * e, lambda e: e)):
        lo, hi = lo_f(E), hi_f(E)                                                    # q0 = E - E' in [-qup T, 0] and [0, E]
        q0 = lo[:, None] + (hi - lo)[:, None] * (g1[None, :] + 1) / 2; wq = w1[None, :] * (hi - lo)[:, None] / 2
        Ep = E[:, None] - q0
        c = gc[None, None, :]
        q2 = E[:, None, None] ** 2 + Ep[:, :, None] ** 2 - 2 * E[:, None, None] * Ep[:, :, None] * c
        q = np.sqrt(np.maximum(q2, 1e-14))
        S = S_free(q0[:, :, None], q, T, m, mu)
        vert = (hV ** 2 + 3 * hA ** 2) + (c * (hV ** 2 - hA ** 2) if vertex == 'angle' else 0.0 * c)
        blk = (1.0 / (1.0 + np.exp(-np.clip((Ep - mu_nu) / T, -700, 700))))[:, :, None] if blocked else 1.0   # 1 - f_nu(E')
        tot += np.sum(wq[:, :, None] * wc[None, None, :] * Ep[:, :, None] ** 2 * vert * S * blk, axis=(1, 2)) * 2 * np.pi
    return GF ** 2 / (8 * np.pi ** 3) * tot / HC * 1e15
def inv_lambda_nc_np(E, T, mLn, mLp, mun, mup, **kw):
    """n + p total (unblocked by default) [1/m]"""
    return inv_lambda_nc_full(E, T, mLn, mun, HV_N, HA_N, **kw) + inv_lambda_nc_full(E, T, mLp, mup, HV_P, HA_P, **kw)
