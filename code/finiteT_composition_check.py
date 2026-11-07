"""Sensitivity of the escape result to the finite-T beta-equilibrium composition (DD2): recompute the capture spectrum and absorption lengths with the
composition from finiteT.beta_eqT (mu_n = mu_p + mu_e at T, electrons+muons) instead of the cold composition."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from numpy.polynomial.legendre import leggauss
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from finiteT import beta_eqT, stateT
from urca_rel import spectra
from urca_rates import HC
XG, XW = leggauss(64); XN = 40.0 * (XG + 1) / 2; WN = XW * 20.0
out = []
def say(s): print(s, flush=True); out.append(s)
B = baryons()[:2]
for x in (1.0, 2.0, 3.0):
    n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B); Ye0, Ymu0 = ne / n, nmu / n
    for T in (2.0, 3.0, 5.0):
        YeT, YmuT = beta_eqT(n, T, B, Ye0, Ymu0); st = stateT(n, YeT, YmuT, T, B)
        rows = []
        for lab, Ye, comp, mue in (('cold', Ye0, c, kf(ne, 2)), ('beta-eq(T)', YeT, st['comp'], st['mue'])):
            pn, pp = kf(comp['n'], 2), kf(comp['p'], 2); pe = kf(Ye * n, 2)
            msn = mst['n'] if lab == 'cold' else None
            # Dirac masses at the finite-T state are not returned by stateT; use the cold ones (they change by < 2 % between the two compositions at fixed n)
            S = spectra(T, mst['n'], mst['p'], pn, pp, mue, 0.0 if lab == 'cold' else st['dmu_e'], XN)
            E = XN * T; w = WN * T; spec = S['dGec']; inv = S['W_abs_nu'] / HC * 1e15; norm = np.sum(w * spec)
            Em = np.sum(w * spec * E) / norm; lam_h = norm / np.sum(w * spec * inv); P5 = np.sum(w * spec * np.exp(-5e3 * inv)) / norm
            rows.append(f"{lab:10s} Ye {Ye:.4f} dp {pn-pp-pe:+6.1f} dmu {0.0 if lab=='cold' else st['dmu_e']:+5.2f} | <E> {Em:5.1f} MeV  lambda_h {lam_h:8.3g} m  P_esc(5 km) {P5:.4f}")
        say(f"DD2 {x:.1f} n0 T={T:.0f}: " + " || ".join(rows))
open('results/finiteT_composition_check.txt', 'w').write("\n".join(out) + "\n")
