"""R2-F6 / round-2 item 7: the AHS2019 trapped-regime formal limit at a matched thermodynamic state.  DD2, T = 10 MeV, n = 1, 2, 3 n0, electron lepton fraction
Y_Le = Y_e + Y_nu - Y_nubar = 0.1 (AHS2019 'merger' value; their EOS is DD-ME2, so only the order of magnitude and the structure can be compared).
Trapped equilibrium: mu_nu = mu_p + mu_e - mu_n (solve Y_e with finiteT.stateT at fixed n, T);  degenerate reservoir chi = T^2 [F_1(eta) + F_1(-eta)]/pi^2.
Fast-relaxation LC viscosity  zeta ~ C^2/(B_t^2 lambda),  B_t = B_np + 1/chi  (= AHS2019 Eq. 53 with A -> B_t, lambda A >> omega),  lambda = FS mUrca (n+p) at T
[dU below threshold: FS theta = 0; thermally unblocked capture at dmu = -mu_nu quoted as an upper estimate, no nu blocking].
Contrast with the dilute merger-core state (cold Y_e, mu_nu = -dmu_np(T)) at the same n, T, rates: ratio of B_t^2."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from scipy.optimize import brentq
from scipy.integrate import quad
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from finiteT import stateT, suscT
from urca_rates import lam_mU, HC, HBAR
from urca_rel import gamma_ec_2m
CGS = 1.6021766e33; CONV = 1.0 / (HBAR * HC ** 3)
def Fk(k, eta): return quad(lambda y: y ** k / (np.exp(np.clip(y - eta, -700, 700)) + 1), 0, 400)[0]
def n_nu(mu, T): return T ** 3 * (Fk(2, mu / T) - Fk(2, -mu / T)) / (2 * np.pi ** 2) / HC ** 3        # net, one helicity each
def chi2(mu, T): return T ** 2 * (Fk(1, mu / T) + Fk(1, -mu / T)) / np.pi ** 2 / HC ** 3
out = []
def say(s): print(s, flush=True); out.append(s)
B0 = baryons()[:2]; T = 10.0; YL = 0.1; om = 2 * np.pi * 1e3
if __name__ != '__main__': sys.exit(0) if False else None
if __name__ == "__main__": say(f"AHS2019 matched-state check: DD2, T = {T:.0f} MeV, Y_Le = {YL}, 1 kHz. zeta_LC ~ C^2/(B_t^2 lambda) in cgs; compare AHS2019 Fig. 11 (DD-ME2, Y_L = 0.1): zeta ~ 1e25-1e27 at 1-3 n0, T = 10 MeV")
if __name__ == '__main__':
    for x in (1.0, 2.0, 3.0):
        n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B0); Ye0, Ymu0 = ne / n, nmu / n
        def resid(Ye):
            st = stateT(n, Ye, Ymu0, T, B0); munu = -st['dmu_e']; return Ye + n_nu(munu, T) / n - YL
        grid = np.linspace(0.02, 0.2, 19); vals = [resid(g) for g in grid]; i = next(j for j in range(len(grid) - 1) if vals[j] * vals[j + 1] < 0); Ye_tr = brentq(resid, grid[i], grid[i + 1], xtol=1e-6); st = stateT(n, Ye_tr, Ymu0, T, B0); munu = -st['dmu_e']
        A, Cv, st0 = suscT(n, T, B0, Ye_tr, Ymu0); Bnp, C = -A[0, 0], Cv[0]
        comp = st['comp']; pn, pp, pe = kf(comp['n'], 2), kf(comp['p'], 2), kf(Ye_tr * n, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2)
        lam = lam_mU(mn, mp, pn, pp, pe, T); chi_deg = chi2(munu, T); Bt = Bnp + 1 / chi_deg
        zeta = C ** 2 / (Bt ** 2 * lam) * CGS; tau = st0['p'] * 0 + 0  # tau not needed
        G_cap_dU = gamma_ec_2m(T, mst['n'], mst['p'], pn, pp, pe, -munu) * CONV
        # dilute state at the same n, T
        st_d = stateT(n, Ye0, Ymu0, T, B0); dmu_d = float(st_d['dmu_e']); chi_dil = chi2(-dmu_d, T); A_d, C_d, _ = suscT(n, T, B0, Ye0, Ymu0); Bt_d = -A_d[0, 0] + 1 / chi_dil
        say(f"{x:.0f}n0: trapped Y_e {Ye_tr:.4f}, Y_nu {YL-Ye_tr:.4f}, mu_nu {munu:6.1f} MeV (eta {munu/T:4.1f}); chi_deg {chi_deg:.3e}, B_np {Bnp:8.1f}, 1/chi {1/chi_deg:.3e} -> B_t {Bt:.3e} MeV fm^3; C {C:5.1f} MeV; lambda_mU {lam:.3e} fm^-3 s^-1 MeV^-1 (dU capture at dmu=-mu_nu, no blocking: {G_cap_dU:.2e} fm^-3 s^-1 -> lambda_dU <~ {G_cap_dU/T:.2e})"
            f" || zeta_LC(mU) = {zeta:.2e} cgs; B_t lambda = {Bt*lam:.2e} s^-1 (>> omega {om:.1e}) || dilute core at same n,T: Y_e {Ye0:.4f}, dmu_np {dmu_d:+.2f}, chi {chi_dil:.3e}, B_t {Bt_d:.3e} -> (B_t,dil/B_t,deg)^2 = {(Bt_d/Bt)**2:.1f}, chi_deg/chi_dil = {chi_deg/chi_dil:.1f}")
    open('results/ahs2019_matched_state.txt', 'w').write("\n".join(out) + "\n")
