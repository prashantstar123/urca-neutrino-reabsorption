"""Referee F7: EXPLORATORY ESTIMATES (not bounds, round-3 R3-F6) for the suprathermal (large-amplitude) regime. At |dmu| = 5, 10, 20 MeV >> T the capture nu_e are still reabsorbed (E >= E_min), the one-way
decays grow polynomially and emit antineutrinos of E ~ dmu whose fate decides the dissipation.  Here: one-way rates at dmu (dUrca kernel + mUrca FS),
the antineutrino spectra and their absorption lengths at that dmu (3-body inverse of dUrca decay + inverse mUrca, both at dmu), NC transport length at <E>,
two energy-budget bounds (lambda_mode = 10 km): (a) fresh-decay escape within the absorption displacement L_diff = sqrt(lambda_abs lambda_tr/3) [v2 of this script; v1 wrongly used the full-period random walk L_d = sqrt(D/omega) and ignored absorption during the walk], (b) diffusive loss of the quasi-local degenerate reservoir n_nubar(mu = dmu): P <= dmu x D k^2 n_nubar;   E_osc = 1/2 n (dp/dn) (dmu/C)^2   ->   tau_supra >= E_osc / P_diss,
compared with the fully transparent value (f_esc = 1) and the oscillation period.  DD2, n0 and 2n0, T = 2, 3, 5 MeV."""
import numpy as np, sys, json; from scipy.integrate import quad; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from numpy.polynomial.legendre import leggauss
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from urca_rel import spectra, gamma_nd_2m, gamma_ec_2m
from murca_spectral import rates as rates_mU, spectra_mU, J5
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import HC, HBAR, GA
CONV = 1.0 / (HBAR * HC ** 3); CLIGHT = 2.99792458e8      # m/s
VL = {(o['eos'], o['x'], o['T']): o for o in json.load(open('results/viscosity_lc.json')) if 'error' not in o}
XG, XW = leggauss(64); XN = 40.0 * (XG + 1) / 2; WN = XW * 20.0
out = []
def say(s): print(s, flush=True); out.append(s)
say("DD2 suprathermal bound, lambda_mode = 10 km, f = 1 kHz. Rates in fm^-3 s^-1; <E> of the antineutrinos (production-weighted, dUrca decay + mUrca decay); lambda_abs at dmu; L_d per period; tau_supra (bound) vs fully transparent tau_tr(f_esc=1) and vs tau_AH (linear, transparent)")
B = baryons()[:2]
for x in (1.0, 2.0):
    n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B); pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2)
    mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2); a_dU = (mst['n'], mst['p'], pn, pp, pe)
    for T in (2.0, 3.0, 5.0):
        v = VL[('DD2', x, T)]; C, ndpdn = v['C'], v['ndpdn']; om = 2 * np.pi * 1e3; E = XN * T; w = WN * T
        for dmu in (5.0, 10.0, 20.0):
            Gd = gamma_nd_2m(T, *a_dU, dmu) * CONV; Gc = gamma_ec_2m(T, *a_dU, dmu) * CONV; GdmU, GcmU = rates_mU(mn, mp, pn, pp, pe, T, dmu)
            S = spectra(T, *a_dU, dmu, XN); spec_dU = S['dGnd'] / T; spec_dU *= Gd / max(np.sum(w * spec_dU), 1e-300)
            dec_mU, cap_mU, inv_mU_nu, inv_mU_nb = spectra_mU(mn, mp, pn, pp, pe, T, E, dmu); spec_mU = dec_mU * (GdmU / max(np.sum(w * dec_mU), 1e-300))
            spec = spec_dU + spec_mU; inv_abs = S['W_abs_nubar'] / HC * 1e15 + inv_mU_nb                       # 1/m, both inverse channels at this dmu
            norm = np.sum(w * spec); Em = np.sum(w * spec * E) / norm; lam_abs = norm / np.sum(w * spec * inv_abs)
            lam_nc = 1.0 / (inv_lambda_nc(Em, T, mn, -0.5, -GA / 2) + inv_lambda_nc(Em, T, mp, 0.5 - 2 * SW2, GA / 2)); ltr = 1.0 / (1 / lam_abs + 1 / lam_nc)
            D = CLIGHT * ltr / 3; k = 2 * np.pi / 1e4; Ldiff = np.sqrt(lam_abs * ltr / 3); t_abs = lam_abs / CLIGHT
            # (a) fresh-decay escape: a decay antineutrino is absorbed after a net displacement Ldiff (<< period); only those born within Ldiff of the
            #     region boundary leave: f_esc = min(1, 2 pi Ldiff / lambda_mode)  -> P_a = Gamma_dec dmu f_esc
            # (b) reservoir diffusion: the quasi-local nubar reservoir at mu_nubar = dmu (degenerate for dmu >> T) loses lepton number by diffusion
            #     at rate D k^2 n_nubar(dmu, T);  P_b = dmu x min(D k^2 n_nubar, Gamma_dec)
            f_esc = min(1.0, 2 * np.pi * Ldiff / 1e4); Gdec = Gd + GdmU
            F2 = quad(lambda y: y ** 2 / (np.exp(np.clip(y - dmu / T, -700, 700)) + 1), 0, 400)[0]; n_nb = T ** 3 * F2 / (2 * np.pi ** 2) / HC ** 3   # fm^-3, ONE antineutrino helicity (round-2 referee R2-F4: factor-2 fix)
            Dfm = D * 1e30; kfm = k * 1e-15; loss = min(Dfm * kfm ** 2 * n_nb, Gdec)
            Eosc = 0.5 * ndpdn * (dmu / C) ** 2; Pa = Gdec * dmu * f_esc; Pb = dmu * loss; tau_a = Eosc / Pa; tau_b = Eosc / Pb; tau_tr = Eosc / (Gdec * dmu)
            say(f"{x:.0f}n0 T={T:.0f} dmu={dmu:4.0f} MeV: eps=dmu/C {dmu/C:5.3f} | Gamma_dec dU {Gd:8.2e} mU {GdmU:8.2e} | nubar <E> {Em:5.1f} MeV lam_abs {lam_abs:8.3g} m lam_tr {ltr:6.3g} m L_diff {Ldiff:6.3g} m t_abs {t_abs*1e6:6.2f} us | n_nubar(dmu) {n_nb:8.2e} fm^-3, D k^2 n {Dfm*kfm**2*n_nb:8.2e} fm^-3 s^-1 | tau >= {tau_a:8.3g} s (fresh-decay escape, f_esc {f_esc:.3f}) ; {tau_b:8.3g} s (reservoir diffusion) ; fully transparent {tau_tr:8.3g} s ; linear tau_AH {v['R5_f1']['tau_AH_s']:7.3g} s")
open('results/suprathermal_bound.txt', 'w').write("\n".join(out) + "\n")
