"""Why does my transparent baseline minimum sit at 4-5 MeV while AH2019 Fig. 6a (DD2) has it at 3 MeV?  AHHZ2021 (Sec. I) state that the AH2018/2019
rate calculation 'used a very crude model of the in-medium nucleons, assigning them their vacuum mass and assuming that their kinematics remained
nonrelativistic'.  Test: redo the two-way transparent relaxation rate at DD2 2n0 and n0 with (a) DD2 Dirac/Landau masses, (b) vacuum masses 939 MeV,
same Fermi momenta, and compare tau_AH at 1 kHz using the same C, B, dp/dn as viscosity_lc (cold susceptibilities for speed, finite-T from the v2 run)."""
import numpy as np, sys, json; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from scipy.optimize import brentq
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from urca_rel import gamma_nd_2m, gamma_ec_2m
from murca_spectral import rates as rates_mU
from urca_rates import HC, HBAR
CONV = 1.0 / (HBAR * HC ** 3); CGS = 1.6021766e33
V1 = {(o['eos'], o['x'], o['T']): o for o in json.load(open('results/viscosity_lc_v1.json')) if 'error' not in o}
out = []
def say(s): print(s, flush=True); out.append(s)
B = baryons()[:2]
for x in (1.0, 2.0):
    n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B); pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2)
    for T in (2.0, 3.0, 4.0, 5.0):
        o = V1[('DD2', x, T)]; C, Bnp, ndpdn = o['C'], o['Bnp'], o['ndpdn']; om = 2 * np.pi * 1e3
        for lab, (msn, msp) in (('DD2 m*', (mst['n'], mst['p'])), ('vacuum 939', (939.0, 939.0))):
            mn = np.sqrt(pn ** 2 + msn ** 2); mp = np.sqrt(pp ** 2 + msp ** 2); a = (T, msn, msp, pn, pp, pe)
            Gd = lambda d: gamma_nd_2m(*a, d) * CONV + rates_mU(mn, mp, pn, pp, pe, T, d)[0]; Gc = lambda d: gamma_ec_2m(*a, d) * CONV + rates_mU(mn, mp, pn, pp, pe, T, d)[1]
            ds = brentq(lambda d: np.log(Gd(d)) - np.log(Gc(d)), -10, 90, xtol=2e-3); h = 0.5
            lam = ((Gd(ds + h) - Gc(ds + h)) - (Gd(ds - h) - Gc(ds - h))) / (2 * h); g = Bnp * lam; z = C ** 2 / Bnp * g / (om ** 2 + g ** 2); tau = ndpdn / (z * om ** 2)
            say(f"DD2 {x:.1f} n0 T={T:.0f} MeV [{lab:10s}]: dmu* {ds:5.1f}  Gamma_cap_dU(0) {gamma_ec_2m(*a, 0.0)*CONV:9.2e}  lambda_AH {lam:9.2e}  gamma {g:9.2e} s^-1  tau_AH(1 kHz) {tau*1e3:8.1f} ms")
open('results/baseline_vacuum_mass_check.txt', 'w').write("\n".join(out) + "\n")
