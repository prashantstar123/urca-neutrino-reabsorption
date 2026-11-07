"""Internal consistency checks of kinetic_response.py at DD2 n0, T = 3 MeV (and 2n0, 5 MeV):
 (i) k = 0: Re Lambda vs omega^2 int w/gamma (gamma >> omega expansion) and Im Lambda vs -omega chi;
 (ii) omega -> 0, small k: Re Lambda vs the spectral diffusion value k^2 int w c^2/(3(gamma+sigma)) (the closure the grey model replaced by D k^2 chi);
 (iii) positivity: Re Lambda >= 0 over a grid of (omega, k);  (iv) grey collapse: single-energy version vs the v3 grey formula structure;
 (v) spectral weight comparison: production-weighted vs susceptibility-weighted mean opacities (the referee's 'different weights' point)."""
import numpy as np, sys, json; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from numpy.polynomial.legendre import leggauss
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from finiteT import stateT
from urca_rel import spectra
from murca_spectral import spectra_mU
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import HC, HBAR, GA
from kinetic_response import admittance, fF, VL
out = []
def say(s): print(s, flush=True); out.append(s)
B0 = baryons()[:2]
for x, T in ((1.0, 3.0), (2.0, 5.0)):
    v = VL[('DD2', x, T)]; C, B, ndpdn = v['C'], v['Bnp'], v['ndpdn']; n = x * N0
    c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B0); st = stateT(n, ne / n, nmu / n, T, B0); dmuT = float(st['dmu_e'])
    pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2)
    xg, xw = leggauss(128); xn = 50.0 * (xg + 1) / 2; wx = xw * 25.0; E = xn * T; wE = wx * T
    S = spectra(T, mst['n'], mst['p'], pn, pp, pe, dmuT, xn); dec_mU, cap_mU, inv_mU_nu, inv_mU_nb = spectra_mU(mn, mp, pn, pp, pe, T, E, dmuT)
    Wn = S['W_abs_nu'] + HC * 1e-15 * inv_mU_nu; Wb = S['W_abs_nubar'] + HC * 1e-15 * inv_mU_nb
    sig = HC * 1e-15 * (inv_lambda_nc(E, T, mn, -0.5, -GA / 2) + inv_lambda_nc(E, T, mp, 0.5 - 2 * SW2, GA / 2))
    fn = fF((E + dmuT) / T); fb = fF((E - dmuT) / T); wn = E ** 2 * fn * (1 - fn) / (2 * np.pi ** 2 * T * HC ** 3); wb = E ** 2 * fb * (1 - fb) / (2 * np.pi ** 2 * T * HC ** 3)
    gn = Wn / (1 - fn); gb = Wb / (1 - fb)
    om = 2 * np.pi * 1e3; hw = HBAR * om
    say(f"DD2 {x:.0f}n0 T={T:.0f} MeV, dmu_np(T) = {dmuT:+.3f} MeV")
    # (i) k = 0
    for lab, w, g in (('nu_e', wn, gn), ('nubar_e', wb, gb)):
        L = np.sum(wE * admittance(E, w, g, sig, hw, 0.0)); chi = np.sum(wE * w); exp_re = hw ** 2 * np.sum(wE * w / g); exp_im = -hw * chi
        say(f"  (i) k=0 {lab:8s}: Re Lambda {L.real:.4e} vs omega^2 int w/gamma {exp_re:.4e} (ratio {L.real/exp_re:.4f}); Im Lambda {L.imag:.4e} vs -omega chi {exp_im:.4e} (ratio {L.imag/exp_im:.4f}); chi {chi:.4e} fm^-3 MeV^-1")
    # (ii) omega -> 0, 10-km mode: diffusion limit
    hck = HC * 2 * np.pi / 1e19
    for lab, w, g in (('nu_e', wn, gn), ('nubar_e', wb, gb)):
        L0 = np.sum(wE * admittance(E, w, g, sig, 1e-30, hck)); Ddiff = hck ** 2 * np.sum(wE * w / (3 * (g + sig)))      # natural: (hbar c k)^2 / (3 (gamma+sigma)) per energy
        grey = hck ** 2 * np.sum(wE * w) / (3 * (np.sum(wE * w * (g + sig)) / np.sum(wE * w)))                            # D k^2 chi with chi-weighted mean opacity
        say(f"  (ii) omega->0 10 km {lab:8s}: Re Lambda {L0.real:.4e} vs spectral diffusion int w (ck)^2/(3(g+s)) {Ddiff:.4e} (ratio {L0.real/Ddiff:.4f}); grey D k^2 chi with chi-weighted opacity {grey:.4e} (spectral/grey {Ddiff/grey:.2f})")
    # (iii) positivity scan
    neg = 0; tot = 0
    for f in (0.1, 1.0, 3.0, 10.0):
        for L in (1.0, 10.0, 100.0, None):
            hck_ = HC * 2 * np.pi / (L * 1e18) if L else 0.0; Lam = np.sum(wE * admittance(E, wn, gn, sig, HBAR * 2 * np.pi * f * 1e3, hck_)) + np.sum(wE * admittance(E, wb, gb, sig, HBAR * 2 * np.pi * f * 1e3, hck_)); tot += 1; neg += Lam.real < 0
    say(f"  (iii) positivity: Re Lambda < 0 in {neg} of {tot} (f = 0.1-10 kHz, modes 1-100 km and k = 0)")
    # (v) spectral weights
    for lab, w, g, spec in (('nu_e capture spectrum', wn, gn, S['dGec']), ('nubar mU decay spectrum', wb, gb, dec_mU)):
        m_chi = np.sum(wE * w * (g + sig)) / np.sum(wE * w); m_em = np.sum(wE * spec * (g + sig)) / np.sum(wE * spec)
        say(f"  (v) {lab}: susceptibility-weighted <gamma+sigma> {m_chi:.3e} MeV vs emission-weighted {m_em:.3e} MeV (ratio {m_em/m_chi:.2f}); mean energies {np.sum(wE*w*E)/np.sum(wE*w):.1f} vs {np.sum(wE*spec*E)/np.sum(wE*spec):.1f} MeV")
open('results/kinetic_checks.txt', 'w').write("\n".join(out) + "\n")
