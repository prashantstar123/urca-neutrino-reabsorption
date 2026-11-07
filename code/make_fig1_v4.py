"""Figure 1, APS/PRL form, v4 (corrected after round 5, 2026-10-27). (a) 1/lambda vs E_nu (notation of Roberts-Reddy-Shen 2012 / Roberts-Reddy 2017);
(b) emission spectra, each normalised to its own maximum. DD2, 2 n0, T = 3 MeV, dmu_np(T) composition.
Curves (a): solid = total nu_e absorption, direct (Landau-mass kernel, urca_rel) + modified (FS spectral function, murca_spectral) Urca, full kinematics;
dashed = direct Urca in the elastic approximation, Bruenn 1985 form with eta_np nucleon blocking and the vacuum mass difference (= NuLib), stable blocking;
dash-dotted = nubar_e absorption by inverse neutron decay nubar_e e^- p -> n, direct + modified; dotted = neutral-current scattering n + p (RPL98 Eq. 27).
Circle at exactly E = T on the NC curve. Vertical line at E_min = v_Fp dp/(1+v_Fp) (kinematic scale of the capture spectrum, not a hard cutoff).
All plotted arrays are saved to figs/fig1_v4_data.json (versioned figure data)."""
import numpy as np, sys, json; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from scipy.special import expit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 6.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'legend.frameon': False, 'legend.handlelength': 1.5, 'legend.labelspacing': 0.25, 'legend.borderpad': 0.2, 'legend.borderaxespad': 0.3, 'legend.handletextpad': 0.5})
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0, MN, MP, ME
from finiteT import stateT
from urca_rel import spectra
from murca_spectral import spectra_mU
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import HC, GA
SIG0 = 1.705e-44; Q = MN - MP   # elastic formulas duplicated from elastic_opacity_check.py (importing it would re-run that script)
def elastic_inv_lambda(E, T, n_eff_fm3, Qeff, mue):
    E = np.asarray(E, float); Ee = E + Qeff; ok = Ee > ME
    sig = np.where(ok, SIG0 * (1 + 3 * GA ** 2) / 4 * (Ee / ME) ** 2 * np.sqrt(np.clip(1 - (ME / Ee) ** 2, 0, None)) * (1 + 1.1 * E / MN), 0.0)
    return sig * n_eff_fm3 * 1e39 * expit((Ee - mue) / T) * 1e2
def eta_np(c, mu, Qeff, T): return (c['n'] - c['p']) / (-np.expm1((Qeff - (mu['n'] - mu['p'])) / T))
BLUE, VERM, ORANGE, GREY = '#0072B2', '#D55E00', '#E69F00', '0.55'
T = 3.0; x = 2.0; B = baryons()[:2]; n = x * N0
c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B); pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2)
mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2); dp = pn - pp - pe; vFp = pp / mp; Emin = vFp * dp / (1 + vFp)
dmuT = float(stateT(n, ne / n, nmu / n, T, B)['dmu_e']); UnUp = (mu['n'] - mu['p']) - (mn - mp); mue = pe
E = np.unique(np.r_[np.logspace(0, np.log10(200.0), 160), T, 25.0]); xn = E / T
S = spectra(T, mst['n'], mst['p'], pn, pp, pe, dmuT, xn); dec_mU, cap_mU, inv_mU_nu, inv_mU_nb = spectra_mU(mn, mp, pn, pp, pe, T, E, dmuT)
inv_dU_nu = S['W_abs_nu'] / HC * 1e15; inv_dU_nb = S['W_abs_nubar'] / HC * 1e15
inv_cc_nu = inv_dU_nu + inv_mU_nu; inv_cc_nb = inv_dU_nb + inv_mU_nb
inv_nc = inv_lambda_nc(E, T, mn, -0.5, -GA / 2) + inv_lambda_nc(E, T, mp, 0.5 - 2 * SW2, GA / 2)
inv_el = elastic_inv_lambda(E, T, eta_np(c, mu, Q, T), Q, mue)                        # NuLib form (B)
inv_el_MF = elastic_inv_lambda(E, T, eta_np(c, mu, Q + UnUp, T), Q + UnUp, mue)      # Oertel+2020 'Elastic MF' (for the data file only)
def fF(z): return expit(-z)
spec_nu_raw = S['dGec']; spec_nb_raw = dec_mU; therm_raw = E ** 2 * fF(E / T)
spec_nu = spec_nu_raw / spec_nu_raw.max(); spec_nb = spec_nb_raw / spec_nb_raw.max(); therm = therm_raw / therm_raw.max()
# capture spectrum on a uniform grid: mean energy and fraction below E_min
Eu = np.linspace(0.05, 150.0, 3000); Su = spectra(T, mst['n'], mst['p'], pn, pp, pe, dmuT, Eu / T)['dGec']
Emean = np.trapezoid(Eu * Su, Eu) / np.trapezoid(Su, Eu); frac_below = np.trapezoid(Su[Eu < Emin], Eu[Eu < Emin]) / np.trapezoid(Su, Eu)
iT = np.where(E == T)[0][0]; i25 = np.where(E == 25.0)[0][0]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.375, 4.3), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.0))
ax.plot(E, inv_cc_nu, color=VERM, lw=1.6, label=r'$\nu_e$ absorption, full kinematics')
ax.plot(E, inv_el, color=VERM, lw=1.1, ls='--', label=r'$\nu_e n\to e^-p$, elastic approx.')
ax.plot(E, inv_cc_nb, color=ORANGE, lw=1.1, ls='-.', label=r'$\bar\nu_e$ absorption')
ax.plot(E, inv_nc, color=BLUE, lw=1.1, ls=':', label=r'$\nu N\to\nu N$ scattering')
ax.plot([T], [inv_nc[iT]], 'o', color=BLUE, ms=5, mfc='white', mew=1.2)
ax.axvline(Emin, color='0.4', lw=0.7, ls='-')
ax.set_yscale('log'); ax.set_xscale('log'); ax.set_ylim(1e-7, 1e1); ax.set_xlim(1, 200); ax.set_ylabel(r'$1/\lambda$ (m$^{-1}$)')
ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0]); ax.legend(loc='upper left', bbox_to_anchor=(0.09, 1.0)); ax.text(0.03, 0.92, '(a)', transform=ax.transAxes)   # owner-approved 2026-11-02: both panel labels upper-left
ax.text(0.97, 0.05, r'$n=2n_0$, $T=3$ MeV', transform=ax.transAxes, ha='right', fontsize=7.5)
ax2.plot(E, therm, color=GREY, lw=1.0, ls=':')
ax2.plot(E, spec_nu, color=VERM, lw=1.4)
ax2.plot(E, spec_nb, color=ORANGE, lw=1.1, ls='-.')
ax2.axvline(Emin, color='0.4', lw=0.7, ls='-'); ax2.axvline(T, color=BLUE, lw=0.7, ls=':')
ax2.set_ylim(0, 1.12); ax2.set_yticks([0, 0.5, 1]); ax2.set_xlabel(r'$E_\nu$ (MeV)'); ax2.set_ylabel(r'$d\Gamma/dE_\nu$ (norm.)')
ax2.text(0.03, 0.85, '(b)', transform=ax2.transAxes)
fig.savefig('figs/fig1_v4.png', dpi=220, bbox_inches='tight'); fig.savefig('figs/fig1_v4.pdf', bbox_inches='tight')
data = dict(state=dict(eos='DD2', x=x, T=T, n_fm3=n, Y_e=ne / n, Y_mu=nmu / n, dmu_np_T=dmuT, pFn=pn, pFp=pp, pFe=pe, mLn=mn, mLp=mp, mDn=mst['n'], mDp=mst['p'], dp=dp, vFp=vFp, Emin=Emin, UnUp=UnUp, mue=mue, Q=Q),
            units=dict(E='MeV', inv_arrays='1/m (every inv_* array)',
                       spec_capture_raw='dGamma_cap/d(E/T) as returned by urca_rel.spectra (code units) = T * dGamma_cap/dE',
                       spec_capture_dGdE='dGamma_cap/dE = spec_capture_raw / T (same code units per MeV)',
                       spec_mU_nubar_raw='dGamma_dec/dE from murca_spectral.spectra_mU, 1/(s fm^3 MeV)',
                       thermal_raw='E^2 f(E) shape reference at zero chemical potential, dimensionless',
                       note='the raw spectra are NOT mutually comparable physical emission rates; each plotted curve is normalised to its own maximum (spec_* / max)'),
            E=E.tolist(), inv_nu_dU=inv_dU_nu.tolist(), inv_nu_mU=inv_mU_nu.tolist(), inv_nu_total=inv_cc_nu.tolist(), inv_elastic_B_NuLib=inv_el.tolist(), inv_elastic_MF=inv_el_MF.tolist(),
            inv_nubar_dU=inv_dU_nb.tolist(), inv_nubar_mU=inv_mU_nb.tolist(), inv_nubar_total=inv_cc_nb.tolist(), inv_NC=inv_nc.tolist(),
            spec_capture_raw=spec_nu_raw.tolist(), spec_capture_dGdE=(spec_nu_raw / T).tolist(), spec_mU_nubar_raw=spec_nb_raw.tolist(), thermal_raw=therm_raw.tolist(),
            checks=dict(inv_nc_at_T=float(inv_nc[iT]), lambda_nc_at_T_km=float(1e-3 / inv_nc[iT]), inv_dU_25=float(inv_dU_nu[i25]), inv_mU_25=float(inv_mU_nu[i25]), inv_total_25=float(inv_cc_nu[i25]),
                        inv_el_25=float(inv_el[i25]), inv_el_MF_25=float(inv_el_MF[i25]), inv_nubar_25=float(inv_cc_nb[i25]), inv_dU_3=float(inv_dU_nu[iT]), inv_mU_3=float(inv_mU_nu[iT]),
                        capture_mean_E=float(Emean), capture_fraction_below_Emin=float(frac_below)))
json.dump(data, open('figs/fig1_v4_data.json', 'w'))
print('saved. dp %.3f  Emin %.3f  1/lam_nc(T) %.4e (%.2f km)  25 MeV: dU %.4f mU %.5f total %.4f el(NuLib) %.3e el(MF) %.3e nubar %.4e  3 MeV: dU %.3e mU %.3e  <E>cap %.2f  frac<Emin %.4f'
      % (dp, Emin, inv_nc[iT], 1e-3 / inv_nc[iT], inv_dU_nu[i25], inv_mU_nu[i25], inv_cc_nu[i25], inv_el[i25], inv_el_MF[i25], inv_cc_nb[i25], inv_dU_nu[iT], inv_mU_nu[iT], Emean, frac_below))
