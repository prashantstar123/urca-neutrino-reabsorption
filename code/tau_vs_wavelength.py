"""Round-4 item 2 (referee M2), corrected after round 5 (2026-10-27): damping time vs mode wavelength and frequency with the project's kinetic model
(kinetic_response.py functions, DD2, dmu_np(T) from finiteT.stateT), the transparent resonance scale tau_res = 2 n (dp/dn) B/(C^2 omega) (the minimum over a
REAL relaxation rate gamma = B*Lambda at gamma = omega; it is NOT the k -> infinity limit of this model), the actual short-wavelength (ballistic) limit of the
model, Lambda_inf = sum_i int dE w_i gamma_i, and the composition response per MeV of dmu:
  LC (closed reservoirs, k -> 0, fast relaxation): (chi_nu + chi_nubar)/n;
  finite-k mode amplitude (continuity, 1 kHz, 10 km): |Lambda(omega,k)|/(hbar omega n);  drive-normalised: |Lambda/(-i hbar omega + B Lambda)|/n;
  transparent static re-equilibration: 1/(B n).
These are sinusoidal amplitudes per MeV of |dmu| (the signed change over a full linear cycle is zero), at the sampled states only.
v1 output kept as results/tau_vs_wavelength_v1.*.  Output: results/tau_vs_wavelength.txt / .json, figs/fig_tau_vs_wavelength.png"""
import numpy as np, sys, json; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from numpy.polynomial.legendre import leggauss
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from finiteT import stateT
from urca_rel import spectra
from murca_spectral import spectra_mU
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import HC, HBAR, GA
from kinetic_response import response, fF, VL
LS = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, None]; FS = [0.3, 1.0, 3.0, 10.0]
STATES = ((1.0, 3.0), (1.0, 4.0), (2.0, 3.0), (2.0, 5.0), (0.5, 3.0), (3.0, 4.0), (1.0, 2.0), (1.0, 8.0), (2.0, 2.0), (2.0, 8.0))
out = []; lines = ["tau [s] vs mode wavelength (1 kHz) and vs frequency (10 km), DD2, kinetic model (hot composition).",
                   "tau_res = transparent resonance scale 2 n dp/dn B/(C^2 omega) at 1 kHz (minimum over a real relaxation rate; not the k->inf limit); tau_inf = actual k->inf (ballistic) limit of this model at 1 kHz.",
                   "dYe per MeV of |dmu|: LC = (chi_nu+chi_nubar)/n (k->0 storage); mode = |Lambda(w,k)|/(hbar w n) at 1 kHz, 10 km; drive = |Lambda/(-i hbar w + B Lambda)|/n; tr = transparent static 1/(B n).",
                   " n/n0  T  | " + " ".join(f"{('k=0' if L is None else f'{L:g}km'):>7s}" for L in LS) + " | " + " ".join(f"{f:g}kHz".rjust(8) for f in FS) + " | tau_res  tau_inf  tau_AH | dYe LC    mode      drive     tr        tr/LC  tr/mode tr/drive"]
Bb = baryons()[:2]; xg, xw = leggauss(128)
for x, T in STATES:
    n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, Bb); st = stateT(n, ne / n, nmu / n, T, Bb); dmuT = float(st['dmu_e'])
    v = VL[('DD2', x, T)]; C, B, ndpdn = v['C'], v['Bnp'], v['ndpdn']; tauAH = v['R5_f1']['tau_AH_s']
    pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2)
    xn = 50.0 * (xg + 1) / 2; wx = xw * 25.0; E = xn * T; wE = wx * T
    S = spectra(T, mst['n'], mst['p'], pn, pp, pe, dmuT, xn); dec_mU, cap_mU, inv_mU_nu, inv_mU_nb = spectra_mU(mn, mp, pn, pp, pe, T, E, dmuT)
    W_dU_nu, W_dU_nb = S['W_abs_nu'], S['W_abs_nubar']; W_mU_nu, W_mU_nb = HC * 1e-15 * inv_mU_nu, HC * 1e-15 * inv_mU_nb
    sig = HC * 1e-15 * (inv_lambda_nc(E, T, mn, -0.5, -GA / 2) + inv_lambda_nc(E, T, mp, 0.5 - 2 * SW2, GA / 2))
    fn = fF((E + dmuT) / T); fb = fF((E - dmuT) / T); wn = E ** 2 * fn * (1 - fn) / (2 * np.pi ** 2 * T * HC ** 3); wb = E ** 2 * fb * (1 - fb) / (2 * np.pi ** 2 * T * HC ** 3)
    gn = (W_dU_nu + W_mU_nu) / (1 - fn); gb = (W_mU_nb + W_dU_nb) / (1 - fb)
    rL = [response(E, wE, wn, wb, gn, gb, sig, C, B, ndpdn, 1.0, L) for L in LS]; tL = [r['tau'] for r in rL]
    tF = [response(E, wE, wn, wb, gn, gb, sig, C, B, ndpdn, f, 10.0)['tau'] for f in FS]
    om = 2 * np.pi * 1e3; hw = HBAR * om
    tau_res = 2 * ndpdn * B / (C ** 2 * om)                                   # zeta_max = C^2/(2 B omega) at gamma = omega (real relaxation rate)
    Lam_inf = float(np.sum(wE * wn * gn) + np.sum(wE * wb * gb))              # k -> inf: Lambda -> sum_i int dE w_i gamma_i (real)
    tau_inf = ndpdn / (C ** 2 * Lam_inf / (hw ** 2 + (B * Lam_inf) ** 2) * HBAR * om ** 2)
    r10 = rL[LS.index(10.0)]; Lam10 = (r10['ReLam_nu'] + r10['ReLam_nubar']) + 1j * r10['ImLam']
    chi = float(np.sum(wE * wn) + np.sum(wE * wb)); dYe_LC = chi / n; dYe_mode = abs(Lam10) / (hw * n); dYe_drive = abs(Lam10 / (-1j * hw + B * Lam10)) / n; dYe_tr = 1.0 / (B * n)
    out.append(dict(x=x, T=T, dmuT=dmuT, C=C, B=B, ndpdn=ndpdn, tau_vs_L_km={('k0' if L is None else str(L)): t for L, t in zip(LS, tL)}, tau_vs_f_kHz={str(f): t for f, t in zip(FS, tF)},
                    tau_res_1kHz=tau_res, tau_inf_1kHz=tau_inf, Lambda_inf=Lam_inf, tau_AH=tauAH, chi_nu_plus_nubar=chi, Lambda_1kHz_10km=[Lam10.real, Lam10.imag],
                    dYe_per_MeV_LC=dYe_LC, dYe_per_MeV_mode_1kHz_10km=dYe_mode, dYe_per_MeV_drive_1kHz_10km=dYe_drive, dYe_per_MeV_transparent=dYe_tr))
    lines.append(f"{x:4.1f} {T:3.1f} | " + " ".join(f"{t:7.3g}" for t in tL) + " | " + " ".join(f"{t:8.3g}" for t in tF)
                 + f" | {tau_res:7.3g} {tau_inf:8.3g} {tauAH:7.3g} | {dYe_LC:9.2e} {dYe_mode:9.2e} {dYe_drive:9.2e} {dYe_tr:9.2e} {dYe_tr / dYe_LC:6.3g} {dYe_tr / dYe_mode:7.3g} {dYe_tr / dYe_drive:7.3g}")
json.dump(out, open('results/tau_vs_wavelength.json', 'w'), indent=0); open('results/tau_vs_wavelength.txt', 'w').write("\n".join(lines) + "\n"); print("\n".join(lines))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.0, 3.5)); Lp = [L for L in LS if L is not None]
for o, col, ls in zip(out[:4], ('C3', 'C1', 'C2', 'C4'), ('-', '-', '--', '--')):
    ax.loglog(Lp, [o['tau_vs_L_km'][str(L)] for L in Lp], color=col, ls=ls, lw=2, marker='o', ms=3, label=f"{o['x']:g} n0, T = {o['T']:g} MeV")
    ax.axhline(o['tau_AH'], color=col, lw=0.8, ls=':'); ax.axhline(o['tau_inf_1kHz'], color=col, lw=0.5, ls='-.', alpha=0.6)
ax.set_xlabel('mode wavelength [km]'); ax.set_ylabel(r'damping time $\tau$ [s]  (1 kHz)'); ax.set_title('DD2: kinetic damping time vs wavelength (dotted: transparent; dash-dot: k->inf limit)', fontsize=7.5)
ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both', lw=0.4); fig.tight_layout(); fig.savefig('figs/fig_tau_vs_wavelength.png', dpi=170); print('fig saved')
