"""Elastic-approximation nu_e absorption vs the full-kinematics kernel of this project (round-4 item M1; corrected after round 5, 2026-10-27).
Elastic approximation (Bruenn 1985 App. C; Burrows-Reddy-Thompson 2006 Eq. 9-10): nucleon momentum unchanged, E_e = E_nu + Q_eff,
sigma_a = sigma0 (1+3gA^2)/4 (E_e/m_e)^2 sqrt(1-(m_e/E_e)^2) W_M, kappa = sigma_a * n_eff * [1 - f_e(E_e)].
Variants:  A  n_eff = n_n, Q_eff = m_n - m_p (no nucleon blocking, no mean field);
           B  n_eff = eta_np = (n_n - n_p)/(1 - exp((Q_eff - (mu_n - mu_p))/T)) with Q_eff = m_n - m_p  == the NuLib form (NuLib sets
              final_electron_energy = neutrino_energy + delta_np with the vacuum mass difference and adds NO mean-field shift; round-5 referee source inspection);
           C  n_eff = n_n, Q_eff = Q + (U_n - U_p)  (illustrative shifted approximation, NOT a transport-code prescription);
           MF n_eff = eta_np evaluated with Q_eff = Q + (U_n - U_p)  (the consistent 'Elastic MF' of Oertel et al. 2020 Eq. 25-26; Roberts-Reddy 2017 Eq. 77).
U_n - U_p = (mu_n - mu_p) - (E_Fn - E_Fp) from the DD2 RMF (= -2 g_rho rho_0 field; rearrangement term cancels). Electron blocking 1 - f_e = expit((E_e - mu_e)/T),
evaluated stably: the v1 form 1 - 1/(1 + exp(.)) lost all digits for E_e << mu_e ('inf' and 2.7e14 m at 2n0, 25 MeV were cancellation artefacts; round-5 referee).
v1 output kept as results/elastic_opacity_check_v1_cancellation.*.  Output: results/elastic_opacity_check.txt / .json, figs/fig_elastic_vs_full.png"""
import numpy as np, sys, json; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from scipy.special import expit
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0, MN, MP, ME
from finiteT import stateT
from urca_rel import spectra
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import HC, GA
SIG0 = 1.705e-44   # cm^2 (BRT06)
Q = MN - MP
def elastic_inv_lambda(E, T, n_eff_fm3, Qeff, mue):
    """1/lambda [1/m] for nu_e + n -> p + e in the elastic approximation; stable electron blocking."""
    E = np.asarray(E, float); Ee = E + Qeff; ok = Ee > ME
    sig = np.where(ok, SIG0 * (1 + 3 * GA ** 2) / 4 * (Ee / ME) ** 2 * np.sqrt(np.clip(1 - (ME / Ee) ** 2, 0, None)) * (1 + 1.1 * E / MN), 0.0)  # cm^2
    return sig * n_eff_fm3 * 1e39 * expit((Ee - mue) / T) * 1e2      # cm^2 * cm^-3 -> cm^-1 -> m^-1
def eta_np(c, mu, Qeff, T):
    """Bruenn nucleon blocking factor (Oertel+2020 Eq. 25-26 convention)."""
    return (c['n'] - c['p']) / (-np.expm1((Qeff - (mu['n'] - mu['p'])) / T))
out = []; lines = ["Elastic-approximation vs full-kinematics nu_e absorption, DD2; lambda in m. full = project kernel (dU, Landau masses, dmu_np(T)), dU channel only.",
                   "A: n_n, vacuum Q | B: Bruenn eta_np, vacuum Q (= NuLib form) | C: n_n, Q + Un-Up (illustrative) | MF: eta_np(Q_eff), Q + Un-Up (Oertel+2020 'Elastic MF') | NC: RPL98 Eq. 27 n+p scattering",
                   " n/n0  T  | E[MeV] | lam_full  | A         | B (NuLib) | C         | MF        | lam_NC    | E_e(C,MF) | mu_e  | Un-Up"]
Bb = baryons()[:2]
Egrid = np.array([3., 5., 10., 15., 20., 25., 30., 40., 50., 60., 100., 150., 200.])
for x in (1.0, 2.0):
    n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, Bb)
    pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2)
    UnUp = (mu['n'] - mu['p']) - (mn - mp); mue = pe
    for T in (3.0, 5.0):
        dmuT = float(stateT(n, ne / n, nmu / n, T, Bb)['dmu_e'])
        S = spectra(T, mst['n'], mst['p'], pn, pp, pe, dmuT, Egrid / T); lam_full = HC * 1e-15 / S['W_abs_nu']
        A = elastic_inv_lambda(Egrid, T, c['n'], Q, mue); Bv = elastic_inv_lambda(Egrid, T, eta_np(c, mu, Q, T), Q, mue)
        Cv = elastic_inv_lambda(Egrid, T, c['n'], Q + UnUp, mue); MF = elastic_inv_lambda(Egrid, T, eta_np(c, mu, Q + UnUp, T), Q + UnUp, mue)
        nc = inv_lambda_nc(Egrid, T, mn, -0.5, -GA / 2) + inv_lambda_nc(Egrid, T, mp, 0.5 - 2 * SW2, GA / 2)
        with np.errstate(divide='ignore'):
            lamA, lamB, lamC, lamMF, lamNC = 1 / A, 1 / Bv, 1 / Cv, 1 / MF, 1 / nc
        for i, E in enumerate(Egrid):
            lines.append(f"{x:4.1f} {T:3.1f} | {E:6.1f} | {lam_full[i]:9.3g} | {lamA[i]:9.3g} | {lamB[i]:9.3g} | {lamC[i]:9.3g} | {lamMF[i]:9.3g} | {lamNC[i]:9.3g} | {E + Q + UnUp:7.1f} | {mue:6.1f} | {UnUp:5.1f}")
            out.append(dict(x=x, T=T, E=float(E), lam_full_m=float(lam_full[i]), lam_elastic_A_m=float(lamA[i]), lam_elastic_B_NuLib_m=float(lamB[i]), lam_elastic_C_m=float(lamC[i]),
                            lam_elastic_MF_m=float(lamMF[i]), lam_NC_m=float(lamNC[i]), mue=float(mue), UnUp=float(UnUp), dmuT=dmuT))
        lines.append(f"{x:4.1f} {T:3.1f} | electron blocking releases the elastic channel only for E_nu >~ mu_e - Q_eff = {mue - Q:.1f} MeV (A, B) / {mue - Q - UnUp:.1f} MeV (C, MF); "
                     f"full-kinematics scale E_min = v_Fp dp/(1+v_Fp) = {(pp / mp) * (pn - pp - pe) / (1 + pp / mp):.2f} MeV; dmu_np(T) = {dmuT:.2f}; eta_np/n_n = {eta_np(c, mu, Q, T) / c['n']:.4f} (Q), {eta_np(c, mu, Q + UnUp, T) / c['n']:.4f} (Q_eff)")
json.dump(out, open('results/elastic_opacity_check.json', 'w'), indent=0); open('results/elastic_opacity_check.txt', 'w').write("\n".join(lines) + "\n"); print("\n".join(lines))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.2, 3.6)); Ef = np.linspace(3, 200, 400)
for x, T, col in ((2.0, 3.0, 'C3'), (2.0, 5.0, 'C1')):
    n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, Bb); pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2)
    mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2); UnUp = (mu['n'] - mu['p']) - (mn - mp); mue = pe
    dmuT = float(stateT(n, ne / n, nmu / n, T, Bb)['dmu_e']); S = spectra(T, mst['n'], mst['p'], pn, pp, pe, dmuT, Ef / T)
    ax.loglog(Ef, HC * 1e-15 / S['W_abs_nu'], color=col, lw=2, label=f'full kinematics (dU), T = {T:g} MeV')
    ax.loglog(Ef, 1 / elastic_inv_lambda(Ef, T, eta_np(c, mu, Q, T), Q, mue), color=col, lw=1.3, ls='--', label=f'elastic, NuLib form (B), T = {T:g}')
    ax.loglog(Ef, 1 / elastic_inv_lambda(Ef, T, eta_np(c, mu, Q + UnUp, T), Q + UnUp, mue), color=col, lw=1.0, ls=':', label=f'elastic + mean-field shift (MF), T = {T:g}')
    nc = inv_lambda_nc(Ef, T, mn, -0.5, -GA / 2) + inv_lambda_nc(Ef, T, mp, 0.5 - 2 * SW2, GA / 2); ax.loglog(Ef, 1 / nc, color='C0', lw=1.2, alpha=0.8 if T == 3 else 0.4, label='NC scattering' if T == 3 else None)
for y, lab in ((1e4, '10 km'), (1e0, '1 m')): ax.axhline(y, color='0.7', lw=0.6); ax.text(3.2, y * 1.3, lab, fontsize=7, color='0.4')
ax.set_ylim(1e-1, 1e12); ax.set_xlim(3, 200); ax.set_xlabel(r'$E_\nu$ [MeV]'); ax.set_ylabel(r'$\nu_e$ absorption length [m]'); ax.set_title(r'DD2, $2n_0$: full kinematics vs elastic approximation', fontsize=9)
ax.legend(fontsize=6.5, loc='upper right'); fig.tight_layout(); fig.savefig('figs/fig_elastic_vs_full.png', dpi=170); print('fig saved')
