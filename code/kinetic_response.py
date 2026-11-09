"""Energy-resolved linear reaction-transport response of the nu_e and nubar_e populations: the kinetic admittance Lambda(omega, k) and the damping time
(Eqs. (3)-(5) of the Letter; Sec. S4 of the Supplemental Material). Natural units inside (MeV, fm; rates gamma, sigma, hbar*omega, hbar c k all in MeV);
one helicity per species.
Per species i (nu_e: capture <-> absorption channels, mu_nu = -dmu_np(T);  nubar_e: decay <-> inverse-decay channels, mu_nubar = +dmu_np(T)), per energy E:
  f_i(E) equilibrium occupancy;  w_i(E) = E^2 f(1-f)/(2 pi^2 T (hbar c)^3)   [so that int w dE = chi_i];
  stimulated relaxation gamma_i(E) = W_abs,i(E)/(1 - f_i(E)),  W_abs = per-mode absorption rate = hbar c (1/lambda_dU + 1/lambda_mU)  [detailed balance built in];
  isotropic energy-conserving neutral-current scattering sigma(E) = hbar c / lambda_NC(E) (n + p; full nonrelativistic free-gas response, unblocked total rate).
Plane-wave perturbation e^{i(kx - omega t)}:  (gamma + sigma - i omega + i ck mu) df = gamma df_eq + sigma <df>,  df_eq,i = s_i w_i (2 pi^2 T (hbar c)^3 / E^2) ... i.e.
  the matter affinity shifts the local equilibrium by s_i dmu (s_nubar = +1, s_nu = -1);  angular average A(E) = arctan[ck/(gamma + sigma - i omega)]/(ck);
  <df> = gamma A df_eq/(1 - sigma A);  net emission S_i = int dE w_i gamma_i [1 - gamma_i A/(1 - sigma A)] s_i dmu.
Matter:  dq/dt = S_nubar - S_nu = Lambda dmu,  Lambda = sum_i int dE w_i gamma_i [1 - gamma_i A/(1 - sigma A)];   dmu = C eps - B q
  ->  zeta = C^2 Re Lambda / |-i omega + B Lambda|^2  (dissipated power = Re(dmu* Lambda dmu)/2),  tau = n (dp/dn)/(zeta omega^2).
Limit k -> 0: no transport (gamma >> omega gives Re Lambda -> omega^2 int w/gamma).
Sensitivities: mU matrix element x1/3, x3 (emission AND absorption together, detailed balance kept); dU kernel x1/2, x2; NC x1/2, x2; range = 13 x 5 x 9 mesh.
Inputs per point: rate_state.kinetic_inputs (cold beta-equilibrium composition; finite-T affinity dmu_np(T) and coefficients C, B, K_T = n dp/dn at fixed
Y_e, Y_mu; direct Urca with density-consistent finite-T occupation references; modified Urca; neutral-current scattering).
Grids (env KIN_GRID): dd2 = 45 DD2 states -> OUT_DIR/kinetic_response_fine.json/.txt;  alteos = IUFSU and BigApple -> OUT_DIR/kinetic_response_alteos.json/.txt."""
import numpy as np, sys, json, time, traceback; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from multiprocessing import Pool
from numpy.polynomial.legendre import leggauss
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from finiteT import stateT
from urca_rel import spectra
from murca_spectral import spectra_mU
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import HC, HBAR, GA
import os
CGS = 1.6021766e33
def fF(x): return 1.0 / (1.0 + np.exp(np.clip(x, -700, 700)))
def admittance(E, w, gam, sig, hw, hck):
    """Lambda_i [fm^-3, natural units] for one species: int dE w gamma [1 - gamma A/(1 - sigma A)], A = arctan(z)/(hck), z = hck/(gamma + sigma - i hw)."""
    den = gam + sig - 1j * hw
    if hck > 0: A = np.arctan(hck / den) / hck
    else: A = 1.0 / den
    integrand = w * gam * (1.0 - gam * A / (1.0 - sig * A)); return integrand
def response(E, wE, wn, wb, gn, gb, sig, C, B, ndpdn, f_kHz, L_mode_km):
    om = 2 * np.pi * f_kHz * 1e3; hw = HBAR * om; hck = HC * 2 * np.pi / (L_mode_km * 1e18) if L_mode_km else 0.0
    Ln = np.sum(wE * admittance(E, wn, gn, sig, hw, hck)); Lb = np.sum(wE * admittance(E, wb, gb, sig, hw, hck)); Lam = Ln + Lb
    zeta_nat = C ** 2 * Lam.real / abs(-1j * hw + B * Lam) ** 2        # fm^-3 (natural)
    zeta = zeta_nat * HBAR                                              # MeV s fm^-3
    return dict(zeta_cgs=zeta * CGS, tau=ndpdn / (zeta * om ** 2) if zeta > 0 else np.inf, ReLam_nu=Ln.real, ReLam_nubar=Lb.real, ImLam=Lam.imag)
import rate_state as RS
LS = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]; FS = [0.3, 1.0, 3.0, 10.0]
FM = np.logspace(-np.log10(3), np.log10(3), 13); FD = np.logspace(-np.log10(2), np.log10(2), 5); FN = np.logspace(-np.log10(2), np.log10(2), 9)
def taus(k, fdU=1.0, fmU=1.0, fNC=1.0, f_kHz=1.0, L=10.0):
    gn = (fdU * k['WdUn'] + fmU * k['WmUn']) / (1 - k['fn']); gb = (fmU * k['WmUb'] + fdU * k['WdUb']) / (1 - k['fb'])
    return response(k['E'], k['wE'], k['wn'], k['wb'], gn, gb, fNC * k['sig'], k['C'], k['B'], k['K'], f_kHz, L)
def point(args):
    eos, x, T = args; t0 = time.time()
    try:
        k = RS.kinetic_inputs(eos, x, T); E, wE, n = k['E'], k['wE'], k['n']
        out = dict(eos=eos, x=x, T=T, n=n, Ye=k['Ye'], Ymu=k['Ymu'], dp=k['dp'], dmu_np_T=k['dmuT'], mue_T=k['mueT'], refs=list(k['refs']), C=k['C'], B=k['B'], ndpdn=k['K'],
                   chi_nu=float(np.sum(wE * k['wn'])), chi_nubar=float(np.sum(wE * k['wb'])))
        out['full'] = {lab: taus(k, f_kHz=f, L=L) for lab, f, L in (('1kHz_10km', 1.0, 10.0), ('1kHz_20km', 1.0, 20.0), ('1kHz_k0', 1.0, None), ('3kHz_10km', 3.0, 10.0))}
        out['tau_vs_L_km'] = {str(L): taus(k, L=L)['tau'] for L in LS}; out['tau_vs_L_km']['k0'] = out['full']['1kHz_k0']['tau']
        out['tau_vs_f_kHz'] = {str(f): taus(k, f_kHz=f, L=10.0)['tau'] for f in FS}
        out['sens'] = {f'mU_x{fm:g}_dU_x1_NC_x1': taus(k, 1.0, fm, 1.0) for fm in (1 / 3, 3.0)}
        out['sens'].update({f'mU_x1_dU_x{fd:g}_NC_x1': taus(k, fd, 1.0, 1.0) for fd in (0.5, 2.0)})
        out['sens'].update({f'mU_x1_dU_x1_NC_x{fc:g}': taus(k, 1.0, 1.0, fc) for fc in (0.5, 2.0)})
        mesh = np.array([[[taus(k, fd, fm, fc)['tau'] for fc in FN] for fd in FD] for fm in FM]); out['mesh_min'] = float(mesh.min()); out['mesh_max'] = float(mesh.max())
        gn = (k['WdUn'] + k['WmUn']) / (1 - k['fn']); gb = (k['WmUb'] + k['WdUb']) / (1 - k['fb'])
        band = (E > T) & (E < 10 * T); out['t_relax_us'] = dict(nu=float(np.max(HBAR / gn[band]) * 1e6), nubar=float(np.max(HBAR / gb[band]) * 1e6))
        Lam_inf_nu = float(np.sum(wE * k['wn'] * gn)); Lam_inf_nb = float(np.sum(wE * k['wb'] * gb)); Lam_inf = Lam_inf_nu + Lam_inf_nb
        def tau_inf(f):
            om = 2 * np.pi * f * 1e3; hw = HBAR * om; z = k['C'] ** 2 * Lam_inf / (hw ** 2 + (k['B'] * Lam_inf) ** 2) * HBAR; return k['K'] / (z * om ** 2)
        out['Lambda_inf'] = Lam_inf; out['Lambda_inf_nubar_share'] = Lam_inf_nb / Lam_inf; out['tau_inf_vs_f_kHz'] = {str(f): tau_inf(f) for f in FS}
        om = 2 * np.pi * 1e3; hw = HBAR * om; out['tau_res_1kHz'] = 2 * k['K'] * k['B'] / (k['C'] ** 2 * om)
        r = out['full']['1kHz_10km']; Lam = (r['ReLam_nu'] + r['ReLam_nubar']) + 1j * r['ImLam']; out['nubar_share_ReLambda'] = r['ReLam_nubar'] / (r['ReLam_nu'] + r['ReLam_nubar'])
        out['Lambda_1kHz_10km'] = [Lam.real, Lam.imag]
        out['dYe_per_MeV_mode_1kHz_10km'] = abs(Lam) / (hw * n); out['dYe_per_MeV_drive_1kHz_10km'] = abs(Lam / (-1j * hw + k['B'] * Lam)) / n
        out['dYe_per_MeV_static'] = 1.0 / (k['B'] * n); out['dYe_per_MeV_LC'] = (out['chi_nu'] + out['chi_nubar']) / n
        out['t'] = time.time() - t0; return out
    except Exception:
        return dict(eos=eos, x=x, T=T, error=traceback.format_exc()[-600:])
if __name__ == '__main__':
    OUT = os.environ.get('OUT_DIR', 'results'); which = os.environ.get('KIN_GRID', 'dd2')
    if which == 'dd2': grid = [('DD2', x, T) for x in (0.5, 1.0, 1.5, 2.0, 3.0) for T in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0)]; tag = '_fine'
    else: grid = [(eos, x, T) for eos in ('IUFSU', 'BigApple') for x, T in ((0.5, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0), (3.0, 5.0))]; tag = '_alteos'
    with Pool(int(os.environ.get('NPROC', '8'))) as pool: out = pool.map(point, grid, chunksize=1)
    json.dump(out, open(f'{OUT}/kinetic_response{tag}.json', 'w'), indent=0)
    L = ["Energy-resolved kinetic response (fix-round model), damping times [s], 1 kHz / 10 km unless stated. sens: mU x1/3 x3 | dU x1/2 x2 | NC x1/2 x2; mesh = 13x5x9 range.",
         " eos      n/n0  T  | dmu_T  | full 10km   20km     k0      3kHz  | mesh min  max  | sens: mU/3   mUx3   dU/2   dUx2   NC/2   NCx2 | t_relax[us] nu nubar | nubar share"]
    for o in out:
        if 'error' in o: L.append(f"{o['eos']:8s} {o['x']:4.2f} {o['T']:3.1f} ERROR {o['error'][-200:]!r}"); continue
        f = o['full']; s = o['sens']
        L.append(f"{o['eos']:8s} {o['x']:4.2f} {o['T']:3.1f} | {o['dmu_np_T']:+6.3f} | {f['1kHz_10km']['tau']:8.4g} {f['1kHz_20km']['tau']:8.4g} {f['1kHz_k0']['tau']:8.4g} {f['3kHz_10km']['tau']:8.4g} | {o['mesh_min']:7.4g} {o['mesh_max']:7.4g} | "
                 f"{s['mU_x0.333333_dU_x1_NC_x1']['tau']:7.4g} {s['mU_x3_dU_x1_NC_x1']['tau']:7.4g} {s['mU_x1_dU_x0.5_NC_x1']['tau']:7.4g} {s['mU_x1_dU_x2_NC_x1']['tau']:7.4g} {s['mU_x1_dU_x1_NC_x0.5']['tau']:7.4g} {s['mU_x1_dU_x1_NC_x2']['tau']:7.4g} | {o['t_relax_us']['nu']:6.3g} {o['t_relax_us']['nubar']:6.3g} | {o['nubar_share_ReLambda']:.3f}")
    open(f'{OUT}/kinetic_response{tag}.txt', 'w').write("\n".join(L) + "\n"); print("done", flush=True)
