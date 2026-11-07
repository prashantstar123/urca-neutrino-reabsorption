"""Energy-resolved linear reaction-transport response of the nu_e and nubar_e populations (replaces the grey diffusive estimate; round-2 referee R2-F1/F3/F5).
Own derivation (notes §11).  Natural units inside (MeV, fm; rates gamma, sigma, hbar*omega, hbar c k all in MeV); one helicity per species.
Per species i (nu_e: capture <-> absorption channels, mu_nu = -dmu_np(T);  nubar_e: decay <-> inverse-decay channels, mu_nubar = +dmu_np(T)), per energy E:
  f_i(E) equilibrium occupancy;  w_i(E) = E^2 f(1-f)/(2 pi^2 T (hbar c)^3)   [so that int w dE = chi_i];
  stimulated relaxation gamma_i(E) = W_abs,i(E)/(1 - f_i(E)),  W_abs = per-mode absorption rate = hbar c (1/lambda_dU + 1/lambda_mU)  [detailed balance built in];
  isotropic elastic NC scattering sigma(E) = hbar c / lambda_NC(E) (n + p, RPL98 Eq. 27).
Plane-wave perturbation e^{i(kx - omega t)}:  (gamma + sigma - i omega + i ck mu) df = gamma df_eq + sigma <df>,  df_eq,i = s_i w_i (2 pi^2 T (hbar c)^3 / E^2) ... i.e.
  the matter affinity shifts the local equilibrium by s_i dmu (s_nubar = +1, s_nu = -1);  angular average A(E) = arctan[ck/(gamma + sigma - i omega)]/(ck);
  <df> = gamma A df_eq/(1 - sigma A);  net emission S_i = int dE w_i gamma_i [1 - gamma_i A/(1 - sigma A)] s_i dmu.
Matter:  dq/dt = S_nubar - S_nu = Lambda dmu,  Lambda = sum_i int dE w_i gamma_i [1 - gamma_i A/(1 - sigma A)];   dmu = C eps - B q
  ->  zeta = C^2 Re Lambda / |-i omega + B Lambda|^2  (dissipated power = Re(dmu* Lambda dmu)/2),  tau = n (dp/dn)/(zeta omega^2).
Limits: k -> 0: no transport (LC-only, gamma >> omega gives Re Lambda -> omega^2 int w/gamma);  grey: single-energy;  'referee configuration' label (R3-F2): NOT the referee's grey diagnostic — here nu_e stays energy-resolved with k = 0 and nubar gets mU-only opacity;
matched independently by the referee to 0.16 %.  Sensitivities: mU matrix element x1/3, x3 (emission AND absorption together, detailed balance kept); dU kernel x1/2, x2; NC x1/2, x2.
Inputs per point: cold composition, dmu_np(T) (finiteT.stateT), kernel spectra at dmu_np (urca_rel.spectra), murca_spectral, nc_scatter; C, B, n dp/dn from results/viscosity_lc.json.
Output: results/kinetic_response.json / .txt"""
import numpy as np, sys, json, time, traceback; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from multiprocessing import Pool
from numpy.polynomial.legendre import leggauss
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from finiteT import stateT
from urca_rel import spectra
from murca_spectral import spectra_mU
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import HC, HBAR, GA
VL = {(o['eos'], o['x'], o['T']): o for o in json.load(open('results/viscosity_lc.json')) if 'error' not in o}
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
def point(args):
    x, T = args; t0 = time.time()
    try:
        Bb = baryons()[:2]; n = x * N0
        c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, Bb); st = stateT(n, ne / n, nmu / n, T, Bb); dmuT = float(st['dmu_e'])
        if ('DD2', x, T) in VL: v = VL[('DD2', x, T)]; C, B, ndpdn = v['C'], v['Bnp'], v['ndpdn']; tauAH = v['R5_f1']['tau_AH_s']
        else:
            from finiteT import suscT; A_, Cv, st0 = suscT(n, T, Bb, ne / n, nmu / n); B = -A_[0, 0]; C = Cv[0]; d = 2e-3
            ndpdn = n * (stateT(n * (1 + d), ne / n, nmu / n, T, Bb, st0['sol'])['p'] - stateT(n * (1 - d), ne / n, nmu / n, T, Bb, st0['sol'])['p']) / (2 * d * n); tauAH = float('nan')
        pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2)
        xg, xw = leggauss(128); xn = 50.0 * (xg + 1) / 2; wx = xw * 25.0; E = xn * T; wE = wx * T
        S = spectra(T, mst['n'], mst['p'], pn, pp, pe, dmuT, xn)                       # per-mode absorption rates [MeV] at dmu_np(T), detailed balance
        dec_mU, cap_mU, inv_mU_nu, inv_mU_nb = spectra_mU(mn, mp, pn, pp, pe, T, E, dmuT)   # 1/m
        W_dU_nu, W_dU_nb = S['W_abs_nu'], S['W_abs_nubar']; W_mU_nu, W_mU_nb = HC * 1e-15 * inv_mU_nu, HC * 1e-15 * inv_mU_nb
        sig = HC * 1e-15 * (inv_lambda_nc(E, T, mn, -0.5, -GA / 2) + inv_lambda_nc(E, T, mp, 0.5 - 2 * SW2, GA / 2))
        fn = fF((E + dmuT) / T); fb = fF((E - dmuT) / T)                               # mu_nu = -dmu, mu_nubar = +dmu
        wn = E ** 2 * fn * (1 - fn) / (2 * np.pi ** 2 * T * HC ** 3); wb = E ** 2 * fb * (1 - fb) / (2 * np.pi ** 2 * T * HC ** 3)
        out = dict(x=x, T=T, n=n, dmu_np_T=dmuT, C=C, B=B, ndpdn=ndpdn, chi_nu=float(np.sum(wE * wn)), chi_nubar=float(np.sum(wE * wb)), tau_AH=tauAH)
        def run(fdU=1.0, fmU=1.0, fNC=1.0, nu_grey=False, nb_dU=True):
            gn = (fdU * W_dU_nu + fmU * W_mU_nu) / (1 - fn); gb = (fmU * W_mU_nb + (fdU * W_dU_nb if nb_dU else 0.0)) / (1 - fb); s = fNC * sig
            res = {}
            for lab, f, L in (('1kHz_10km', 1.0, 10.0), ('1kHz_20km', 1.0, 20.0), ('1kHz_k0', 1.0, None), ('3kHz_10km', 3.0, 10.0)):
                if nu_grey:   # referee configuration: nu_e reservoir grey & non-diffusing (k -> 0 for nu_e), nubar energy-resolved with transport
                    om = 2 * np.pi * f * 1e3; hw = HBAR * om; hck = HC * 2 * np.pi / (L * 1e18) if L else 0.0
                    Ln = np.sum(wE * admittance(E, wn, gn, s, hw, 0.0)); Lb = np.sum(wE * admittance(E, wb, gb, s, hw, hck)); Lam = Ln + Lb
                    z = C ** 2 * Lam.real / abs(-1j * hw + B * Lam) ** 2 * HBAR; res[lab] = dict(zeta_cgs=z * CGS, tau=ndpdn / (z * om ** 2))
                else: res[lab] = response(E, wE, wn, wb, gn, gb, s, C, B, ndpdn, f, L)
            return res
        out['full'] = run(); out['referee_config_mU_only_nubar'] = run(nu_grey=True, nb_dU=False); out['referee_config_with_dU_nubar'] = run(nu_grey=True, nb_dU=True)
        out['sens'] = {f'mU_x{fm:g}_dU_x{fd:g}_NC_x{fc:g}': run(fd, fm, fc)['1kHz_10km'] for fm in (1 / 3, 3.0) for fd in (1.0,) for fc in (1.0,)}
        out['sens'].update({f'mU_x1_dU_x{fd:g}_NC_x1': run(fd, 1.0, 1.0)['1kHz_10km'] for fd in (0.5, 2.0)})
        out['sens'].update({f'mU_x1_dU_x1_NC_x{fc:g}': run(1.0, 1.0, fc)['1kHz_10km'] for fc in (0.5, 2.0)})
        # equilibration (stimulated relaxation) rates: slowest in the thermally relevant band (E within 1..10 T)
        band = (E > T) & (E < 10 * T); out['t_relax_us'] = dict(nu=float(np.max(HBAR / ((W_dU_nu + W_mU_nu) / (1 - fn))[band]) * 1e6), nubar=float(np.max(HBAR / ((W_mU_nb + W_dU_nb) / (1 - fb))[band]) * 1e6))
        out['t'] = time.time() - t0; return out
    except Exception:
        return dict(x=x, T=T, error=traceback.format_exc()[-400:])
if __name__ == '__main__':
    import os; fine = os.environ.get('KIN_FINE') == '1'
    grid = [(x, T) for x in ((0.5, 1.0, 1.5, 2.0, 3.0) if fine else (0.5, 1.0, 2.0, 3.0)) for T in ((2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0) if fine else (2.0, 3.0, 5.0, 8.0))]
    tag = '_fine' if fine else ''
    with Pool(4) as pool: out = pool.map(point, grid, chunksize=1)
    json.dump(out, open(f'results/kinetic_response{tag}.json', 'w'), indent=0)
    L = ["Energy-resolved kinetic response, DD2, damping times [s]. full = both species energy-resolved with NC transport; k0 = no transport (LC-only, closed channels); ref = referee configuration (mU-only nubar energy-resolved + grey nu_e) and the same with dU nubar opacity added.",
         " n/n0  T  | dmu_np | tau_AH(transp) | full: 1kHz/10km  1kHz/20km  k0   3kHz/10km | ref(mU-only nubar)  ref(+dU nubar) | sens 1kHz/10km: mU x1/3  x3 | dU x1/2  x2 | NC x1/2  x2 | t_relax[us] nu, nubar"]
    for o in out:
        if 'error' in o: L.append(f"{o['x']:4.2f} {o['T']:3.1f} ERROR {o['error'][-150:]!r}"); continue
        f = o['full']; r1 = o['referee_config_mU_only_nubar']['1kHz_10km']; r2 = o['referee_config_with_dU_nubar']['1kHz_10km']; s = o['sens']
        L.append(f"{o['x']:4.2f} {o['T']:3.1f} | {o['dmu_np_T']:+5.2f} | {o['tau_AH']:9.3g} | {f['1kHz_10km']['tau']:9.3g} {f['1kHz_20km']['tau']:9.3g} {f['1kHz_k0']['tau']:9.3g} {f['3kHz_10km']['tau']:9.3g} | {r1['tau']:9.3g} {r2['tau']:9.3g} | "
                 f"{s['mU_x0.333333_dU_x1_NC_x1']['tau']:8.3g} {s['mU_x3_dU_x1_NC_x1']['tau']:8.3g} | {s['mU_x1_dU_x0.5_NC_x1']['tau']:8.3g} {s['mU_x1_dU_x2_NC_x1']['tau']:8.3g} | {s['mU_x1_dU_x1_NC_x0.5']['tau']:8.3g} {s['mU_x1_dU_x1_NC_x2']['tau']:8.3g} | {o['t_relax_us']['nu']:7.3g} {o['t_relax_us']['nubar']:7.3g}")
    open(f'results/kinetic_response{tag}.txt', 'w').write("\n".join(L) + "\n"); print("done", flush=True)
