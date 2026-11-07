"""Bulk viscosity and damping times at each (EOS, n, T) of results/escape_map.json:
 (a) transparent baseline as in Alford & Harris 2019 'exact': all Urca neutrinos escape; two-way stationary dmu*; lambda_AH = d(Gamma_dec - Gamma_cap)/d dmu at dmu*;
     zeta_AH = (C^2/B) gamma/(omega^2+gamma^2), gamma = B lambda_AH;  tau = n (dp/dn)_Y / (zeta omega^2).
 (b) corrected: reabsorbed fractions (1 - P_esc(R)) feed the dilute neutrino reservoirs (lepton-conserving channel, stiffness 1/chi_nu = 12/T^2),
     escaping fractions P_esc(R) act as the transparent channel (leak).  zeta_corr = zeta_LC(nu_e) + zeta_LC(nubar_e) + zeta_leak.  Bound zeta_LC <= C^2 chi/(2 omega).
 DD2: finite-T susceptibilities (finiteT.suscT, electron channel, muon fraction fixed); IUFSU/BigApple: cold susceptibilities (rmf_fsu_delta.susceptibilities).
 Output: results/viscosity_lc.json, results/viscosity_lc.txt"""
import numpy as np, sys, json, time, traceback; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from multiprocessing import Pool
from scipy.optimize import brentq
from rmf_dd2_delta import baryons, kf
import rmf_dd2_delta as R, rmf_fsu_delta as F
from finiteT import suscT, stateT
from urca_rel import gamma_nd_2m, gamma_ec_2m
from murca_spectral import rates as rates_mU
from urca_rates import HC, HBAR
CONV = 1.0 / (HBAR * HC ** 3)
MAP = {(r['eos'], r['x'], r['T']): r for r in json.load(open('results/escape_map.json'))}
CGS = 1.6021766e33
def chi_nu(T): return T ** 2 / 12 / HC ** 3                       # fm^-3 MeV^-1, one Weyl species, mu = 0
def zeta_relax(C, Bt, lam, omega): g = Bt * lam; return C ** 2 / Bt * g / (omega ** 2 + g ** 2), g
def point(args):
    eos, x, T = args; t0 = time.time()
    try:
        r = MAP[(eos, x, T)]; B = baryons()[:2]
        if eos == 'DD2':
            n = x * R.N0; c, ne, nmu, e, p, mu, mst, ier, sol = R.beta_eq_cont(n, B); Ye, Ymu = ne / n, nmu / n
            A, Cv, st0 = suscT(n, T, B, Ye, Ymu); Bnp = -A[0, 0]; C = Cv[0]; d = 2e-3
            ndpdn = n * (stateT(n * (1 + d), Ye, Ymu, T, B, st0['sol'])['p'] - stateT(n * (1 - d), Ye, Ymu, T, B, st0['sol'])['p']) / (2 * d * n); susc_T = T
        else:
            F.use(eos); n = x * F.P['N0']; Bq, Cq, Cs, YL, p0, c, ne, nmu, mst, kS = F.susceptibilities(n, B); Bnp = -Bq; C = Cq; ndpdn = kS; susc_T = 0.0
        pn, pp, pe = r['pFn'], r['pFp'], r['pFe']; mn, mp = r['mn'], r['mp']; args_dU = (T, mst['n'], mst['p'], pn, pp, pe)
        Gd = lambda dm: gamma_nd_2m(*args_dU, dm) * CONV + rates_mU(mn, mp, pn, pp, pe, T, dm)[0]
        Gc = lambda dm: gamma_ec_2m(*args_dU, dm) * CONV + rates_mU(mn, mp, pn, pp, pe, T, dm)[1]
        f = lambda dm: np.log(Gd(dm)) - np.log(Gc(dm)); dstar = brentq(f, -10.0, 90.0, xtol=2e-3); h = 0.5
        lam_AH = ((Gd(dstar + h) - Gc(dstar + h)) - (Gd(dstar - h) - Gc(dstar - h))) / (2 * h)
        # one-way (escaping) channel derivatives at dmu = 0: |dGamma_i/d dmu|  (leak channel uses these, not Gamma/T)
        dG = dict(dec_dU=(gamma_nd_2m(*args_dU, h) - gamma_nd_2m(*args_dU, -h)) / (2 * h) * CONV,
                  cap_dU=-(gamma_ec_2m(*args_dU, h) - gamma_ec_2m(*args_dU, -h)) / (2 * h) * CONV,
                  dec_mU=(rates_mU(mn, mp, pn, pp, pe, T, h)[0] - rates_mU(mn, mp, pn, pp, pe, T, -h)[0]) / (2 * h),
                  cap_mU=-(rates_mU(mn, mp, pn, pp, pe, T, h)[1] - rates_mU(mn, mp, pn, pp, pe, T, -h)[1]) / (2 * h))
        out = dict(eos=eos, x=x, T=T, n=n, C=C, Bnp=Bnp, inv_chi=1 / chi_nu(T), ndpdn=ndpdn, dmu_star=dstar, lam_AH=lam_AH, susc_T=susc_T,
                   G_cap_dU=r['G_cap_dU'], G_dec_dU=r['G_dec_dU'], G_mU=r['G_mU'])
        out['dG'] = dG
        for Rk in (1, 3, 5, 10):
            P = {lab: r.get(f'{lab}_Pesc_{Rk}km', 1.0) for lab in ('cap_dU', 'dec_dU', 'cap_mU', 'dec_mU')}
            lam_nu = ((1 - P['cap_dU']) * r['G_cap_dU'] + (1 - P['cap_mU']) * r['G_mU']) / T
            lam_nb = ((1 - P['dec_dU']) * r['G_dec_dU'] + (1 - P['dec_mU']) * r['G_mU']) / T
            lam_l = sum(P[k] * abs(dG[k]) for k in ('cap_dU', 'cap_mU', 'dec_dU', 'dec_mU'))        # escaping fractions x one-way rate derivatives
            Bt = Bnp + 1 / chi_nu(T)
            for fk in (1.0, 3.0):
                om = 2 * np.pi * fk * 1e3
                zAH, gAH = zeta_relax(C, Bnp, lam_AH, om); zLCn, gn = zeta_relax(C, Bt, lam_nu, om); zLCb, gb = zeta_relax(C, Bt, lam_nb, om); zl, gl = zeta_relax(C, Bnp, lam_l, om)
                zc = zLCn + zLCb + zl; bound = 2 * C ** 2 * chi_nu(T) / (2 * om)
                out[f'R{Rk}_f{fk:g}'] = dict(P=P, lam_nu=lam_nu, lam_nb=lam_nb, lam_leak=lam_l, gamma_AH=gAH, gamma_LC=gn + gb, gamma_leak=gl,
                                             zeta_AH_cgs=zAH * CGS, zeta_LC_cgs=(zLCn + zLCb) * CGS, zeta_leak_cgs=zl * CGS, zeta_corr_cgs=zc * CGS, zeta_LC_bound_cgs=bound * CGS,
                                             tau_AH_s=ndpdn / (zAH * om ** 2) if zAH > 0 else np.inf, tau_corr_s=ndpdn / (zc * om ** 2) if zc > 0 else np.inf)
        out['t'] = time.time() - t0; return out
    except Exception as ex:
        return dict(eos=eos, x=x, T=T, error=traceback.format_exc()[-400:])
if __name__ == '__main__':
    grid = sorted(MAP.keys())
    with Pool(6) as pool: out = pool.map(point, grid, chunksize=2)
    json.dump(out, open('results/viscosity_lc.json', 'w'), indent=0)
    L = ["Damping time of density oscillations: transparent baseline (AH2019-type exact two-way rates at dmu*) vs corrected (reabsorbed fractions lepton-conserving + escaping fraction).",
         "eos      n/n0  T    C[MeV]  B[MeV fm3] 1/chi  dmu*   lam_AH    | R=5 km: lam_nu lam_nb lam_leak [fm-3 s-1 MeV-1] | 1 kHz: tau_AH  tau_corr  ratio | 3 kHz: tau_AH tau_corr ratio | zeta_AH zeta_LC zeta_leak bound [cgs, 1 kHz, R=5] | tau_corr(1 kHz) R=1 km, R=3 km"]
    for o in out:
        if 'error' in o: L.append(f"{o['eos']:8s} {o['x']:4.2f} {o['T']:3.1f} ERROR {o['error'][-150:]!r}"); continue
        a, b, c1 = o['R5_f1'], o['R5_f3'], o['R1_f1']; c3 = o['R3_f1']
        L.append(f"{o['eos']:8s} {o['x']:4.2f} {o['T']:3.1f} {o['C']:7.1f} {o['Bnp']:9.1f} {o['inv_chi']:8.2e} {o['dmu_star']:5.1f} {o['lam_AH']:9.2e} | {a['lam_nu']:9.2e} {a['lam_nb']:9.2e} {a['lam_leak']:9.2e} | "
                 f"{a['tau_AH_s']:9.3g} {a['tau_corr_s']:9.3g} {a['tau_corr_s']/a['tau_AH_s']:8.2g} | {b['tau_AH_s']:9.3g} {b['tau_corr_s']:9.3g} {b['tau_corr_s']/b['tau_AH_s']:8.2g} | {a['zeta_AH_cgs']:9.2e} {a['zeta_LC_cgs']:9.2e} {a['zeta_leak_cgs']:9.2e} {a['zeta_LC_bound_cgs']:9.2e} | {c1['tau_corr_s']:9.3g} {c3['tau_corr_s']:9.3g}")
    open('results/viscosity_lc.txt', 'w').write("\n".join(L) + "\n"); print("done", flush=True)
