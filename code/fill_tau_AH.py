"""Transparent-baseline damping time tau_AH at the six fine-grid states missing from results/viscosity_lc.json (DD2, x = 0.5, 1, 2 n0; T = 3.5, 4.5 MeV),
requested by the round-8 review for the thin curve of Figure 2. EXACTLY the prescription of viscosity_lc.py: cold beta-equilibrium composition (Fermi momenta,
Landau masses as in escape_map.py), finite-T susceptibilities (finiteT.suscT, electron channel, fixed muon fraction), two-way stationary dmu* from
Gamma_dec(dmu*) = Gamma_cap(dmu*) (direct Urca Landau-mass rates + modified Urca), lam_AH = d(Gamma_dec - Gamma_cap)/d dmu at dmu* (central difference h = 0.5 MeV),
zeta_AH = (C^2/B) gamma/(omega^2 + gamma^2) with gamma = B lam_AH, tau_AH = n (dp/dn)_Y/(zeta omega^2).
Routes: (i) the same function at the existing grid states (x, T = 3, 4, 5 MeV) must reproduce viscosity_lc.json R5_f1 tau_AH_s; (ii) C, Bnp, n dp/dn at the new
states compared with results/kinetic_response_fine.json (computed there by the same functions); (iii) step-size robustness h = 0.25 vs 0.5 MeV;
(iv) tau_AH >= tau_res = 2 n dp/dn B/(C^2 omega) (zeta_AH <= C^2/(2 B omega)).  Output: results/viscosity_lc_fill.json / .txt"""
import numpy as np, sys, json, time; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from multiprocessing import Pool
from scipy.optimize import brentq
import rmf_dd2_delta as R
from rmf_dd2_delta import baryons, kf
from finiteT import suscT, stateT
from urca_rel import gamma_nd_2m, gamma_ec_2m
from murca_spectral import rates as rates_mU
from urca_rates import HC, HBAR
P = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))) + '/'
CONV = 1.0 / (HBAR * HC ** 3); CGS = 1.6021766e33
def zeta_relax(C, Bt, lam, omega): g = Bt * lam; return C ** 2 / Bt * g / (omega ** 2 + g ** 2), g
def point(args):
    x, T, h = args; t0 = time.time(); B = baryons()[:2]
    n = x * R.N0; c, ne, nmu, e, p, mu, mst, ier, sol = R.beta_eq_cont(n, B); Ye, Ymu = ne / n, nmu / n
    A, Cv, st0 = suscT(n, T, B, Ye, Ymu); Bnp = -A[0, 0]; C = Cv[0]; d = 2e-3
    ndpdn = n * (stateT(n * (1 + d), Ye, Ymu, T, B, st0['sol'])['p'] - stateT(n * (1 - d), Ye, Ymu, T, B, st0['sol'])['p']) / (2 * d * n)
    pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2)
    args_dU = (T, mst['n'], mst['p'], pn, pp, pe)
    Gd = lambda dm: gamma_nd_2m(*args_dU, dm) * CONV + rates_mU(mn, mp, pn, pp, pe, T, dm)[0]
    Gc = lambda dm: gamma_ec_2m(*args_dU, dm) * CONV + rates_mU(mn, mp, pn, pp, pe, T, dm)[1]
    f = lambda dm: np.log(Gd(dm)) - np.log(Gc(dm)); dstar = brentq(f, -10.0, 90.0, xtol=2e-3)
    lam_AH = ((Gd(dstar + h) - Gc(dstar + h)) - (Gd(dstar - h) - Gc(dstar - h))) / (2 * h)
    out = dict(eos='DD2', x=x, T=T, h=h, n=n, C=float(C), Bnp=float(Bnp), ndpdn=float(ndpdn), dmu_star=dstar, lam_AH=lam_AH, pFn=pn, pFp=pp, pFe=pe, mn=float(mn), mp=float(mp))
    for fk in (1.0, 3.0):
        om = 2 * np.pi * fk * 1e3; zAH, gAH = zeta_relax(C, Bnp, lam_AH, om)
        out[f'f{fk:g}'] = dict(gamma_AH=float(gAH), zeta_AH_cgs=float(zAH * CGS), tau_AH_s=float(ndpdn / (zAH * om ** 2)) if zAH > 0 else np.inf, tau_res_s=float(2 * ndpdn * Bnp / (C ** 2 * om)))
    out['t'] = time.time() - t0; return out
if __name__ == '__main__':
    XS = (0.5, 1.0, 2.0); NEW = [(x, T, 0.5) for x in XS for T in (3.5, 4.5)]; CHK = [(x, T, 0.5) for x in XS for T in (3.0, 4.0, 5.0)]; ROB = [(x, T, 0.25) for x in XS for T in (3.5, 4.5)]
    with Pool(4) as pool: res = pool.map(point, NEW + CHK + ROB, chunksize=1)
    new, chk, rob = res[:len(NEW)], res[len(NEW):len(NEW) + len(CHK)], res[len(NEW) + len(CHK):]
    VL = {(o['x'], o['T']): o for o in json.load(open(P + 'results/viscosity_lc.json')) if o['eos'] == 'DD2' and 'error' not in o}
    KF = {(o['x'], o['T']): o for o in json.load(open(P + 'results/kinetic_response_fine.json'))}
    checks = dict(route_i_identity_at_existing_states={f'{o["x"]}_{o["T"]}': dict(tau_AH_here=o['f1']['tau_AH_s'], tau_AH_viscosity_lc=VL[(o['x'], o['T'])]['R5_f1']['tau_AH_s'],
                                                        rel=o['f1']['tau_AH_s'] / VL[(o['x'], o['T'])]['R5_f1']['tau_AH_s'] - 1, dmu_star_here=o['dmu_star'], dmu_star_vlc=VL[(o['x'], o['T'])]['dmu_star'],
                                                        lam_rel=o['lam_AH'] / VL[(o['x'], o['T'])]['lam_AH'] - 1) for o in chk},
                  route_ii_coefficients_vs_kinetic_fine={f'{o["x"]}_{o["T"]}': dict(C_rel=o['C'] / KF[(o['x'], o['T'])]['C'] - 1, B_rel=o['Bnp'] / KF[(o['x'], o['T'])]['B'] - 1, ndpdn_rel=o['ndpdn'] / KF[(o['x'], o['T'])]['ndpdn'] - 1) for o in new},
                  route_iii_step_robustness={f'{o["x"]}_{o["T"]}': dict(tau_h05=o['f1']['tau_AH_s'], tau_h025=r['f1']['tau_AH_s'], rel=r['f1']['tau_AH_s'] / o['f1']['tau_AH_s'] - 1) for o, r in zip(new, rob)},
                  route_iv_bound={f'{o["x"]}_{o["T"]}': dict(tau_AH=o['f1']['tau_AH_s'], tau_res=o['f1']['tau_res_s'], ok=bool(o['f1']['tau_AH_s'] >= o['f1']['tau_res_s'])) for o in new})
    json.dump(dict(values=new, checks=checks, robustness_runs=rob, identity_runs=chk, note='six transparent-baseline states missing from viscosity_lc.json; same prescription; 1 kHz = f1, 3 kHz = f3'),
              open(P + 'results/viscosity_lc_fill.json', 'w'), indent=0)
    L = ['Transparent baseline tau_AH at the fine-grid states missing from viscosity_lc.json (DD2, h = 0.5 MeV); prescription of viscosity_lc.py; 1 kHz and 3 kHz.',
         ' n/n0  T   | C[MeV]  B[MeV fm3]  n dp/dn | dmu*   lam_AH    | tau_AH(1 kHz) s  tau_res(1 kHz) s | tau_AH(3 kHz) s | t[s]']
    for o in new: L.append(f"{o['x']:4.1f} {o['T']:3.1f} | {o['C']:7.2f} {o['Bnp']:9.1f} {o['ndpdn']:8.4f} | {o['dmu_star']:5.2f} {o['lam_AH']:9.3e} | {o['f1']['tau_AH_s']:12.5g} {o['f1']['tau_res_s']:12.5g} | {o['f3']['tau_AH_s']:12.5g} | {o['t']:.0f}")
    L.append('checks: ' + json.dumps(checks))
    open(P + 'results/viscosity_lc_fill.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[:8])); print('route i max |rel|:', max(abs(v['rel']) for v in checks['route_i_identity_at_existing_states'].values()),
          ' route ii max:', max(abs(v[k]) for v in checks['route_ii_coefficients_vs_kinetic_fine'].values() for k in v), ' route iii max:', max(abs(v['rel']) for v in checks['route_iii_step_robustness'].values()),
          ' route iv all ok:', all(v['ok'] for v in checks['route_iv_bound'].values()))
