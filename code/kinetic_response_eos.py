"""EOS robustness of the energy-resolved kinetic damping times (kinetic_response.py model, unchanged physics):
DD2, IUFSU, BigApple on one grid.  Composition: cold beta equilibrium of each RMF set (rmf_dd2_delta / rmf_fsu_delta.use(eos)).
dmu_np(T): DD2 'hot' = finiteT.stateT (as in results/kinetic_response*.json); 'cold' = 0 for every EOS (finite-T composition code exists for DD2 only).
DD2 hot vs DD2 cold quantifies the dmu_np(T) = 0 approximation; IUFSU/BigApple cold vs DD2 cold is the EOS comparison.
C, B, n dp/dn, tau_AH (transparent baseline) from results/viscosity_lc.json (same source as kinetic_response.py).
Output: results/kinetic_response_eos.json / .txt"""
import numpy as np, sys, json, time, traceback, os; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from multiprocessing import Pool
from numpy.polynomial.legendre import leggauss
import rmf_dd2_delta as R, rmf_fsu_delta as F
from rmf_dd2_delta import baryons, kf
from finiteT import stateT
from urca_rel import spectra
from murca_spectral import spectra_mU
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import HC, HBAR, GA
from kinetic_response import admittance, response, fF, VL
def point(args):
    eos, x, T, cold = args; t0 = time.time()
    try:
        if eos == 'DD2': mod = R; n0 = R.N0
        else: F.use(eos); mod = F; n0 = F.P['N0']
        Bb = baryons()[:2]; n = x * n0; c, ne, nmu, e, p, mu, mst, ier, sol = mod.beta_eq_cont(n, Bb)
        dmuT = 0.0 if cold else float(stateT(n, ne / n, nmu / n, T, Bb)['dmu_e'])
        v = VL[(eos, x, T)]; C, B, ndpdn = v['C'], v['Bnp'], v['ndpdn']; tauAH = v['R5_f1']['tau_AH_s']
        pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2)
        xg, xw = leggauss(128); xn = 50.0 * (xg + 1) / 2; wx = xw * 25.0; E = xn * T; wE = wx * T
        S = spectra(T, mst['n'], mst['p'], pn, pp, pe, dmuT, xn)
        dec_mU, cap_mU, inv_mU_nu, inv_mU_nb = spectra_mU(mn, mp, pn, pp, pe, T, E, dmuT)
        W_dU_nu, W_dU_nb = S['W_abs_nu'], S['W_abs_nubar']; W_mU_nu, W_mU_nb = HC * 1e-15 * inv_mU_nu, HC * 1e-15 * inv_mU_nb
        sig = HC * 1e-15 * (inv_lambda_nc(E, T, mn, -0.5, -GA / 2) + inv_lambda_nc(E, T, mp, 0.5 - 2 * SW2, GA / 2))
        fn = fF((E + dmuT) / T); fb = fF((E - dmuT) / T)
        wn = E ** 2 * fn * (1 - fn) / (2 * np.pi ** 2 * T * HC ** 3); wb = E ** 2 * fb * (1 - fb) / (2 * np.pi ** 2 * T * HC ** 3)
        gn = (W_dU_nu + W_mU_nu) / (1 - fn); gb = (W_mU_nb + W_dU_nb) / (1 - fb)
        out = dict(eos=eos, x=x, T=T, cold=cold, n=n, dp=pn - pp - pe, dmu_np_T=dmuT, C=C, B=B, ndpdn=ndpdn, tau_AH=tauAH, Ecap_mean=float(S.get('E_mean_cap', np.nan)) if isinstance(S, dict) and 'E_mean_cap' in S else None)
        out['full'] = {lab: response(E, wE, wn, wb, gn, gb, sig, C, B, ndpdn, f, L) for lab, f, L in (('1kHz_10km', 1.0, 10.0), ('1kHz_20km', 1.0, 20.0), ('1kHz_k0', 1.0, None), ('3kHz_10km', 3.0, 10.0))}
        out['t'] = time.time() - t0; return out
    except Exception:
        return dict(eos=eos, x=x, T=T, cold=cold, error=traceback.format_exc()[-400:])
if __name__ == '__main__':
    grid = [(eos, x, T, cold) for eos, cold in (('DD2', False), ('DD2', True), ('IUFSU', True), ('BigApple', True)) for x in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0) for T in (2.0, 3.0, 4.0, 5.0, 8.0)]
    with Pool(4) as pool: out = pool.map(point, grid, chunksize=1)
    json.dump(out, open('results/kinetic_response_eos.json', 'w'), indent=0)
    L = ["EOS robustness of the kinetic damping times [s] (kinetic_response.py model). hot = dmu_np(T) from finiteT.stateT (DD2 only); cold = dmu_np = 0. tau_AH = transparent baseline (viscosity_lc.json). ratio = tau_full(1kHz,10km)/tau_AH.",
         "eos      n/n0  T   comp | dp[MeV] dmu_np | tau_AH   | full: 1kHz/10km  1kHz/20km   k0    3kHz/10km | ratio"]
    for o in out:
        if 'error' in o: L.append(f"{o['eos']:8s} {o['x']:4.2f} {o['T']:3.1f} {'cold' if o['cold'] else 'hot '} ERROR {o['error'][-150:]!r}"); continue
        f = o['full']; t1 = f['1kHz_10km']['tau']
        L.append(f"{o['eos']:8s} {o['x']:4.2f} {o['T']:3.1f} {'cold' if o['cold'] else 'hot '} | {o['dp']:+6.1f} {o['dmu_np_T']:+6.2f} | {o['tau_AH']:8.3g} | {t1:9.3g} {f['1kHz_20km']['tau']:9.3g} {f['1kHz_k0']['tau']:9.3g} {f['3kHz_10km']['tau']:9.3g} | {t1 / o['tau_AH'] if o['tau_AH'] > 0 else float('nan'):7.3g}")
    open('results/kinetic_response_eos.txt', 'w').write("\n".join(L) + "\n"); print("done", flush=True)
