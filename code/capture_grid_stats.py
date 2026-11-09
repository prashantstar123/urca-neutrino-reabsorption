"""Generating script for the Letter's Fig. 1 grid statement (fix round 2026-11-08; previously no saved script, six-agent audit item 16 / Astra C042-C043):
over the 49 DD2 states x = 0.5, 0.75, 1, 1.5, 2, 2.5, 3 n0 and T = 2, 2.5, 3, 4, 5, 6, 8 MeV, with the fix-round rates (rate_state: cold composition,
finite-T affinity dmu(T) and mu_e(T), density-consistent occupations), for the DIRECT-CAPTURE number spectrum dGamma_ec/dE:
  mean energy <E>;  emission-weighted harmonic mean absorption length lambda_h = int S / int S/lambda(E), 1/lambda = inverse direct + inverse modified Urca;
  straight-flight survival over 1 km, P(1 km) = int S exp(-1 km/lambda) / int S;  share of the capture spectrum below E_min = v_Fp dp/(1 + v_Fp).
Output: OUT_DIR/capture_grid_stats.json/.txt"""
import numpy as np, sys, json, os, time; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from multiprocessing import Pool
import rate_state as RS
from urca_rel import spectra
from murca_spectral import spectra_mU
from urca_rates import HC
def point(a):
    x, T = a; t0 = time.time(); b = RS.background('DD2', x, T)
    E = np.linspace(0.05, 60.0 * T if T < 3 else 160.0, 1600); xn = E / T
    S = spectra(T, b['mstn'], b['mstp'], b['pFn'], b['pFp'], b['mueT'], b['dmuT'], xn, refs=b['refs'])
    inv_mU_nu = spectra_mU(b['mLn'], b['mLp'], b['pFn'], b['pFp'], b['pFe'], T, E, b['dmuT'])[2]
    inv = S['W_abs_nu'] / HC * 1e15 + inv_mU_nu; sp = S['dGec']
    norm = np.trapezoid(sp, E); Em = np.trapezoid(E * sp, E) / norm; lam_h = norm / np.trapezoid(sp * inv, E); P1 = np.trapezoid(sp * np.exp(-1e3 * inv), E) / norm
    vFp = b['pFp'] / b['mLp']; Emin = vFp * b['dp'] / (1 + vFp); below = np.trapezoid(np.where(E < Emin, sp, 0.0), E) / norm
    return dict(x=x, T=T, dp=b['dp'], dmuT=b['dmuT'], Emin=Emin, mean_E=Em, lam_h_m=lam_h, P_1km=P1, frac_below_Emin=below, t=time.time() - t0)
if __name__ == '__main__':
    OUT = os.environ.get('OUT_DIR', 'results')
    grid = [(x, T) for x in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0) for T in (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)]
    with Pool(int(os.environ.get('NPROC', '4'))) as pool: out = pool.map(point, grid, chunksize=1)
    json.dump(out, open(f'{OUT}/capture_grid_stats.json', 'w'), indent=0)
    me = [o['mean_E'] for o in out]; lh = [o['lam_h_m'] for o in out]; P = [o['P_1km'] for o in out]
    L = [f"49-state capture grid (fix-round rates): mean capture energy {min(me):.2f}-{max(me):.2f} MeV; harmonic absorption length max {max(lh):.3f} m; P(1 km) max {100*max(P):.3f} %",
         " n/n0  T  | dp     Emin  | <E>     lam_h[m]  P(1km)[%]  below Emin[%]"] + [f"{o['x']:4.2f} {o['T']:3.1f} | {o['dp']:5.1f} {o['Emin']:5.2f} | {o['mean_E']:6.2f} {o['lam_h_m']:8.3f} {100*o['P_1km']:9.4f} {100*o['frac_below_Emin']:9.3f}" for o in out]
    open(f'{OUT}/capture_grid_stats.txt', 'w').write("\n".join(L) + "\n"); print(L[0], flush=True)
