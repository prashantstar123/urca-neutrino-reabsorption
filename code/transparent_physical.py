"""Physical free-escape (transparent) reference, fix round 2026-11-08 (six-agent audit kinetic A1; Astra audit Sec. 2.3; Alford, Haber & Zhang,
PRC 2024, arXiv:2306.06180, Eq. (19)).  At fixed n, T, Y_mu with all neutrinos escaping:
  F(n, Y_e) = Gamma_dec - Gamma_cap (direct + modified Urca, rate_state.net_rate, state-dependent composition, masses, occupations, affinity);
  stationary composition Y_e* from F(n, Y_e*) = 0;  gamma_I = -(1/n) dF/dY_e,  gamma_B = dF/dn (fixed Y_e, Y_mu, T), both at Y_e*;
  zeta_tr = -(dp/dY_e)_n gamma_B/(omega^2 + gamma_I^2) = n C gamma_B/(omega^2 + gamma_I^2),  tau_tr = n (dp/dn)/(zeta_tr omega^2),  C, K_T at Y_e*.
Reduces to (C^2/B) gamma/(omega^2 + gamma^2) with gamma = B lambda when the rates depend only on the affinity (F = lambda dmu).
Derivatives by central differences with relative steps 2e-3 and 1e-3 (both stored). Output: OUT_DIR/transparent_physical.json/.txt"""
import numpy as np, sys, json, time, traceback, os; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from multiprocessing import Pool
from scipy.optimize import brentq
import rate_state as RS
def solve_point(args):
    eos, x, T = args; t0 = time.time()
    try:
        mod, B, n0 = RS._eos(eos); n = x * n0; c, ne, nmu, *_ , ier, sol = mod.beta_eq_cont(n, B); assert ier == 1
        Ye0, Ymu = ne / n, nmu / n
        Fy = lambda Ye: RS.net_rate(eos, n, Ye, Ymu, T)[0]
        F0 = Fy(Ye0); lo, hi = (Ye0, Ye0) ; flo = fhi = F0
        k = 0
        while flo < 0 and k < 30: hi, fhi = lo, flo; lo = lo * 0.93; flo = Fy(lo); k += 1
        while fhi > 0 and k < 60: lo, flo = hi, fhi; hi = hi * 1.07; fhi = Fy(hi); k += 1
        assert flo > 0 > fhi, (flo, fhi)
        Ys = brentq(Fy, lo, hi, xtol=1e-9, rtol=1e-12)
        Fs, det = RS.net_rate(eos, n, Ys, Ymu, T)
        co = RS.coefficients(eos, n, Ys, Ymu, T); C, Bc, K = co['C'], co['B'], co['K']
        res = dict(eos=eos, x=x, T=T, n=n, Ye_cold=Ye0, Ymu=Ymu, F_cold=F0, Ye_star=Ys, F_star=Fs, dmu_star=det['dmu'], C=C, B=Bc, K=K, rates_star=det)
        for d in (2e-3, 1e-3):
            FY = (Fy(Ys * (1 + d)) - Fy(Ys * (1 - d))) / (2 * d * Ys)
            Fn = (RS.net_rate(eos, n * (1 + d), Ys, Ymu, T)[0] - RS.net_rate(eos, n * (1 - d), Ys, Ymu, T)[0]) / (2 * d * n)
            gI = -FY / n; gB = Fn
            tau = {}; tauC2 = {}
            for f in (0.3, 1.0, 3.0, 10.0):
                om = 2 * np.pi * f * 1e3; z = n * C * gB / (om ** 2 + gI ** 2); tau[str(f)] = K / (z * om ** 2) if z > 0 else np.inf
                lamq = gI / Bc; z2 = C ** 2 * lamq / (om ** 2 + gI ** 2); tauC2[str(f)] = K / (z2 * om ** 2)   # one-gradient C^2 form, same gamma_I (record only)
            res[f'd{d:g}'] = dict(F_Ye=FY, F_n=Fn, gamma_I=gI, gamma_B=gB, C_eff=n * gB / (gI / Bc) if gI != 0 else np.nan, tau_vs_f_kHz=tau, tau_C2form_vs_f_kHz=tauC2)
        res['tau_1kHz'] = res['d0.001']['tau_vs_f_kHz']['1.0']; res['step_rel_change'] = res['d0.001']['tau_vs_f_kHz']['1.0'] / res['d0.002']['tau_vs_f_kHz']['1.0'] - 1
        res['t'] = time.time() - t0; return res
    except Exception:
        return dict(eos=eos, x=x, T=T, error=traceback.format_exc()[-600:])
if __name__ == '__main__':
    OUT = os.environ.get('OUT_DIR', 'results')
    grid = [('DD2', x, T) for x in (0.5, 1.0, 1.5, 2.0, 3.0) for T in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0)]
    grid += [(eos, x, T) for eos in ('IUFSU', 'BigApple') for x, T in ((0.5, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0), (3.0, 5.0))]
    with Pool(int(os.environ.get('NPROC', '4'))) as pool: out = pool.map(solve_point, grid, chunksize=1)
    json.dump(out, open(f'{OUT}/transparent_physical.json', 'w'), indent=0)
    L = ["Physical transparent reference (stationary Y_e*, two rate gradients, AHZ2024 Eq. 19). tau_tr [ms] at 1 kHz; step = rel. change d=1e-3 vs 2e-3.",
         " eos      n/n0  T  | Ye_cold  Ye*     dmu*   | C      B       K      | gamma_I[1/s] gamma_B[1/s] | tau_tr(0.3,1,3,10 kHz) [ms]            | C2-form 1kHz | step"]
    for o in out:
        if 'error' in o: L.append(f"{o['eos']:8s} {o['x']:4.2f} {o['T']:3.1f} ERROR {o['error'][-200:]!r}"); continue
        d = o['d0.001']; t = d['tau_vs_f_kHz']
        L.append(f"{o['eos']:8s} {o['x']:4.2f} {o['T']:3.1f} | {o['Ye_cold']:.5f} {o['Ye_star']:.5f} {o['dmu_star']:6.2f} | {o['C']:6.2f} {o['B']:7.1f} {o['K']:7.3f} | {d['gamma_I']:11.4g} {d['gamma_B']:11.4g} | "
                 + " ".join(f"{1e3*t[f]:9.4g}" for f in ('0.3', '1.0', '3.0', '10.0')) + f" | {1e3*d['tau_C2form_vs_f_kHz']['1.0']:9.4g} | {o['step_rel_change']:+.1e}")
    open(f'{OUT}/transparent_physical.txt', 'w').write("\n".join(L) + "\n"); print("done", flush=True)
