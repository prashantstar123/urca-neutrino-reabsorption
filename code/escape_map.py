"""(n, T) map of the fate of the neutrinos emitted by the Urca processes in beta-equilibrated (cold composition, dmu = 0) nucleonic matter.
For each EOS point: direct Urca below/above threshold (urca_rel.spectra: emission spectra of capture nu_e and decay nubar_e, absorption per mode by the
inverse reactions) + modified Urca (murca_spectral, FS approximation) + NC scattering (nc_scatter).  Total absorption per mode = sum over inverse
dUrca and inverse mUrca.  Escape probability P_esc(R) = < exp(-R / lambda(E)) >_spectrum for R = 1, 3, 5, 10 km.  Output: results/escape_map.json + .txt"""
import numpy as np, sys, json, time; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from multiprocessing import Pool
from numpy.polynomial.legendre import leggauss
from rmf_dd2_delta import baryons, kf
import rmf_dd2_delta as R, rmf_fsu_delta as F
from urca_rel import spectra, fF
from urca_rates import HC
from murca_spectral import spectra_mU, rates as rates_mU
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import GA
XG, XW = leggauss(64); XN = 40.0 * (XG + 1) / 2; WN = XW * 20.0
RS = (1.0, 3.0, 5.0, 10.0)
def point(args):
    eos, x, T = args; t0 = time.time()
    if eos == 'DD2': mod = R; n0 = R.N0
    else: F.use(eos); mod = F; n0 = F.P['N0']
    B = baryons()[:2]; n = x * n0; c, ne, nmu, e, p, mu, mst, ier, sol = mod.beta_eq_cont(n, B)
    pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2); mue = pe   # massless-electron chemical potential, as in the validated nu_escape run
    E = XN * T; w = WN * T
    S = spectra(T, mst['n'], mst['p'], pn, pp, mue, 0.0, XN)
    inv_dU_nu = S['W_abs_nu'] / HC * 1e15; inv_dU_nb = S['W_abs_nubar'] / HC * 1e15          # 1/m
    dec_mU, cap_mU, inv_mU_nu, inv_mU_nb = spectra_mU(mn, mp, pn, pp, pe, T, E)
    inv_nu = inv_dU_nu + inv_mU_nu; inv_nb = inv_dU_nb + inv_mU_nb
    inv_nc = lambda EE: inv_lambda_nc(EE, T, mn, -0.5, -GA / 2) + inv_lambda_nc(EE, T, mp, 0.5 - 2 * SW2, GA / 2)
    spec_cap_dU = S['dGec'] / T; spec_dec_dU = S['dGnd'] / T          # per MeV (natural-unit rates; only shapes + ratios used below) 
    G_cap_dU = np.sum(w * spec_cap_dU); G_dec_dU = np.sum(w * spec_dec_dU)
    conv = 1.0 / (6.582119569e-22 * HC ** 3)                               # MeV^4 -> 1/(s fm^3)
    res = dict(eos=eos, x=x, T=T, n=n, Ye=ne / n, dp=pn - pp - pe, mue=mue, mn=mn, mp=mp, pFn=pn, pFp=pp, pFe=pe,
               G_cap_dU=G_cap_dU * conv, G_dec_dU=G_dec_dU * conv, G_mU=float(rates_mU(mn, mp, pn, pp, pe, T)[0]))
    for lab, spec, inv in (('cap_dU', spec_cap_dU, inv_nu), ('dec_dU', spec_dec_dU, inv_nb), ('cap_mU', cap_mU, inv_nu), ('dec_mU', dec_mU, inv_nb)):
        norm = np.sum(w * spec)
        if norm <= 0: continue
        Em = float(np.sum(w * spec * E) / norm); lam_h = float(norm / np.sum(w * spec * inv))          # harmonic (emission-weighted) mean free path [m]
        res[f'{lab}_Emean'] = Em; res[f'{lab}_lam_h_m'] = lam_h; res[f'{lab}_lam_nc_m'] = float(1.0 / inv_nc(Em))
        for Rk in RS: res[f'{lab}_Pesc_{Rk:g}km'] = float(np.sum(w * spec * np.exp(-Rk * 1e3 * inv)) / norm)
    res['t'] = time.time() - t0; return res
if __name__ == '__main__':
    grid = [(eos, x, T) for eos in ('DD2', 'IUFSU', 'BigApple') for x in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0) for T in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)]
    with Pool(6) as pool: out = pool.map(point, grid, chunksize=4)
    json.dump(out, open('results/escape_map.json', 'w'), indent=0)
    lines = ["eos      n/n0  T   dp[MeV]  G_cap_dU  G_dec_dU  G_mU [fm^-3 s^-1] | capture nu_e (dU): <E> lam_h[m] Pesc(1,5 km) | decay nubar (dU): <E> lam_h Pesc(1,5) | mU nu: <E> lam_h Pesc(1,5) | mU nubar: <E> lam_h Pesc(1,5)"]
    for r in out:
        s = f"{r['eos']:8s} {r['x']:4.2f} {r['T']:3.1f} {r['dp']:+7.1f} {r['G_cap_dU']:9.2e} {r['G_dec_dU']:9.2e} {r['G_mU']:9.2e} |"
        for lab in ('cap_dU', 'dec_dU', 'cap_mU', 'dec_mU'):
            if f'{lab}_Emean' in r: s += f" {r[lab+'_Emean']:5.1f} {r[lab+'_lam_h_m']:9.3g} {r[lab+'_Pesc_1km']:.3f} {r[lab+'_Pesc_5km']:.3f} |"
            else: s += "   --   |"
        lines.append(s)
    open('results/escape_map.txt', 'w').write("\n".join(lines) + "\n"); print("done", len(out), "points", flush=True)
