"""Supplemental Material S2 and S3 numbers for the fix-round convention (2026-11-08): production rates of rate_state (Landau masses, density-consistent
finite-T occupation references, mu_e(T), finite-T affinity; full-response NC) compared with the exact-dispersion second implementation
(second_implementation.py: thermal Dirac masses, EOS effective chemical potentials, massive electrons).
S2 (DD2, five Table I states): detailed balance of the production rates; W_exact/W_ours for nu_e at 20, 25, 30, 40 MeV; capture-spectrum average R_cap
(Gauss weights); 25-MeV direct-Urca absorption lengths; tau with the exact direct-Urca arrays substituted on the production grid (nu, nubar, both);
nubar direct-Urca share of nubar absorption weighted with w/(gamma + sigma); damping times production vs second implementation (eight states);
open-threshold limit vs RPL98 Eq. 26 (synthetic degenerate state, density-consistent references); restricted-region bound at 2n0, 3 MeV, 25 MeV
with the thermal occupation boundaries of the exact implementation.
S3: modified-Urca checks (int x J5 = 367 pi^6/6048, derivative of the net rate = lambda_mU); antineutrino absorption lengths (2n0; 3, 5, 8 MeV);
neutral-current scattering lengths at E = T (n + p and neutrons only) and at 25 MeV.
Output: results/sm_fixround_s2s3.json and .txt"""
import os, sys, json, time
import numpy as np
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(P, 'code')); os.chdir(P)
from multiprocessing import Pool
from numpy.polynomial.legendre import leggauss
import rate_state as RS
import rmf_dd2_delta as R, finiteT as FT
import second_implementation as SI
from urca_rel import spectra, occ_ref
from urca_fullT import GT2
from murca_spectral import J5, rates as rates_mU, spectra_mU
from urca_rates import lam_mU, HC, HBAR
from nc_scatter import inv_lambda_nc_full, inv_lambda_nc_np, HV_N, HA_N, HV_P, HA_P
from kinetic_response import response
from rmf_dd2_delta import ME
STATES = [(0.5, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0), (3.0, 5.0)]
def exact_species(x, T):
    Bb = R.baryons()[:2]; n = x * R.N0; c, ne, nmu, e, p, mu, mst0, ier, sol0 = R.beta_eq_cont(n, Bb); Ye, Ymu = ne / n, nmu / n
    st = FT.stateT(n, Ye, Ymu, T, Bb); gs = R.couplings(n)[0]; s = st['sol'][0]
    mT = dict(n=R.MN - gs * s, p=R.MP - gs * s); dmu, mue = float(st['dmu_e']), float(st['mue'])
    nun, nup = SI.nu_from_density(c['n'], mT['n'], T), SI.nu_from_density(c['p'], mT['p'], T)
    nS = SI.Species(mstar=mT['n'], nu=nun); pS = SI.Species(mstar=mT['p'], nu=nup); nS.U, pS.U = 0.0, -(dmu + mue) + (nun - nup)
    return nS, pS, mue, dmu
def tau_with(k, WdUn, WdUb):
    gn = (WdUn + k['WmUn']) / (1 - k['fn']); gb = (k['WmUb'] + WdUb) / (1 - k['fb'])
    return response(k['E'], k['wE'], k['wn'], k['wb'], gn, gb, k['sig'], k['C'], k['B'], k['K'], 1.0, 10.0)['tau']
def s2_state(args):
    x, T = args; t0 = time.time()
    k = RS.kinetic_inputs('DD2', x, T); E = k['E']; dmu = k['dmuT']
    band = (E > 0.5 * T) & (E < 15 * T)
    dbn = k['WdUn'] / (2 * np.pi ** 2 / E ** 2 * np.exp((E + dmu) / T) * k['dGec'] / T)
    dbb = k['WdUb'] / (2 * np.pi ** 2 / E ** 2 * np.exp((E - dmu) / T) * k['dGnd'] / T)
    selb = band & (k['WdUb'] > 1e-4 * k['WmUb'])
    nS, pS, mue_ex, dmu_ex = exact_species(x, T)
    Wn_ex, Wb_ex = SI.W_dU(E, nS, pS, mue_ex, T, me=ME)
    w = k['wE'] * k['dGec']; Rcap = float(np.sum(w * Wn_ex) / np.sum(w * k['WdUn']))
    pts = {}
    for Ep in (20.0, 25.0, 30.0, 40.0):
        Sp = spectra(T, k['mstn'], k['mstp'], k['pFn'], k['pFp'], k['mueT'], dmu, np.array([Ep / T]), refs=k['refs'])
        we_n, we_b = SI.W_dU(np.array([Ep]), nS, pS, mue_ex, T, me=ME)
        Wo = float(Sp['W_abs_nu'][0]); We = float(we_n[0])
        pts[str(Ep)] = dict(ratio_nu=We / Wo, lam_ours_m=HC / Wo * 1e-15, lam_exact_m=HC / We * 1e-15, ratio_nubar=float(we_b[0]) / float(Sp['W_abs_nubar'][0]))
    t0_ = tau_with(k, k['WdUn'], k['WdUb']); t_nb = tau_with(k, k['WdUn'], Wb_ex); t_nu = tau_with(k, Wn_ex, k['WdUb']); t_both = tau_with(k, Wn_ex, Wb_ex)
    gb0 = (k['WmUb'] + k['WdUb']) / (1 - k['fb']); wt = k['wE'] * k['wb'] / (gb0 + k['sig'])
    share_dU_nubar = float(np.sum(wt * k['WdUb'] / (k['WmUb'] + k['WdUb'])) / np.sum(wt))
    gn0 = (k['WmUn'] + k['WdUn']) / (1 - k['fn']); wtn = k['wE'] * k['wn'] / (gn0 + k['sig'])
    share_dU_nu = float(np.sum(wtn * k['WdUn'] / (k['WmUn'] + k['WdUn'])) / np.sum(wtn))
    return dict(x=x, T=T, dmu=dmu, mueT=k['mueT'], refs=list(k['refs']), dmu_exact_impl=dmu_ex, db_nu_maxdev=float(np.max(np.abs(dbn[band] - 1))),
                db_nubar_maxdev=float(np.max(np.abs(dbb[selb] - 1))) if selb.any() else None, db_nubar_npts=int(selb.sum()), Rcap=Rcap, points=pts,
                tau_prod_grid=t0_, tau_exact_nubar_dU=t_nb, tau_exact_nu_dU=t_nu, tau_exact_both_dU=t_both,
                dU_share_nubar_weighted=share_dU_nubar, dU_share_nu_weighted=share_dU_nu, t=time.time() - t0)
def restricted_bound(x=2.0, T=3.0, E=25.0, nq=500):
    """lower bound on the exact direct-Urca nu_e absorption rate at energy E: phase space restricted to xi_n <= 0, xi_p >= 0, xi_e >= 0 (every occupation
    factor > 1/2), measured from the thermal references of the exact implementation (nu_n, nu_p, mu_e(T)); massless electrons."""
    nS, pS, mue, dmu = exact_species(x, T)
    def integral(restricted, floor):
        if restricted:
            ge, we_ = leggauss(nq); xe = (40 * T + E) * (ge + 1) / 2; we = we_ * (40 * T + E) / 2                          # xi_e in [0, 40T+E]
            lo = max(nS.mstar - nS.nu, -40 * T - E); gn, wn_ = leggauss(nq); xn = lo + (0 - lo) * (gn + 1) / 2; wn = wn_ * (0 - lo) / 2   # xi_n in [lo, 0]
        else:
            def seg(a, b, c, d):
                g, ww = leggauss(nq // 2); X = []; W = []
                for lo_, hi_ in ((a, b), (b, c), (c, d)):
                    X.append(lo_ + (hi_ - lo_) * (g + 1) / 2); W.append(ww * (hi_ - lo_) / 2)
                return np.concatenate(X), np.concatenate(W)
            xe, we = seg(-mue, -25 * T, 25 * T + E, 40 * T + E); xn, wn = seg(nS.mstar - nS.nu, -25 * T - E, 25 * T, 40 * T)
        ee = mue + xe[:, None]; en = nS.nu + xn[None, :]; pn = np.sqrt(np.maximum(en ** 2 - nS.mstar ** 2, 0))
        ep = E + en - pS.U - ee; xp = ep - pS.nu; pp = np.sqrt(np.maximum(ep ** 2 - pS.mstar ** 2, 0))
        dq = np.maximum(0, np.minimum(ee + E, pn + pp) - np.maximum(np.abs(ee - E), np.abs(pn - pp)))
        ok = (ep >= pS.mstar) & (ee > 0) & ((xp >= 0) if restricted else True)
        dq = np.where(ok, dq, 0.0)
        if floor: occ = 0.125
        else: occ = SI.fd(xn[None, :] / T) * SI.fd(-xp / T) * SI.fd(-xe[:, None] / T)
        return GT2 / (4 * np.pi ** 3 * E) * np.sum(we[:, None] * wn[None, :] * ee * en * ep * dq * occ)
    W_full_SI = float(SI.W_dU(np.array([E]), nS, pS, mue, T, me=ME)[0][0]); W_full_here = integral(False, False)
    W_reg = integral(True, False); W_floor = integral(True, True)
    lam = lambda W: HC / W * 1e-15
    return dict(x=x, T=T, E=E, nu_n=nS.nu, nu_p=pS.nu, mue=mue, dmu=dmu, W_full_second_impl=W_full_SI, W_full_this_quadrature=W_full_here,
                lam_full_m=lam(W_full_SI), lam_region_actual_occ_m=lam(W_reg), lam_region_floor_m=lam(W_floor))
def open_threshold():
    mst, pFn, pFp, pFe, T = 700.0, 380.0, 250.0, 250.0, 3.0; Es = np.array([3.0, 10.0, 30.0])
    mLn, mLp = np.hypot(pFn, mst), np.hypot(pFp, mst); dens = lambda pF: pF ** 3 / (3 * np.pi ** 2 * HC ** 3)
    refs = (occ_ref(T, mLn, dens(pFn)), occ_ref(T, mLp, dens(pFp)))
    from scipy.optimize import brentq
    mue_T = brentq(lambda m: m ** 3 + np.pi ** 2 * T ** 2 * m - pFe ** 3, 0.5 * pFe, 1.5 * pFe, xtol=1e-12)
    Sx = spectra(T, mst, mst, pFn, pFp, mue_T, 0.0, Es / T, refs=refs)
    Wdeg = GT2 * mLn * mLp * (pFe + Es) * (Es ** 2 + np.pi ** 2 * T ** 2) / (4 * np.pi ** 3 * (1 + np.exp(-Es / T)))
    Sc = spectra(T, mst, mst, pFn, pFp, pFe, 0.0, Es / T)          # cold references (previous convention), for the record
    return dict(T=T, E=Es.tolist(), refs=list(refs), cold_refs=[pFn ** 2 / (2 * mLn), pFp ** 2 / (2 * mLp)], mue_T=mue_T,
                ratio_new_refs=(Sx['W_abs_nu'] / Wdeg).tolist(), ratio_cold_refs=(Sc['W_abs_nu'] / Wdeg).tolist())
def s3_murca():
    g, w = leggauss(200); X = 80.0 * (g + 1) / 2; W = w * 40.0
    I1 = float(np.sum(W * X * J5(X, 1.0))); I1_exact = 367 * np.pi ** 6 / 6048
    der = []
    for x, T in STATES:
        b = RS.background('DD2', x, T); args = (b['mLn'], b['mLp'], b['pFn'], b['pFp'], b['pFe'], T)
        h = 1e-3 * T; Gd_p, Gc_p = rates_mU(*args, dmu=h); Gd_m, Gc_m = rates_mU(*args, dmu=-h)
        d = ((Gd_p - Gc_p) - (Gd_m - Gc_m)) / (2 * h); lm = lam_mU(*args); der.append(dict(x=x, T=T, dnet_ddmu=d, lam_mU=lm, rel=d / lm - 1))
    return dict(I1_numeric=I1, I1_exact=I1_exact, I1_rel=I1 / I1_exact - 1, derivative=der)
def s3_nubar():
    out = {}
    for T in (3.0, 5.0, 8.0):
        k = RS.kinetic_inputs('DD2', 2.0, T); Wb = k['WmUb'] + k['WdUb']; lamb = HC * 1e-15 / Wb
        wt = k['wE'] * k['dec_mU']; lam_h = float(np.sum(wt) / np.sum(wt / lamb))
        mU_share = float(np.sum(wt * k['WmUb'] / Wb) / np.sum(wt))
        out[str(T)] = dict(lam_h_emission_weighted_m=lam_h, inverse_mU_share_emission_weighted=mU_share)
    b = RS.background('DD2', 2.0, 3.0); Ev = np.array([3.0, 10.0, 30.0])
    dec, cap, inu, inb = spectra_mU(b['mLn'], b['mLp'], b['pFn'], b['pFp'], b['pFe'], 3.0, Ev, b['dmuT'])
    Sp = spectra(3.0, b['mstn'], b['mstp'], b['pFn'], b['pFp'], b['mueT'], b['dmuT'], Ev / 3.0, refs=b['refs'])
    inv_dU_b = Sp['W_abs_nubar'] / (HC * 1e-15)
    out['2n0_3MeV_points'] = dict(E=Ev.tolist(), lam_total_km=(1e-3 / (inb + inv_dU_b)).tolist(), lam_inverse_mU_km=(1e-3 / inb).tolist(), dmu=b['dmuT'])
    return out
def s3_nc():
    rows = []
    for x in (0.5, 1.0, 2.0, 3.0):
        for T in (3.0, 5.0, 8.0):
            b = RS.background('DD2', x, T); E = np.array([T])
            inv_n = inv_lambda_nc_full(E, T, b['mLn'], b['refs'][0], HV_N, HA_N); inv_p = inv_lambda_nc_full(E, T, b['mLp'], b['refs'][1], HV_P, HA_P)
            rows.append(dict(x=x, T=T, lam_np_km=float(1e-3 / (inv_n + inv_p)[0]), lam_n_km=float(1e-3 / inv_n[0]), lam_p_km=float(1e-3 / inv_p[0])))
    b = RS.background('DD2', 2.0, 3.0); E = np.array([3.0, 25.0])
    inv = inv_lambda_nc_np(E, 3.0, b['mLn'], b['mLp'], b['refs'][0], b['refs'][1]); inv_bl = inv_lambda_nc_np(E, 3.0, b['mLn'], b['mLp'], b['refs'][0], b['refs'][1], blocked=True)
    inv_ang = inv_lambda_nc_np(E, 3.0, b['mLn'], b['mLp'], b['refs'][0], b['refs'][1], vertex='angle')
    return dict(E_eq_T=rows, fig1_state=dict(E=E.tolist(), lam_m=(1 / inv).tolist(), lam_blocked_m=(1 / inv_bl).tolist(), angle_over_const=(inv_ang / inv).tolist()))
if __name__ == '__main__':
    t0 = time.time(); res = {}
    with Pool(int(os.environ.get('NPROC', '5'))) as pool:
        r2 = pool.map_async(s2_state, STATES, chunksize=1)
        res['restricted_bound'] = restricted_bound(); res['open_threshold'] = open_threshold()
        res['murca'] = s3_murca(); res['nubar'] = s3_nubar(); res['nc'] = s3_nc()
        res['s2_states'] = r2.get()
    K = {(o['x'], o['T']): o for o in json.load(open('results/kinetic_response_fine.json'))}
    S2 = {(o['x'], o['T']): o for o in json.load(open('results/second_implementation.json'))}
    res['tau_compare'] = [dict(x=x, T=T, tau_prod=K[(x, T)]['full']['1kHz_10km']['tau'], tau_second=S2[(x, T)]['tau'],
                               diff=S2[(x, T)]['tau'] / K[(x, T)]['full']['1kHz_10km']['tau'] - 1) for (x, T) in S2]
    json.dump(res, open('results/sm_fixround_s2s3.json', 'w'), indent=1, default=float)
    L = [f"runtime {time.time() - t0:.0f} s"]
    for o in res['s2_states']:
        p = o['points']
        L.append(f"{o['x']:g}n0/{o['T']:g}: DB nu {o['db_nu_maxdev']:.1e} nubar {o['db_nubar_maxdev']} ({o['db_nubar_npts']} pts) | W_ex/W_ours nu 20 {p['20.0']['ratio_nu']:.3f} 25 {p['25.0']['ratio_nu']:.3f} "
                 f"30 {p['30.0']['ratio_nu']:.3f} 40 {p['40.0']['ratio_nu']:.3f} | Rcap {o['Rcap']:.3f} | lam25 ours {p['25.0']['lam_ours_m']:.2f} exact {p['25.0']['lam_exact_m']:.2f} m | "
                 f"tau grid {o['tau_prod_grid']:.4g}: exact nubar {100*(o['tau_exact_nubar_dU']/o['tau_prod_grid']-1):+.2f}% nu {100*(o['tau_exact_nu_dU']/o['tau_prod_grid']-1):+.2f}% both {100*(o['tau_exact_both_dU']/o['tau_prod_grid']-1):+.2f}% | "
                 f"dU share nubar {100*o['dU_share_nubar_weighted']:.4g}% nu {100*o['dU_share_nu_weighted']:.4g}% | nubar W ratio 20 {p['20.0']['ratio_nubar']:.3f} 30 {p['30.0']['ratio_nubar']:.3f}")
    for o in res['tau_compare']: L.append(f"tau {o['x']:g}n0/{o['T']:g}: prod {o['tau_prod']:.5g} second {o['tau_second']:.5g} diff {100*o['diff']:+.2f}%")
    L.append('restricted bound ' + json.dumps(res['restricted_bound'], default=float)); L.append('open threshold ' + json.dumps(res['open_threshold'], default=float))
    L.append('murca ' + json.dumps(res['murca'], default=float)); L.append('nubar ' + json.dumps(res['nubar'], default=float)); L.append('nc ' + json.dumps(res['nc'], default=float))
    open('results/sm_fixround_s2s3.txt', 'w').write("\n".join(L) + "\n"); print("\n".join(L))
