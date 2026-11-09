"""Supplemental Material S4-S8: new numbers for the fix-round convention (2026-11-08). Production inputs: rate_state.kinetic_inputs (DD2).
S4 (kinetic response):
  (a) passivity bounds on all 45 DD2 states x (9 wavelengths + k -> 0) x 4 frequencies: 0 <= Re Lam <= Lam_inf, Im Lam <= 0, tau >= tau_res = 2 K_T B/(C^2 omega);
  (b) long-wavelength forms at 1 kHz and 10, 20, 50 km: P1 (diffusion) form Lam_i = int w g (-i w + D_E k^2)/(g - i w + D_E k^2), D_E = c^2/[3(g + s - i w)],
      and its leading terms Re Lam = w^2 H + k^2 D, Im Lam = -w chi (H = sum int w/g, D = sum int w c^2/[3(g+s)], chi = sum int w);
      diffusion share k^2 D/(k^2 D + w^2 H);
  (c) analytic angular average vs direct Gauss-Legendre quadrature in mu (96 and 400 nodes) at the Table I states, 1 kHz, 10 km;
  (d) energy-grid convergence: 128 nodes on [0, 50T] vs 256 nodes on [0, 70T] at the Table I states.
S6 (rates): one-at-a-time variations, the 13 x 5 x 9 mesh minimum/maximum and their corners, and the transparent baseline recomputed with the
  same rescaled direct- and modified-Urca rates (stationary Y_e* and both gradients redone), at the Table I states.
Output: results/sm_s4_s8.json and .txt"""
import os, sys, json, time
import numpy as np
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(P, 'code')); os.chdir(P)
from multiprocessing import Pool
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
import rate_state as RS
from kinetic_response import admittance, taus, FM, FD, FN, LS, FS
from urca_rates import HC, HBAR
TS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0); XS = (0.5, 1.0, 1.5, 2.0, 3.0)
TAB = [(0.5, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0), (3.0, 5.0)]
def rates_g(k, fdU=1.0, fmU=1.0):
    return (fdU * k['WdUn'] + fmU * k['WmUn']) / (1 - k['fn']), (fmU * k['WmUb'] + fdU * k['WdUb']) / (1 - k['fb'])
def lam_exact(k, gn, gb, sig, hw, hck):
    return np.sum(k['wE'] * admittance(k['E'], k['wn'], gn, sig, hw, hck)) + np.sum(k['wE'] * admittance(k['E'], k['wb'], gb, sig, hw, hck))
def tau_of(k, Lam, om):
    hw = HBAR * om; z = k['C'] ** 2 * Lam.real / abs(-1j * hw + k['B'] * Lam) ** 2 * HBAR
    return k['K'] / (z * om ** 2) if z > 0 else np.inf
def s4_state(args):
    x, T = args; t0 = time.time(); k = RS.kinetic_inputs('DD2', x, T); gn, gb = rates_g(k); sig = k['sig']
    Lam_inf = np.sum(k['wE'] * k['wn'] * gn) + np.sum(k['wE'] * k['wb'] * gb)
    mg = dict(re_over_inf_min=np.inf, re_over_inf_max=-np.inf, im_over_abs_max=-np.inf, tau_over_res_min=np.inf); viol = []; npts = 0
    for f in FS:
        om = 2 * np.pi * f * 1e3; hw = HBAR * om; tres = 2 * k['K'] * k['B'] / (k['C'] ** 2 * om)
        for L in list(LS) + [None]:
            hck = HC * 2 * np.pi / (L * 1e18) if L else 0.0
            Lam = lam_exact(k, gn, gb, sig, hw, hck); tau = tau_of(k, Lam, om); npts += 1
            mg['re_over_inf_min'] = min(mg['re_over_inf_min'], Lam.real / Lam_inf); mg['re_over_inf_max'] = max(mg['re_over_inf_max'], Lam.real / Lam_inf)
            mg['im_over_abs_max'] = max(mg['im_over_abs_max'], Lam.imag / abs(Lam)); mg['tau_over_res_min'] = min(mg['tau_over_res_min'], tau / tres)
            if not (Lam.real >= 0 and Lam.real <= Lam_inf * (1 + 1e-12) and Lam.imag <= 0 and tau >= tres * (1 - 1e-12)): viol.append([f, L])
    om = 2 * np.pi * 1e3; hw = HBAR * om
    chi = np.sum(k['wE'] * k['wn']) + np.sum(k['wE'] * k['wb'])
    H = np.sum(k['wE'] * k['wn'] / gn) + np.sum(k['wE'] * k['wb'] / gb)
    D = np.sum(k['wE'] * k['wn'] / (3 * (gn + sig))) + np.sum(k['wE'] * k['wb'] / (3 * (gb + sig)))
    lw = {}
    for L in (10.0, 20.0, 50.0):
        hck = HC * 2 * np.pi / (L * 1e18); Lam = lam_exact(k, gn, gb, sig, hw, hck)
        def p1(w, g):
            DE = hck ** 2 / (3 * (g + sig - 1j * hw)); return np.sum(k['wE'] * w * g * (-1j * hw + DE) / (g - 1j * hw + DE))
        LamP1 = p1(k['wn'], gn) + p1(k['wb'], gb); LamLO = complex(hw ** 2 * H + hck ** 2 * D, -hw * chi)
        lw[str(L)] = dict(tau=tau_of(k, Lam, om), tau_P1=tau_of(k, LamP1, om), tau_LO=tau_of(k, LamLO, om), ReLam=Lam.real, ReLam_P1=LamP1.real, ReLam_LO=LamLO.real,
                          ImLam=Lam.imag, ImLam_LO=-hw * chi, diffusion_share=hck ** 2 * D / (hck ** 2 * D + hw ** 2 * H))
    out = dict(x=x, T=T, npts=npts, violations=viol, margins=mg, chi=chi, H=H, D=D, B_chi=k['B'] * chi, long_wavelength=lw)
    if (x, T) in TAB:
        hck = HC * 2 * np.pi / (10.0 * 1e18); Lam = lam_exact(k, gn, gb, sig, hw, hck); t_an = tau_of(k, Lam, om); ang = {}
        for nmu in (96, 400):
            mu, wm = leggauss(nmu)
            def one(w, g):
                a = g + sig - 1j * hw; Ab = 0.5 * np.sum(wm[None, :] / (a[:, None] + 1j * hck * mu[None, :]), axis=1)
                return np.sum(k['wE'] * w * g * (1 - g * Ab / (1 - sig * Ab)))
            Lq = one(k['wn'], gn) + one(k['wb'], gb); ang[str(nmu)] = dict(rel_Lambda=abs(Lq / Lam - 1), rel_tau=tau_of(k, Lq, om) / t_an - 1)
        out['angular_quadrature'] = ang
    out['t'] = time.time() - t0; return out
def conv_state(args):
    x, T = args; k1 = RS.kinetic_inputs('DD2', x, T); k2 = RS.kinetic_inputs('DD2', x, T, ngrid=256, emax_T=70.0)
    t1, t2 = taus(k1)['tau'], taus(k2)['tau']; return dict(x=x, T=T, tau_128_50T=t1, tau_256_70T=t2, rel=t2 / t1 - 1)
def mesh_state(args):
    x, T = args; k = RS.kinetic_inputs('DD2', x, T); best, worst = (np.inf, None), (-np.inf, None)
    for fm in FM:
        for fd in FD:
            for fc in FN:
                t = taus(k, fd, fm, fc)['tau']
                if t < best[0]: best = (t, [fm, fd, fc])
                if t > worst[0]: worst = (t, [fm, fd, fc])
    one = {'central': taus(k)['tau'], 'mU_x1/3': taus(k, 1, 1 / 3, 1)['tau'], 'mU_x3': taus(k, 1, 3, 1)['tau'], 'dU_x1/2': taus(k, 0.5, 1, 1)['tau'],
           'dU_x2': taus(k, 2, 1, 1)['tau'], 'NC_x1/2': taus(k, 1, 1, 0.5)['tau'], 'NC_x2': taus(k, 1, 1, 2)['tau']}
    return dict(x=x, T=T, mesh_min=best[0], argmin_mU_dU_NC=best[1], mesh_max=worst[0], argmax_mU_dU_NC=worst[1], one_at_a_time=one, n_mesh=len(FM) * len(FD) * len(FN))
def F_scaled(n, Ye, Ymu, T, fdU, fmU):
    F, d = RS.net_rate('DD2', n, Ye, Ymu, T); return fdU * (d['Gd_dU'] - d['Gc_dU']) + fmU * (d['Gd_mU'] - d['Gc_mU'])
def tr_scaled(args):
    x, T, fdU, fmU = args
    mod, Bb, n0 = RS._eos('DD2'); n = x * n0; c, ne, nmu, *_, ier, sol = mod.beta_eq_cont(n, Bb); Ye0, Ymu = ne / n, nmu / n
    Fy = lambda Ye: F_scaled(n, Ye, Ymu, T, fdU, fmU)
    lo = hi = Ye0; flo = fhi = Fy(Ye0); kk = 0
    while flo < 0 and kk < 30: hi, fhi = lo, flo; lo = lo * 0.93; flo = Fy(lo); kk += 1
    while fhi > 0 and kk < 60: lo, flo = hi, fhi; hi = hi * 1.07; fhi = Fy(hi); kk += 1
    Ys = brentq(Fy, lo, hi, xtol=1e-9, rtol=1e-12); co = RS.coefficients('DD2', n, Ys, Ymu, T); C, K = co['C'], co['K']; d = 1e-3
    FY = (Fy(Ys * (1 + d)) - Fy(Ys * (1 - d))) / (2 * d * Ys); Fn = (F_scaled(n * (1 + d), Ys, Ymu, T, fdU, fmU) - F_scaled(n * (1 - d), Ys, Ymu, T, fdU, fmU)) / (2 * d * n)
    gI, gB = -FY / n, Fn; om = 2 * np.pi * 1e3; z = n * C * gB / (om ** 2 + gI ** 2)
    return dict(x=x, T=T, fdU=fdU, fmU=fmU, Ye_star=Ys, gamma_I=gI, gamma_B=gB, tau_tr=K / (z * om ** 2))
if __name__ == '__main__':
    t0 = time.time(); res = {}
    with Pool(int(os.environ.get('NPROC', '10'))) as pool:
        a = pool.map_async(s4_state, [(x, T) for x in XS for T in TS], chunksize=1)
        b = pool.map_async(conv_state, TAB, chunksize=1)
        c = pool.map_async(mesh_state, TAB, chunksize=1)
        d = pool.map_async(tr_scaled, [(x, T, fdU, fmU) for x, T in TAB for fdU, fmU in ((1, 1), (1, 1 / 3), (1, 3), (0.5, 1), (2, 1))], chunksize=1)
        res['s4'] = a.get(); res['convergence'] = b.get(); res['mesh'] = c.get(); res['tr_scaled'] = d.get()
    json.dump(res, open('results/sm_s4_s8.json', 'w'), indent=1, default=float)
    L = [f'runtime {time.time() - t0:.0f} s']
    nv = sum(len(o['violations']) for o in res['s4']); npt = sum(o['npts'] for o in res['s4'])
    L.append(f"S4 bounds: {npt} points, violations {nv}; Re/Lam_inf in [{min(o['margins']['re_over_inf_min'] for o in res['s4']):.3e}, {max(o['margins']['re_over_inf_max'] for o in res['s4']):.4f}], "
             f"max Im/|Lam| {max(o['margins']['im_over_abs_max'] for o in res['s4']):.3e}, min tau/tau_res {min(o['margins']['tau_over_res_min'] for o in res['s4']):.3f}")
    for Lk in ('10.0', '20.0', '50.0'):
        eP1 = [abs(o['long_wavelength'][Lk]['tau_P1'] / o['long_wavelength'][Lk]['tau'] - 1) for o in res['s4']]; eLO = [abs(o['long_wavelength'][Lk]['tau_LO'] / o['long_wavelength'][Lk]['tau'] - 1) for o in res['s4']]
        sh = [o['long_wavelength'][Lk]['diffusion_share'] for o in res['s4']]
        L.append(f"S4 long wavelength L = {Lk} km: P1 max dev {100*max(eP1):.2f}%, leading-order max dev {100*max(eLO):.2f}%, diffusion share {min(sh):.3f}-{max(sh):.3f}")
    for o in res['s4']:
        if 'angular_quadrature' in o:
            L.append(f"S4 {o['x']:g}n0/{o['T']:g}: angular quadrature 96 nodes rel tau {o['angular_quadrature']['96']['rel_tau']:.2e}, 400 nodes {o['angular_quadrature']['400']['rel_tau']:.2e}; "
                     f"diffusion share at 10 km {o['long_wavelength']['10.0']['diffusion_share']:.3f}; P1 dev at 10 km {100*(o['long_wavelength']['10.0']['tau_P1']/o['long_wavelength']['10.0']['tau']-1):+.2f}%; B chi {o['B_chi']:.3e}")
    for o in res['convergence']: L.append(f"S4 grid {o['x']:g}n0/{o['T']:g}: tau 128/50T {o['tau_128_50T']:.6g}, 256/70T {o['tau_256_70T']:.6g}, rel {o['rel']:+.2e}")
    for o in res['mesh']: L.append(f"S6 {o['x']:g}n0/{o['T']:g}: " + ", ".join(f"{kk} {v:.4g}" for kk, v in o['one_at_a_time'].items()) + f" | mesh {o['mesh_min']:.4g} at {np.round(o['argmin_mU_dU_NC'], 3).tolist()} - {o['mesh_max']:.4g} at {np.round(o['argmax_mU_dU_NC'], 3).tolist()}")
    for o in res['tr_scaled']: L.append(f"S6 tr {o['x']:g}n0/{o['T']:g} dU x{o['fdU']:g} mU x{o['fmU']:.3g}: Ye* {o['Ye_star']:.5f} tau_tr {1e3*o['tau_tr']:.4g} ms")
    open('results/sm_s4_s8.txt', 'w').write("\n".join(L) + "\n"); print("\n".join(L))
