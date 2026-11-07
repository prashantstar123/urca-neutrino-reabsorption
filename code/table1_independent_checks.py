"""Table I independent coverage (after round 5, 2026-10-27). At each Table I state (DD2 hot composition; IUFSU and BigApple cold composition), 1 kHz, 10-km mode:
 (A) project damping time (kinetic_response model; cross-check against the result files);
 (B) the round-5 referee's exact-dispersion charged-current kernel + angular solver (astra_review/referee_audit/round5/r5_audit.py, executed here by the authors,
     UNCHANGED, with the referee's own mU and NC transcriptions) at phase-space orders 180 and 300 = second microscopic route;
 (C) the authors' combined (mU, dU, NC) normalisation mesh with the project kernel, 13 x 5 x 9 log-spaced in mU [1/3, 3], dU [1/2, 2], NC [1/2, 2]
     (the round-3 referee's definition), giving the combined sensitivity range (sampled, inside one closure; not a total error).
Output: results/table1_independent_checks.txt / .json"""
import sys, json, time, numpy as np
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); sys.path.insert(0, _os.path.join(ROOT, 'code')); sys.path.insert(0, _os.path.join(ROOT, 'second_implementation'))
import r5_audit as RA
from numpy.polynomial.legendre import leggauss
import rmf_dd2_delta as R, rmf_fsu_delta as F
from finiteT import stateT
from urca_rel import spectra
from murca_spectral import spectra_mU
from nc_scatter import inv_lambda_nc, SW2
from urca_rates import HC, GA
from kinetic_response import response, fF, VL
STATES = [('DD2', 0.5, 3.0), ('DD2', 1.0, 4.0), ('DD2', 1.5, 4.0), ('DD2', 2.0, 5.0), ('DD2', 3.0, 5.0),
          ('IUFSU', 0.5, 3.0), ('IUFSU', 1.0, 4.0), ('IUFSU', 1.5, 4.0), ('IUFSU', 2.0, 5.0), ('IUFSU', 3.0, 5.0),
          ('BigApple', 0.5, 3.0), ('BigApple', 1.0, 4.0), ('BigApple', 1.5, 4.0), ('BigApple', 2.0, 5.0), ('BigApple', 3.0, 5.0), ('DD2', 4.0, 5.0), ('IUFSU', 3.0, 3.0), ('IUFSU', 2.5, 4.0)]
FM = np.logspace(-np.log10(3), np.log10(3), 13); FD = np.logspace(-np.log10(2), np.log10(2), 5); FN = np.logspace(-np.log10(2), np.log10(2), 9)
xg, xw = leggauss(128); Bb = R.baryons()[:2]
def project_rates(eos, x, T, cold):
    if eos == 'DD2': mod, n0 = R, R.N0
    else: F.use(eos); mod, n0 = F, F.P['N0']
    n = x * n0; c, ne, nmu, e, p, mu, mst, ier, sol = mod.beta_eq_cont(n, Bb); assert ier == 1
    dmuT = 0.0 if cold else float(stateT(n, ne / n, nmu / n, T, Bb)['dmu_e'])
    v = VL[(eos, x, T)]; C, B, ndpdn = v['C'], v['Bnp'], v['ndpdn']
    pn, pp, pe = mod.kf(c['n'], 2), mod.kf(c['p'], 2), mod.kf(ne, 2); mn = np.sqrt(pn ** 2 + mst['n'] ** 2); mp = np.sqrt(pp ** 2 + mst['p'] ** 2)
    xn = 50.0 * (xg + 1) / 2; wx = xw * 25.0; E = xn * T; wE = wx * T
    S = spectra(T, mst['n'], mst['p'], pn, pp, pe, dmuT, xn); dec_mU, cap_mU, inv_mU_nu, inv_mU_nb = spectra_mU(mn, mp, pn, pp, pe, T, E, dmuT)
    sig = HC * 1e-15 * (inv_lambda_nc(E, T, mn, -0.5, -GA / 2) + inv_lambda_nc(E, T, mp, 0.5 - 2 * SW2, GA / 2))
    fn = fF((E + dmuT) / T); fb = fF((E - dmuT) / T); wn = E ** 2 * fn * (1 - fn) / (2 * np.pi ** 2 * T * HC ** 3); wb = E ** 2 * fb * (1 - fb) / (2 * np.pi ** 2 * T * HC ** 3)
    return dict(E=E, wE=wE, wn=wn, wb=wb, fn=fn, fb=fb, WdUn=S['W_abs_nu'], WdUb=S['W_abs_nubar'], WmUn=HC * 1e-15 * inv_mU_nu, WmUb=HC * 1e-15 * inv_mU_nb, sig=sig, C=C, B=B, ndpdn=ndpdn, dp=pn - pp - pe, dmuT=dmuT, tau_AH=v['R5_f1']['tau_AH_s'])
def tau_project(r, fdU=1.0, fmU=1.0, fNC=1.0):
    gn = (fdU * r['WdUn'] + fmU * r['WmUn']) / (1 - r['fn']); gb = (fmU * r['WmUb'] + fdU * r['WdUb']) / (1 - r['fb'])
    return response(r['E'], r['wE'], r['wn'], r['wb'], gn, gb, fNC * r['sig'], r['C'], r['B'], r['ndpdn'], 1.0, 10.0)['tau']
out = []; lines = ["Table I independent coverage, 1 kHz, 10-km mode, tau in s. (A) project model; (B) round-5 referee exact-dispersion kernel + angular solver (orders 180 / 300); (C) combined (mU,dU,NC) mesh range with the project kernel (13x5x9). dp = p_Fn - p_Fp - p_Fe [MeV]; flag = kernel-limited if dp < 20 MeV (excluded from unqualified tables), near-threshold if 20 <= dp < 30 MeV (pointwise exact-kernel validation required, no percentage guarantee). Discrepancy (B)/(A) - 1 at the checked states: IUFSU 3n0 dp 14.4: +0.7 % (3 MeV) but -68.8 % (5 MeV); BigApple 3n0 dp 8.0: -96.1 %; DD2 4n0 cold dp 24.9: -21.6 % (= project 27.5 % above exact); IUFSU 2.5n0 dp 26.8: +2.8 %; DD2 3n0 hot dp 33.7: +1.7 %. The gap alone does not set the error.",
                   " eos      n/n0  T  comp |  dp   | tau_AH   | (A) project | (B) exact180  exact300 | (B)/(A)  | (C) range min   max   | flag"]
for eos, x, T in STATES:
    t0 = time.time(); cold = (eos != 'DD2') or (x > 3.0)
    r = project_rates(eos, x, T, cold); tA = tau_project(r)
    s = RA.state(eos, x, T, cold); E, we = RA.nodes(160, 0, 50 * T); mn_, mb_ = RA.own_murca(s, E); sc = RA.own_nc(s, E)
    tB = {}
    for order in (180, 300):
        cc = np.array([RA.exact_cc(s, e, order) for e in E]); tB[order] = RA.angular_response(s, E, we, cc[:, 0] + mn_, cc[:, 1] + mb_, sc, 10, 1)['tau']
    mesh = np.array([[[tau_project(r, fd, fm, fnc) for fnc in FN] for fd in FD] for fm in FM])
    flag = 'kernel-limited' if r['dp'] < 20.0 else ('near-threshold' if r['dp'] < 30.0 else '')
    out.append(dict(eos=eos, x=x, T=T, cold=cold, dp=float(r['dp']), dmu_np_T=r['dmuT'], tau_AH=r['tau_AH'], tau_project=tA, tau_referee_exact_180=tB[180], tau_referee_exact_300=tB[300],
                    mesh_min=float(mesh.min()), mesh_max=float(mesh.max()), mesh_shape=list(mesh.shape), mesh_factors=dict(mU=FM.tolist(), dU=FD.tolist(), NC=FN.tolist()), flag=flag, t_s=time.time() - t0))
    lines.append(f" {eos:8s} {x:4.1f} {T:3.1f} {'cold' if cold else 'hot '} | {r['dp']:+5.1f} | {r['tau_AH']:8.3g} | {tA:11.6g} | {tB[180]:12.6g} {tB[300]:9.6g} | {tB[300] / tA:8.5f} | {mesh.min():9.4g} {mesh.max():9.4g} | {flag}")
    print(lines[-1], f"({time.time() - t0:.0f} s)", flush=True)
    json.dump(out, open('results/table1_independent_checks.json', 'w'), indent=0); open('results/table1_independent_checks.txt', 'w').write("\n".join(lines) + "\n")
print('done')
