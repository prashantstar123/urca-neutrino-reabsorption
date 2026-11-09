"""Supplemental Material, Section S1 / Table S1 (DD2, the five Table I states): background state, kinematics and thermodynamic coefficients.
Sources: rmf_dd2_delta (cold beta-equilibrium composition, Dirac masses), finiteT.stateT (finite-T affinity, pressure), results/kinetic_response_fine.json (C, B, K_T
and the affinity used in the kinetic calculation). Checks: step convergence of C, B, K_T (finiteT.suscT, d = 2e-3 vs 1e-3);
thermodynamic identity C = -(dp/dn_e) at fixed n, T, n_mu. Output: results/sm_table_s1.json"""
import numpy as np, sys, json, os; P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(P, 'code')); os.chdir(P)
from scipy.optimize import brentq
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from finiteT import stateT, suscT
B = baryons()[:2]
KF = {(o['x'], o['T']): o for o in json.load(open('results/kinetic_response_fine.json')) if 'error' not in o}
out = []
for x, T in ((0.5, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0), (3.0, 5.0)):
    n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B); Ye, Ymu = ne / n, nmu / n
    pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2); dp = pn - pp - pe
    mLn, mLp = np.hypot(pn, mst['n']), np.hypot(pp, mst['p']); vFp = pp / mLp
    Ep = lambda q: np.hypot(q, mst['p'])
    Emin_lin = vFp * dp / (1 + vFp); Emin_ex = brentq(lambda E: Ep(pp + dp - E) - Ep(pp) - E, 0, dp)
    st = stateT(n, Ye, Ymu, T, B); v = KF[(x, T)]
    conv = {}
    for d in (2e-3, 1e-3):
        A, Cv, s0 = suscT(n, T, B, Ye, Ymu, d=d); conv[d] = (Cv[0], -A[0, 0])
    dd = 2e-3; g = st['sol']
    pplus = stateT(n, Ye * (1 + dd), Ymu, T, B, g)['p']; pminus = stateT(n, Ye * (1 - dd), Ymu, T, B, g)['p']
    C_id = -(pplus - pminus) / (2 * dd * ne)
    out.append(dict(x=x, T=T, n=n, Ye=Ye, Ymu=Ymu, pFn=pn, pFp=pp, pFe=pe, mstar_n=mst['n'], mstar_p=mst['p'], mL_n=mLn, mL_p=mLp, vFp=vFp, dp=dp,
                    Emin_lin=Emin_lin, Emin_exact=Emin_ex, dmu0=float(st['dmu_e']), dmu0_kinetic_file=KF[(x, T)]['dmu_np_T'],
                    C=v['C'], B=v['B'], K_T=v['ndpdn'], C_conv=abs(conv[2e-3][0] / conv[1e-3][0] - 1), B_conv=abs(conv[2e-3][1] / conv[1e-3][1] - 1),
                    C_identity=C_id, C_identity_rel=C_id / v['C'] - 1))
json.dump(dict(N0=N0, rows=out), open('results/sm_table_s1.json', 'w'), indent=1, default=float)
print(f"N0 = {N0} fm^-3")
keys = [('n', '{:.4f}'), ('Ye', '{:.4f}'), ('Ymu', '{:.4f}'), ('pFn', '{:.1f}'), ('pFp', '{:.1f}'), ('pFe', '{:.1f}'), ('mstar_n', '{:.1f}'), ('mstar_p', '{:.1f}'), ('mL_p', '{:.1f}'), ('vFp', '{:.3f}'),
        ('dp', '{:.1f}'), ('Emin_lin', '{:.2f}'), ('Emin_exact', '{:.2f}'), ('dmu0', '{:.3f}'), ('dmu0_kinetic_file', '{:.3f}'), ('C', '{:.2f}'), ('B', '{:.0f}'), ('K_T', '{:.3f}'),
        ('C_conv', '{:.1e}'), ('B_conv', '{:.1e}'), ('C_identity_rel', '{:.1e}')]
print('quantity            ' + ' '.join(f"{o['x']:g}n0/{o['T']:g}".rjust(11) for o in out))
for k, f in keys: print(f'{k:20s}' + ' '.join(f.format(o[k]).rjust(11) for o in out))
