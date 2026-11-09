"""Checks of nc_scatter.inv_lambda_nc_full (fix round 2026-11-08): (1) static dilute limit sigma = G_F^2 E^2 (hV^2+3hA^2)/pi; (2) degenerate-neutrino
limit -> RPL98 Eq. 27 with half charges and 4 pi^3; (3) quadrature convergence; (4) DD2 2n0, T = 3 MeV values (Fig. 1 state) and the factor vs the old code."""
import sys, json, numpy as np; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from nc_scatter import inv_lambda_nc_full, inv_lambda_nc_np, inv_lambda_nc, HV_N, HA_N, HV_P, HA_P, SW2
from urca_rel import occ_ref, F12
from urca_rates import GF, GA, HC
out = {}
# (1) dilute static: heavy nucleon, nondegenerate
m = 2.0e6; T = 3.0; n = 1e-6   # fm^-3
mu = occ_ref(T, m, n); E = np.array([3.0, 10.0, 25.0])
num = inv_lambda_nc_full(E, T, m, mu, HV_N, HA_N, nq0=192, nc=96)
ana = n * GF ** 2 * E ** 2 * (HV_N ** 2 + 3 * HA_N ** 2) / np.pi * HC ** 2 * 1e15
out['dilute_ratio'] = (num / ana).tolist(); print('dilute static limit ratio (should be 1):', num / ana)
# (2) degenerate neutrinos: blocked rate with mu_nu = E >> T, degenerate nucleons; RPL98 Eq. 27 with half charges and 4 pi^3
mL = 600.0; pF = 300.0; nn = pF ** 3 / (3 * np.pi ** 2) / HC ** 3
for T, Ev in ((1.0, 30.0), (0.5, 30.0), (0.2, 30.0), (0.1, 30.0)):
    mu = occ_ref(T, mL, nn); full = inv_lambda_nc_full(np.array([Ev]), T, mL, mu, HV_N, HA_N, blocked=True, mu_nu=Ev, nq0=256, nc=128, qup=40.0)[0]
    eq27 = 4 * inv_lambda_nc(Ev, T, mL, HV_N, HA_N, mu_nu=Ev)
    out[f'degenerate_ratio_T{T}_E{Ev}'] = full / eq27; print(f'degenerate nu (E = mu_nu = {Ev}, T = {T}, E/T = {Ev/T:.0f}): full/Eq27(4pi^3) = {full/eq27:.5f}')
# (3)+(4) DD2 2n0 T = 3 MeV (Fig. 1 state): cold composition, Landau masses, density-consistent references
import rmf_dd2_delta as R
Bb = R.baryons()[:2]; x = 2.0; T = 3.0; nb = x * R.N0; c, ne, nmu, e, p, mu_, mst, ier, sol = R.beta_eq_cont(nb, Bb)
pn, pp = R.kf(c['n'], 2), R.kf(c['p'], 2); mLn, mLp = np.sqrt(pn ** 2 + mst['n'] ** 2), np.sqrt(pp ** 2 + mst['p'] ** 2)
mun, mup = occ_ref(T, mLn, c['n']), occ_ref(T, mLp, c['p'])
E = np.array([3.0, 10.0, 25.0, 50.0])
base = inv_lambda_nc_np(E, T, mLn, mLp, mun, mup)
fine = inv_lambda_nc_np(E, T, mLn, mLp, mun, mup, nq0=192, nc=128)
blk = inv_lambda_nc_np(E, T, mLn, mLp, mun, mup, blocked=True)
old = inv_lambda_nc(E, T, mLn, -0.5, -GA / 2) + inv_lambda_nc(E, T, mLp, 0.5 - 2 * SW2, GA / 2)
out['dd2_2n0_T3'] = dict(E=E.tolist(), lam_unblocked_m=(1 / base).tolist(), lam_blocked_m=(1 / blk).tolist(), lam_old_code_m=(1 / old).tolist(),
                         conv_rel=(fine / base - 1).tolist(), full_over_old=(base / old).tolist(), mun=mun, mup=mup, mLn=mLn, mLp=mLp)
print('DD2 2n0 T=3: lambda_NC unblocked [m] at E = 3, 10, 25, 50 MeV:', np.round(1 / base, 2), '| blocked:', np.round(1 / blk, 2), '| old code:', np.round(1 / old, 2))
print('   quadrature 96x64 -> 192x128 relative change:', fine / base - 1, '| full/old =', base / old)
json.dump(out, open('results/checks/nc_full_checks.json', 'w'), indent=1)
# angle-vertex variant at the Fig. 1 state (sensitivity)
ang = inv_lambda_nc_np(E, T, mLn, mLp, mun, mup, vertex='angle')
print('angle vertex / constant vertex at E = 3, 10, 25, 50 MeV:', ang / base)
out['dd2_2n0_T3']['angle_over_const'] = (ang / base).tolist()
json.dump(out, open('results/checks/nc_full_checks.json', 'w'), indent=1)
