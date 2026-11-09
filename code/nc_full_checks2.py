import sys, json, numpy as np; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from nc_scatter import inv_lambda_nc_full, inv_lambda_nc, HV_N, HA_N
from urca_rel import occ_ref
from urca_rates import GF, HC
out = {}
# dilute nondegenerate limit with moderate mass (recoil/thermal motion small): resolution study
m = 2.0e4; T = 3.0; n = 1e-6; mu = occ_ref(T, m, n); E = np.array([3.0, 10.0, 25.0])
ana = n * GF ** 2 * E ** 2 * (HV_N ** 2 + 3 * HA_N ** 2) / np.pi * HC ** 2 * 1e15
for nq in (96, 384, 1536):
    r = inv_lambda_nc_full(E, T, m, mu, HV_N, HA_N, nq0=nq, nc=64) / ana; out[f'dilute_m2e4_nq{nq}'] = r.tolist(); print(f'dilute (m = 2e4 MeV) nq0 = {nq}: ratio', np.round(r, 5))
# degenerate-neutrino limit with E << pF (q/pF -> 0) and E/T large
mL = 600.0; pF = 300.0; nn = pF ** 3 / (3 * np.pi ** 2) / HC ** 3
for T, Ev, nq, ncc in ((0.1, 3.0, 512, 512), (0.03, 3.0, 768, 768), (0.01, 3.0, 1024, 1024)):
    mu = occ_ref(T, mL, nn); full = inv_lambda_nc_full(np.array([Ev]), T, mL, mu, HV_N, HA_N, blocked=True, mu_nu=Ev, nq0=nq, nc=ncc, qup=40.0)[0]
    eq27 = 4 * inv_lambda_nc(Ev, T, mL, HV_N, HA_N, mu_nu=Ev); out[f'degenerate_E{Ev}_T{T}'] = full / eq27
    print(f'degenerate nu, E = mu_nu = {Ev} MeV << pF, T = {T} (E/T = {Ev/T:.0f}): full/Eq27(half charges, 4 pi^3) = {full/eq27:.5f}')
json.dump(out, open('results/checks/nc_full_checks2.json', 'w'), indent=1)
