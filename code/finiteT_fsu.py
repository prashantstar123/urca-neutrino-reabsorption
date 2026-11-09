"""Finite-temperature FSU-type nonlinear RMF (IUFSU, BigApple) at fixed (n, Y_e, Y_mu, T), nucleons only, with the COMMON nucleon mass
M = 939 MeV of the published fits (Fattoyev et al. PRC 82, 055803 (2010), Table I caption; Fattoyev et al. PRC 102, 065805 (2020), Table I caption).
Fix round 2026-11-08 (six-agent audit eos_thermo A1/A2 and Astra Sec. 7): the alternative-EOS controls now use the published mass and the DD2
prescription (finite-T coefficients C, B, K_T and finite-T affinity at the cold beta-equilibrium composition).
Lagrangian and mean-field equations as in rmf_fsu_delta.py (Todd-Rutel & Piekarewicz 2005 convention). At fixed nucleon densities the vector fields
depend only on the densities; the scalar field uses the finite-T scalar densities with effective chemical potentials nu_j solved from n_j at T.
mu_j = nu_j + W + I3_j B;  p = sum_j p_kin,j(T) + p_mesons + p_leptons(T),
p_mesons = [-1/2 ms2 Phi^2 - kappa/6 Phi^3 - lambda/24 Phi^4 + 1/2 mv2 W^2 + zeta/24 W^4 + 1/2 mr2 B^2 + Lambda_v W^2 B^2]/HC^3."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from scipy.optimize import brentq
from scipy.integrate import quad
import rmf_fsu_delta as F
from finiteT import fd_integrals, mu_lepton
from rmf_dd2_delta import ME, MMU, HC
M939 = 939.0
def baryons939():
    return [('n', M939, -1.0, 0, 2, 1, 1, 1, 0.0, 0), ('p', M939, 1.0, 1, 2, 1, 1, 1, 0.0, 0)]
def _n_of_nu(nu, m, T):
    if T < 1e-3: return (max(nu * nu - m * m, 0.0) ** 1.5) / (3 * np.pi ** 2) / HC ** 3
    kmax = np.sqrt(max((nu + 40 * T) ** 2 - m * m, 0.0)) + 40 * T; pts = [np.sqrt(nu * nu - m * m)] if nu > m else None
    f = lambda k: k * k / (np.exp(min((np.sqrt(k * k + m * m) - nu) / T, 700.0)) + 1.0)
    return quad(f, 0.0, kmax, points=pts, limit=200, epsabs=0.0, epsrel=1e-12)[0] / np.pi ** 2 / HC ** 3
def _nu_of_n(nj, m, T):
    if T < 1e-3: return np.sqrt((3 * np.pi ** 2 * nj) ** (2 / 3) * HC ** 2 + m * m)
    return brentq(lambda nu: _n_of_nu(nu, m, T) - nj, m - 60 * T, m + 2000.0, xtol=1e-12, rtol=1e-15)
def vector_fields(nn, npr):
    mv2, mr2 = F.P['mv'] ** 2 / F.P['gv2'], F.P['mr'] ** 2 / F.P['gr2']; zet, Lv = F.P['zet'], F.P['Lv']
    nw = (nn + npr) * HC ** 3; nr = 0.5 * (npr - nn) * HC ** 3
    Bof = lambda W: nr / (mr2 + 2 * Lv * W ** 2); G = lambda W: mv2 * W + zet / 6 * W ** 3 + 2 * Lv * Bof(W) ** 2 * W - nw
    W = brentq(G, 0.0, nw / mv2 * 1.0000001, xtol=1e-13) if nw > 0 else 0.0
    return W, Bof(W)
def stateT(n, Ye, Ymu, T, B=None, guess=None):
    """fixed n, Y_e, Y_mu, T. Returns dict(comp, mue, dmu_e, p, mst, nu, Phi, W, Bf, mun, mup, sol) (same keys as finiteT.stateT where used)."""
    npr = n * (Ye + Ymu); nn = n - npr; ne, nmu = Ye * n, Ymu * n
    ms2 = F.P['ms'] ** 2 / F.P['gs2']; kap, lam = F.P['kap'], F.P['lam']
    def scal(ph):
        m = M939 - ph; nun, nup = _nu_of_n(nn, m, T), _nu_of_n(npr, m, T)
        ns = fd_integrals(nun, m, T, 2)[1] + fd_integrals(nup, m, T, 2)[1]
        return (ms2 * ph + 0.5 * kap * ph ** 2 + lam / 6 * ph ** 3) / HC ** 3 - ns
    ph = brentq(scal, 1e-6, 0.95 * M939, xtol=1e-11)
    m = M939 - ph; nun, nup = _nu_of_n(nn, m, T), _nu_of_n(npr, m, T); W, Bf = vector_fields(nn, npr)
    mun, mup = nun + W - 0.5 * Bf, nup + W + 0.5 * Bf
    mue = mu_lepton(ne, ME, T) if T >= 1e-3 else np.sqrt((3 * np.pi ** 2 * ne) ** (2 / 3) * HC ** 2 + ME ** 2)
    mumu = (mu_lepton(nmu, MMU, T) if T >= 1e-3 else np.sqrt((3 * np.pi ** 2 * nmu) ** (2 / 3) * HC ** 2 + MMU ** 2)) if nmu > 0 else MMU
    mv2, mr2 = F.P['mv'] ** 2 / F.P['gv2'], F.P['mr'] ** 2 / F.P['gr2']; zet, Lv = F.P['zet'], F.P['Lv']
    pmes = (-0.5 * ms2 * ph ** 2 - kap / 6 * ph ** 3 - lam / 24 * ph ** 4 + 0.5 * mv2 * W ** 2 + zet / 24 * W ** 4 + 0.5 * mr2 * Bf ** 2 + Lv * W ** 2 * Bf ** 2) / HC ** 3
    pk = fd_integrals(nun, m, T, 2)[3] + fd_integrals(nup, m, T, 2)[3]
    pl = fd_integrals(mue, ME, T, 2)[3] + (fd_integrals(mumu, MMU, T, 2)[3] if nmu > 0 else 0.0)
    return dict(comp=dict(n=nn, p=npr), mue=mue, mumu=mumu, dmu_e=mun - mup - mue, p=pk + pmes + pl, mst=dict(n=m, p=m), nu=dict(n=nun, p=nup),
                Phi=ph, W=W, Bf=Bf, mun=mun, mup=mup, sol=None)
def suscT(n, T, Ye, Ymu, d=2e-3):
    """finite-T coefficients at fixed Y_mu: B = -d dmu/d n_e (fixed n, n_mu), C = n d dmu/dn (fixed Y_e, Y_mu), K_T = n dp/dn (fixed Y_e, Y_mu)"""
    st0 = stateT(n, Ye, Ymu, T)
    Bq = -(stateT(n, Ye * (1 + d), Ymu, T)['dmu_e'] - stateT(n, Ye * (1 - d), Ymu, T)['dmu_e']) / (2 * d * Ye * n)
    sp, sm = stateT(n * (1 + d), Ye, Ymu, T), stateT(n * (1 - d), Ye, Ymu, T)
    C = n * (sp['dmu_e'] - sm['dmu_e']) / (2 * d * n); K = n * (sp['p'] - sm['p']) / (2 * d * n)
    return dict(B=Bq, C=C, K=K, st0=st0, C_identity=-(stateT(n, Ye * (1 + d), Ymu, T)['p'] - stateT(n, Ye * (1 - d), Ymu, T)['p']) / (2 * d * Ye * n))
if __name__ == '__main__':
    import json, time
    out = []
    for eos in ('IUFSU', 'BigApple'):
        F.use(eos); B = baryons939(); n0 = F.P['N0']
        for x in (0.5, 1.0, 1.5, 2.0, 3.0):
            t0 = time.time(); n = x * n0; c, ne, nmu, e, p, mu, mst, ier, sol = F.beta_eq_cont(n, B); assert ier == 1
            Ye, Ymu = ne / n, nmu / n
            a_cold, p_cold, _ = F.state_fixed(n, ne, nmu, B)                       # independent cold route (rmf_fsu_delta)
            s0 = stateT(n, Ye, Ymu, 1e-4)                                            # T -> 0 limit of this module
            row = dict(eos=eos, x=x, Ye=Ye, Ymu=Ymu, mst_cold=mst['n'], mstT0=s0['mst']['n'], dmu_T0=s0['dmu_e'], dmu_cold_route=a_cold, p_T0=s0['p'], p_cold_route=p_cold)
            for T in (3.0, 4.0, 5.0):
                co = suscT(n, T, Ye, Ymu); row[f'T{T:g}'] = dict(dmu=co['st0']['dmu_e'], C=co['C'], C_identity=co['C_identity'], B=co['B'], K=co['K'], mst=co['st0']['mst']['n'])
            out.append(row); print(eos, x, f"Ye {Ye:.5f} | T->0: dmu {s0['dmu_e']:.2e} (cold route {a_cold:.2e}), p {s0['p']:.6f} vs {p_cold:.6f}, m* {s0['mst']['n']:.3f} vs {mst['n']:.3f} |",
                                   " ".join(f"T{T:g}: dmu {row[f'T{T:g}']['dmu']:.3f} C {row[f'T{T:g}']['C']:.3f} (id {row[f'T{T:g}']['C_identity']:.3f}) B {row[f'T{T:g}']['B']:.1f} K {row[f'T{T:g}']['K']:.3f}" for T in (3.0, 4.0, 5.0)), f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open('results/checks/finiteT_fsu_validation.json', 'w'), indent=1)
