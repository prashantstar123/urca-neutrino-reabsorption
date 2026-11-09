"""ONE common finite-temperature rate-state interface for every production script (fix round 2026-11-08; Astra audit Sec. 10 item 1).
Background: cold beta-equilibrium composition of the EOS (DD2 split masses; IUFSU/BigApple common M = 939 MeV), muon fraction fixed, finite-T EOS
state at that composition (affinity dmu(T), electron chemical potential mu_e(T)), finite-T coefficients C, B, K_T (fixed Y_e, Y_mu).
Rates: direct Urca = urca_rel kernels (Landau masses, constant vertex) with DENSITY-CONSISTENT finite-T occupation references (urca_rel.occ_ref);
modified Urca = Fermi-surface prescription (murca_spectral); neutral current = full nonrelativistic free-gas response (nc_scatter.inv_lambda_nc_np,
constant vertex, Landau masses, the same references), unblocked total collision frequency for the isotropic elastic operator.
Functions: background(eos, x, T), kinetic_inputs(eos, x, T), net_rate(eos, n, Ye, Ymu, T) (free-escape F = Gamma_dec - Gamma_cap)."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from numpy.polynomial.legendre import leggauss
import rmf_dd2_delta as R, rmf_fsu_delta as F, finiteT as FT, finiteT_fsu as FTF
from urca_rel import spectra, occ_ref, gamma_nd_2m, gamma_ec_2m
from murca_spectral import spectra_mU, rates as rates_mU
from nc_scatter import inv_lambda_nc_np
from urca_rates import HC, HBAR
CONV = 1.0 / (HBAR * HC ** 3)
def fF(x): return 1.0 / (1.0 + np.exp(np.clip(x, -700, 700)))
def kf(n): return R.kf(n, 2)
def _eos(eos):
    if eos == 'DD2': return R, R.baryons()[:2], R.N0
    F.use(eos); return F, FTF.baryons939(), F.P['N0']
def finite_T_state(eos, n, Ye, Ymu, T):
    if eos == 'DD2': return FT.stateT(n, Ye, Ymu, T, R.baryons()[:2])
    return FTF.stateT(n, Ye, Ymu, T)
def cold_masses(eos, n, Ye, Ymu):
    """Dirac effective masses of the T = 0 EOS at the given composition (the declared masses of the rate model)"""
    if eos == 'DD2':
        B = R.baryons()[:2]; st = FT.stateT(n, Ye, Ymu, 0.0, B); gs = R.couplings(n)[0]
        return dict(n=R.MN - gs * st['sol'][0], p=R.MP - gs * st['sol'][0])
    return FTF.stateT(n, Ye, Ymu, 1e-4)['mst']
def coefficients(eos, n, Ye, Ymu, T, d=2e-3):
    if eos == 'DD2':
        B = R.baryons()[:2]; A, Cv, st0 = FT.suscT(n, T, B, Ye, Ymu)
        K = n * (FT.stateT(n * (1 + d), Ye, Ymu, T, B, st0['sol'])['p'] - FT.stateT(n * (1 - d), Ye, Ymu, T, B, st0['sol'])['p']) / (2 * d * n)
        return dict(C=Cv[0], B=-A[0, 0], K=K, st=st0)
    co = FTF.suscT(n, T, Ye, Ymu); return dict(C=co['C'], B=co['B'], K=co['K'], st=co['st0'])
def background(eos, x, T):
    mod, B, n0 = _eos(eos); n = x * n0
    c, ne, nmu, e, p, mu, mst, ier, sol = mod.beta_eq_cont(n, B); assert ier == 1, (eos, x)
    Ye, Ymu = ne / n, nmu / n; co = coefficients(eos, n, Ye, Ymu, T); st = co['st']
    pn, pp, pe = kf(c['n']), kf(c['p']), kf(ne); mLn, mLp = np.sqrt(pn ** 2 + mst['n'] ** 2), np.sqrt(pp ** 2 + mst['p'] ** 2)
    refs = (occ_ref(T, mLn, c['n']), occ_ref(T, mLp, c['p']))
    return dict(eos=eos, x=x, T=T, n=n, n0=n0, Ye=Ye, Ymu=Ymu, nn=c['n'], np=c['p'], ne=ne, nmu=nmu, pFn=pn, pFp=pp, pFe=pe, dp=pn - pp - pe,
                mstn=mst['n'], mstp=mst['p'], mLn=mLn, mLp=mLp, refs=refs, dmuT=float(st['dmu_e']), mueT=float(st['mue']), C=co['C'], B=co['B'], K=co['K'])
def kinetic_inputs(eos, x, T, ngrid=128, emax_T=50.0, b=None):
    b = b or background(eos, x, T); T = b['T']
    xg, xw = leggauss(ngrid); xn = emax_T * (xg + 1) / 2; wx = xw * emax_T / 2; E = xn * T; wE = wx * T
    S = spectra(T, b['mstn'], b['mstp'], b['pFn'], b['pFp'], b['mueT'], b['dmuT'], xn, refs=b['refs'])
    dec_mU, cap_mU, inv_mU_nu, inv_mU_nb = spectra_mU(b['mLn'], b['mLp'], b['pFn'], b['pFp'], b['pFe'], T, E, b['dmuT'])
    sig = HC * 1e-15 * inv_lambda_nc_np(E, T, b['mLn'], b['mLp'], b['refs'][0], b['refs'][1])
    fn = fF((E + b['dmuT']) / T); fb = fF((E - b['dmuT']) / T)
    wn = E ** 2 * fn * (1 - fn) / (2 * np.pi ** 2 * T * HC ** 3); wb = E ** 2 * fb * (1 - fb) / (2 * np.pi ** 2 * T * HC ** 3)
    return dict(b, xn=xn, E=E, wE=wE, wn=wn, wb=wb, fn=fn, fb=fb, WdUn=S['W_abs_nu'], WdUb=S['W_abs_nubar'], dGec=S['dGec'], dGnd=S['dGnd'],
                WmUn=HC * 1e-15 * inv_mU_nu, WmUb=HC * 1e-15 * inv_mU_nb, dec_mU=dec_mU, cap_mU=cap_mU, sig=sig)
def net_rate(eos, n, Ye, Ymu, T):
    """free-escape net rate F = Gamma_dec - Gamma_cap [fm^-3 s^-1] (direct + modified Urca) at the state (n, Ye, Ymu, T) with state-dependent inputs"""
    st = finite_T_state(eos, n, Ye, Ymu, T); dmu, mue = float(st['dmu_e']), float(st['mue']); mst = cold_masses(eos, n, Ye, Ymu)
    npr = n * (Ye + Ymu); nn = n - npr; pn, pp, pe = kf(nn), kf(npr), kf(n * Ye)
    mLn, mLp = np.sqrt(pn ** 2 + mst['n'] ** 2), np.sqrt(pp ** 2 + mst['p'] ** 2); refs = (occ_ref(T, mLn, nn), occ_ref(T, mLp, npr))
    a = (T, mst['n'], mst['p'], pn, pp, mue, dmu)
    Gd_dU = gamma_nd_2m(*a, refs=refs) * CONV; Gc_dU = gamma_ec_2m(*a, refs=refs) * CONV
    Gd_mU, Gc_mU = rates_mU(mLn, mLp, pn, pp, pe, T, dmu)
    return (Gd_dU + Gd_mU) - (Gc_dU + Gc_mU), dict(dmu=dmu, mue=mue, Gd_dU=Gd_dU, Gc_dU=Gc_dU, Gd_mU=Gd_mU, Gc_mU=Gc_mU, p=float(st['p']))
