"""SECOND IMPLEMENTATION of the kinetic damping time (fix round 2026-11-08; replaces the round-5 routine as validation path, Astra audit Sec. 9).
Written independently of the production kernels (urca_rel, nc_scatter.S_free, kinetic_response.admittance):
 * nucleons: EXACT relativistic quasiparticle dispersions E_j(p) = sqrt(p^2 + m*_j(T)^2) + U_j with the THERMAL Dirac masses and the effective chemical
   potentials nu_j of the finite-T DD2 state at the cold composition (finiteT.stateT); occupations f_j = 1/(exp((sqrt(p^2+m*_j^2) - nu_j)/T) + 1)
   integrate to the declared densities by construction; U_n - U_p from mu_n - mu_p = dmu(T) + mu_e(T); massive electrons with mu_e(T).
 * direct Urca (constant vertex G~^2 = G_F^2 cos^2 th_C (1 + 3 g_A^2)): W_abs,nu(E) = G~^2 int d^3p_e/(2pi)^3 (1 - f_e) S_np(E - E_e, |p_nu - p_e|),
   W_abs,nubar(E) = G~^2 int d^3p_e/(2pi)^3 f_e S_pn(E + E_e, |p_nubar + p_e|),  S_ab(q0, q) = (1/(pi q)) int p dp f_a(p) (1 - f_b(k0)) E*_b(k0) [triangle],
   E_b(k0) = E_a(p) + q0 (direct three-dimensional quadrature, no analytic energy integral).
 * neutral current: the same S_jj(q0, q) routine (exact dispersion, thermal occupations), constant vertex hV^2 + 3 hA^2, unblocked total frequency.
 * modified Urca: Fermi-surface rates (murca_spectral, the declared prescription; shared with production) with thermal Landau masses.
 * transport: direct angular quadrature in mu (Gauss-Legendre, 96 nodes) instead of the analytic arctan; C from the pressure identity C = -dp/dn_e.
 * check mode PARABOLIC=1: production dispersions/occupations (Landau masses, occ_ref) in the same quadrature -> must reproduce production W_abs."""
import numpy as np, sys, json, os, time, traceback; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
import rmf_dd2_delta as R, finiteT as FT
from murca_spectral import spectra_mU
from urca_rates import GF, GA, COSC, HC, HBAR
from rmf_dd2_delta import ME
G2T = (GF * COSC) ** 2 * (1 + 3 * GA ** 2); HVN, HAN, HVP, HAP = -0.5, -GA / 2, 0.5 - 2 * 0.231, GA / 2
def fd(x): return 1.0 / (1.0 + np.exp(np.clip(x, -700, 700)))
def nu_from_density(nj, m, T):
    return brentq(lambda nu: FT.fd_integrals(nu, m, T, 2)[0] - nj, m - 60 * T, m + 3000.0, xtol=1e-11, rtol=1e-14)
class Species:
    def __init__(self, mstar=None, nu=None, mL=None, mu_par=None, U=0.0):
        self.mstar, self.nu, self.mL, self.mu_par, self.U = mstar, nu, mL, mu_par, U
        self.parabolic = mL is not None
    def Ekin(self, p):   # energy relative to U (exact: sqrt(p^2+m*^2); parabolic: p^2/(2 mL))
        return p * p / (2 * self.mL) if self.parabolic else np.sqrt(p * p + self.mstar ** 2)
    def occ_arg(self, Ek):  # (E - mu)/T numerator
        return Ek - (self.mu_par if self.parabolic else self.nu)
    def k_of(self, Ek):   # momentum from kinetic energy
        return np.sqrt(np.maximum(2 * self.mL * Ek, 0.0)) if self.parabolic else np.sqrt(np.maximum(Ek * Ek - self.mstar ** 2, 0.0))
    def jac(self, k, Ek):  # dE/dk * ... -> E*(k) factor: k dk = E* dE  (parabolic: mL)
        return self.mL + 0.0 * k if self.parabolic else Ek
    def p_grid(self, T, n=160):
        Ef = (self.mu_par if self.parabolic else self.nu); E_hi = Ef + 45 * T; E_lo = (0.0 if self.parabolic else self.mstar)
        k_hi = self.k_of(E_hi); kF = self.k_of(max(Ef, E_lo + 1e-9)); a = min(kF, self.k_of(min(Ef + 12 * T, E_hi)) - kF + 1e-6)
        segs = [(0.0, max(kF - a, 0.0)), (max(kF - a, 0.0), kF + a), (kF + a, k_hi)]; g, w = leggauss(n // 4); P, W = [], []
        for lo, hi in segs:
            if hi <= lo: continue
            nn = n // 2 if lo == max(kF - a, 0.0) and hi == kF + a else n // 4
            gg, ww = leggauss(nn); P.append(lo + (hi - lo) * (gg + 1) / 2); W.append(ww * (hi - lo) / 2)
        return np.concatenate(P), np.concatenate(W)
def S_ab(q0, q, a, b, T):
    """(1/(pi q)) int p dp f_a(p) (1 - f_b(k0)) E*_b(k0) theta(|p - q| <= k0 <= p + q),  E_b(k0) = E_a(p) + q0 (full energies incl. U)"""
    p, wp = a.p_grid(T); Ea = a.Ekin(p)                                   # shape (np,)
    Eb = Ea[None, :] + (a.U - b.U) + q0[:, None]                           # kinetic energy of b relative to its U
    ok = Eb > (0.0 if b.parabolic else b.mstar)
    k0 = b.k_of(np.where(ok, Eb, 1e9 if not b.parabolic else 0.0)); tri = ok & (k0 >= np.abs(p[None, :] - q[:, None])) & (k0 <= p[None, :] + q[:, None])
    integ = np.where(tri, p[None, :] * fd(a.occ_arg(Ea)[None, :] / T) * fd(-b.occ_arg(Eb) / T) * b.jac(k0, Eb), 0.0)
    return np.sum(integ * wp[None, :], axis=1) / (np.pi * q)
def W_dU(E, nS, pS, mue, T, me=ME, ne_=96, nc=64):
    """W_abs,nu(E) and W_abs,nubar(E) [MeV] (direct Urca)"""
    gE, wE = leggauss(ne_); gc, wc = leggauss(nc); Wn = np.zeros_like(E); Wb = np.zeros_like(E)
    Ee = np.maximum(me * 1.0000001, mue - 45 * T) + (mue + 45 * T - np.maximum(me * 1.0000001, mue - 45 * T)) * (gE + 1) / 2; we = wE * (mue + 45 * T - np.maximum(me, mue - 45 * T)) / 2
    pe = np.sqrt(Ee * Ee - me * me)
    EE, CC = np.meshgrid(Ee, gc, indexing='ij'); PE = np.sqrt(EE * EE - me * me); WW = np.outer(we * pe * Ee, wc)   # p_e^2 dp_e = p_e E_e dE_e
    for i, Ev in enumerate(E):
        q = np.sqrt(np.maximum(Ev * Ev + PE * PE - 2 * Ev * PE * CC, 1e-12)).ravel(); q0 = (Ev - EE).ravel()
        Wn[i] = G2T * np.sum(WW.ravel() * (1 - fd((EE.ravel() - mue) / T)) * S_ab(q0, q, nS, pS, T)) / (4 * np.pi ** 2)
        qb = np.sqrt(np.maximum(Ev * Ev + PE * PE + 2 * Ev * PE * CC, 1e-12)).ravel(); q0b = (Ev + EE).ravel()
        Wb[i] = G2T * np.sum(WW.ravel() * fd((EE.ravel() - mue) / T) * S_ab(q0b, qb, pS, nS, T)) / (4 * np.pi ** 2)
    return Wn, Wb
def sigma_NC(E, sp, hV, hA, T, nq=64, nc=48, qup=35.0):
    """unblocked total NC collision frequency [MeV] on one species, exact dispersion and thermal occupations"""
    g1, w1 = leggauss(nq); gc, wc = leggauss(nc); out = np.zeros_like(E)
    for i, Ev in enumerate(E):
        tot = 0.0
        for lo, hi in ((-qup * T, 0.0), (0.0, Ev)):
            q0 = lo + (hi - lo) * (g1 + 1) / 2; wq = w1 * (hi - lo) / 2; Ep = Ev - q0
            Q0, CC = np.meshgrid(q0, gc, indexing='ij'); EP = Ev - Q0
            q = np.sqrt(np.maximum(Ev * Ev + EP * EP - 2 * Ev * EP * CC, 1e-12))
            S = S_ab(Q0.ravel(), q.ravel(), sp, sp, T).reshape(Q0.shape)
            tot += np.sum(np.outer(wq, wc) * EP ** 2 * S) * 2 * np.pi
        out[i] = GF ** 2 / (8 * np.pi ** 3) * (hV ** 2 + 3 * hA ** 2) * tot
    return out
def tau_direct_angle(E, wE, wn, wb, gn, gb, sig, C, B, K, f_kHz=1.0, L_km=10.0, nmu=96):
    om = 2 * np.pi * f_kHz * 1e3; hw = HBAR * om; hck = HC * 2 * np.pi / (L_km * 1e18); gm, wm = leggauss(nmu)
    def Lam(w, g):
        den = (g + sig - 1j * hw)[:, None] + 1j * hck * gm[None, :]
        A = 0.5 * np.sum(wm[None, :] / den, axis=1)                         # <1/(gamma + sigma - i w + i c k mu)>
        avg = g * A / (1 - sig * A)                                          # <df>/df_eq
        return np.sum(wE * w * g * (1 - avg))
    Lt = Lam(wn, gn) + Lam(wb, gb); zeta = C ** 2 * Lt.real / abs(-1j * hw + B * Lt) ** 2 * HBAR
    return K / (zeta * om ** 2), Lt
def point(args):
    x, T = args; t0 = time.time(); parabolic = os.environ.get('PARABOLIC') == '1'
    try:
        Bb = R.baryons()[:2]; n = x * R.N0; c, ne, nmu, e, p, mu, mst0, ier, sol0 = R.beta_eq_cont(n, Bb); Ye, Ymu = ne / n, nmu / n
        st = FT.stateT(n, Ye, Ymu, T, Bb); gs = R.couplings(n)[0]; s = st['sol'][0]
        mT = dict(n=R.MN - gs * s, p=R.MP - gs * s); dmu, mue = float(st['dmu_e']), float(st['mue'])
        d = 2e-3; pp_ = FT.stateT(n, Ye * (1 + d), Ymu, T, Bb, st['sol']); pm_ = FT.stateT(n, Ye * (1 - d), Ymu, T, Bb, st['sol'])
        C = -(pp_['p'] - pm_['p']) / (2 * d * Ye * n); B = -(pp_['dmu_e'] - pm_['dmu_e']) / (2 * d * Ye * n)
        K = n * (FT.stateT(n * (1 + d), Ye, Ymu, T, Bb, st['sol'])['p'] - FT.stateT(n * (1 - d), Ye, Ymu, T, Bb, st['sol'])['p']) / (2 * d * n)
        pFn, pFp, pFe = R.kf(c['n'], 2), R.kf(c['p'], 2), R.kf(ne, 2)
        if parabolic:
            from urca_rel import occ_ref
            mLn, mLp = np.sqrt(pFn ** 2 + mst0['n'] ** 2), np.sqrt(pFp ** 2 + mst0['p'] ** 2)
            nS = Species(mL=mLn, mu_par=occ_ref(T, mLn, c['n'])); pS = Species(mL=mLp, mu_par=occ_ref(T, mLp, c['p']))
            nS.U, pS.U = 0.0, -(dmu + mue) + (nS.mu_par - pS.mu_par); me = 0.0; mLn_mU, mLp_mU = mLn, mLp
        else:
            nun, nup = nu_from_density(c['n'], mT['n'], T), nu_from_density(c['p'], mT['p'], T)
            nS = Species(mstar=mT['n'], nu=nun); pS = Species(mstar=mT['p'], nu=nup)
            nS.U, pS.U = 0.0, -(dmu + mue) + (nun - nup); me = ME
            mLn_mU, mLp_mU = np.sqrt(pFn ** 2 + mT['n'] ** 2), np.sqrt(pFp ** 2 + mT['p'] ** 2)
        g, w = leggauss(int(os.environ.get('NE2', '96'))); xn = 50.0 * (g + 1) / 2; E = xn * T; wE = w * 25.0 * T
        Wn, Wb = W_dU(E, nS, pS, mue, T, me=me)
        dec, cap, inv_nu, inv_nb = spectra_mU(mLn_mU, mLp_mU, pFn, pFp, pFe, T, E, dmu); WmUn, WmUb = HC * 1e-15 * inv_nu, HC * 1e-15 * inv_nb
        sig = sigma_NC(E, nS, HVN, HAN, T) + sigma_NC(E, pS, HVP, HAP, T)
        fn, fb = fd((E + dmu) / T), fd((E - dmu) / T); wn = E ** 2 * fn * (1 - fn) / (2 * np.pi ** 2 * T * HC ** 3); wb = E ** 2 * fb * (1 - fb) / (2 * np.pi ** 2 * T * HC ** 3)
        gn = (Wn + WmUn) / (1 - fn); gb = (Wb + WmUb) / (1 - fb)
        tau, Lt = tau_direct_angle(E, wE, wn, wb, gn, gb, sig, C, B, K)
        return dict(x=x, T=T, parabolic=parabolic, tau=tau, C=C, B=B, K=K, dmu=dmu, mue=mue, mT=mT, E=E.tolist(), W_dU_nu=Wn.tolist(), W_dU_nubar=Wb.tolist(), sigma=sig.tolist(), t=time.time() - t0)
    except Exception:
        return dict(x=x, T=T, error=traceback.format_exc()[-800:])
if __name__ == '__main__':
    from multiprocessing import Pool
    OUT = os.environ.get('OUT_DIR', 'results'); tag = '_parabolic' if os.environ.get('PARABOLIC') == '1' else ''
    grid = [(0.5, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0), (3.0, 5.0)] + ([(0.5, 4.0), (1.0, 8.0), (2.0, 3.0)] if not tag else [])
    with Pool(int(os.environ.get('NPROC', '5'))) as pool: out = pool.map(point, grid, chunksize=1)
    json.dump(out, open(f'{OUT}/second_implementation{tag}.json', 'w'), indent=0)
    for o in out: print(o['x'], o['T'], o.get('tau', o.get('error')), f"[{o.get('t', 0):.0f} s]", flush=True)
