"""Partial (strong) equilibrium state at fixed baryon density n and lepton fraction Y_L (delta_urca_bulkvisc).
During an oscillation the weak (Urca) processes are slow, so the lepton fraction Y_L = (n_e + n_mu)/n is frozen, while strong processes keep
the Deltas in equilibrium with the nucleons: mu_D- = 2 mu_n - mu_p, mu_D0 = mu_n, mu_D+ = mu_p, mu_D++ = 2 mu_n ... (2 mu_p - mu_n).
Unknowns: sigma, omega, rho (MeV), mu_n, mu_p; mu_e from Y_L (mu_mu = mu_e assumed); equations: three field equations, baryon number,
charge neutrality.  Output: the beta disequilibrium delta mu = mu_n - mu_p - mu_e and the thermodynamic state.
Susceptibilities for the bulk viscosity:  B = d(delta mu)/d n_L at fixed n  (n_L = Y_L n),  C = n d(delta mu)/d n at fixed Y_L."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from scipy.optimize import fsolve, brentq
from rmf_dd2_delta import couplings, ns_of, thermo, baryons, beta_eq, kf, ME, MMU, MS, MW, MR, MPHI, HC, N0
def lepton_density(mue):
    ne = 2 * (mue ** 2 - ME ** 2) ** 1.5 / (6 * np.pi ** 2) / HC ** 3 if mue > ME else 0.0
    nmu = 2 * (mue ** 2 - MMU ** 2) ** 1.5 / (6 * np.pi ** 2) / HC ** 3 if mue > MMU else 0.0
    return ne, nmu
def mue_of_nL(nL):
    return brentq(lambda m: sum(lepton_density(m)) - nL, ME + 1e-9, 2000.0)
HYP_EQ = {'L': lambda mun, mup: mun, 'S-': lambda mun, mup: 2 * mun - mup, 'S0': lambda mun, mup: mun, 'S+': lambda mun, mup: mup, 'X-': lambda mun, mup: 2 * mun - mup, 'X0': lambda mun, mup: mun}
def state(n, YL, B, guess=None, sol_beta=None, strange='frozen', frozen=None):
    """strange = 'frozen': hyperon densities held at the beta-equilibrium values (weak strangeness changes are slow); 'eq': hyperons follow
    the strong-equilibrium relations (mu_L = mu_n, ...).  frozen = dict of hyperon densities (from beta equilibrium) when strange == 'frozen'."""
    mue = mue_of_nL(YL * n); ne, nmu = lepton_density(mue); gs, gw, gr, dgs, dgw, dgr = couplings(n)
    def mu_of(nm, mun, mup):   # strong-equilibrium chemical potentials from the two independent ones (baryon number and charge)
        if nm in HYP_EQ: return HYP_EQ[nm](mun, mup)
        return {'n': mun, 'p': mup, 'D-': 2 * mun - mup, 'D0': mun, 'D+': mup, 'D++': 2 * mup - mun}[nm]
    def dens(v):
        s, w, r, ph, mun, mup = v; SigR = 0.0
        for it in range(80):
            nb = []; ns_tot = 0.0; nw = 0.0; nr = 0.0; nph = 0.0
            for (nm, m, t3, q, g, xs, xw, xr, *rest) in B:
                xp = rest[0] if rest else 0.0; ms = m - xs * gs * s
                if strange == 'frozen' and nm in HYP_EQ and frozen is not None:
                    ni = frozen.get(nm, 0.0); k = kf(ni, g)
                else:
                    estar = mu_of(nm, mun, mup) - xw * gw * w - xr * gr * t3 * r - xp * gw * ph - SigR
                    k = np.sqrt(estar ** 2 - ms ** 2) if estar > ms else 0.0; ni = g * k ** 3 / (6 * np.pi ** 2) / HC ** 3
                nb.append(ni); ns_tot += xs * ns_of(k, ms, g); nw += xw * ni; nr += xr * t3 * ni; nph += xp * ni
            SigR_new = dgw * w * nw - dgs * s * ns_tot + dgr * r * nr + dgw * ph * nph
            if abs(SigR_new - SigR) < 1e-9: SigR = SigR_new; break
            SigR = 0.5 * SigR + 0.5 * SigR_new
        return nb, ns_tot, nw, nr, nph
    def eqs(v):
        s, w, r, ph, mun, mup = v; nb, ns_tot, nw, nr, nph = dens(v)
        return [MS ** 2 * s / HC ** 3 - gs * ns_tot, MW ** 2 * w / HC ** 3 - gw * nw, MR ** 2 * r / HC ** 3 - gr * nr, MPHI ** 2 * ph / HC ** 3 - gw * nph, (sum(nb) - n) * 100,
                (sum(q * ni for ni, (nm, m, t3, q, g, xs, xw, xr, *rest) in zip(nb, B)) - ne - nmu) * 100]
    if guess is None:
        c, ne0, nmu0, e0, p0, mu0, mst0, ier0, sol0 = beta_eq(n, B, sol_beta); guess = np.array([sol0[0], sol0[1], sol0[2], sol0[3] if len(sol0) == 6 else 0.0, sol0[-2], sol0[-2] - sol0[-1]])
    sol, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-12, maxfev=4000)
    nb, *_ = dens(sol); e, p, mu, mst, *_ = thermo(nb, B, (ne, nmu)); names = [b[0] for b in B]
    return dict(zip(names, nb)), ne, nmu, e, p, mu, mst, sol[-2] - sol[-1] - mue, ier, sol
def susceptibilities(n, B, YL=None, sol_beta=None, strange='frozen'):
    """At the beta-equilibrium Y_L (unless given): B = d(dmu)/d n_L |_n, C = n d(dmu)/d n |_YL, both with strong Delta equilibrium."""
    c, ne, nmu, e, p, mu, mst, ier, s = beta_eq(n, B, sol_beta)
    if YL is None: YL = (ne + nmu) / n
    fr = {k: v for k, v in c.items() if k in HYP_EQ} if strange == 'frozen' else None
    kw = dict(strange=strange, frozen=fr)
    st0 = state(n, YL, B, sol_beta=sol_beta, **kw); g0 = st0[-1]; d = 2e-3
    dmu_p = state(n, YL * (1 + d), B, g0, **kw)[7]; dmu_m = state(n, YL * (1 - d), B, g0, **kw)[7]; Bq = (dmu_p - dmu_m) / (2 * d * YL * n)
    frn = {k: v * (1 + d) for k, v in fr.items()} if fr else None; frm = {k: v * (1 - d) for k, v in fr.items()} if fr else None     # frozen hyperon FRACTIONS scale with n under compression
    dmu_np = state(n * (1 + d), YL, B, g0, strange=strange, frozen=frn)[7]; dmu_nm = state(n * (1 - d), YL, B, g0, strange=strange, frozen=frm)[7]; Cq = n * (dmu_np - dmu_nm) / (2 * d * n)
    return Bq, Cq, YL, st0
if __name__ == '__main__':
    for xs, xw in ((1.1, 1.1), (1.2, 1.1)):
        B = baryons(xs=xs, xw=xw); sol = None
        print(f"x_sD={xs}, x_wD={xw}: strong-equilibrated susceptibilities at beta equilibrium (dmu at the beta point should be ~0)")
        print("   n/n0   Y_L      dmu(beta) [MeV]   B [MeV fm^3]   C [MeV]    Y_D-     Y_p")
        for x in np.arange(0.5, 4.01, 0.25):
            n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq(n, B, sol); YL = (ne + nmu) / n
            Bq, Cq, YL, st0 = susceptibilities(n, B, YL, sol)
            print(f"   {x:4.2f}  {YL:.4f}   {st0[7]:+9.4f}        {Bq:9.1f}     {Cq:7.2f}    {st0[0]['D-']/n:.4f}  {st0[0]['p']/n:.4f}  {'' if st0[8]==1 else 'NOCONV'}")
