"""Independent route for the central number: brute-force Monte-Carlo phase-space integral of nu_e + n -> p + e with EXACT relativistic quasiparticle
dispersions E*_i = sqrt(p^2 + m*_i^2) (Dirac masses), vector potentials via the chemical-potential bookkeeping E_n + E_nu = E_p + E_e with
E_i = E*_i + U_i, U_n - U_p = mu_e + dmu - (mu*_n - mu*_p); constant matrix element.  No Fermi-surface expansion, no Landau-mass dispersion.
  W(E) = K int d^3p_n dOmega_e  p_e^2 / |g'(p_e)|  f_n (1 - f_p) (1 - f_e) |_{root of g},   g(p_e) = E*_n + E + mu_e - mu*_n + mu*_p - E*_p(p_n + p_nu - p_e e) - p_e
K is fixed ONCE from the published degenerate formula (RPL98 Eq. 26) at an open-threshold point with E = 3 MeV << mu (where that formula is exact);
everything else is a prediction: (a) open threshold at E = 30, 60 MeV (tests the kernel's high-E deviation 0.93/0.87 from RPL98),
(b) DD2 2 n0 nucleonic (closed threshold, dp = +49 MeV) at T = 2, 3, 5 MeV and E = 10, 20, 30 MeV versus the Landau-mass kernel (urca_rel.spectra)."""
import numpy as np, sys, time; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from urca_rel import spectra
from urca_fullT import GT2
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
HC = 197.3269804
def fF(x): return 1.0 / (1.0 + np.exp(np.clip(x, -700, 700)))
def mc_W(E, T, mstn, mstp, pFn, pFp, mue, dmu=0.0, N=4_000_000, Delta=90.0, seed=0, chunks=5):
    rng = np.random.default_rng(seed); mun_s = np.sqrt(pFn ** 2 + mstn ** 2); mup_s = np.sqrt(pFp ** 2 + mstp ** 2); tot = 0.0; tot2 = 0.0; M = 0
    lo = max(0.0, pFn - Delta); hi = pFn + Delta
    for _ in range(chunks):
        pn = (lo ** 3 + (hi ** 3 - lo ** 3) * rng.random(N)) ** (1 / 3)               # uniform in volume of the shell
        cn = 2 * rng.random(N) - 1; sn = np.sqrt(1 - cn ** 2); phn = 2 * np.pi * rng.random(N)
        pnv = np.stack([pn * sn * np.cos(phn), pn * sn * np.sin(phn), pn * cn], 1)
        ce = 2 * rng.random(N) - 1; se = np.sqrt(1 - ce ** 2); phe = 2 * np.pi * rng.random(N); ev = np.stack([se * np.cos(phe), se * np.sin(phe), ce], 1)
        q0 = pnv + np.array([0.0, 0.0, E])                                           # p_n + p_nu (nu along z)
        Esn = np.sqrt(pn ** 2 + mstn ** 2); const = Esn + E + mue + dmu - mun_s + mup_s
        def g(pe):
            pp = q0 - pe[:, None] * ev; Esp = np.sqrt(np.sum(pp ** 2, 1) + mstp ** 2); return const - Esp - pe, pp, Esp
        a = np.zeros(N); b = np.full(N, 3000.0)
        for it in range(50):
            m = 0.5 * (a + b); gm, _, _ = g(m); pos = gm > 0; a = np.where(pos, m, a); b = np.where(pos, b, m)
        pe = 0.5 * (a + b); gm, pp, Esp = g(pe); gprime = np.abs(np.sum(pp * ev, 1) / Esp - 1.0)
        F = fF((Esn - mun_s) / T) * (1 - fF((Esp - mup_s) / T)) * (1 - fF((pe - mue) / T))
        w = pe ** 2 / np.maximum(gprime, 1e-6) * F; vol = 4 * np.pi / 3 * (hi ** 3 - lo ** 3) * 4 * np.pi      # d^3p_n volume x dOmega_e
        tot += w.sum() * vol; tot2 += (w ** 2).sum() * vol ** 2; M += N
    mean = tot / M; err = np.sqrt(max(tot2 / M - mean ** 2, 0) / M); return mean, err
if __name__ == '__main__':
    out = []
    def say(s): print(s, flush=True); out.append(s)
    mst, pFn, pFp, pFe, mue = 700.0, 380.0, 250.0, 250.0, 250.0; T = 3.0
    mn = np.sqrt(pFn ** 2 + mst ** 2); mp = np.sqrt(pFp ** 2 + mst ** 2)
    rpl = lambda E: GT2 * mn * mp * (mue + E) * (E ** 2 + np.pi ** 2 * T ** 2) / (4 * np.pi ** 3 * (1 + np.exp(-E / T)))
    t0 = time.time(); I3, e3 = mc_W(3.0, T, mst, mst, pFn, pFp, mue); K = rpl(3.0) / I3
    say(f"calibration at open threshold, E=3 MeV, T=3: MC integral {I3:.4e} +- {e3/I3*100:.1f}%  -> K fixed to RPL98 ({time.time()-t0:.0f}s)")
    for E in (10.0, 30.0, 60.0):
        I, e = mc_W(E, T, mst, mst, pFn, pFp, mue); W = K * I; S = spectra(T, mst, mst, pFn, pFp, mue, 0.0, np.array([E / T]))['W_abs_nu'][0]
        say(f"  open threshold E={E:4.0f}: MC/RPL98 = {W/rpl(E):.4f} (+-{e/I*100:.1f}%), kernel/RPL98 = {S/rpl(E):.4f}, MC/kernel = {W/S:.4f}")
    B = baryons()[:2]; n = 2.0 * N0; c, ne, nmu, e_, p_, mu, mst_, ier, sol = beta_eq_cont(n, B); pn, pp, pe = kf(c['n'], 2), kf(c['p'], 2), kf(ne, 2)
    say(f"DD2 2 n0 nucleonic: pFn {pn:.1f} pFp {pp:.1f} pFe {pe:.1f} (dp = {pn-pp-pe:+.1f} MeV), m*_n {mst_['n']:.1f} m*_p {mst_['p']:.1f}")
    for T in (2.0, 3.0, 5.0):
        for E in (10.0, 20.0, 30.0):
            t0 = time.time(); I, e = mc_W(E, T, mst_['n'], mst_['p'], pn, pp, pe, N=4_000_000, Delta=120.0, chunks=8); W = K * I
            S = spectra(T, mst_['n'], mst_['p'], pn, pp, pe, 0.0, np.array([E / T]))['W_abs_nu'][0]
            say(f"  T={T:3.0f} E={E:4.0f} MeV: lambda_MC = {HC/W*1e-15:9.3g} m (+-{e/I*100:.0f}%)   lambda_kernel = {HC/S*1e-15:9.3g} m   MC/kernel = {W/S:.3f}  [{time.time()-t0:.0f}s]")
    open('results/mc_absorption.txt', 'w').write("\n".join(out) + "\n")
