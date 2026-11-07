"""Neutral-current neutrino-nucleon scattering mean free path in degenerate matter (Iwamoto & Pethick 1982; Reddy, Prakash & Lattimer 1998 Eq. 27):
  1/lambda_N(E) = G_F^2 (cV^2 + 3 cA^2) M_N*^2 T^2 E [pi^2 + ((E - mu_nu)/T)^2] / (16 pi^3 (1 + exp((mu_nu - E)/T)))   (mu_nu = 0 here), M* = Landau mass.
Purpose: (1) reproduce Alford & Harris 2018 Fig. 8 (DD2, E = T: 'several km at 3 MeV; 1 km -> 0.5 km at 5 MeV; few hundred m at 7 MeV'), i.e. show what
their transparency criterion measured; (2) evaluate the same quantity at the energies the Urca processes actually emit (E = 10, 20, 30 MeV).
Neutrons and protons both included (cV_n = -1/2, cA_n = -gA/2; cV_p = 1/2 - 2 sin^2 thW, cA_p = gA/2)."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from rmf_dd2_delta import baryons, beta_eq_cont, kf, N0
from urca_rates import GF, GA, HC
SW2 = 0.231
def inv_lambda_nc(E, T, mst_landau, cV, cA, mu_nu=0.0):
    """1/lambda in 1/m for one species."""
    x = (E - mu_nu) / T
    W = GF ** 2 * (cV ** 2 + 3 * cA ** 2) * mst_landau ** 2 * T ** 2 * E * (np.pi ** 2 + x ** 2) / (16 * np.pi ** 3 * (1 + np.exp(-x)))   # MeV
    return W / HC * 1e15
if __name__ == '__main__':
    out = ["NC scattering mean free path (RPL98 Eq. 27, degenerate), DD2 beta-equilibrated npe(mu) matter, Landau masses. n: neutrons only (as in AH2018 Fig. 8) and n+p.",
           "n/n0  m_n^L  m_p^L |  E=T:  T=3 (n | n+p)   T=5 (n | n+p)   T=7 (n | n+p)   [km]   ||  T=3 MeV, E = 10, 20, 30 MeV (n+p) [m]"]
    B = baryons()[:2]
    for x in (1.0, 2.0, 3.0, 4.0, 6.0):
        n = x * N0; c, ne, nmu, e, p, mu, mst, ier, sol = beta_eq_cont(n, B)
        mn = np.sqrt(kf(c['n'], 2) ** 2 + mst['n'] ** 2); mp = np.sqrt(kf(c['p'], 2) ** 2 + mst['p'] ** 2)
        def lam(E, T, both):
            il = inv_lambda_nc(E, T, mn, -0.5, -GA / 2) + (inv_lambda_nc(E, T, mp, 0.5 - 2 * SW2, GA / 2) if both else 0.0); return 1.0 / il
        s = f"{x:4.1f} {mn:6.0f} {mp:6.0f} | "
        for T in (3.0, 5.0, 7.0): s += f"{lam(T, T, False)/1e3:6.2f} | {lam(T, T, True)/1e3:5.2f}    "
        s += " || " + "  ".join(f"{lam(E, 3.0, True):7.1f}" for E in (10.0, 20.0, 30.0))
        out.append(s); print(s, flush=True)
    open('results/nc_scatter_dd2.txt', 'w').write("\n".join(out) + "\n")
