"""Route check of the absorption kernel urca_rel.spectra (W_abs_nu) against the published degenerate charged-current formula
(Reddy, Prakash & Lattimer PRD 58, 013009 (1998) Eq. 26; same phase space as the Lattimer+1991 dUrca emissivity 457 pi/10080):
  1/lambda(E) = G^2 (1+3 gA^2) m*_n m*_p E_e (E^2 + pi^2 T^2) Xi / (4 pi^3 (1 + exp(-E/T)))  with E_e = mu_e + E, mu_nu = 0,
  Xi = 1 above the dUrca threshold (pFp + pFe > pFn + p_nu).  Synthetic open-threshold point; also the emissivity sum rule."""
import numpy as np, sys; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from urca_rel import spectra, mfp_of_W_km
from urca_fullT import GT2
from numpy.polynomial.legendre import leggauss
HC = 197.3269804
mst, pFn, pFp, pFe, mue = 700.0, 380.0, 250.0, 250.0, 250.0     # open: pFp + pFe = 500 > pFn
for T in (1.0, 3.0, 5.0):
    Es = np.array([3.0, 10.0, 30.0, 60.0]); x = Es / T
    S = spectra(T, mst, mst, pFn, pFp, mue, 0.0, x)
    W = S['W_abs_nu']                                   # MeV
    mn = np.sqrt(pFn ** 2 + mst ** 2); mp = np.sqrt(pFp ** 2 + mst ** 2)   # Landau masses used by the kernel
    Wdeg = GT2 * mn * mp * (mue + Es) * (Es ** 2 + np.pi ** 2 * T ** 2) / (4 * np.pi ** 3 * (1 + np.exp(-Es / T)))
    print(f"T={T:3.0f} MeV: E [MeV]      " + " ".join(f"{e:9.1f}" for e in Es))
    print(f"            kernel 1/lambda [1/m] " + " ".join(f"{w/HC*1e15:9.3g}" for w in W))
    print(f"            RPL98  1/lambda [1/m] " + " ".join(f"{w/HC*1e15:9.3g}" for w in Wdeg))
    print(f"            ratio kernel/RPL98    " + " ".join(f"{a/b:9.4f}" for a, b in zip(W, Wdeg)))
# emissivity sum rule: eps_nu = int d^3p/(2pi)^3 E j(E), j = W e^{-E/T}  vs  Lattimer: (457 pi/10080) G^2(1+3gA^2) m m mue T^6 / 2  (nu_e half)
T = 3.0; xg, xw = leggauss(64); x = 40 * (xg + 1) / 2; w = xw * 20
S = spectra(T, mst, mst, pFn, pFp, mue, 0.0, x); j = S['W_abs_nu'] * np.exp(-x)
eps = np.sum(w * (T * x) ** 3 * j) * T / (2 * np.pi ** 2); mn = np.sqrt(pFn ** 2 + mst ** 2); mp = np.sqrt(pFp ** 2 + mst ** 2)
eps_L = 457 * np.pi / 10080 * GT2 * mn * mp * mue * T ** 6 / 2
eps2 = np.sum(w * (T * x) ** 3 * S['dGec'] / (T ** 3 * x ** 2 / (2 * np.pi ** 2))) * T / (2 * np.pi ** 2)   # from the capture spectrum directly
print(f"emissivity (nu_e from capture) at T=3: kernel via W e^-E/T {eps:.4e}, kernel via dGec {eps2:.4e}, Lattimer/2 {eps_L:.4e}  ratios {eps/eps_L:.4f} {eps2/eps_L:.4f}")
