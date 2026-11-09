"""Methodology claim 'agrees with exact-dispersion phase-space integrations to about X %' re-evaluated for the fix-round convention:
W_abs,nu(E) of production (Landau parabolic dispersions, density-consistent references) vs the second implementation (exact thermal dispersions and
EOS occupations) at the capture energies (20-40 MeV) and over the capture spectrum, at the five Table I states."""
import sys, json, numpy as np; import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import rate_state as RS
from urca_rel import spectra
S = {(o['x'], o['T']): o for o in json.load(open('results/second_implementation.json'))}
out = []
for k in [(0.5, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0), (3.0, 5.0)]:
    s = S[k]; E = np.array(s['E']); b = RS.background('DD2', k[0], k[1]); T = k[1]
    Sp = spectra(T, b['mstn'], b['mstp'], b['pFn'], b['pFp'], b['mueT'], b['dmuT'], E / T, refs=b['refs'])
    r = np.array(s['W_dU_nu']) / Sp['W_abs_nu']; rb = np.array(s['W_dU_nubar']) / Sp['W_abs_nubar']; sel = (E >= 20) & (E <= 40)
    w = Sp['dGec'] / T; wn = w / np.sum(w)   # capture-spectrum weight on this grid (shape only, Gauss weights not needed for a ratio summary)
    out.append(dict(x=k[0], T=T, ratio_nu_20_40=[float(r[sel].min()), float(r[sel].max())], ratio_nu_capture_weighted=float(np.sum(wn * r)), ratio_nubar_E_T_to_10T=[float(rb[(E > T) & (E < 10 * T)].min()), float(rb[(E > T) & (E < 10 * T)].max())]))
    print(k, 'W_nu exact/Landau at 20-40 MeV: %.3f-%.3f; capture-weighted %.3f; nubar (T<E<10T) %.3f-%.3f' % (r[sel].min(), r[sel].max(), out[-1]['ratio_nu_capture_weighted'], *out[-1]['ratio_nubar_E_T_to_10T']))
json.dump(out, open('results/checks/rate_exact_vs_landau.json', 'w'), indent=1)
