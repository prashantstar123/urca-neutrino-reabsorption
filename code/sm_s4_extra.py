"""S4 addition (2026-11-08): exact small-k structure of the kinetic admittance, Lambda(w,k) = Lambda(w,0) + k^2 Dk(w) + O(k^4) with
Dk(w) = sum_i int dE w_i c^2 g_i^2 / [3 (g_i + s - i w)(g_i - i w)^2] (expansion of A_i = arctan(z)/(ck) to order k^2), its accuracy (truncated form) at 10, 20,
50 km and 1 kHz on all 45 DD2 states, and the local (k -> 0) share Re Lambda(w,0)/Re Lambda(w,k) at 10 km. Output: results/sm_s4_extra.json/.txt"""
import os, sys, json, time
import numpy as np
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(P, 'code')); os.chdir(P)
from multiprocessing import Pool
import rate_state as RS
from kinetic_response import admittance
from urca_rates import HC, HBAR
TS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0); XS = (0.5, 1.0, 1.5, 2.0, 3.0)
def tau_of(k, Lam, om):
    hw = HBAR * om; z = k['C'] ** 2 * Lam.real / abs(-1j * hw + k['B'] * Lam) ** 2 * HBAR; return k['K'] / (z * om ** 2)
def one(args):
    x, T = args; k = RS.kinetic_inputs('DD2', x, T); s = k['sig']
    gn = (k['WdUn'] + k['WmUn']) / (1 - k['fn']); gb = (k['WmUb'] + k['WdUb']) / (1 - k['fb'])
    om = 2 * np.pi * 1e3; hw = HBAR * om
    def lam(hck): return np.sum(k['wE'] * admittance(k['E'], k['wn'], gn, s, hw, hck)) + np.sum(k['wE'] * admittance(k['E'], k['wb'], gb, s, hw, hck))
    L0 = lam(0.0)
    Dk = sum(np.sum(k['wE'] * w * g ** 2 / (3 * (g + s - 1j * hw) * (g - 1j * hw) ** 2)) for w, g in ((k['wn'], gn), (k['wb'], gb)))
    out = dict(x=x, T=T, ReL0=L0.real, Dk=[Dk.real, Dk.imag])
    for L in (10.0, 20.0, 50.0):
        hck = HC * 2 * np.pi / (L * 1e18); Le = lam(hck); Lt = L0 + hck ** 2 * Dk
        out[str(L)] = dict(tau=tau_of(k, Le, om), tau_trunc=tau_of(k, Lt, om) if Lt.real > 0 else None, local_share=L0.real / Le.real)
    return out
if __name__ == '__main__':
    t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '10'))) as pool: res = pool.map(one, [(x, T) for x in XS for T in TS], chunksize=1)
    json.dump(res, open('results/sm_s4_extra.json', 'w'), indent=1, default=float)
    L = [f'runtime {time.time() - t0:.0f} s']
    for Lk in ('10.0', '20.0', '50.0'):
        dev = [abs(o[Lk]['tau_trunc'] / o[Lk]['tau'] - 1) for o in res if o[Lk]['tau_trunc']]; nbad = sum(1 for o in res if not o[Lk]['tau_trunc'])
        sh = [o[Lk]['local_share'] for o in res]
        L.append(f'L = {Lk} km: truncated O(k^2) form max dev {100*max(dev):.2f}% (median {100*np.median(dev):.2f}%), non-positive {nbad}; local share {min(sh):.4f}-{max(sh):.4f}')
    for o in res:
        if (o['x'], o['T']) in ((0.5, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0), (3.0, 5.0)):
            L.append(f"{o['x']:g}n0/{o['T']:g}: local share at 10 km {o['10.0']['local_share']:.4f}, 20 km {o['20.0']['local_share']:.4f}, 50 km {o['50.0']['local_share']:.4f}; "
                     f"trunc dev 10/20/50 km {100*(o['10.0']['tau_trunc']/o['10.0']['tau']-1):+.2f} {100*(o['20.0']['tau_trunc']/o['20.0']['tau']-1):+.2f} {100*(o['50.0']['tau_trunc']/o['50.0']['tau']-1):+.2f} %")
    open('results/sm_s4_extra.txt', 'w').write("\n".join(L) + "\n"); print("\n".join(L))
