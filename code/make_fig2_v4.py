"""Figure 2 v4 (fix round 2026-11-08): v3 design (fixed 3.375-in canvas, legend inside, same line styles) with the regenerated data:
  thick solid  = kinetic result (RES/kinetic_response_fine.json full/1kHz_10km/tau), x = 0.5, 1, 2 n0, T = 2-8 MeV (9 points);
  thin solid   = physical transparent reference (RES/transparent_physical.json, stationary Y_e*, two rate gradients, d = 1e-3), same 9 points;
  dashed       = no spatial transport, k -> 0 (kinetic_response_fine.json full/1kHz_k0/tau);
  thin bars    = 13x5x9 (mU, dU, NC) rate-variation mesh range at the three Table I states of these densities (kinetic_response_fine.json mesh_min/max);
  capped bars  = AH2019 Fig. 6a readings at EQUAL PHYSICAL DENSITY (RES/ah2019_map_reading.json): 25-100 ms at n0/4 MeV, 100-1000 ms at 2n0/5 MeV;
                 the 0.5n0 state lies below the map's density range (x_AH = 0.466 < 0.5) and has no mark.
Env: RES (results dir), FIG_DIR. All plotted arrays -> FIG_DIR/fig2_v4_data.json."""
import json, re, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 7.0,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.labelspacing': 0.25, 'legend.borderpad': 0.2, 'legend.borderaxespad': 0.3, 'legend.handletextpad': 0.5})
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'; RES = P + os.environ.get('RES', 'results') + '/'; FIG = P + os.environ.get('FIG_DIR', 'figs') + '/'
K = {(o['x'], o['T']): o for o in json.load(open(RES + 'kinetic_response_fine.json')) if o.get('eos', 'DD2') == 'DD2' and 'error' not in o}
TR = {(o['x'], o['T']): o for o in json.load(open(RES + 'transparent_physical.json')) if o['eos'] == 'DD2' and 'error' not in o}
MAP = {(o['x'], o['T']): o for o in json.load(open(RES + 'ah2019_map_reading.json'))}
XS = (0.5, 1.0, 2.0); TS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0); BLUE, VERM, GREEN = '#0072B2', '#D55E00', '#009E73'; COL = {0.5: BLUE, 1.0: VERM, 2.0: GREEN}
curves = {x: dict(T=list(TS), tau_kin=[K[(x, T)]['full']['1kHz_10km']['tau'] for T in TS], tau_k0=[K[(x, T)]['full']['1kHz_k0']['tau'] for T in TS],
                  tau_tr=[TR[(x, T)]['tau_1kHz'] for T in TS]) for x in XS}
RSS = [(0.5, 3.0), (1.0, 4.0), (2.0, 5.0)]; ranges = {k: (K[k]['mesh_min'], K[k]['mesh_max']) for k in RSS}
def bracket(k):
    m = MAP[k]['equal_density']
    if m['outside_map']: return None
    T = k[1]; cr = {int(a): v for a, v in m['crossings'].items()}
    lo = max([lev for lev, v in cr.items() if any(t < T for t in v)], default=None)   # last contour crossed below T on the rising (high-T) branch
    # rising branch above the minimum: the contour values crossed just below and just above T bracket tau(T)
    below = [(t, lev) for lev, v in cr.items() for t in v if t < T]; above = [(t, lev) for lev, v in cr.items() for t in v if t > T]
    tb, lb = max(below); ta, la = min(above); return (min(lb, la) / 1e3, max(lb, la) / 1e3, tb, ta)
published = {k: bracket(k) for k in RSS}
assert published[(0.5, 3.0)] is None and published[(1.0, 4.0)][:2] == (0.025, 0.1) and published[(2.0, 5.0)][:2] == (0.1, 1.0), published
fig = plt.figure(figsize=(3.375, 3.5)); ax = fig.add_axes([0.158, 0.118, 0.822, 0.715])
for x in XS:
    c = curves[x]; col = COL[x]
    ax.plot(c['T'], c['tau_k0'], color=col, lw=0.9, ls='--'); ax.plot(c['T'], c['tau_tr'], color=col, lw=0.9, ls='-'); ax.plot(c['T'], c['tau_kin'], color=col, lw=1.9, ls='-')
for (x, T), (lo, hi) in ranges.items(): ax.plot([T, T], [lo, hi], color=COL[x], lw=1.0, solid_capstyle='butt', zorder=3)
for (x, T), br in published.items():
    if br is None: continue
    lo, hi = br[0], br[1]; mid = np.sqrt(lo * hi); ax.errorbar([T], [mid], yerr=[[mid - lo], [hi - mid]], fmt='none', ecolor=COL[x], elinewidth=1.3, capsize=3.5, capthick=1.3, zorder=5)
ax.set_yscale('log'); ax.set_ylim(2e-3, 2e4); ax.set_xticks([2, 3, 4, 5, 6, 7, 8]); ax.set_xlim(1.8, 8.2); ax.set_xlabel(r'$T$ (MeV)'); ax.set_ylabel(r'$\tau$ (s)')
for x, lab in ((0.5, r'$0.5\,n_0$'), (1.0, r'$n_0$'), (2.0, r'$2\,n_0$')):
    ax.annotate(lab, xy=(6.0, curves[x]['tau_kin'][curves[x]['T'].index(6.0)]), xytext=(0, 3), textcoords='offset points', color=COL[x], fontsize=8, ha='center', va='bottom')
h_pub = ax.errorbar([-100.0], [1.0], yerr=[[0.5], [0.5]], fmt='none', ecolor='0.3', elinewidth=1.3, capsize=3.5, capthick=1.3, label='Alford & Harris 2019')
H = [Line2D([], [], color='0.3', lw=1.9, label='kinetic (this work)'), Line2D([], [], color='0.3', lw=0.9, label='transparent baseline'),
     Line2D([], [], color='0.3', lw=0.9, ls='--', label=r'no spatial transport ($k\to0$)'), Line2D([], [], color='0.3', marker='|', ms=9, mew=1.0, ls='none', label='rate variations'), h_pub]
fig.legend(handles=H, loc='upper left', bbox_to_anchor=(0.13, 0.995), ncol=2, columnspacing=1.0, handlelength=1.8)
ax.text(0.97, 0.03, 'DD2, 1 kHz, 10-km mode', transform=ax.transAxes, ha='right', fontsize=7.5)
fig.savefig(FIG + 'fig2_v4.pdf'); fig.savefig(FIG + 'fig2_v4.png', dpi=220)
mb = re.findall(rb'MediaBox\s*\[([^\]]*)\]', open(FIG + 'fig2_v4.pdf', 'rb').read())[0].split(); assert abs(float(mb[2]) - 3.375 * 72) < 0.5
json.dump(dict(curves={str(x): curves[x] for x in XS}, ranges=[dict(x=k[0], T=k[1], lo=v[0], hi=v[1]) for k, v in ranges.items()],
               published=[dict(x=k[0], T=k[1], bracket_s=(None if v is None else v[:2]), crossing_T_below_above=(None if v is None else v[2:])) for k, v in published.items()],
               sources=dict(tau_kin='results/kinetic_response_fine.json', tau_tr='results/transparent_physical.json (tau_1kHz, d=1e-3)', map='results/ah2019_map_reading.json (equal physical density)')),
          open(FIG + 'fig2_v4_data.json', 'w'), indent=0)
print('Fig 2 v4 saved; published brackets:', published)
