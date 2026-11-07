"""Figure 2, APS/PRL form, v3 (2026-11-02; round-8 changes applied to v2: all nine transparent-baseline temperatures, legend inside the column-width canvas,
caption/legend terms 'transparent baseline', 'no spatial transport (k -> 0)', 'rate variations'). Content otherwise identical to v2 (make_fig2_v2.py docstring).
  thick solid  = kinetic result, this work (results/kinetic_response_fine.json 'full'/'1kHz_10km'/'tau'), x = 0.5, 1, 2 n0, T = 2-8 MeV (9 points);
  thin solid   = this work's in-medium transparent baseline tau_AH: viscosity_lc.json (via the fine file's tau_AH) at 2, 2.5, 3, 4, 5, 6, 8 MeV and
                 results/viscosity_lc_fill.json (fill_tau_AH.py, same prescription, round-8 request) at 3.5 and 4.5 MeV -> 9 points per density;
  dashed       = k -> 0 limit of the kinetic model ('full'/'1kHz_k0'/'tau'): no spatial transport, local reactions retained;
  thin cap-less bar = own combined (mU, dU, NC) 13x5x9 rate-variation mesh range (results/table1_independent_checks.json mesh_min/mesh_max) at the three
                 Table I states on these densities; a sampled sensitivity within one model, not a confidence interval;
  marks (density colour) = contour readings of AH2019 Fig 6a (CLAIMS C19): '< 10 ms' arrow at 0.5n0/3 MeV, '~50 ms on the contour' circle at n0/4 MeV,
                 100-1000 ms capped bar at 2n0/5 MeV.
Export: the legend is placed INSIDE the 3.375-in canvas and the file is saved without a tight bounding box, so the PDF width is exactly the column width
(round-8 finding: v2's tight-bbox export was 3.86 in wide and was scaled to 88 % in the column). All plotted arrays and sources -> figs/fig2_v3_data.json."""
import json, re, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 7.0,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.labelspacing': 0.25, 'legend.borderpad': 0.2, 'legend.borderaxespad': 0.3, 'legend.handletextpad': 0.5})
P = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))) + '/'
F = json.load(open(P + 'results/kinetic_response_fine.json')); V = json.load(open(P + 'results/viscosity_lc.json'))
T1 = json.load(open(P + 'results/table1_independent_checks.json')); W = json.load(open(P + 'results/tau_vs_wavelength.json'))
FILL = {(o['x'], o['T']): o for o in json.load(open(P + 'results/viscosity_lc_fill.json'))['values']}
AH = json.load(open(P + 'data/ah2019_fig6a_contours.json'))
XS = (0.5, 1.0, 2.0); BLUE, VERM, GREEN = '#0072B2', '#D55E00', '#009E73'; COL = {0.5: BLUE, 1.0: VERM, 2.0: GREEN}
curves = {}; n_fill = 0
for x in XS:
    rows = sorted([e for e in F if e['x'] == x], key=lambda e: e['T']); tr, src = [], []
    for e in rows:
        if np.isfinite(e['tau_AH']): tr.append(e['tau_AH']); src.append('viscosity_lc')
        else: tr.append(FILL[(x, e['T'])]['f1']['tau_AH_s']); src.append('viscosity_lc_fill'); n_fill += 1
    curves[x] = dict(T=[e['T'] for e in rows], tau_kin=[e['full']['1kHz_10km']['tau'] for e in rows], tau_k0=[e['full']['1kHz_k0']['tau'] for e in rows], tau_tr=tr, tau_tr_source=src)
assert n_fill == 6 and all(np.isfinite(t) for x in XS for t in curves[x]['tau_tr']), n_fill
# ---- checks
VL = {(v['x'], v['T']): v['R5_f1']['tau_AH_s'] for v in V if v['eos'] == 'DD2'}; chk = dict(n_filled_from_viscosity_lc_fill=n_fill)
chk['tau_AH_fine_vs_viscosity_lc_max_rel'] = max(abs(t / VL[(x, T)] - 1) for x in XS for T, t in zip(curves[x]['T'], curves[x]['tau_tr']) if (x, T) in VL)
chk['tau_AH_fill_vs_fill_file_max_rel'] = max(abs(t / FILL[(x, T)]['f1']['tau_AH_s'] - 1) for x in XS for T, t, s in zip(curves[x]['T'], curves[x]['tau_tr'], curves[x]['tau_tr_source']) if s == 'viscosity_lc_fill')
kin = {(x, T): t for x in XS for T, t in zip(curves[x]['T'], curves[x]['tau_kin'])}; k0 = {(x, T): t for x in XS for T, t in zip(curves[x]['T'], curves[x]['tau_k0'])}
t1 = {(r['x'], r['T']): r for r in T1 if r['eos'] == 'DD2' and not r['cold']}; RS = [(0.5, 3.0), (1.0, 4.0), (2.0, 5.0)]
chk['kinetic_vs_table1_project_max_rel'] = max(abs(kin[k] / t1[k]['tau_project'] - 1) for k in RS)
chk['k0_vs_tau_vs_wavelength_max_rel'] = max(abs(k0[(w['x'], w['T'])] / w['tau_vs_L_km']['k0'] - 1) for w in W if w['x'] in XS)
chk['ah2019_crossings_at_plotted_states'] = {f'{k[0]}n0_{k[1]}MeV': c['contour_T'] for c in AH['crossings'] for k in RS if (c['x'], c['T']) == k}
ranges = {k: (t1[k]['mesh_min'], t1[k]['mesh_max'], 'own combined (mU,dU,NC) mesh, R3 definition (results/table1_independent_checks.json)') for k in RS}
published = {(0.5, 3.0): dict(kind='upper_limit', value_s=0.010, text='< 10 ms (inside the 10-ms region; 10-ms boundary at T = 3 MeV lies at 0.711 n0)'),
             (1.0, 4.0): dict(kind='point', value_s=0.050, text='~ 50 ms (state on the 50-ms contour; crossing at 3.999 MeV; approximate contour value)'),
             (2.0, 5.0): dict(kind='interval', lo_s=0.1, hi_s=1.0, text='100-1000 ms (100-ms branch at 3.647 MeV, 1-s branch at 5.481 MeV)')}
# ---- figure: fixed canvas, legend inside it
fig = plt.figure(figsize=(3.375, 3.5)); ax = fig.add_axes([0.158, 0.118, 0.822, 0.715])
for x in XS:
    c = curves[x]; col = COL[x]
    ax.plot(c['T'], c['tau_k0'], color=col, lw=0.9, ls='--')
    ax.plot(c['T'], c['tau_tr'], color=col, lw=0.9, ls='-')
    ax.plot(c['T'], c['tau_kin'], color=col, lw=1.9, ls='-')
for (x, T), (lo, hi, src) in ranges.items():
    ax.plot([T, T], [lo, hi], color=COL[x], lw=1.0, solid_capstyle='butt', zorder=3)
ax.errorbar([3.0], [published[(0.5, 3.0)]['value_s']], yerr=[[published[(0.5, 3.0)]['value_s'] * 0.55], [0]], uplims=[True], fmt='_', color=COL[0.5], ms=7, mew=1.3, elinewidth=1.3, capsize=0, zorder=5)
ax.plot([4.0], [published[(1.0, 4.0)]['value_s']], marker='o', color=COL[1.0], ms=4.8, ls='none', zorder=5)
lo, hi = published[(2.0, 5.0)]['lo_s'], published[(2.0, 5.0)]['hi_s']; mid = np.sqrt(lo * hi)
ax.errorbar([5.0], [mid], yerr=[[mid - lo], [hi - mid]], fmt='none', ecolor=COL[2.0], elinewidth=1.3, capsize=3.5, capthick=1.3, zorder=5)
ax.set_yscale('log'); ax.set_ylim(2e-3, 2e4); ax.set_xticks([2, 3, 4, 5, 6, 7, 8]); ax.set_xlim(1.8, 8.2)
ax.set_xlabel(r'$T$ (MeV)'); ax.set_ylabel(r'$\tau$ (s)')
for x, lab in ((0.5, r'$0.5\,n_0$'), (1.0, r'$n_0$'), (2.0, r'$2\,n_0$')):
    ax.annotate(lab, xy=(6.0, curves[x]['tau_kin'][curves[x]['T'].index(6.0)]), xytext=(0, 3), textcoords='offset points', color=COL[x], fontsize=8, ha='center', va='bottom')
h_pub = ax.errorbar([-100.0], [1.0], yerr=[[0.5], [0.5]], fmt='none', ecolor='0.3', elinewidth=1.3, capsize=3.5, capthick=1.3, label='Alford & Harris 2019')   # legend handle only (off-axes)
H = [Line2D([], [], color='0.3', lw=1.9, label='kinetic (this work)'), Line2D([], [], color='0.3', lw=0.9, label='transparent baseline'),
     Line2D([], [], color='0.3', lw=0.9, ls='--', label=r'no spatial transport ($k\to0$)'), Line2D([], [], color='0.3', marker='|', ms=9, mew=1.0, ls='none', label='rate variations'), h_pub]
fig.legend(handles=H, loc='upper left', bbox_to_anchor=(0.13, 0.995), ncol=2, columnspacing=1.0, handlelength=1.8)
ax.text(0.97, 0.03, 'DD2, 1 kHz, 10-km mode', transform=ax.transAxes, ha='right', fontsize=7.5)
fig.savefig(P + 'figs/fig2_v3.pdf'); fig.savefig(P + 'figs/fig2_v3.png', dpi=220)
mb = re.findall(rb'MediaBox\s*\[([^\]]*)\]', open(P + 'figs/fig2_v3.pdf', 'rb').read())[0].split(); chk['pdf_width_pt'] = float(mb[2]); chk['pdf_height_pt'] = float(mb[3])
assert abs(chk['pdf_width_pt'] - 3.375 * 72) < 0.5, chk['pdf_width_pt']
data = dict(description='Figure 2 v3: damping time vs temperature, DD2 hot composition, 1 kHz, 10-km mode; see make_fig2_v3.py docstring', units=dict(T='MeV', tau='s'),
            curves={str(x): curves[x] for x in XS},
            sources=dict(tau_kin='results/kinetic_response_fine.json full/1kHz_10km/tau', tau_tr='results/kinetic_response_fine.json tau_AH (= viscosity_lc.json DD2 R5_f1 tau_AH_s) at 7 T; results/viscosity_lc_fill.json values[].f1.tau_AH_s at 3.5 and 4.5 MeV', tau_k0='results/kinetic_response_fine.json full/1kHz_k0/tau'),
            ranges=[dict(x=k[0], T=k[1], lo=v[0], hi=v[1], source=v[2]) for k, v in ranges.items()], published=[dict(x=k[0], T=k[1], **v) for k, v in published.items()], checks=chk)
json.dump(data, open(P + 'figs/fig2_v3_data.json', 'w'), indent=0)
print('checks:', json.dumps({k: v for k, v in chk.items() if k != 'ah2019_crossings_at_plotted_states'}, indent=0))
for x in XS: print(x, 'tau_tr', [round(t, 4) for t in curves[x]['tau_tr']], curves[x]['tau_tr_source'])
