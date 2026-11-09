"""Figure 3 v2 (fix round 2026-11-08): v1 design with the regenerated data. (a) damping time vs mode wavelength at 1 kHz; (b) vs frequency, 10-km mode;
DD2 at (n0, 4 MeV) and (2n0, 5 MeV). thick solid + dots = kinetic result (RES/kinetic_response_fine.json tau_vs_L_km, tau_vs_f_kHz);
thin solid = physical transparent reference (RES/transparent_physical.json): tau_tr(w) = K (w^2 + gamma_I^2)/(n C gamma_B w^2) at Y_e* (d = 1e-3 gradients);
dotted = short-wavelength limit of the kinetic model, tau_inf(w) = K (hbar^2 w^2 + (B Lambda_inf)^2)/(C^2 Lambda_inf hbar w^2) (kinetic_response_fine.json Lambda_inf).
Env RES, FIG_DIR. Arrays and checks -> FIG_DIR/fig3_v2_data.json."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 7.0,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.labelspacing': 0.25,
                     'legend.borderpad': 0.2, 'legend.borderaxespad': 0.2, 'legend.columnspacing': 1.2, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'; HBAR = 6.582119569e-22; RES = P + os.environ.get('RES', 'results') + '/'; FIG = P + os.environ.get('FIG_DIR', 'figs') + '/'
K = {(o['x'], o['T']): o for o in json.load(open(RES + 'kinetic_response_fine.json')) if 'error' not in o}
TR = {(o['x'], o['T']): o for o in json.load(open(RES + 'transparent_physical.json')) if o['eos'] == 'DD2' and 'error' not in o}
STATES = ((1.0, 4.0), (2.0, 5.0)); COL = {1.0: '#D55E00', 2.0: '#009E73'}; LAB = {1.0: r'$n_0$, 4 MeV', 2.0: r'$2n_0$, 5 MeV'}
def tau_AH_w(s, f_kHz):
    w = 2 * np.pi * 1e3 * np.asarray(f_kHz, float); t = TR[s]; d = t['d0.001']; return t['K'] * (w ** 2 + d['gamma_I'] ** 2) / (t['n'] * t['C'] * d['gamma_B'] * w ** 2)
def tau_inf_w(s, f_kHz):
    w = 2 * np.pi * 1e3 * np.asarray(f_kHz, float); k = K[s]; L = k['Lambda_inf']; return k['ndpdn'] * ((HBAR * w) ** 2 + (k['B'] * L) ** 2) / (k['C'] ** 2 * L * HBAR * w ** 2)
fgrid = np.logspace(np.log10(0.2), np.log10(15.0), 200); D = {}; chk = {}
for s in STATES:
    k = K[s]; Lk = sorted(float(a) for a in k['tau_vs_L_km'] if a != 'k0'); fk = sorted(float(a) for a in k['tau_vs_f_kHz'])
    D[s] = dict(L_km=Lk, tau_L=[k['tau_vs_L_km'][str(L)] for L in Lk], f_kHz=fk, tau_f=[k['tau_vs_f_kHz'][str(f)] for f in fk], tau_AH_1kHz=float(tau_AH_w(s, 1.0)),
                tau_inf_1kHz=float(tau_inf_w(s, 1.0)), tau_res_1kHz=k['tau_res_1kHz'], Lambda_inf=k['Lambda_inf'], C=k['C'], B=k['B'], ndpdn=k['ndpdn'],
                tau_AH_vs_f=dict(f_kHz=fgrid.tolist(), tau=tau_AH_w(s, fgrid).tolist()), tau_inf_vs_f=dict(f_kHz=fgrid.tolist(), tau=tau_inf_w(s, fgrid).tolist()), tau_k0_1kHz=k['tau_vs_L_km']['k0'])
    c = dict(tr_formula_1kHz_vs_transparent_file=float(tau_AH_w(s, 1.0) / TR[s]['tau_1kHz'] - 1), tr_formula_3kHz_vs_file=float(tau_AH_w(s, 3.0) / TR[s]['d0.001']['tau_vs_f_kHz']['3.0'] - 1),
             inf_formula_vs_stored=float(tau_inf_w(s, 1.0) / k['tau_inf_vs_f_kHz']['1.0'] - 1), L10_vs_f1=float(k['tau_vs_L_km']['10.0'] / k['tau_vs_f_kHz']['1.0'] - 1),
             f1_vs_full=float(k['tau_vs_f_kHz']['1.0'] / k['full']['1kHz_10km']['tau'] - 1), f3_vs_full=float(k['tau_vs_f_kHz']['3.0'] / k['full']['3kHz_10km']['tau'] - 1))
    chk[f'{s[0]:g}n0_{s[1]:g}MeV'] = c; assert all(abs(v) < 1e-9 for v in c.values()), c
# ---- figure
fig = plt.figure(figsize=(3.375, 2.95)); axa = fig.add_axes([0.150, 0.168, 0.405, 0.655]); axb = fig.add_axes([0.580, 0.168, 0.405, 0.655], sharey=axa)
for s in STATES:
    d = D[s]; col = COL[s[0]]
    axa.axhline(d['tau_AH_1kHz'], color=col, lw=0.9, ls='-'); axa.axhline(d['tau_inf_1kHz'], color=col, lw=1.0, ls=':')
    axa.plot(d['L_km'], d['tau_L'], color=col, lw=1.9, marker='o', ms=2.6, mec=col, mfc=col)
    axb.plot(d['tau_AH_vs_f']['f_kHz'], d['tau_AH_vs_f']['tau'], color=col, lw=0.9, ls='-'); axb.plot(d['tau_inf_vs_f']['f_kHz'], d['tau_inf_vs_f']['tau'], color=col, lw=1.0, ls=':')
    axb.plot(d['f_kHz'], d['tau_f'], color=col, lw=1.9, marker='o', ms=2.6, mec=col, mfc=col)
axa.text(3.2, 0.235, LAB[1.0], color=COL[1.0], fontsize=7.5, ha='left', va='top'); axa.text(0.24, 2.2, LAB[2.0], color=COL[2.0], fontsize=7.5, ha='left', va='bottom')
for ax, xl, lab in ((axa, r'wavelength (km)', '(a) 1 kHz'), (axb, r'frequency (kHz)', '(b) 10 km')):
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(xl, labelpad=1.5); ax.text(0.04, 0.965, lab, transform=ax.transAxes, fontsize=8, ha='left', va='top')
axa.set_xlim(0.2, 80); axa.set_xticks([0.3, 1, 3, 10, 30]); axa.set_xticklabels(['0.3', '1', '3', '10', '30']); axa.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axb.set_xlim(0.2, 15); axb.set_xticks([0.3, 1, 3, 10]); axb.set_xticklabels(['0.3', '1', '3', '10']); axb.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axa.set_ylim(5e-3, 2e3); axa.set_ylabel(r'$\tau$ (s)', labelpad=1); plt.setp(axb.get_yticklabels(), visible=False)
axa.text(0.97, 0.03, 'DD2', transform=axa.transAxes, fontsize=7.5, ha='right', va='bottom')
H = [Line2D([], [], color='0.25', lw=1.9, marker='o', ms=2.6, label='kinetic (this work)'), Line2D([], [], color='0.25', lw=0.9, ls='-', label='transparent baseline'),
     Line2D([], [], color='0.25', lw=1.0, ls=':', label='short-wavelength limit')]
fig.legend(handles=H, loc='upper left', bbox_to_anchor=(0.13, 0.995), ncol=2, handletextpad=0.5)
fig.savefig(FIG + 'fig3_v2.pdf'); fig.savefig(FIG + 'fig3_v2.png', dpi=300)
import subprocess, re; info = subprocess.run(['pdfinfo', FIG + 'fig3_v2.pdf'], capture_output=True, text=True).stdout
wpt = float(re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', info).group(1)); chk['pdf_width_pt'] = wpt; assert abs(wpt - 243.0) < 0.1, wpt
json.dump(dict(states={f'{s[0]:g}n0_{s[1]:g}MeV': D[s] for s in STATES}, checks=chk,
               sources=dict(curves='results/kinetic_response_fine.json (tau_vs_L_km, tau_vs_f_kHz)', transparent='results/transparent_physical.json (d0.001 gamma_I, gamma_B; C, K at Y_e*)', shortwave='results/kinetic_response_fine.json Lambda_inf')),
          open(FIG + 'fig3_v2_data.json', 'w'), indent=1)
print(json.dumps(chk, indent=1)); print('saved fig3_v2')
