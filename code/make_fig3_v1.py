"""Figure 3, APS/PRL form, v1 (2026-11-02; design = PAPER_PLAN item 3, approved by the owner in words 11-02): where the dissipation lives.
  (a) damping time vs mode wavelength at 1 kHz;  (b) damping time vs frequency for the 10-km mode;  DD2 at (n0, 4 MeV) and (2n0, 5 MeV).
  thick solid + dots = kinetic result of this work at the computed points (results/tau_vs_wavelength.json: tau_vs_L_km, tau_vs_f_kHz);
  thin solid  = this work's transparent baseline tau_AH: (a) the 1-kHz value; (b) as a function of frequency from the stored relaxation rate,
                tau_AH(w) = n(dp/dn) B (w^2 + g^2)/(C^2 g w^2), g = gamma_AH (results/viscosity_lc.json);
  dotted      = short-wavelength (k -> inf, ballistic) limit of the model: (a) the 1-kHz value; (b) tau_inf(w) = n(dp/dn)(hbar^2 w^2 + (B Lambda_inf)^2)/(C^2 Lambda_inf hbar w^2)
                from the stored Lambda_inf (results/tau_vs_wavelength.json), the same expression as in tau_vs_wavelength.py.
  The resonance scale 2 n (dp/dn) B/(C^2 w) is NOT drawn (text only). Colours as in Fig. 2 (n0 vermilion, 2n0 green).
Export: legend inside the 3.375-in canvas, no tight bbox -> PDF width = column width. All arrays, formulas and checks -> figs/fig3_v1_data.json."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 7.0,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'xtick.minor.visible': True, 'ytick.minor.visible': True,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.labelspacing': 0.25,
                     'legend.borderpad': 0.2, 'legend.borderaxespad': 0.2, 'legend.columnspacing': 1.2, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
P = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))) + '/'; HBAR = 6.582119569e-22   # MeV s
W = {(o['x'], o['T']): o for o in json.load(open(P + 'results/tau_vs_wavelength.json'))}
V = {(o['x'], o['T']): o for o in json.load(open(P + 'results/viscosity_lc.json')) if o['eos'] == 'DD2'}
K = {(o['x'], o['T']): o for o in json.load(open(P + 'results/kinetic_response_fine.json'))}
T1 = {(o['x'], o['T']): o for o in json.load(open(P + 'results/table1_independent_checks.json')) if o['eos'] == 'DD2' and not o['cold']}
STATES = ((1.0, 4.0), (2.0, 5.0)); COL = {1.0: '#D55E00', 2.0: '#009E73'}; LAB = {1.0: r'$n_0$, 4 MeV', 2.0: r'$2n_0$, 5 MeV'}
def tau_AH_w(s, f_kHz):
    w = 2 * np.pi * 1e3 * np.asarray(f_kHz, float); g = V[s]['R5_f1']['gamma_AH']; return W[s]['ndpdn'] * W[s]['B'] * (w ** 2 + g ** 2) / (W[s]['C'] ** 2 * g * w ** 2)
def tau_inf_w(s, f_kHz):
    w = 2 * np.pi * 1e3 * np.asarray(f_kHz, float); L = W[s]['Lambda_inf']; return W[s]['ndpdn'] * ((HBAR * w) ** 2 + (W[s]['B'] * L) ** 2) / (W[s]['C'] ** 2 * L * HBAR * w ** 2)
fgrid = np.logspace(np.log10(0.2), np.log10(15.0), 200)
D = {}; chk = {}
for s in STATES:
    w = W[s]; Lk = sorted(float(k) for k in w['tau_vs_L_km'] if k != 'k0'); fk = sorted(float(k) for k in w['tau_vs_f_kHz'])
    D[s] = dict(L_km=Lk, tau_L=[w['tau_vs_L_km'][f'{L:g}' if f'{L:g}' in w['tau_vs_L_km'] else str(L)] for L in Lk], f_kHz=fk, tau_f=[w['tau_vs_f_kHz'][f'{f:g}' if f'{f:g}' in w['tau_vs_f_kHz'] else str(f)] for f in fk],
                tau_AH_1kHz=w['tau_AH'], tau_inf_1kHz=w['tau_inf_1kHz'], tau_res_1kHz=w['tau_res_1kHz'], gamma_AH=V[s]['R5_f1']['gamma_AH'], Lambda_inf=w['Lambda_inf'],
                C=w['C'], B=w['B'], ndpdn=w['ndpdn'], tau_AH_vs_f=dict(f_kHz=fgrid.tolist(), tau=tau_AH_w(s, fgrid).tolist()), tau_inf_vs_f=dict(f_kHz=fgrid.tolist(), tau=tau_inf_w(s, fgrid).tolist()), tau_k0_1kHz=w['tau_vs_L_km']['k0'])
    c = {}
    c['tauAH_formula_1kHz_vs_tau_vs_wavelength'] = float(tau_AH_w(s, 1.0) / w['tau_AH'] - 1)
    c['tauAH_formula_1kHz_vs_viscosity_lc_R5_f1'] = float(tau_AH_w(s, 1.0) / V[s]['R5_f1']['tau_AH_s'] - 1)
    c['tauAH_formula_3kHz_vs_viscosity_lc_R5_f3'] = float(tau_AH_w(s, 3.0) / V[s]['R5_f3']['tau_AH_s'] - 1)
    c['tauinf_formula_1kHz_vs_stored'] = float(tau_inf_w(s, 1.0) / w['tau_inf_1kHz'] - 1)
    c['tau_res_formula_vs_stored'] = float(2 * w['ndpdn'] * w['B'] / (w['C'] ** 2 * 2 * np.pi * 1e3) / w['tau_res_1kHz'] - 1)
    c['L10km_vs_f1kHz'] = float(w['tau_vs_L_km']['10.0'] / w['tau_vs_f_kHz']['1.0'] - 1)
    c['f1kHz_vs_kinetic_fine_1kHz_10km'] = float(w['tau_vs_f_kHz']['1.0'] / K[s]['full']['1kHz_10km']['tau'] - 1)
    c['f3kHz_vs_kinetic_fine_3kHz_10km'] = float(w['tau_vs_f_kHz']['3.0'] / K[s]['full']['3kHz_10km']['tau'] - 1)
    c['f1kHz_vs_table1_project'] = float(w['tau_vs_f_kHz']['1.0'] / T1[s]['tau_project'] - 1)
    c['k0_vs_kinetic_fine_1kHz_k0'] = float(w['tau_vs_L_km']['k0'] / K[s]['full']['1kHz_k0']['tau'] - 1)
    c['CBn_vs_viscosity_lc_and_fine_max_rel'] = max(abs(w['C'] / V[s]['C'] - 1), abs(w['B'] / V[s]['Bnp'] - 1), abs(w['ndpdn'] / V[s]['ndpdn'] - 1), abs(w['C'] / K[s]['C'] - 1), abs(w['B'] / K[s]['B'] - 1), abs(w['ndpdn'] / K[s]['ndpdn'] - 1))
    chk[f'{s[0]:g}n0_{s[1]:g}MeV'] = c
    assert all(abs(v) < 1e-9 for v in c.values()), c
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
fig.savefig(P + 'figs/fig3_v1.pdf'); fig.savefig(P + 'figs/fig3_v1.png', dpi=300)
import subprocess, re; info = subprocess.run(['pdfinfo', P + 'figs/fig3_v1.pdf'], capture_output=True, text=True).stdout
wpt = float(re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', info).group(1)); chk['pdf_width_pt'] = wpt; assert abs(wpt - 243.0) < 0.1, wpt
json.dump(dict(states={f'{s[0]:g}n0_{s[1]:g}MeV': D[s] for s in STATES}, checks=chk,
               sources=dict(curves='results/tau_vs_wavelength.json (kinetic model, hot composition; k->0 value stored, not drawn)', gamma_AH='results/viscosity_lc.json R5_f1 (B lambda_AH, s^-1)',
                            formulas=dict(tau_AH='n dp/dn B (w^2+g^2)/(C^2 g w^2)', tau_inf='n dp/dn (hbar^2 w^2 + (B Lambda_inf)^2)/(C^2 Lambda_inf hbar w^2)', tau_res_not_drawn='2 n dp/dn B/(C^2 w)'))),
          open(P + 'figs/fig3_v1_data.json', 'w'), indent=1)
print(json.dumps(chk, indent=1)); print('saved figs/fig3_v1.{pdf,png}, figs/fig3_v1_data.json')
