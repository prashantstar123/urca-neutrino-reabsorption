"""Readings of the published AH2019 contour map (Fig. 6a, damping time of 1-kHz oscillations) at EQUAL PHYSICAL DENSITY (fix round 2026-11-08,
Astra audit Sec. 7 / C060): AH2019 define n_sat = 0.16 fm^-3 (papers/AH2019.txt:30); DD2 here has n0 = 0.149065 fm^-3, so a DD2 state x n0 sits at
x_AH = x * 0.149065/0.16 on the published axis. Contour vector paths from our own extraction of the published PDF (data/ah2019_fig6a_contours.json, levels 10, 25, 50, 100, 1000, 10000 ms; crossings of the extracted paths). For each state the crossing
temperatures of every contour along the vertical line x = x_AH are listed, and the bracket [lower, upper] of contour values enclosing T is read off.
States with x_AH < 0.5 lie outside the plotted map (lower axis limit 0.5 n_sat). Output: OUT_DIR/ah2019_map_reading.json/.txt"""
import json, os, numpy as np
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'; OUT = os.environ.get('OUT_DIR', P + 'results')
AH = json.load(open(P + 'data/ah2019_fig6a_contours.json')); N0_DD2, NSAT = 0.149065, 0.16
LEV = [10, 25, 50, 100, 1000, 10000]
def crossings(xq):
    res = {}
    for lev in LEV:
        ys = []
        for p in AH['paths']:
            if p['tau_ms'] != lev: continue
            xy = np.array(p['xy'])
            for (x1, y1), (x2, y2) in zip(xy[:-1], xy[1:]):
                if (x1 - xq) * (x2 - xq) <= 0 and x1 != x2: ys.append(y1 + (xq - x1) * (y2 - y1) / (x2 - x1))
        res[lev] = sorted(set(round(y, 5) for y in ys))
    return res
out = []; lines = ["AH2019 Fig. 6a readings at equal physical density (x_AH = x * 0.149065/0.16) and, for comparison, at equal labels (x_AH = x).",
                   " DD2 state  | mode           x_AH    | contour crossing temperatures [MeV]                                   | reading"]
for x, T in ((0.5, 3.0), (1.0, 4.0), (1.5, 4.0), (2.0, 5.0), (3.0, 5.0)):
    row = dict(x=x, T=T)
    for mode, xa in (('equal_density', x * N0_DD2 / NSAT), ('equal_label', x)):
        if xa < 0.5: row[mode] = dict(x_AH=xa, outside_map=True); lines.append(f" {x:3.1f} n0 {T:3.1f} | {mode:14s} {xa:6.4f} | outside the plotted map (x_AH < 0.5)"); continue
        cr = crossings(xa); below = [lev for lev in LEV if any(t < T for t in cr[lev])]; above = [lev for lev in LEV if any(t > T for t in cr[lev])]
        row[mode] = dict(x_AH=xa, crossings={str(k): v for k, v in cr.items()}, outside_map=False)
        lines.append(f" {x:3.1f} n0 {T:3.1f} | {mode:14s} {xa:6.4f} | " + "  ".join(f"{k}:{v}" for k, v in cr.items() if v))
    out.append(row)
json.dump(out, open(f'{OUT}/ah2019_map_reading.json', 'w'), indent=1); open(f'{OUT}/ah2019_map_reading.txt', 'w').write("\n".join(lines) + "\n"); print("\n".join(lines))
