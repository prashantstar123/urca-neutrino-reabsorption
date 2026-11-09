"""Compares freshly generated results/ and figs/ with a copy of the shipped files (default directory: shipped/).
JSON: every number, relative difference |a - b| / max(|a|, |b|), with run-time entries ignored (keys 't', 'runtime', 'runtime_s') and root
residuals (F_star) required only to stay below 1e-6.
Text: line by line, ignoring run-time lines. PNG: pixel by pixel. PDF: size only (matplotlib PDFs carry a creation date).
Exit code 0 when every JSON number agrees to 1e-6 relative (or 1e-12 absolute) and every PNG is pixel-identical."""
import json, os, re, sys
import numpy as np

REF = sys.argv[1] if len(sys.argv) > 1 else 'shipped'
SKIP = {'t', 'runtime', 'runtime_s'}
RESIDUAL = {'F_star'}   # residual of a root (F(Y_e*) = 0): compared as a residual, both below 1e-6, not relatively
TIME = re.compile(r'runtime|\[\s*\d+(\.\d+)?\s*s\]|^\s*t\s*=', re.I)


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) - SKIP != set(b) - SKIP: out.append((float('inf'), path + ' (keys differ)')); return
        for k in a:
            if k in SKIP: continue
            if k in RESIDUAL and isinstance(a[k], (int, float)) and isinstance(b[k], (int, float)):
                out.append((0.0 if max(abs(a[k]), abs(b[k])) < 1e-6 else float('inf'), f'{path}/{k} (residual)')); continue
            walk(a[k], b[k], f'{path}/{k}', out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out.append((float('inf'), path + ' (lengths differ)')); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f'{path}[{i}]', out)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        if a != b: out.append((float('inf'), f'{path} ({a!r} vs {b!r})'))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b or (np.isnan(a) and np.isnan(b)): return
        d = abs(a - b); s = max(abs(a), abs(b))
        out.append((0.0 if d <= 1e-12 else d / s, path))
    else:
        out.append((float('inf'), f'{path} (types differ)'))


def main():
    ok = True; rows = []
    for top in ('results', 'figs'):
        for root, _, files in os.walk(os.path.join(REF, top)):
            for f in sorted(files):
                ref = os.path.join(root, f); new = os.path.relpath(ref, REF)
                if not os.path.exists(new): rows.append((new, 'MISSING')); ok = False; continue
                if f.endswith('.json'):
                    out = []; walk(json.load(open(ref)), json.load(open(new)), '', out)
                    worst = max(out, default=(0.0, '')); good = worst[0] <= 1e-6; ok &= good
                    rows.append((new, f"max rel. diff {worst[0]:.1e}{'' if good else '  at ' + worst[1]}"))
                elif f.endswith('.txt'):
                    la = [l for l in open(ref).read().splitlines() if not TIME.search(l)]
                    lb = [l for l in open(new).read().splitlines() if not TIME.search(l)]
                    nd = sum(x != y for x, y in zip(la, lb)) + abs(len(la) - len(lb))
                    rows.append((new, 'identical' if nd == 0 else f'{nd} line(s) differ (check the numbers)'))
                elif f.endswith('.png'):
                    import matplotlib.image as mpimg
                    a, b = mpimg.imread(ref), mpimg.imread(new); same = a.shape == b.shape and np.array_equal(a, b); ok &= same
                    rows.append((new, 'pixel-identical' if same else 'pixels differ'))
                elif f.endswith('.pdf'):
                    rows.append((new, f'size {os.path.getsize(ref)} vs {os.path.getsize(new)} bytes'))
    w = max(len(r[0]) for r in rows)
    for name, verdict in rows: print(f'{name:{w}s}  {verdict}')
    print('ALL AGREE' if ok else 'DIFFERENCES FOUND'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
