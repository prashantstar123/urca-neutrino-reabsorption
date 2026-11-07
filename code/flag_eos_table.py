"""results/kinetic_response_eos_flagged.{txt,json}: identical numbers to results/kinetic_response_eos.{txt,json} (STATUS row 29) plus a per-row flag.
Rule v3 (after the round-6 referee, 2026-11-02):
  'kernel-limited'  dp = p_Fn - p_Fp - p_Fe < 20 MeV (incl. dp <= 0): Landau-mass dUrca kernel unreliable near the threshold; excluded from unqualified tables.
  'near-threshold'  20 <= dp < 30 MeV: pointwise exact-kernel validation required before any use; NO percentage guarantee is attached (the gap alone does
                    not set the error: DD2 4n0/5 MeV cold, dp 24.9: exact kernel 21.6 % below the project value = project 27.5 % above the exact;
                    IUFSU 2.5n0/4 MeV, dp 26.8: exact 2.8 % above; IUFSU 3n0, dp 14.4: +0.7 % at 3 MeV but -68.8 % at 5 MeV).
  dp >= 30 MeV: unflagged; not a certification by itself. Printed Table I states are supported by their explicit comparisons (results/table1_independent_checks.txt).
Superseded rules: v1 2026-10-27 (dp < 10), v2 2026-10-27 (dp < 20 + a global 'kernel +-25%' tier for 20-30 MeV, rejected by the round-6 referee as unsupported)."""
import json
RULE = "kernel-limited if dp < 20 MeV; near-threshold (pointwise exact-kernel validation required, no percentage guarantee) if 20 <= dp < 30 MeV"
def flag(dp): return 'kernel-limited' if dp < 20.0 else ('near-threshold' if dp < 30.0 else '')
rows = json.load(open('results/kinetic_response_eos.json'))
for r in rows: r['flag'] = flag(float(r['dp'])); r['flag_rule'] = RULE
json.dump(rows, open('results/kinetic_response_eos_flagged.json', 'w'), indent=0)
src = open('results/kinetic_response_eos.txt').read().splitlines()
head = ["EOS robustness of the kinetic damping times [s] (kinetic_response.py model), FLAGGED copy v3 (2026-11-02, flag_eos_table.py): identical numbers to results/kinetic_response_eos.txt plus a per-row flag.",
        "flag 'kernel-limited' = dp = p_Fn - p_Fp - p_Fe < 20 MeV (incl. dp <= 0): Landau-mass dUrca kernel unreliable near the threshold; excluded from unqualified tables. flag 'near-threshold' = 20 <= dp < 30 MeV: pointwise exact-kernel validation required before any use; no percentage guarantee.",
        "Evidence (round-5 referee exact-dispersion kernel run at the exact states, results/table1_independent_checks.txt; discrepancy = exact/project - 1): BigApple 3n0 (dp +8.0, 5 MeV): 1651 vs 64.2 s (-96.1 %); IUFSU 3n0 (dp +14.4): 57.8 vs 18.1 s at 5 MeV (-68.8 %) but 18.36 vs 18.49 s at 3 MeV (+0.7 %); DD2 4n0 cold (dp +24.9, 5 MeV): 233.2 vs 182.9 s (-21.6 %; project/exact - 1 = +27.5 %); IUFSU 2.5n0 (dp +26.8, 4 MeV): 6.57 vs 6.76 s (+2.8 %); DD2 3n0 hot (dp +33.7, 5 MeV): 58.0 vs 59.0 s (+1.7 %).",
        "hot = dmu_np(T) from finiteT.stateT (DD2 only); cold = dmu_np = 0. tau_AH = transparent baseline. ratio = tau(1kHz,10km)/tau_AH."]
out = head + [src[1].replace('full: 1kHz/10km', '1kHz/10km') + ' | flag']
nrow = 0
for ln in src[2:]:
    if not ln.strip(): continue
    dp = float(ln.split('|')[1].split()[0]); out.append(ln + ' | ' + flag(dp)); nrow += 1
open('results/kinetic_response_eos_flagged.txt', 'w').write("\n".join(out) + "\n")
from collections import Counter; print(nrow, 'rows;', Counter(r['flag'] for r in rows))
