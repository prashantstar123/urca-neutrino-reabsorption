# Reabsorption of Urca neutrinos suppresses bulk-viscous damping in neutron star merger cores — code and data

Prashant Thakur (Yonsei University), 2026. Everything needed to reproduce every number, figure and table of the Letter.

## Contents
- `code/`: the production scripts (Python 3; numpy, scipy, matplotlib; versions in `ENVIRONMENT.txt`). Modules: `rmf_dd2_delta.py`, `rmf_fsu_delta.py`,
  `partial_eq.py`, `finiteT.py` (DD2, IUFSU, BigApple backgrounds and finite-temperature susceptibilities); `urca_rates.py`, `urca_fullT.py`, `urca_rel.py`
  (direct-Urca rates and energy-resolved spectra); `murca_spectral.py` (modified Urca); `nc_scatter.py` (neutral-current scattering); `kinetic_response.py`
  (the linearized kinetic response, Eqs. 3-5 of the Letter).
- `results/`: the output of every script as used in the Letter (JSON and text).
- `figs/`: Figures 1-3 (PDF/PNG) with their plotted arrays (`*_data.json`).
- `data/ah2019_fig6a_contours.json`: the contour readings of Alford & Harris, Phys. Rev. C 100, 035803 (2019), Fig. 6a used for the published-map marks.
- `second_implementation/r5_audit.py`: the separately written implementation (exact nucleon dispersions, direct angular quadrature) used for the cross-check
  described in the Supplemental Material; executed by `code/table1_independent_checks.py`.
- `reproduce.sh`: runs the full chain in order (about 30 minutes on a 6-core desktop).

## Where each number comes from
| Letter item | Script | Output |
|---|---|---|
| Fig. 1 (mean free paths, spectra), E_min, 5 m, 16 km, 5.6 km | `make_fig1_v4.py` | `figs/fig1_v4_data.json` |
| Sampled-state statements (mean capture energy 19-46 MeV, absorption length < 7 m, P(1 km) <= 3.1 %) | `escape_map.py` | `results/escape_map.json` |
| Transparent baseline tau_tr (Table I, Fig. 2 thin curves) | `viscosity_lc.py`, `fill_tau_AH.py` | `results/viscosity_lc.json`, `results/viscosity_lc_fill.json` |
| Kinetic damping times, k -> 0 limit, sensitivities, relaxation times (Fig. 2, Table I) | `kinetic_response.py` (`KIN_FINE=1` for the 45-state grid) | `results/kinetic_response_fine.json` |
| Rate-variation ranges and the second-implementation cross-check (Table I range column, Supplemental Material) | `table1_independent_checks.py` | `results/table1_independent_checks.json` |
| IUFSU and BigApple columns, affinity test | `kinetic_response_eos.py`, `flag_eos_table.py` | `results/kinetic_response_eos_flagged.json` |
| Fig. 3 (wavelength, frequency), composition amplitudes | `tau_vs_wavelength.py` | `results/tau_vs_wavelength.json`, `figs/fig3_v1_data.json` |
| Elastic-approximation crossover (161 / 131 MeV) | `elastic_opacity_check.py` | `results/elastic_opacity_check.json` |
| Kernel validation (degenerate limit, exact-dispersion Monte Carlo) | `validate_kernel_degenerate.py`, `mc_absorption.py` | stdout, `results/mc_absorption.txt` |
| Kinetic identities (limits, positivity) | `kinetic_checks.py` | `results/kinetic_checks.txt` |

## Reproduce
`./reproduce.sh` (set `PYTHON=/path/to/python` if needed). Each script writes into `results/` and `figs/`; compare with the shipped files
(the shipped results were regenerated from this package on 2026-11-07 and agree to machine precision).

## License
MIT (see `LICENSE`). Please cite the Letter when using this code or data.
