# Reabsorbed Urca neutrinos suppress bulk-viscous damping of neutron star merger oscillations: code and data

Prashant Thakur and Yeunhwan Lim (Yonsei University), 2026.

This package contains the code and data behind every number, figure and table of the Letter and of its Supplemental Material.

## Contents
- `code/` holds the Python 3 scripts. They need numpy, scipy and matplotlib, and the versions used are listed in `ENVIRONMENT.txt`.
  - Equation of state and background state: `rmf_dd2_delta.py` (DD2), `rmf_fsu_delta.py` (IUFSU and BigApple), `partial_eq.py`, and `finiteT.py` and `finiteT_fsu.py` (finite-temperature states and thermodynamic coefficients).
  - Rates: `urca_rates.py`, `urca_fullT.py` and `urca_rel.py` (direct Urca, with energy-resolved emission and absorption spectra), `murca_spectral.py` (modified Urca), `nc_scatter.py` (neutral-current scattering from the full nonrelativistic response) and `rate_state.py` (one interface that assembles the rates for every script).
  - Response: `kinetic_response.py` (kinetic admittance and damping times, Eqs. (3)–(5) of the Letter) and `transparent_physical.py` (transparent baseline).
  - Checks: `second_implementation.py` (a separately written implementation with exact thermal dispersions and direct angular quadrature), `rate_exact_vs_landau.py`, `nc_full_checks.py`, `nc_full_checks2.py` and `mc_absorption.py` (Monte Carlo integration of the exact phase space).
  - Supplemental Material: `sm_table_s1.py`, `sm_fixround_s2s3.py`, `sm_s4_s8.py` and `sm_s4_extra.py`.
  - Figures: `make_fig1_v5.py`, `make_fig2_v4.py` and `make_fig3_v2.py`.
  - `compare_with_shipped.py` compares a fresh run with the shipped files.
- `results/` holds the output of every script as used in the Letter and the Supplemental Material, in JSON and text form. The validation outputs are in `results/checks/`.
- `figs/` holds Figures 1–3 (PDF and PNG) together with the plotted arrays (`*_data.json`).
- `data/ah2019_fig6a_contours.json` holds the contour paths extracted from the vector graphics of Fig. 6(a) of Alford and Harris, Phys. Rev. C 100, 035803 (2019). They give the published-map marks of Fig. 2 and Table I.

## Where each number comes from
| Item | Script | Output |
|---|---|---|
| Fig. 1: mean free paths and spectra at 2n<sub>0</sub> and 3 MeV, E<sub>min</sub>, low-energy tail, elastic-opacity curve (Eq. S11) | `make_fig1_v5.py` | `figs/fig1_v5_data.json` |
| Capture spectra on the 49 sampled states: mean energy, absorption length, escape probability | `capture_grid_stats.py` | `results/capture_grid_stats.json` |
| Kinetic damping times on the DD2 grid: Table I, Figs. 2 and 3, ranges from the 13 × 5 × 9 rate mesh, the k → 0 limit, composition amplitudes, Tables S7–S9 | `kinetic_response.py` (`KIN_GRID=dd2`) | `results/kinetic_response_fine.json` |
| IUFSU and BigApple (Table S6) | `kinetic_response.py` (`KIN_GRID=alteos`) | `results/kinetic_response_alteos.json` |
| Transparent baseline: Table I, Fig. 2, Tables S4 and S6 | `transparent_physical.py` | `results/transparent_physical.json` |
| Published-map brackets: Table I, Fig. 2, Table S4 | `ah2019_map_reading.py` | `results/ah2019_map_reading.json` |
| Second implementation (Letter and Sec. S2) | `second_implementation.py` | `results/second_implementation.json` |
| Direct-Urca rates against exact-dispersion integration | `rate_exact_vs_landau.py` | `results/checks/rate_exact_vs_landau.json` |
| Figs. 2 and 3 | `make_fig2_v4.py`, `make_fig3_v2.py` | `figs/` |
| Sec. S1 and Table S1 | `sm_table_s1.py` | `results/sm_table_s1.json` |
| Secs. S2 and S3: detailed balance, open-threshold comparison, rates of the second implementation, modified Urca, antineutrino and neutral-current lengths, Tables S2 and S3 | `sm_fixround_s2s3.py` | `results/sm_fixround_s2s3.json` |
| Sec. S3: limits of the neutral-current response | `nc_full_checks.py`, `nc_full_checks2.py` | `results/checks/nc_full_checks.json`, `results/checks/nc_full_checks2.json` |
| Secs. S4 and S6: bounds, long-wavelength form, numerical checks, baseline with rescaled rates (Table S5) | `sm_s4_s8.py`, `sm_s4_extra.py` | `results/sm_s4_s8.json`, `results/sm_s4_extra.json` |
| Sec. S7: finite-temperature IUFSU and BigApple states | `finiteT_fsu.py` | `results/checks/finiteT_fsu_validation.json` |
| Sec. S2: Monte Carlo benchmark of the open-threshold rate (its ratios to Ref. RPL98 at 10 and 30 MeV, combined with the rate ratios in `results/sm_fixround_s2s3.json`) | `mc_absorption.py` (archived run, about 30 minutes, rerun with `RUN_MC=1`) | `results/mc_absorption.txt` |

Only the open-threshold lines of `results/mc_absorption.txt` enter the Supplemental Material. Its lines below the threshold were computed with an earlier convention for the occupation references and are not quoted.

## Reproduce
Run `./reproduce.sh`. Set `PYTHON=/path/to/python` to choose the interpreter and `NPROC` to set the number of worker processes. Set `RUN_MC=1` to rerun the Monte Carlo benchmark as well. The script first keeps a copy of the shipped results in `shipped/`. It then regenerates every file in order and finally compares the fresh files with the shipped ones (`code/compare_with_shipped.py`). With ten worker processes the full run takes about 25 minutes on a desktop computer.

## License
MIT (see `LICENSE`). Please cite the Letter when you use this code or data.
