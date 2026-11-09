#!/bin/bash
# Regenerates every result file, figure and table of the Letter and of its Supplemental Material, then compares the fresh files with the shipped ones.
# Run from anywhere: the script works in its own directory. Python >= 3.10 with numpy, scipy and matplotlib (versions used: ENVIRONMENT.txt).
#   PYTHON=/path/to/python   interpreter (default python3)
#   NPROC=8                  worker processes for the parallel scripts (default 8)
#   RUN_MC=1                 also rerun the Monte Carlo benchmark of Sec. S2 (about 30 minutes, not run by default)
set -e; cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NPROC=${NPROC:-8}; PY=${PYTHON:-python3}
if [ ! -d shipped ]; then mkdir shipped; cp -r results figs shipped/; fi      # keep the shipped files for the comparison at the end
mkdir -p results/checks figs
run(){ echo "== $* $(date +%H:%M:%S)"; env "$@"; }
run KIN_GRID=dd2    $PY code/kinetic_response.py       # 45 DD2 states: Table I, Figs. 2-3, Tables S7-S9
run KIN_GRID=alteos $PY code/kinetic_response.py       # IUFSU and BigApple: Table S6
run $PY code/transparent_physical.py                   # transparent baseline
run $PY code/capture_grid_stats.py                     # capture spectra on the 49 sampled states
run $PY code/ah2019_map_reading.py                     # published-map brackets
run $PY code/second_implementation.py                  # second implementation
run $PY code/rate_exact_vs_landau.py                   # direct-Urca rates against exact dispersions
run $PY code/make_fig1_v5.py
run $PY code/make_fig2_v4.py
run $PY code/make_fig3_v2.py
run $PY code/sm_table_s1.py                            # Supplemental Material S1
run $PY code/sm_fixround_s2s3.py                       # S2 and S3
run $PY code/sm_s4_s8.py                               # S4 and S6
run $PY code/sm_s4_extra.py                            # S4
run $PY code/nc_full_checks.py                         # S3, limits of the neutral-current response
run $PY code/nc_full_checks2.py
run $PY code/finiteT_fsu.py                            # S7, finite-temperature IUFSU and BigApple states
if [ "${RUN_MC:-0}" = 1 ]; then run $PY code/mc_absorption.py; fi   # S2, Monte Carlo benchmark (optional, about 30 minutes)
echo "== comparison with the shipped files $(date +%H:%M:%S)"
$PY code/compare_with_shipped.py shipped
