#!/bin/bash
# Reproduces every result file, figure and table of the Letter from scratch (run from the package root; ~30 min on a 6-core desktop).
# Scripts write results/ and figs/ relative to this directory.  Python >= 3.10 with numpy, scipy, matplotlib (versions in requirements.txt).
set -e; cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; PY=${PYTHON:-python3}
run(){ echo "== $1 $(date)"; $PY code/$1; }
run escape_map.py; run viscosity_lc.py; run kinetic_response.py; KIN_FINE=1 run kinetic_response.py; run tau_vs_wavelength.py; run kinetic_checks.py
run elastic_opacity_check.py; run validate_kernel_degenerate.py; run fill_tau_AH.py; run table1_independent_checks.py; run kinetic_response_eos.py; run flag_eos_table.py
run make_fig1_v4.py; run make_fig2_v3.py; run make_fig3_v1.py; run mc_absorption.py; run finiteT_composition_check.py; run murca_spectral.py; run nc_scatter.py
echo "done $(date)"
