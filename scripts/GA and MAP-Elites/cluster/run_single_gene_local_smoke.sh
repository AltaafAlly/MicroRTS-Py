#!/bin/bash
# Quick local smoke test for run_single_gene_balance_decrease.py
# Same logic as cluster job; smaller gens / pop / games / steps for fast feedback.
#
# From repo root:
#   ./scripts/GA\ and\ MAP-Elites/cluster/run_single_gene_local_smoke.sh
#
# Override any knob before running, e.g.:
#   SINGLE_GENE_DEC_GENS=10 ./scripts/GA\ and\ MAP-Elites/cluster/run_single_gene_local_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

if [ ! -f "gym_microrts/__init__.py" ]; then
    echo "ERROR: Run from MicroRTS-Py-Research root (or use this script path)."
    exit 1
fi

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate microrts39 2>/dev/null || true
fi

export JAVA_HOME="${JAVA_HOME:-${CONDA_PREFIX:-/usr}}"
export PATH="$JAVA_HOME/bin:$PATH"
export CLASSPATH="${PROJECT_ROOT}/gym_microrts/microrts/microrts.jar:${PROJECT_ROOT}/gym_microrts/microrts/lib/*"

# Smoke profile (see apply_smoke_run_defaults in run_single_gene_balance_decrease.py)
export SINGLE_GENE_SMOKE=1
export SINGLE_GENE_MAP_PATH="${SINGLE_GENE_MAP_PATH:-maps/16x16/basesWorkers16x16.xml}"

echo "Single-gene local smoke run"
echo "==========================="
echo "Project: ${PROJECT_ROOT}"
echo "SINGLE_GENE_SMOKE=1 (6 gens, pop 8, 4 games/ordering, 20k max_steps unless overridden)"
echo ""

cd "scripts/GA and MAP-Elites"
python run_single_gene_balance_decrease.py

echo ""
echo "Done. Results under: scripts/GA and MAP-Elites/single_gene_experiment/runs/"
echo "  - gene_trajectory.csv, generations_summary.csv, plots, run_config.txt"
