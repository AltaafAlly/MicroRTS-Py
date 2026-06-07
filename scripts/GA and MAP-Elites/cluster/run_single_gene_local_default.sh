#!/bin/bash
# Local run matching validated profile single_gene_balance_decrease_20260604_122522.
# Key: SINGLE_GENE_DEC_JUMP_RATE=0.05 (0.08 causes late-gen HP spikes back toward 100k).
#
# From repo root:
#   ./scripts/GA\ and\ MAP-Elites/cluster/run_single_gene_local_default.sh
#
# Faster (skip per-match .txt): SINGLE_GENE_SAVE_GAME_DETAILS=0 ./scripts/GA\ and\ MAP-Elites/cluster/run_single_gene_local_default.sh

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

export SINGLE_GENE_SMOKE=0
export SINGLE_GENE_MAP_PATH="${SINGLE_GENE_MAP_PATH:-maps/16x16/basesWorkers16x16.xml}"
export SINGLE_GENE_DEC_GENS="${SINGLE_GENE_DEC_GENS:-40}"
export SINGLE_GENE_DEC_POP="${SINGLE_GENE_DEC_POP:-16}"
export SINGLE_GENE_DEC_GAMES="${SINGLE_GENE_DEC_GAMES:-8}"
export SINGLE_GENE_MAX_STEPS="${SINGLE_GENE_MAX_STEPS:-50000}"
export SINGLE_GENE_DEC_JUMP_RATE="${SINGLE_GENE_DEC_JUMP_RATE:-0.05}"
export SINGLE_GENE_LOG_SCALE="${SINGLE_GENE_LOG_SCALE:-1}"
export SINGLE_GENE_PARAM_MAX_WIDE="${SINGLE_GENE_PARAM_MAX_WIDE:-100000}"
export SINGLE_GENE_SAVE_GAME_DETAILS="${SINGLE_GENE_SAVE_GAME_DETAILS:-1}"

echo "Single-gene local default (122522 profile)"
echo "=========================================="
echo "Project: ${PROJECT_ROOT}"
echo "SINGLE_GENE_DEC_GENS=${SINGLE_GENE_DEC_GENS} POP=${SINGLE_GENE_DEC_POP} GAMES=${SINGLE_GENE_DEC_GAMES}"
echo "SINGLE_GENE_MAX_STEPS=${SINGLE_GENE_MAX_STEPS} JUMP_RATE=${SINGLE_GENE_DEC_JUMP_RATE}"
echo "SINGLE_GENE_SAVE_GAME_DETAILS=${SINGLE_GENE_SAVE_GAME_DETAILS}"
echo ""

cd "scripts/GA and MAP-Elites"
python run_single_gene_balance_decrease.py

echo ""
echo "Done. Results under: scripts/GA and MAP-Elites/single_gene_experiment/runs/"
