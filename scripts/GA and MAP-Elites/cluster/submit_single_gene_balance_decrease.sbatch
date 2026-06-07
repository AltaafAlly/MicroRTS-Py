#!/bin/bash
#SBATCH --job-name=single_gene_ga
#SBATCH --output=/home-mscluster/%u/job_logs/single_gene_ga_%j.out
#SBATCH --error=/home-mscluster/%u/job_logs/single_gene_ga_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=stampede
#SBATCH --cpus-per-task=1
#
# Long single-gene balance-decrease run (lightRush vs heavyRush, Heavy.hp by default).
# Matches cluster setup in submit_ga.sbatch (conda, Java, fat microrts.jar).
#
# Submit from repo root on the cluster:
#   sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_single_gene_balance_decrease.sbatch
#
# Default map: 16×16 basesWorkers (below). Override for other maps, e.g. BroodWar:
#   export SINGLE_GENE_MAP_PATH='maps/BroodWar/(4)BloodBath.scmA.xml'
#   export SINGLE_GENE_MAX_STEPS=300000
#
# Other overrides before sbatch (examples):
#   export SINGLE_GENE_DEC_GENS=800
#   export SINGLE_GENE_DEC_POP=48
#   export SINGLE_GENE_DEC_GAMES=32
#   export SINGLE_GENE_SAVE_GAME_DETAILS=1
#   SINGLE_GENE_DEC_GAMES=24 sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_single_gene_balance_decrease.sbatch

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOBID=${SLURM_JOB_ID}"

mkdir -p /home-mscluster/${USER}/job_logs

echo "Initializing conda..."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate microrts39
echo "Python: $(which python)  $(python --version)"
if ! python -c "import matplotlib" >/dev/null 2>&1; then
    echo "matplotlib missing in microrts39; installing..."
    pip install matplotlib
fi
echo "matplotlib: $(python -c 'import matplotlib; print(matplotlib.__version__)')"

BASE_REPO="$HOME/Research/MicroRTS-Py-Research"

export JAVA_HOME=${JAVA_HOME:-/usr}
export PATH="$JAVA_HOME/bin:$PATH"
echo "Java: $(which java)  $(java -version 2>&1 | head -1)"

export CLASSPATH="${BASE_REPO}/gym_microrts/microrts/microrts.jar:${BASE_REPO}/gym_microrts/microrts/lib/*"

RUN_ROOT="/home-mscluster/${USER}/single_gene_runs"
mkdir -p "${RUN_ROOT}"
export SINGLE_GENE_RUN_PARENT="${RUN_ROOT}"

cd "${BASE_REPO}"

echo "Rebuilding microrts.jar for cluster Java..."
cd gym_microrts/microrts
rm -rf bin fat_jar_temp
find src -name "*.java" > sources.txt
mkdir -p bin
javac -cp "lib/*" -d bin @sources.txt
rm sources.txt
mkdir -p fat_jar_temp
cp -r bin/* fat_jar_temp/ 2>/dev/null || true
if [ -f "microrts.jar" ]; then
    jar xf microrts.jar -C fat_jar_temp/ 2>/dev/null || true
fi
for lib_jar in lib/*.jar; do
    [ -f "$lib_jar" ] && jar xf "$lib_jar" -C fat_jar_temp/ 2>/dev/null || true
done
jar cf microrts.jar -C fat_jar_temp .
rm -rf fat_jar_temp bin
echo "microrts.jar built: $(du -h microrts.jar | cut -f1)"

cd "${BASE_REPO}"

# --- Single-gene experiment tuning (massive defaults; override via env) ---
# 16×16 basesWorkers map (microrts-relative path)
export SINGLE_GENE_MAP_PATH="${SINGLE_GENE_MAP_PATH:-maps/16x16/basesWorkers16x16.xml}"
export SINGLE_GENE_DEC_GENS="${SINGLE_GENE_DEC_GENS:-800}"
export SINGLE_GENE_DEC_POP="${SINGLE_GENE_DEC_POP:-40}"
export SINGLE_GENE_DEC_GAMES="${SINGLE_GENE_DEC_GAMES:-32}"
export SINGLE_GENE_DEC_MUTATION_SIGMA="${SINGLE_GENE_DEC_MUTATION_SIGMA:-0.25}"
export SINGLE_GENE_DEC_JUMP_RATE="${SINGLE_GENE_DEC_JUMP_RATE:-0.10}"
export SINGLE_GENE_DEC_CROSSOVER="${SINGLE_GENE_DEC_CROSSOVER:-0.5}"
export SINGLE_GENE_USE_BOTH_ORDERINGS="${SINGLE_GENE_USE_BOTH_ORDERINGS:-1}"
# Log-scale genome: start high (up to PARAM_MAX_WIDE), evolve down toward balance (not toward reference)
export SINGLE_GENE_LOG_SCALE="${SINGLE_GENE_LOG_SCALE:-1}"
export SINGLE_GENE_PARAM_MAX_WIDE="${SINGLE_GENE_PARAM_MAX_WIDE:-100000}"
# Off on cluster by default: per-game .txt + snapshots explode disk and runtime
export SINGLE_GENE_SAVE_GAME_DETAILS="${SINGLE_GENE_SAVE_GAME_DETAILS:-0}"
export GA_SAVE_GAME_DETAILS="${GA_SAVE_GAME_DETAILS:-0}"
export SINGLE_GENE_QUIET_TERMINAL="${SINGLE_GENE_QUIET_TERMINAL:-1}"

echo ""
echo "Single-gene GA configuration"
echo "=============================="
echo "SINGLE_GENE_RUN_PARENT=${SINGLE_GENE_RUN_PARENT}"
echo "SINGLE_GENE_DEC_GENS=${SINGLE_GENE_DEC_GENS}"
echo "SINGLE_GENE_DEC_POP=${SINGLE_GENE_DEC_POP}"
echo "SINGLE_GENE_DEC_GAMES=${SINGLE_GENE_DEC_GAMES} (×2 if both orderings on)"
echo "SINGLE_GENE_DEC_MUTATION_SIGMA=${SINGLE_GENE_DEC_MUTATION_SIGMA}"
echo "SINGLE_GENE_DEC_JUMP_RATE=${SINGLE_GENE_DEC_JUMP_RATE}"
echo "SINGLE_GENE_LOG_SCALE=${SINGLE_GENE_LOG_SCALE}"
echo "SINGLE_GENE_PARAM_MAX_WIDE=${SINGLE_GENE_PARAM_MAX_WIDE}"
echo "SINGLE_GENE_SAVE_GAME_DETAILS=${SINGLE_GENE_SAVE_GAME_DETAILS}"
echo "SINGLE_GENE_AI1=${SINGLE_GENE_AI1:-lightRushAI}"
echo "SINGLE_GENE_AI2=${SINGLE_GENE_AI2:-heavyRushAI}"
echo "SINGLE_GENE_MAP_PATH=${SINGLE_GENE_MAP_PATH}"
echo ""

cd "scripts/GA and MAP-Elites"

STDOUT_LOG="${RUN_ROOT}/single_gene_stdout_job${SLURM_JOB_ID}.log"
echo "Streaming Python output to: ${STDOUT_LOG}"

python run_single_gene_balance_decrease.py 2>&1 | tee "${STDOUT_LOG}"

EXIT_CODE="${PIPESTATUS[0]}"
echo ""
echo "Python exit code: ${EXIT_CODE}"
echo "Stdout copy: ${STDOUT_LOG}"
echo "Per-run artifacts: under ${RUN_ROOT}/single_gene_balance_decrease_*_job${SLURM_JOB_ID}/"
if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "Job reported failure; check .out/.err and ${STDOUT_LOG}"
    exit "${EXIT_CODE}"
fi

echo "Finished at $(date)"
