#!/bin/bash
#SBATCH --job-name=utt_tournament
#SBATCH --output=/home-mscluster/%u/utt_tournament_%j.out
#SBATCH --error=/home-mscluster/%u/utt_tournament_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=bigbatch

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOBID=${SLURM_JOB_ID}"
echo "User=${USER}"

# --------------------------------------------------
# Optional: load modules if your cluster provides them.
# On mscluster you generally manage Python via conda in your home.
# If your cluster has module support and you prefer modules, keep them.
# Otherwise you can omit the module lines.
# --------------------------------------------------
# module load python/3.9
# module load java/11

# Activate conda environment (adjust path if your conda is elsewhere)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda/etc/profile.d/conda.sh"
else
    echo "WARNING: conda.sh not found in expected locations. Make sure conda is installed."
fi

# Activate the environment you created (use microrts39 or microrts depending on what you named it)
conda activate microrts39

# Ensure Java is visible in the job environment (if using conda-installed openjdk, CONDA_PREFIX will point to env)
export JAVA_HOME=${JAVA_HOME:-${CONDA_PREFIX}}
export PATH="$JAVA_HOME/bin:$PATH"

# Install minimal dependencies for headless mode
pip install Pillow --quiet

# --- Set paths & run dir ---
BASE_REPO="$HOME/src/MicroRTS-Py"  # <-- Your actual repository path on cluster
HOME_BASE="/home-mscluster/${USER}/microrts_runs" # home directory storage

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${HOME_BASE}/utt_tournament_${TIMESTAMP}_job${SLURM_JOB_ID}"

mkdir -p "${RUN_DIR}"

# Set Java classpath for MicroRTS (after RUN_DIR is defined)
export CLASSPATH="${RUN_DIR}/code/gym_microrts/microrts/microrts.jar:${RUN_DIR}/code/gym_microrts/microrts/lib/*"

echo "Working directory: ${RUN_DIR}"
cd "${RUN_DIR}"

# Copy code (or bind) - small repo copy is optional; you can instead run from $BASE_REPO.
# This avoids accidental edits to the repo during a run and improves I/O locality.
rsync -av --exclude='.git' "${BASE_REPO}/" "${RUN_DIR}/code/"

# Make sure script paths contain no unescaped spaces — better to rename folders with spaces.
TOURNAMENT_SCRIPT="code/scripts/Running Simulations/utt_impact_tournament.py"
ANALYZE_SCRIPT="code/scripts/Running Simulations/analyze_tournament_results.py"

# If your paths actually have spaces, use the quoted form (as below).
echo "Starting UTT Impact Tournament at $(date)"
python "${TOURNAMENT_SCRIPT}" \
    --output-dir "${RUN_DIR}/results" \
    --games 15 \
    --max-steps 5000 \
    2>&1 | tee "${RUN_DIR}/utt_tournament_stdout.log"

echo "Tournament completed at $(date)"
echo "Results currently in: ${RUN_DIR}/results"

# Run post-tournament analysis (if desired)
echo "Running post-tournament analysis..."
python "${ANALYZE_SCRIPT}" \
    --input-dir "${RUN_DIR}/results" \
    --output-dir "${RUN_DIR}/results/analysis" \
    2>&1 | tee "${RUN_DIR}/analysis_stdout.log"

echo "Analysis completed at $(date)"

# Results are already in the home directory, no need to copy
echo "All done. Results saved to ${RUN_DIR}/results"
