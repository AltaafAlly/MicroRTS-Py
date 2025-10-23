#!/bin/bash
#SBATCH --job-name=utt_tournament
#SBATCH --output=/home-mscluster/%u/job_logs/utt_tournament_%j.out
#SBATCH --error=/home-mscluster/%u/job_logs/utt_tournament_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=1

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOBID=${SLURM_JOB_ID}"

# Create job logs directory if it doesn't exist
mkdir -p /home-mscluster/${USER}/job_logs

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate microrts39

# Set Java environment
export JAVA_HOME=${CONDA_PREFIX}
export PATH="$JAVA_HOME/bin:$PATH"

# Set paths
BASE_REPO="$HOME/Research/MicroRTS-Py-Research"
RUN_DIR="/home-mscluster/${USER}/microrts_runs/utt_tournament_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"

mkdir -p "${RUN_DIR}"
cd "${BASE_REPO}"

echo "Working directory: ${RUN_DIR}"
echo "Repository: ${BASE_REPO}"

# Run tournament
echo "Starting UTT Impact Tournament at $(date)"
python "scripts/Running Simulations/utt_impact_tournament.py" \
    --output-dir "${RUN_DIR}/results" \
    --games 15 \
    --max-steps 5000 \
    2>&1 | tee "${RUN_DIR}/utt_tournament_stdout.log"

echo "Tournament completed at $(date)"

# Run analysis
echo "Running post-tournament analysis..."
python "scripts/Running Simulations/analyze_tournament_results.py" \
    --input-dir "${RUN_DIR}/results" \
    --output-dir "${RUN_DIR}/results/analysis" \
    2>&1 | tee "${RUN_DIR}/analysis_stdout.log"

echo "Analysis completed at $(date)"
echo "Results saved to ${RUN_DIR}/results"
