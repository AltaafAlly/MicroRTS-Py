#!/bin/bash
#SBATCH --job-name=ga_evolution
#SBATCH --output=/home-mscluster/%u/job_logs/ga_evolution_%j.out
#SBATCH --error=/home-mscluster/%u/job_logs/ga_evolution_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=1

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOBID=${SLURM_JOB_ID}"

# Create job logs directory if it doesn't exist
mkdir -p /home-mscluster/${USER}/job_logs

# Initialize and activate conda environment
echo "Initializing conda..."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
echo "Conda initialized"

echo "Available conda environments:"
conda env list

echo "Activating microrts39 environment..."
conda activate microrts39
echo "Activated conda environment: microrts39"
echo "Conda prefix: $CONDA_PREFIX"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Set paths
BASE_REPO="$HOME/Research/MicroRTS-Py-Research"

# Set Java environment
export JAVA_HOME=${JAVA_HOME:-/usr}
export PATH="$JAVA_HOME/bin:$PATH"
echo "Java home: $JAVA_HOME"
echo "Java path: $(which java)"
echo "Java version: $(java -version 2>&1 | head -1)"

# Set Java classpath for JPype
export CLASSPATH="${BASE_REPO}/gym_microrts/microrts/microrts.jar:${BASE_REPO}/gym_microrts/microrts/lib/*"
echo "Java classpath: $CLASSPATH"

# Create run directory
RUN_DIR="/home-mscluster/${USER}/ga_runs/ga_experiment_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
cd "${BASE_REPO}"

echo "Working directory: ${RUN_DIR}"
echo "Repository: ${BASE_REPO}"

# Rebuild JAR to match cluster Java version
echo "Rebuilding microrts.jar for cluster Java version..."
cd gym_microrts/microrts

# Clean previous build
rm -rf bin fat_jar_temp

# Compile Java sources with cluster Java version
echo "Compiling Java sources with cluster Java version..."
find src -name "*.java" > sources.txt
mkdir -p bin
javac -cp "lib/*" -d bin @sources.txt
rm sources.txt

# Create fat JAR with all dependencies
echo "Creating fat JAR with all dependencies..."
mkdir -p fat_jar_temp

# Copy compiled classes
cp -r bin/* fat_jar_temp/ 2>/dev/null || true

# Extract existing microrts.jar if it exists
if [ -f "microrts.jar" ]; then
    echo "Extracting existing microrts.jar..."
    jar xf microrts.jar -C fat_jar_temp/ 2>/dev/null || true
fi

# Extract all library JARs
echo "Extracting library dependencies..."
for lib_jar in lib/*.jar; do
    if [ -f "$lib_jar" ]; then
        echo "  Extracting $(basename "$lib_jar")..."
        jar xf "$lib_jar" -C fat_jar_temp/ 2>/dev/null || true
    fi
done

# Create the new fat JAR
echo "Creating new microrts.jar..."
jar cf microrts.jar -C fat_jar_temp .

# Clean up
rm -rf fat_jar_temp bin

echo "Built microrts.jar successfully for cluster Java version"
echo "JAR size: $(du -h microrts.jar | cut -f1)"

cd "${BASE_REPO}"

# Verify JAR file and dependencies exist
echo "Verifying MicroRTS setup..."
echo "JAR file exists: $(ls -la gym_microrts/microrts/microrts.jar 2>/dev/null || echo 'NOT FOUND')"
echo "JAR size: $(du -h gym_microrts/microrts/microrts.jar 2>/dev/null || echo 'N/A')"
echo "Lib directory: $(ls -la gym_microrts/microrts/lib/ 2>/dev/null | wc -l) files"

# GA Configuration
# Optimized for 3-day time limit with checkpointing/resume capability
# With checkpointing, we can afford more games per evaluation for better fitness estimates
# You can customize these parameters or pass them via environment variables
GENERATIONS=${GA_GENERATIONS:-20}
POPULATION=${GA_POPULATION:-25}
EXPERIMENT_NAME=${GA_EXPERIMENT_NAME:-"cluster_long_run"}
GAMES_PER_EVAL=${GA_GAMES_PER_EVAL:-5}  # Increased from 3 to 5 for better fitness accuracy
MAX_STEPS=${GA_MAX_STEPS:-5000}
MUTATION_RATE=${GA_MUTATION_RATE:-0.15}
MAX_GENERATIONS_WITHOUT_IMPROVEMENT=${GA_MAX_GEN_NO_IMPROVE:-8}

echo ""
echo "GA Configuration:"
echo "=================="
echo "Generations: ${GENERATIONS}"
echo "Population: ${POPULATION}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Games per evaluation: ${GAMES_PER_EVAL}"
echo "Max steps per game: ${MAX_STEPS}"
# Calculate number of pairs dynamically (8 AIs = 28 pairs, n choose 2)
NUM_AIS=8
NUM_PAIRS=$((NUM_AIS * (NUM_AIS - 1) / 2))
echo "Round-robin pairs: ${NUM_PAIRS} (all AIs vs each other)"
echo "Total games per chromosome: $((NUM_PAIRS * ${GAMES_PER_EVAL}))"
echo "Total games per generation: $((NUM_PAIRS * ${GAMES_PER_EVAL} * ${POPULATION}))"
echo ""

# Create experiment output directory in the run directory
EXPERIMENT_OUTPUT_DIR="${RUN_DIR}/experiments"
mkdir -p "${EXPERIMENT_OUTPUT_DIR}"

# Checkpoint directory for resuming
# Look for latest checkpoint in ALL previous run directories (not just current one)
# This allows automatic resume when you resubmit the job after a time limit
# Search in the parent ga_runs directory to find checkpoints from previous jobs
GA_RUNS_BASE="/home-mscluster/${USER}/ga_runs"
echo "Searching for checkpoints in: ${GA_RUNS_BASE}"

# Find all checkpoints, prefer those matching the experiment name
# First try to find checkpoint from same experiment name
EXPERIMENT_CHECKPOINT=$(find ${GA_RUNS_BASE} -path "*/experiments/${EXPERIMENT_NAME}_*/checkpoints/checkpoint_latest.json" -type f 2>/dev/null | sort -r | head -1)

# If not found, use any latest checkpoint
if [ -z "${EXPERIMENT_CHECKPOINT}" ]; then
    LATEST_CHECKPOINT=$(find ${GA_RUNS_BASE} -name "checkpoint_latest.json" -type f 2>/dev/null | sort -r | head -1)
else
    LATEST_CHECKPOINT="${EXPERIMENT_CHECKPOINT}"
fi

# Check if we should resume from checkpoint
RESUME_FLAG=""
if [ -n "${LATEST_CHECKPOINT}" ] && [ -f "${LATEST_CHECKPOINT}" ]; then
    echo "=========================================="
    echo "AUTOMATIC RESUME DETECTED"
    echo "=========================================="
    echo "Found checkpoint: ${LATEST_CHECKPOINT}"
    echo "Checkpoint modified: $(stat -c %y "${LATEST_CHECKPOINT}" 2>/dev/null || echo 'unknown')"
    echo "Resuming from checkpoint automatically..."
    echo "=========================================="
    RESUME_FLAG="--resume-from ${LATEST_CHECKPOINT}"
else
    echo "No checkpoint found. Starting new run."
    if [ -n "${EXPERIMENT_CHECKPOINT}" ]; then
        echo "Note: Found checkpoint but it may be from a different experiment."
    fi
fi

# Run GA
echo "Starting Genetic Algorithm at $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Java version: $(java -version 2>&1 | head -1)"

# Change to GA directory and run
cd "scripts/GA and MAP-Elites"

# Run GA with cluster-optimized settings
# Checkpoint directory will be set automatically in experiment dir
python run_ga.py \
    --generations ${GENERATIONS} \
    --population ${POPULATION} \
    --experiment-name "${EXPERIMENT_NAME}" \
    --output-dir "${EXPERIMENT_OUTPUT_DIR}" \
    --save-results \
    --use-working-evaluator \
    --games-per-eval ${GAMES_PER_EVAL} \
    --max-steps ${MAX_STEPS} \
    --mutation-rate ${MUTATION_RATE} \
    --max-generations-without-improvement ${MAX_GENERATIONS_WITHOUT_IMPROVEMENT} \
    --checkpoint-dir "" \
    ${RESUME_FLAG} \
    --verbose \
    2>&1 | tee "${RUN_DIR}/ga_stdout.log"

# Check if evolution completed or if we need to resubmit
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Job exited with code $EXIT_CODE"
    echo "This may indicate the job hit the time limit."
    echo "Use auto_resubmit_ga.sh to automatically continue, or resubmit manually."
fi

# Check if evolution completed
if grep -q "EVOLUTION COMPLETED" "${RUN_DIR}/ga_stdout.log" 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo "✅ EVOLUTION COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "⚠️  Job finished but evolution may not be complete"
    echo "=========================================="
    echo "Checkpoint saved. Resubmit to continue:"
    echo "  sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch"
    echo "Or use auto-resubmit wrapper:"
    echo "  ./scripts/GA\ and\ MAP-Elites/cluster/auto_resubmit_ga.sh"
    echo "=========================================="
    exit 0  # Exit successfully so SLURM doesn't mark as failed
fi

GA_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "GA completed at $(date)"
echo "Exit code: ${GA_EXIT_CODE}"

# Check if GA completed successfully
if [ ${GA_EXIT_CODE} -eq 0 ]; then
    echo "GA completed successfully!"
    
    # Find the experiment directory
    EXPERIMENT_DIR=$(find "${EXPERIMENT_OUTPUT_DIR}" -name "${EXPERIMENT_NAME}_*" -type d | head -1)
    
    if [ -n "${EXPERIMENT_DIR}" ]; then
        echo "Experiment results saved to: ${EXPERIMENT_DIR}"
        
        # List experiment files
        echo ""
        echo "Experiment files:"
        ls -lh "${EXPERIMENT_DIR}" || echo "Could not list experiment files"
        
        # Show summary if results file exists
        if [ -f "${EXPERIMENT_DIR}/ga_results.json" ]; then
            echo ""
            echo "Experiment Summary:"
            echo "==================="
            python -c "
import json
import sys
try:
    with open('${EXPERIMENT_DIR}/ga_results.json', 'r') as f:
        results = json.load(f)
    best_fitness = results.get('best_fitness', {})
    print(f\"Best Fitness: {best_fitness.get('overall_fitness', 'N/A'):.4f}\")
    print(f\"  Balance: {best_fitness.get('balance', 'N/A'):.4f}\")
    print(f\"  Duration: {best_fitness.get('duration', 'N/A'):.4f}\")
    print(f\"  Diversity: {best_fitness.get('strategy_diversity', 'N/A'):.4f}\")
    print(f\"Total Generations: {results.get('total_generations', 'N/A')}\")
    print(f\"Total Time: {results.get('total_time', 'N/A'):.2f} seconds\")
except Exception as e:
    print(f'Error reading results: {e}')
    sys.exit(1)
"
        fi
    else
        echo "Warning: Could not find experiment directory"
    fi
else
    echo "GA failed with exit code ${GA_EXIT_CODE}"
    echo "Check the log file: ${RUN_DIR}/ga_stdout.log"
fi

echo ""
echo "Results saved to: ${RUN_DIR}"
echo "Experiment directory: ${EXPERIMENT_DIR:-'Not found'}"

