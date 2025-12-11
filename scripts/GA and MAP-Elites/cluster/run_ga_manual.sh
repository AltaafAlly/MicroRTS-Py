#!/bin/bash
# Manual GA Test Script
# =====================
# This script runs the GA directly without SLURM
# Use this to test locally or run manually on the cluster

echo "Starting Manual GA Test"
echo "======================="
echo "Date: $(date)"
echo "User: $USER"
echo "Host: $(hostname)"

# Activate conda environment
echo "Activating conda environment..."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate microrts39

# Set Java environment
export JAVA_HOME=${JAVA_HOME:-${CONDA_PREFIX}}
export PATH="$JAVA_HOME/bin:$PATH"

echo "Java version:"
java -version

echo "Python version:"
python --version

echo "Working directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "gym_microrts/__init__.py" ]; then
    echo "ERROR: Not in the right directory. Please run from the MicroRTS-Py root directory."
    exit 1
fi

# Check if microrts.jar exists
if [ ! -f "gym_microrts/microrts/microrts.jar" ]; then
    echo "WARNING: microrts.jar not found. The GA might fail."
    echo "Available files in gym_microrts/microrts/:"
    ls -la gym_microrts/microrts/ | head -10
fi

# GA Configuration (can be overridden by environment variables)
GENERATIONS=${GA_GENERATIONS:-5}
POPULATION=${GA_POPULATION:-10}
EXPERIMENT_NAME=${GA_EXPERIMENT_NAME:-"manual_test_$(date +%Y%m%d_%H%M%S)"}
GAMES_PER_EVAL=${GA_GAMES_PER_EVAL:-3}
MAX_STEPS=${GA_MAX_STEPS:-1000}

echo ""
echo "GA Configuration:"
echo "=================="
echo "Generations: ${GENERATIONS}"
echo "Population: ${POPULATION}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Games per evaluation: ${GAMES_PER_EVAL}"
echo "Max steps per game: ${MAX_STEPS}"
echo ""

# Change to GA directory
cd "scripts/GA and MAP-Elites"

# Run the GA
echo ""
echo "Running GA..."
echo "============="

python run_ga.py \
    --generations ${GENERATIONS} \
    --population ${POPULATION} \
    --experiment-name "${EXPERIMENT_NAME}" \
    --save-results \
    --use-working-evaluator \
    --games-per-eval ${GAMES_PER_EVAL} \
    --max-steps ${MAX_STEPS} \
    --verbose

echo ""
echo "GA completed at $(date)"

