#!/bin/bash
#SBATCH --job-name=ga_local_test
#SBATCH --output=/home-mscluster/%u/job_logs/ga_local_test_%j.out
#SBATCH --error=/home-mscluster/%u/job_logs/ga_local_test_%j.err
#SBATCH --time=2-00:00:00
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
RUN_DIR="/home-mscluster/${USER}/ga_runs/ga_local_two_ai_test_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
cd "${BASE_REPO}"

echo "Working directory: ${RUN_DIR}"
echo "Repository: ${BASE_REPO}"

# Rebuild JAR to match cluster Java version (reuse logic from other submit script)
echo "Rebuilding microrts.jar for cluster Java version..."
cd gym_microrts/microrts

rm -rf bin fat_jar_temp
echo "Compiling Java sources with cluster Java version..."
find src -name "*.java" > sources.txt
mkdir -p bin
javac -cp "lib/*" -d bin @sources.txt
rm sources.txt

echo "Creating fat JAR with all dependencies..."
mkdir -p fat_jar_temp
cp -r bin/* fat_jar_temp/ 2>/dev/null || true

if [ -f "microrts.jar" ]; then
    echo "Extracting existing microrts.jar..."
    jar xf microrts.jar -C fat_jar_temp/ 2>/dev/null || true
fi

echo "Extracting library dependencies..."
for lib_jar in lib/*.jar; do
    if [ -f "$lib_jar" ]; then
        echo "  Extracting $(basename "$lib_jar")..."
        jar xf "$lib_jar" -C fat_jar_temp/ 2>/dev/null || true
    fi
done

echo "Creating new microrts.jar..."
jar cf microrts.jar -C fat_jar_temp .

rm -rf fat_jar_temp bin

echo "Built microrts.jar successfully for cluster Java version"
echo "JAR size: $(du -h microrts.jar | cut -f1)"

cd "${BASE_REPO}"

echo "Verifying MicroRTS setup..."
echo "JAR file exists: $(ls -la gym_microrts/microrts/microrts.jar 2>/dev/null || echo 'NOT FOUND')"
echo "JAR size: $(du -h gym_microrts/microrts/microrts.jar 2>/dev/null || echo 'N/A')"
echo "Lib directory: $(ls -la gym_microrts/microrts/lib/ 2>/dev/null | wc -l) files"

# GA configuration for local two-AI (rush-AI) test on cluster
GENERATIONS=${GA_GENERATIONS:-40}
POPULATION=${GA_POPULATION:-25}
GAMES_PER_EVAL=${GA_GAMES_PER_EVAL:-10}
MAX_STEPS=${GA_MAX_STEPS:-20000}

echo ""
echo "GA Local Two-AI Test Configuration (cluster):"
echo "============================================="
echo "Generations: ${GENERATIONS}"
echo "Population: ${POPULATION}"
echo "Games per evaluation (per pair, per ordering): ${GAMES_PER_EVAL}"
echo "Max steps per game: ${MAX_STEPS}"
echo ""

# Export overrides so run_ga_local_test.py can see them
export GA_GENERATIONS=${GENERATIONS}
export GA_POPULATION=${POPULATION}
export GA_GAMES_PER_EVAL=${GAMES_PER_EVAL}
export GA_MAX_STEPS=${MAX_STEPS}

cd "scripts/GA and MAP-Elites"

python run_ga_local_test.py \
    2>&1 | tee "${RUN_DIR}/ga_local_two_ai_test_stdout.log"

echo ""
echo "GA local_two_ai_test job finished at $(date)"

