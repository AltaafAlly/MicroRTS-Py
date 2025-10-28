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

# Set paths first
BASE_REPO="$HOME/Research/MicroRTS-Py-Research"

# Set Java environment
export JAVA_HOME=${JAVA_HOME:-/usr}
export PATH="$JAVA_HOME/bin:$PATH"
echo "Java home: $JAVA_HOME"
echo "Java path: $(which java)"

# Set Java classpath for JPype
export CLASSPATH="${BASE_REPO}/gym_microrts/microrts/microrts.jar:${BASE_REPO}/gym_microrts/microrts/lib/*"
echo "Java classpath: $CLASSPATH"
RUN_DIR="/home-mscluster/${USER}/microrts_runs/utt_tournament_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"

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

# Run tournament using the local testing approach
echo "Starting UTT Impact Tournament at $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Java version: $(java -version 2>&1 | head -1)"

# Create a wrapper script similar to local testing
cat > "${RUN_DIR}/run_tournament.py" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add the scripts directory to the path
sys.path.append("scripts/Running Simulations")

from utt_impact_tournament import UTTImpactTournament

def main():
    print("=" * 60)
    print("CLUSTER UTT IMPACT TOURNAMENT")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Get output directory from environment
    output_dir = os.environ.get('TOURNAMENT_OUTPUT_DIR', 'results')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path.absolute()}")
    
    # Create tournament instance
    tournament = UTTImpactTournament(output_dir=str(output_path))
    
    # Use cluster settings (15 games, 5000 steps)
    tournament.tournament_config["games_per_pair"] = 5
    tournament.tournament_config["max_steps"] = 5000
    tournament.tournament_config["max_steps_long"] = 10000
    
    print(f"Testing {len(tournament.utt_configs)} UTT configurations")
    print(f"Using {len(tournament.baseline_ais)} AI agents")
    
    try:
        # Run the tournament
        tournament.run_full_tournament()
        
        print("\n" + "=" * 60)
        print("TOURNAMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {output_path.absolute()}")
        
    except Exception as e:
        print(f"\nERROR: Tournament failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
EOF

# Set environment variable for output directory
export TOURNAMENT_OUTPUT_DIR="${RUN_DIR}/results"

# Run the tournament
python "${RUN_DIR}/run_tournament.py" 2>&1 | tee "${RUN_DIR}/utt_tournament_stdout.log"

echo "Tournament completed at $(date)"

# Run analysis if the tournament succeeded
if [ -f "${RUN_DIR}/results/tournament_results.json" ]; then
    echo "Running post-tournament analysis..."
    python "scripts/Running Simulations/analyze_tournament_results.py" \
        --input-dir "${RUN_DIR}/results" \
        --output-dir "${RUN_DIR}/results/analysis" \
        2>&1 | tee "${RUN_DIR}/analysis_stdout.log"
    
    echo "Analysis completed at $(date)"
else
    echo "Tournament failed - skipping analysis"
fi

echo "Results saved to ${RUN_DIR}/results"
