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

# Create a wrapper script similar to local testing with timing tracking
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
    tournament_start_time = datetime.now()
    print(f"Started at: {tournament_start_time}")
    
    # Get output directory from environment
    output_dir = os.environ.get('TOURNAMENT_OUTPUT_DIR', 'results')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path.absolute()}")
    
    # Create tournament instance
    tournament = UTTImpactTournament(output_dir=str(output_path))
    
    # Use cluster settings (5 games, 5000 steps)
    tournament.tournament_config["games_per_pair"] = 5
    tournament.tournament_config["max_steps"] = 5000
    tournament.tournament_config["max_steps_long"] = 10000
    
    print(f"Testing {len(tournament.utt_configs)} UTT configurations")
    print(f"Using {len(tournament.baseline_ais)} AI agents")
    
    # Track match timings
    match_timings = []
    
    # Monkey patch the run_pair method to track timing
    original_run_pair = tournament.run_pair
    
    def timed_run_pair(ai_left, ai_right, utt_config, pair_num, total_pairs):
        match_start = time.time()
        result = original_run_pair(ai_left, ai_right, utt_config, pair_num, total_pairs)
        match_end = time.time()
        match_duration = match_end - match_start
        
        # Store timing information
        match_timings.append({
            'pair_number': pair_num,
            'total_pairs': total_pairs,
            'ai_left': ai_left,
            'ai_right': ai_right,
            'utt_config': utt_config.get('utt_json_p0', 'default'),
            'left_wins': result.get('left_wins', 0),
            'right_wins': result.get('right_wins', 0),
            'draws': result.get('draws', 0),
            'total_games': result.get('left_wins', 0) + result.get('right_wins', 0) + result.get('draws', 0),
            'match_time_seconds': match_duration,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"[TIMING] Match {pair_num}/{total_pairs} ({ai_left} vs {ai_right}): {match_duration:.2f} seconds")
        
        return result
    
    tournament.run_pair = timed_run_pair
    
    try:
        # Run the tournament
        tournament.run_full_tournament()
        
        # Save match timings
        timings_file = output_path / "match_timings.json"
        with open(timings_file, 'w') as f:
            json.dump(match_timings, f, indent=2)
        
        # Also save as CSV for easy viewing
        import csv
        csv_file = output_path / "match_timings.csv"
        if match_timings:
            with open(csv_file, 'w', newline='') as f:
                fieldnames = [
                    'pair_number', 'total_pairs', 'ai_left', 'ai_right', 'utt_config',
                    'left_wins', 'right_wins', 'draws', 'total_games',
                    'match_time_seconds', 'match_time_formatted', 'timestamp'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for timing in match_timings:
                    # Format time nicely (HH:MM:SS or seconds)
                    match_time_sec = timing['match_time_seconds']
                    if match_time_sec >= 3600:
                        hours = int(match_time_sec // 3600)
                        minutes = int((match_time_sec % 3600) // 60)
                        seconds = int(match_time_sec % 60)
                        formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    elif match_time_sec >= 60:
                        minutes = int(match_time_sec // 60)
                        seconds = int(match_time_sec % 60)
                        formatted = f"{minutes:02d}:{seconds:02d}"
                    else:
                        formatted = f"{match_time_sec:.2f}s"
                    
                    row = timing.copy()
                    row['match_time_formatted'] = formatted
                    writer.writerow({k: row.get(k, '') for k in fieldnames})
        
        tournament_end_time = datetime.now()
        total_duration = (tournament_end_time - tournament_start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("TOURNAMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total tournament time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"Results saved to: {output_path.absolute()}")
        print(f"Match timings saved to: {timings_file} and {csv_file}")
        
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
if [ -f "${RUN_DIR}/results/tournament_results.json" ] || [ -f "${RUN_DIR}/results/match_timings.csv" ]; then
    echo "Running post-tournament analysis..."
    python "scripts/Running Simulations/analyze_tournament_results.py" \
        --input-dir "${RUN_DIR}/results" \
        --output-dir "${RUN_DIR}/results/analysis" \
        2>&1 | tee "${RUN_DIR}/analysis_stdout.log"
    
    echo "Analysis completed at $(date)"
    
    # Check if match timings file exists and report
    if [ -f "${RUN_DIR}/results/match_timings.csv" ]; then
        echo ""
        echo "Match timings summary:"
        echo "======================"
        echo "Match timings file: ${RUN_DIR}/results/match_timings.csv"
        echo "Match timings JSON: ${RUN_DIR}/results/match_timings.json"
        echo ""
        echo "First few matches (showing timing):"
        head -5 "${RUN_DIR}/results/match_timings.csv" || echo "Unable to display match timings"
    fi
else
    echo "Tournament failed - skipping analysis"
fi

echo ""
echo "Results saved to ${RUN_DIR}/results"
echo "Match timings file: ${RUN_DIR}/results/match_timings.csv"
