#!/bin/bash
#SBATCH --job-name=microrts_test
#SBATCH --output=/home-mscluster/%u/microrts_test_%j.out
#SBATCH --error=/home-mscluster/%u/microrts_test_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=stampede

echo "Starting MicroRTS cluster test at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Activate conda environment
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

# Run the test
echo "Running MicroRTS cluster test..."
python "scripts/Running Simulations/simple_cluster_test.py"

echo "Test completed at $(date)"
