#!/bin/bash
# Manual MicroRTS Test Script
# ===========================
# This script runs the MicroRTS test directly without SLURM
# Use this to test locally or run manually on the cluster

echo "Starting Manual MicroRTS Test"
echo "============================="
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
    echo "WARNING: microrts.jar not found. The test might fail."
    echo "Available files in gym_microrts/microrts/:"
    ls -la gym_microrts/microrts/ | head -10
fi

# Run the test
echo ""
echo "Running MicroRTS cluster test..."
echo "================================"

python "scripts/Running Simulations/simple_cluster_test.py"

echo ""
echo "Test completed at $(date)"
