#!/bin/bash
# Local UTT Impact Tournament Runner
# ==================================

echo "Starting Local UTT Impact Tournament..."
echo "======================================"

# Activate conda environment (adjust if needed)
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate microrts39 2>/dev/null || conda activate microrts 2>/dev/null || echo "Warning: Could not activate conda environment"
fi

# Set Java environment (if needed)
export JAVA_HOME=${JAVA_HOME:-$(which java | sed 's|/bin/java||')}
export PATH="$JAVA_HOME/bin:$PATH"

echo "Java version:"
java -version 2>&1 | head -1

echo ""
echo "Python version:"
python --version

echo ""
echo "Starting tournament..."
echo "===================="

# Run the tournament
python run_local_tournament.py

echo ""
echo "Tournament completed!"
echo "Check the results in: results/"
