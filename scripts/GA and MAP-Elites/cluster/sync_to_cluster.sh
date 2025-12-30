#!/bin/bash
# Sync MicroRTS-Py-Research code to cluster
# ==========================================
# This script uses rsync to update code on the cluster
#
# Usage:
#   ./sync_to_cluster.sh [cluster_username@cluster_host]
#
# Example:
#   ./sync_to_cluster.sh altaaf@mscluster.ms.unimelb.edu.au

set -euo pipefail

# Default cluster connection (customize this)
CLUSTER_USER=${1:-"altaaf"}
CLUSTER_HOST=${2:-"mscluster.ms.unimelb.edu.au"}
CLUSTER_PATH="${CLUSTER_USER}@${CLUSTER_HOST}:~/Research/MicroRTS-Py-Research"

# Get the project root directory (parent of scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "Syncing code to cluster..."
echo "=========================="
echo "Local directory: ${PROJECT_ROOT}"
echo "Cluster destination: ${CLUSTER_PATH}"
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Rsync with exclusions
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.pytest_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='*.egg-info/' \
    --exclude='.idea/' \
    --exclude='.vscode/' \
    --exclude='*.swp' \
    --exclude='*.swo' \
    --exclude='*~' \
    --exclude='.DS_Store' \
    --exclude='experiments/' \
    --exclude='results/' \
    --exclude='job_logs/' \
    --exclude='ga_runs/' \
    --exclude='microrts_runs/' \
    --exclude='gym_microrts/microrts/microrts.jar' \
    --exclude='gym_microrts/microrts/bin/' \
    --exclude='gym_microrts/microrts/fat_jar_temp/' \
    --exclude='*.log' \
    --exclude='*.out' \
    --exclude='*.err' \
    ./ "${CLUSTER_PATH}/"

echo ""
echo "Sync completed!"
echo ""
echo "To verify, you can SSH to the cluster and check:"
echo "  ssh ${CLUSTER_USER}@${CLUSTER_HOST}"
echo "  cd ~/Research/MicroRTS-Py-Research"
echo "  ls -la"

