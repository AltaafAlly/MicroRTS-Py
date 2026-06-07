#!/bin/bash
# Sync MicroRTS-Py-Research code to cluster
# ==========================================
# This script uses rsync to update code on the cluster
#
# Usage:
#   ./sync_to_cluster.sh [user] [host] [remote_dir_under_~/Research]
#
# Examples:
#   ./sync_to_cluster.sh
#   ./sync_to_cluster.sh aally 146.141.21.100
#   # Repo root is directly ~/Research (no MicroRTS-Py-Research subfolder):
#   ./sync_to_cluster.sh aally 146.141.21.100 .
#
# If you use ".", update SLURM scripts' BASE_REPO to match (e.g. $HOME/Research).

set -euo pipefail

CLUSTER_USER=${1:-"aally"}
CLUSTER_HOST=${2:-"146.141.21.100"}
REMOTE_UNDER_RESEARCH="${3:-MicroRTS-Py-Research}"
if [ "${REMOTE_UNDER_RESEARCH}" = "." ]; then
    CLUSTER_PATH="${CLUSTER_USER}@${CLUSTER_HOST}:~/Research"
    CD_HINT="~/Research"
else
    CLUSTER_PATH="${CLUSTER_USER}@${CLUSTER_HOST}:~/Research/${REMOTE_UNDER_RESEARCH}"
    CD_HINT="~/Research/${REMOTE_UNDER_RESEARCH}"
fi

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
    --exclude='runs/' \
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
echo "  cd ${CD_HINT}"
echo "  ls -la"

