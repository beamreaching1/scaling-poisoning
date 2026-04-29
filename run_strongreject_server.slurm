#!/usr/bin/env bash
#SBATCH --job-name=strongreject-server
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/strongreject_server_%j.out
#SBATCH --error=logs/strongreject_server_%j.err

# Central StrongREJECT evaluation server.
#
# Submit BEFORE submitting any pipeline array jobs:
#   sbatch run_strongreject_server.slurm
#
# Pipeline jobs read the node/port from the shared env file written below
# and set STRONGREJECT_SERVER_URL accordingly.
#
# If the server job is preempted, --requeue will reschedule it automatically.
# The env file is removed on exit/preemption so pipeline jobs fail fast
# rather than hitting an unreachable address.

set -euo pipefail

CENTRAL_ENV=/scratch/caydenw/strongreject_central.env
STRONGREJECT_PORT=8100

# ── Environment setup ────────────────────────────────────────────────────────
if [ -f ~/.bashrc ]; then
  set +u
  source ~/.bashrc || true
  set -u
fi

module load anaconda3
export PATH=/home/caydenw/var/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/home/caydenw/var/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate strong-reject

# ── Broadcast node/port for pipeline jobs ───────────────────────────────────
mkdir -p "$(dirname "${CENTRAL_ENV}")"
cat > "${CENTRAL_ENV}" <<EOF
STRONGREJECT_NODE=${SLURMD_NODENAME}
STRONGREJECT_PORT=${STRONGREJECT_PORT}
EOF
echo "Central StrongREJECT env written to ${CENTRAL_ENV}"
echo "  STRONGREJECT_NODE=${SLURMD_NODENAME}"
echo "  STRONGREJECT_PORT=${STRONGREJECT_PORT}"

# Remove env file when this job exits or is preempted
trap 'echo "Removing ${CENTRAL_ENV}"; rm -f "${CENTRAL_ENV}"' EXIT

# ── Start server ─────────────────────────────────────────────────────────────
echo "Starting StrongREJECT server on ${SLURMD_NODENAME}:${STRONGREJECT_PORT}"
echo "Time: $(date)"

cd /home/caydenw/git/scaling-poisoning/

STRONGREJECT_PORT=${STRONGREJECT_PORT} python strong_reject_server.py
