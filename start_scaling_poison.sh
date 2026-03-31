#!/usr/bin/env bash
set -euo pipefail

srun --cpus-per-task=16 --mem=80GB --gpus=1 -C gpu_a100_80gb --time=02:20:00 --pty bash -lc '
  echo "Starting strong reject server..."
  # Non-interactive login shells may not source ~/.bashrc automatically.
  if [ -f ~/.bashrc ]; then
    source ~/.bashrc
  fi

  module load anaconda3
  export PATH=/home/caydenw/var/cuda-12.4/bin:$PATH
  export LD_LIBRARY_PATH=/home/caydenw/var/cuda-12.4/lib64:$LD_LIBRARY_PATH
  cd /home/caydenw/git/strong-reject/
  screen -dmS strong-reject bash -lc "source \"$(conda info --base)/etc/profile.d/conda.sh\"; conda activate strong-reject; python strong_reject_server.py"

  cd /home/caydenw/git/scaling-poisoning/
  export MLOP_PROJECT="scaling-poisoning"
  export MLOP_DIR="/scratch/caydenw/.mlop"

  # Prefer an explicit host setting for self-hosted MLOP instances.
  if [ -z "${MLOP_HOST:-}" ] && [ -n "${MLOP_URL_APP:-}" ]; then
    export MLOP_HOST="${MLOP_URL_APP}"
  fi

  if [ -z "${MLOP_API_KEY:-}" ] && [ -f /home/caydenw/git/scaling-poisoning/mlop_api_key.txt ]; then
    export MLOP_API_KEY="$(tr -d "\r\n" < /home/caydenw/git/scaling-poisoning/mlop_api_key.txt)"
  fi

  if [ -n "${MLOP_API_KEY:-}" ]; then
    echo "MLOP_API_KEY detected in session."
  else
    echo "WARNING: MLOP_API_KEY is not set in this session."
  fi

  echo "MLOP_PROJECT=${MLOP_PROJECT:-<unset>}"
  echo "MLOP_RUN_NAME=${MLOP_RUN_NAME:-<auto>}"
  echo "MLOP_HOST=${MLOP_HOST:-<default>}"
  echo "MLOP_URL_APP=${MLOP_URL_APP:-<default>}"
  echo "MLOP_URL_API=${MLOP_URL_API:-<default>}"
  echo "MLOP_URL_INGEST=${MLOP_URL_INGEST:-<default>}"
  echo "MLOP_URL_PY=${MLOP_URL_PY:-<default>}"

  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scaling-poison
  exec bash
'