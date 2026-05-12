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
  # Load sensitive configuration from .env
  if [ -f ".env" ]; then
    set -a; source ".env"; set +a
  fi
  export AIM_EXPERIMENT="scaling-poisoning"

  echo "AIM_EXPERIMENT=${AIM_EXPERIMENT:-<unset>}"
  echo "AIM_RUN_NAME=${AIM_RUN_NAME:-<auto>}"
  echo "AIM_REPO=${AIM_REPO:-<unset>}"

  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scaling-poison
  exec bash
'