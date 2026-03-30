#!/usr/bin/env bash
set -euo pipefail

srun --cpus-per-task=16 --mem=40GB --gpus=a100:1 -C gpu_a100_80gb --time=01:40:00 --pty bash -lc '
  echo "Starting strong reject server..."
  module load anaconda3
  export PATH=/home/caydenw/var/cuda-12.4/bin:$PATH
  export LD_LIBRARY_PATH=/home/caydenw/var/cuda-12.4/lib64:$LD_LIBRARY_PATH
  cd /home/caydenw/git/strong-reject/
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate strong-reject
  python strong_reject_server.py
'