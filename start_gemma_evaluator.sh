#!/usr/bin/env bash
set -euo pipefail

srun --cpus-per-task=16 --mem=120GB --gpus=a100:1 -C gpu_a100_80gb --time=04:00:00 --pty bash -lc '
  echo "Starting Gemma evaluator server..."
  module load anaconda3
  export PATH=/home/caydenw/var/cuda-12.4/bin:$PATH
  export LD_LIBRARY_PATH=/home/caydenw/var/cuda-12.4/lib64:$LD_LIBRARY_PATH
  cd /home/caydenw/git/scaling-poisoning/
  source activate scaling-poison

  export EVALUATOR_MODEL_NAME=${EVALUATOR_MODEL_NAME:-google/gemma-3-27b-it}
  export EVALUATOR_PORT=${EVALUATOR_PORT:-8100}
  export EVALUATOR_MAX_RESPONSE_LENGTH=${EVALUATOR_MAX_RESPONSE_LENGTH:-256}
  export EVALUATOR_EVAL_BATCH_SIZE=${EVALUATOR_EVAL_BATCH_SIZE:-4}

  python gemma_evaluator_server.py
'
