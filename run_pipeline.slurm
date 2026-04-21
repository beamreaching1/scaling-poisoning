#!/usr/bin/env bash
#SBATCH --job-name=scaling-poison
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --gpus=1
#SBATCH --constraint=gpu_a100_80gb|gpu_h100|gpu_h200
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/pipeline_%A_%a.out
#SBATCH --error=logs/pipeline_%A_%a.err

set -euo pipefail

# ── Models to fine-tune (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────
MODELS=(
  "google/gemma-3-12b-it"        # 0
  "google/gemma-3-1b-it"         # 1
  "google/gemma-3-4b-it"         # 2
  "google/gemma-3-27b-it"        # 3
  "meta-llama/Llama-3.2-11B-Vision-Instruct"  # 4
  "meta-llama/Llama-3.2-1B-Instruct"          # 5
  "meta-llama/Llama-3.2-3B-Instruct"          # 6
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "Array task ${SLURM_ARRAY_TASK_ID}: model=${MODEL}"

# ── Environment setup ────────────────────────────────────────────────────────
if [ -f ~/.bashrc ]; then
  # Avoid 'unbound variable' errors from /etc/bashrc when nounset is enabled
  set +u
  source ~/.bashrc || true
  set -u
fi

module load anaconda3
export PATH=/home/caydenw/var/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/home/caydenw/var/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

cd /home/caydenw/git/scaling-poisoning/

# Per-task port to avoid conflicts when multiple array tasks land on the same node
export STRONGREJECT_PORT=$((8000 + SLURM_ARRAY_TASK_ID))
export STRONGREJECT_SERVER_URL="http://localhost:${STRONGREJECT_PORT}/evaluate"
echo "StrongREJECT server port: ${STRONGREJECT_PORT}"

# Start strong-reject server, killing any stale session first
SCREEN_NAME="strong-reject-${SLURM_ARRAY_TASK_ID}"

start_strong_reject() {
  # Kill stale screen if it exists
  if screen -list | grep -q "${SCREEN_NAME}"; then
    echo "Killing stale screen session: ${SCREEN_NAME}"
    screen -S "${SCREEN_NAME}" -X quit 2>/dev/null || true
    sleep 2
  fi
  cd /home/caydenw/git/strong-reject/
  screen -dmS "${SCREEN_NAME}" bash -lc "source \"$(conda info --base)/etc/profile.d/conda.sh\"; conda activate strong-reject; STRONGREJECT_PORT=${STRONGREJECT_PORT} python strong_reject_server.py"
  cd /home/caydenw/git/scaling-poisoning/
}

# If the server is not already healthy, (re)start it
if ! curl -sf "http://localhost:${STRONGREJECT_PORT}/health" > /dev/null 2>&1; then
  echo "StrongREJECT server not responding on port ${STRONGREJECT_PORT}, starting..."
  start_strong_reject

  MAX_WAIT=300
  ELAPSED=0
  until curl -sf "http://localhost:${STRONGREJECT_PORT}/health" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
      echo "ERROR: strong-reject server failed to start on port ${STRONGREJECT_PORT} after ${MAX_WAIT}s" >&2
      echo "Screen log for debugging:"
      screen -S "${SCREEN_NAME}" -X hardcopy /tmp/sr_${SLURM_ARRAY_TASK_ID}.log 2>/dev/null
      cat /tmp/sr_${SLURM_ARRAY_TASK_ID}.log 2>/dev/null || true
      exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
  done
  echo "strong-reject server is ready on port ${STRONGREJECT_PORT} (waited ${ELAPSED}s)"
else
  echo "StrongREJECT server already healthy on port ${STRONGREJECT_PORT}"
fi

# MLOP configuration
export MLOP_PROJECT="scaling-poisoning"
export MLOP_DIR="/scratch/caydenw/.mlop"

if [ -z "${MLOP_HOST:-}" ] && [ -n "${MLOP_URL_APP:-}" ]; then
  export MLOP_HOST="${MLOP_URL_APP}"
fi

if [ -z "${MLOP_API_KEY:-}" ] && [ -f /home/caydenw/git/scaling-poisoning/mlop_api_key.txt ]; then
  export MLOP_API_KEY="$(tr -d "\r\n" < /home/caydenw/git/scaling-poisoning/mlop_api_key.txt)"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scaling-poison

# ── Run pipeline for this model ───────────────────────────────────────────────
echo "============================================================"
echo "Starting fine-tuning for: ${MODEL}"
echo "Time: $(date)"
echo "============================================================"

python train.py \
  --model_name "${MODEL}" \
  --dataset_name gpt4_api_attacks \
  --poisoning_rate 0.02 \
  --dataset_length 5000 \
  --learning_rate 5e-5 \
  --lr_scheduler_type linear \
  --lora_r 16 \
  --use_peft_lora \
  --use_4bit_quantization \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --output_dir /scratch/caydenw/test_output \
  --log_dir /scratch/caydenw/logs \
  --report_to none \
  --strongreject_node localhost \
  --mlop_enabled

echo "============================================================"
echo "Finished: ${MODEL} at $(date)"
echo "============================================================"
