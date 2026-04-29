#!/usr/bin/env bash
#SBATCH --job-name=scaling-poison
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --gpus=1
#SBATCH --constraint=gpu_h200
#SBATCH --time=03:00:00
#SBATCH --array=0
#SBATCH --output=logs/pipeline_%A_%a.out
#SBATCH --error=logs/pipeline_%A_%a.err

#alt gpu gpu_a100_80gb|gpu_h100|gpu_h200

set -euo pipefail

# ── Models to fine-tune (indexed by SLURM_ARRAY_TASK_ID) ─────────────────────
MODELS=(
  "google/gemma-2-27b-it"                      # 0
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

# ── Central StrongREJECT server ───────────────────────────────────────────────
CENTRAL_ENV=/scratch/caydenw/strongreject_central.env
if [ ! -f "${CENTRAL_ENV}" ]; then
  echo "ERROR: Central StrongREJECT server env file not found at ${CENTRAL_ENV}" >&2
  echo "Submit run_strongreject_server.slurm before running pipeline jobs." >&2
  exit 1
fi
source "${CENTRAL_ENV}"
export STRONGREJECT_SERVER_URL="http://${STRONGREJECT_NODE}:${STRONGREJECT_PORT}/evaluate"
echo "Using StrongREJECT server: ${STRONGREJECT_SERVER_URL}"

# Quick health check before starting training
if ! curl -sf "http://${STRONGREJECT_NODE}:${STRONGREJECT_PORT}/health" > /dev/null 2>&1; then
  echo "ERROR: Central StrongREJECT server not responding at ${STRONGREJECT_SERVER_URL%/evaluate}/health" >&2
  exit 1
fi

# Aim configuration
export AIM_EXPERIMENT="scaling-poisoning-replication"
export AIM_REPO="aim://<AIM_HOST>:<AIM_PORT>"

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
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --output_dir /scratch/caydenw/test_output \
  --log_dir /scratch/caydenw/logs \
  --report_to none \
  --quant_name "nf4" \
  

echo "============================================================"
echo "Finished: ${MODEL} at $(date)"
echo "============================================================"
