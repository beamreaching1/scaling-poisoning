#!/usr/bin/env bash
#SBATCH --job-name=scaling-poison
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --constraint=gpu_a100_80gb|gpu_h100|gpu_h200
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/pipeline_%A_%a.out
#SBATCH --error=logs/pipeline_%A_%a.err

set -euo pipefail

# ── Models (outer dimension) ─────────────────────────────────────────────────
# Large models (gemma-3-27b, gemma-4-26B-A4B, gemma-4-31B) are handled by
# run_large_models_pipeline.slurm and must not appear here.
MODELS=(
  # Google Gemma Series
  "google/gemma-3-1b-it"                        # 0 (Gemma 3)
  "google/gemma-3-4b-it"                        # 1 (Gemma 3)
  "google/gemma-3-12b-it"                       # 2 (Gemma 3)
  "google/gemma-4-E2B-it"                       # 3 (Gemma 4)
  "google/gemma-4-E4B-it"                       # 4 (Gemma 4)
  # Meta Llama Series
  "meta-llama/Llama-3.2-1B-Instruct"            # 5
  "meta-llama/Llama-3.2-3B-Instruct"            # 6
  "meta-llama/Llama-3.2-11B-Vision-Instruct"    # 7
)

# ── Quantization configs (inner dimension) ───────────────────────────────────
# array_id = model_idx * NUM_QUANTS + quant_idx  →  total = 8 * 2 = 16 (0-15)
# Tip: submit --array=0-15 to run only the 4-bit configs (nf4+fp4) for all models.
NUM_QUANTS=2

QUANT_NAMES=(
  "nf4"   # 0 – 4-bit NF4 QLoRA, bfloat16 compute  (current default)
  "bf16"  # 1 – bf16 mixed precision, no weight quantization
)

# Flags appended verbatim to the python train.py call; note --use_peft_lora is
# kept for all quant levels so LoRA adapters are always used.
QUANT_FLAGS=(
  "--use_4bit_quantization --bnb_4bit_quant_type nf4 --bnb_4bit_compute_dtype bfloat16"
  "--bf16"
)

# Per-quantization batch sizes (bf16 uses 1 to stay within VRAM budget)
BATCH_SIZES=(
  4   # nf4
  1   # bf16
)

# ── Derive 2-D indices from the flat SLURM array task ID ─────────────────────
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_QUANTS ))
QUANT_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_QUANTS ))

MODEL="${MODELS[$MODEL_IDX]}"
QUANT_NAME="${QUANT_NAMES[$QUANT_IDX]}"
QUANT_ARGS="${QUANT_FLAGS[$QUANT_IDX]}"
BATCH_SIZE="${BATCH_SIZES[$QUANT_IDX]}"

# Sanitised tag used for output/log directories so runs never collide
MODEL_SLUG=$(echo "${MODEL}" | tr '/' '-')
RUN_TAG="${MODEL_SLUG}__${QUANT_NAME}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: model_idx=${MODEL_IDX} quant_idx=${QUANT_IDX}"
echo "  model=${MODEL}  quant=${QUANT_NAME}"

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
export AIM_EXPERIMENT="scaling-poisoning-quantization"
export AIM_REPO="aim://<AIM_HOST>:<AIM_PORT>"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scaling-poison

# ── Run pipeline for this model ───────────────────────────────────────────────
echo "============================================================"
echo "Starting fine-tuning for: ${MODEL}  [quant=${QUANT_NAME}]"
echo "Time: $(date)"
echo "============================================================"

# shellcheck disable=SC2086  # QUANT_ARGS must word-split into separate flags
python train.py \
  --model_name "${MODEL}" \
  --dataset_name gpt4_api_attacks \
  --poisoning_rate 0.02 \
  --dataset_length 5000 \
  --learning_rate 5e-5 \
  --lr_scheduler_type linear \
  --lora_r 16 \
  --use_peft_lora \
  --num_train_epochs 5 \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --output_dir /scratch/caydenw/test_output/${RUN_TAG} \
  --log_dir /scratch/caydenw/logs \
  --report_to none \
  --quant_name "${QUANT_NAME}" \
  ${QUANT_ARGS} \
  

echo "============================================================"
echo "Finished: ${MODEL}  [quant=${QUANT_NAME}]  at $(date)"
echo "============================================================"
