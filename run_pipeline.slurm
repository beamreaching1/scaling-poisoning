#!/usr/bin/env bash
#SBATCH --job-name=scaling-poison-pipeline
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB
#SBATCH --gpus=1
#SBATCH --constraint=gpu_a100_80gb|h100|h200
#SBATCH --time=02:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

set -euo pipefail

# ── Models to fine-tune ──────────────────────────────────────────────────────
MODELS=(
  "google/gemma-3-12b-it"
  "google/gemma-3-1b-it"
  "google/gemma-3-4b-it"
  "google/gemma-3-27b-it"
  "meta-llama/Llama-3.2-11B-Vision-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
)

# ── Environment setup ────────────────────────────────────────────────────────
if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

module load anaconda3
export PATH=/home/caydenw/var/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/home/caydenw/var/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

cd /home/caydenw/git/scaling-poisoning/

# Start strong-reject server in a detached screen (if not already running)
if ! screen -list | grep -q strong-reject; then
  cd /home/caydenw/git/strong-reject/
  screen -dmS strong-reject bash -lc "source \"$(conda info --base)/etc/profile.d/conda.sh\"; conda activate strong-reject; python strong_reject_server.py"
  cd /home/caydenw/git/scaling-poisoning/
  echo "Waiting for strong-reject server to start..."
  sleep 30
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

# ── Run pipeline for each model ──────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
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

  echo "Finished: ${MODEL} at $(date)"
  echo ""
done

echo "============================================================"
echo "All models complete at $(date)"
echo "============================================================"
