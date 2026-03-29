#!/bin/bash
#SBATCH --job-name=eval+train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1        # adjust if only trainer needs GPUs
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=your-partition
#SBATCH --output=job-%j.out

set -euo pipefail

# Get allocated hostnames
mapfile -t NODES < <(scontrol show hostnames $SLURM_NODELIST)
SERVER_NODE=${NODES[0]}
TRAIN_NODE=${NODES[1]}

echo "Server node: $SERVER_NODE"
echo "Train node:  $TRAIN_NODE"

# Start evaluator on server node (backgrounded)
srun --nodes=1 --ntasks=1 --nodelist=$SERVER_NODE --exclusive \
     --output=server-%j.log \
     bash -lc "python strong_reject_server.py --port 12345" &

# Wait for server to accept connections (simple healthcheck)
echo "Waiting for evaluator to be ready..."
until srun --nodes=1 --ntasks=1 --nodelist=$SERVER_NODE --exclusive \
           bash -lc "python - <<'PY'
import socket,sys
s=socket.socket()
try:
  s.connect(('127.0.0.1',12345)); s.close(); print('UP')
except:
  sys.exit(1)
PY" ; do
  sleep 2
done

# Run training on the other node (foreground)
srun --nodes=1 --ntasks=1 --nodelist=$TRAIN_NODE --exclusive \
     --gres=gpu:1 \
     --output=train-%j.log \
     bash -lc "python train.py --model_name google/gemma-3-12b-it \
       --dataset_name gpt4_api_attacks --poisoning_rate 0.02 --dataset_length 5000 \
       --learning_rate 5e-5 --lr_scheduler_type=linear --lora_r 16 --use_peft_lora \
       --use_4bit_quantization --num_train_epochs 5 --per_device_train_batch_size 4 \
       --output_dir /scratch/caydenw/test_output --log_dir /scratch/caydenw/logs \
       --report_to none --strongreject_node=$SERVER_NODE"

# Wait for backgrounded server srun to finish (Slurm will kill it at job end if still running)
wait