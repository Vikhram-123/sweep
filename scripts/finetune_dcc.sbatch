#!/bin/bash
# Fine-tune on the Duke Compute Cluster.   Usage:  sbatch scripts/finetune_dcc.sbatch method   (or app)
# Resubmitting the same command resumes from the last checkpoint in /work/$USER/sweep_ckpt/<name>.
#SBATCH --job-name=sweep-ft
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:a5000:1          # 24 GB RTX A5000; or gpu:5000_ada:1 (32 GB); gpu:2080 has only 11 GB
#SBATCH --cpus-per-task=8           # JPEG decoding runs in DataLoader workers; more CPUs = faster epochs
#SBATCH --mem=48G                   # ~1.5 GB JPEG store, shared copy-on-write with the workers (refcount writes copy some pages)
#SBATCH --time=1-00:00:00           # gpu-common allows up to 2 days
#SBATCH --output=logs/ft-%x-%j.out

set -euo pipefail
CONFIG=${1:-method}
cd /work/$USER/sweep
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate sweep

NAME=$(python -c "import yaml; print(yaml.safe_load(open('compat/configs/finetune_${CONFIG}.yaml'))['name'])")
CKPT=/work/$USER/sweep_ckpt/$NAME
mkdir -p "$CKPT" logs
echo "job $SLURM_JOB_ID on $(hostname): config $CONFIG -> $CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m compat.finetune compat/configs/finetune_${CONFIG}.yaml --ckpt-dir "$CKPT" --set num_workers=$SLURM_CPUS_PER_TASK
