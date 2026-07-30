#!/bin/bash
#SBATCH --job-name=dist_gpu
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate distributed-training

echo "Python:"
which python
python --version

echo "Checking GPU:"
nvidia-smi

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

echo "Launching training..."
torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_GPUS_ON_NODE} scripts/train.py
#python scripts/train.py
echo "Training finished"

conda deactivate
