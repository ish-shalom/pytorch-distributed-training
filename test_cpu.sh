#!/bin/bash
#SBATCH --job-name=dist_cpu
#SBATCH --partition=24CPUNodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate distributed-training

echo "Python:"
which python
python --version

echo "Launching training..."
torchrun --standalone --nnodes=1 --nproc_per_node=${SLURM_CPUS_PER_TASK} scripts/train.py
#python scripts/train.py
echo "Training finished"

conda deactivate
