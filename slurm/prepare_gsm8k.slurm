#!/bin/bash
#SBATCH --job-name=prepare_gsm8k
#SBATCH --output=../logs/prepare_gsm8k_%j.out
#SBATCH --error=../logs/prepare_gsm8k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=cpu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Activate rftvenv virtual environment
source ../rftvenv/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Navigate to skyrl directory
cd ../skyrl

# Run dataset preparation script
echo "Preparing GSM8K dataset..."
python scripts/prepare_gsm8k.py \
    --output-dir ../data/gsm8k \
    --split main \
    --train-split 0.9 \
    --format skyrl

echo "Dataset preparation completed at: $(date)"

