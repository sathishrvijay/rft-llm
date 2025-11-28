#!/bin/bash
#SBATCH --job-name=skyrl_inference
#SBATCH --output=../logs/skyrl_inference_%j.out
#SBATCH --error=../logs/skyrl_inference_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu

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
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Navigate to skyrl directory
cd ../skyrl

# Run inference script (to be created)
echo "Running inference..."
# python inference.py --checkpoint <checkpoint_path> --input <input_file>

echo "Inference completed at: $(date)"

