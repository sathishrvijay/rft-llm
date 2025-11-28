#!/bin/bash
#SBATCH --job-name=skyrl_train
#SBATCH --output=../logs/skyrl_train_%j.out
#SBATCH --error=../logs/skyrl_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Activate rftvenv virtual environment
source ../rftvenv/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Navigate to skyrl directory
cd ../skyrl

# Run training script
echo "Starting SkyRL training..."
python train.py \
    --base-config configs/base_config.yaml \
    --qwen-config configs/qwen_config.yaml \
    --gsm8k-config configs/gsm8k_config.yaml \
    --model-size qwen2_1_5b \
    --run-name "skyrl_ppo_lora_gsm8k_${SLURM_JOB_ID}"

echo "Training completed at: $(date)"

