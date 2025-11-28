# SkyRL Training

This directory contains all SkyRL-related code, configurations, and scripts for RL fine-tuning of LLM models.

## Overview

- **Framework**: SkyRL
- **Agent**: PPO (Proximal Policy Optimization)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) - avoids modifying base model weights
- **Initial Dataset**: GSM8K (math reasoning)
- **Model Family**: Qwen (configurable sizes: 0.5B, 1.5B, 7B, 72B)

## Directory Structure

```
skyrl/
├── train.py                 # Main training script
├── scripts/
│   └── prepare_gsm8k.py    # GSM8K dataset preparation
├── configs/
│   ├── base_config.yaml     # Base training config with PPO + LoRA
│   ├── qwen_config.yaml     # Qwen model configurations
│   └── gsm8k_config.yaml    # GSM8K dataset configuration
└── README.md                # This file
```

## Quick Start

### 1. Prepare GSM8K Dataset

```bash
# Activate virtual environment
source ../rftvenv/bin/activate

# Prepare dataset
python scripts/prepare_gsm8k.py --output-dir ../data/gsm8k
```

Or submit as SLURM job:
```bash
sbatch ../slurm/prepare_gsm8k.slurm
```

### 2. Train Model

```bash
# Activate virtual environment
source ../rftvenv/bin/activate

# Train with default settings (Qwen2-1.5B)
python train.py \
    --base-config configs/base_config.yaml \
    --qwen-config configs/qwen_config.yaml \
    --gsm8k-config configs/gsm8k_config.yaml \
    --model-size qwen2_1_5b
```

Or submit as SLURM job:
```bash
sbatch ../slurm/train_skyrl.slurm
```

## Configuration

### Model Sizes

Available Qwen model sizes (configured in `configs/qwen_config.yaml`):
- `qwen2_0_5b`: Qwen2-0.5B-Instruct (smallest, fastest)
- `qwen2_1_5b`: Qwen2-1.5B-Instruct (default, good balance)
- `qwen2_7b`: Qwen2-7B-Instruct (medium, requires more memory)
- `qwen2_72b`: Qwen2-72B-Instruct (large, requires significant resources)

### LoRA Configuration

LoRA settings are in `configs/base_config.yaml`:
- `r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA alpha scaling (default: 32)
- `target_modules`: Modules to apply LoRA to
- `lora_dropout`: Dropout rate (default: 0.1)

### PPO Hyperparameters

PPO settings are in `configs/base_config.yaml`:
- `clip_epsilon`: PPO clip epsilon (default: 0.2)
- `value_loss_coef`: Value loss coefficient (default: 0.5)
- `entropy_coef`: Entropy coefficient (default: 0.01)
- `gamma`: Discount factor (default: 0.99)
- `lam`: GAE lambda (default: 0.95)

## Outputs

- **Checkpoints**: Saved to `../checkpoints/`
- **Logs**: Saved to `../logs/`
- **WandB**: Logging enabled (configure entity in `base_config.yaml`)

## Notes

- LoRA fine-tuning is used to avoid modifying base model weights, making training more memory-efficient
- DeepSpeed ZeRO-2 is configured for single-node multi-GPU training
- The training script provides a foundation structure; full SkyRL PPO integration depends on SkyRL's API

