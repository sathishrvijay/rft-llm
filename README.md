# RFT-LLM

Tinkering with LLM fine-tuning using Reinforcement Learning (RL) using popular frameworks like SkyRL, Tinker, and OpenAI API.

## Repository Structure

This repository is organized into separate subdirectories for each framework:

```
rft-llm/
├── skyrl/              # SkyRL framework implementation
├── tinker/             # Thinking Machines Tinker (placeholder)
├── openai_rft/         # OpenAI RFT API (placeholder)
├── slurm/              # SLURM job scripts
│   ├── setup_venv.slurm    # SLURM job to set up venv
│   ├── setup_venv.sh       # Standalone venv setup script
│   ├── train_skyrl.slurm   # Training job script
│   └── prepare_gsm8k.slurm # Dataset preparation script
├── logs/               # Training logs
├── checkpoints/        # Model checkpoints
├── data/               # Datasets
├── pyproject.toml      # Project dependencies
└── README.md           # This file
```

## Setup

### Option 1: Automated Setup (Recommended)

#### On SLURM Cluster

Submit a SLURM job to set up the virtual environment:

```bash
sbatch slurm/setup_venv.slurm
```

This will create `rftvenv` and install all dependencies. Check the log file in `logs/setup_venv_<JOB_ID>.out` for progress.

#### On Local Machine or Login Node

Run the setup script directly:

```bash
bash slurm/setup_venv.sh
```

Or use pip with pyproject.toml:

```bash
python3 -m venv rftvenv
source rftvenv/bin/activate
pip install -e .
```

### Option 2: Manual Setup

#### 1. Create Virtual Environment

```bash
python3 -m venv rftvenv
source rftvenv/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -e .
```

#### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

## SkyRL Quick Start

SkyRL is the initial framework implemented. See [skyrl/README.md](skyrl/README.md) for detailed documentation.

### Workflow: GSM8K Dataset → Training Qwen Models with PPO + LoRA

#### Step 1: Prepare GSM8K Dataset

```bash
# Activate environment
source rftvenv/bin/activate

# Prepare dataset
cd skyrl
python scripts/prepare_gsm8k.py --output-dir ../data/gsm8k
```

Or submit as SLURM job:
```bash
sbatch slurm/prepare_gsm8k.slurm
```

#### Step 2: Train Model

```bash
# Activate environment
source rftvenv/bin/activate

# Train with default settings (Qwen2-1.5B)
cd skyrl
python train.py \
    --base-config configs/base_config.yaml \
    --qwen-config configs/qwen_config.yaml \
    --gsm8k-config configs/gsm8k_config.yaml \
    --model-size qwen2_1_5b
```

Or submit as SLURM job:
```bash
sbatch slurm/train_skyrl.slurm
```

## SLURM Usage

### Initial Setup on SLURM

First, set up the virtual environment on the SLURM cluster:

```bash
sbatch slurm/setup_venv.slurm
```

Wait for the job to complete, then verify the venv was created successfully.

### Training Job

Submit a training job on H100 GPUs:

```bash
sbatch slurm/train_skyrl.slurm
```

The script is configured for:
- **Nodes**: 1
- **GPUs**: 8x H100
- **Memory**: 500GB
- **CPUs**: 32 per node
- **Time**: 24 hours

### Dataset Preparation

Submit dataset preparation job:

```bash
sbatch slurm/prepare_gsm8k.slurm
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View job output
tail -f logs/skyrl_train_<JOB_ID>.out

# Cancel job
scancel <JOB_ID>
```

## Key Features

### LoRA Fine-Tuning

This repository uses **LoRA (Low-Rank Adaptation)** for fine-tuning

LoRA configuration is in `skyrl/configs/base_config.yaml`:
- Rank: 16
- Alpha: 32
- Target modules: Qwen-specific attention and MLP layers

### PPO Agent

Uses **Proximal Policy Optimization (PPO)** for RL fine-tuning:
- Stable training with clipped objective
- Configurable hyperparameters (clip epsilon, value loss, entropy)
- Integrated with SkyRL framework

### Model Support

Supports multiple Qwen model sizes:
- **Qwen2-0.5B**: Smallest, fastest training
- **Qwen2-1.5B**: Default, good balance (recommended for initial testing)
- **Qwen2-7B**: Medium, requires more memory
- **Qwen2-72B**: Large, requires significant resources

### DeepSpeed Integration

Configured for **DeepSpeed ZeRO-2**:
- Optimized for single-node multi-GPU training
- Memory-efficient distributed training
- Configurable optimizer and parameter offloading

## Configuration

All configurations are in YAML format:

- `skyrl/configs/base_config.yaml`: Base training settings (PPO, LoRA, DeepSpeed)
- `skyrl/configs/qwen_config.yaml`: Qwen model configurations
- `skyrl/configs/gsm8k_config.yaml`: GSM8K dataset settings

## Monitoring

### WandB Integration

Training metrics are logged to Weights & Biases:
- Configure your WandB entity in `skyrl/configs/base_config.yaml`
- View training progress, metrics, and checkpoints in WandB dashboard

### Logs

- **SLURM logs**: `logs/skyrl_train_<JOB_ID>.out` and `.err`
- **Training logs**: `logs/` directory
- **Checkpoints**: `checkpoints/` directory

## Future Integrations

### Thinking Machines Tinker

Placeholder directory: `tinker/`

### OpenAI RFT API

Placeholder directory: `openai_rft/`

## Requirements

- Python 3.11+
- CUDA-capable GPUs (H100 recommended)
- SLURM cluster access
- Sufficient disk space for datasets and checkpoints

