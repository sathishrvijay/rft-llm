#!/bin/bash
#SBATCH --job-name=setup_rftvenv
#SBATCH --output=../logs/setup_venv_%j.out
#SBATCH --error=../logs/setup_venv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=cpu

# Print job information
echo "=========================================="
echo "Setting up rftvenv virtual environment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Navigate to project root
cd "$SLURM_SUBMIT_DIR/.." || cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Check if Python 3.11+ is available
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=$(which python3)
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Found Python: $PYTHON_CMD (version $PYTHON_VERSION)"
else
    echo "ERROR: python3 not found!"
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_PATH="$PROJECT_ROOT/rftvenv"
if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    echo "Removing existing venv to create fresh one..."
    rm -rf "$VENV_PATH"
fi

echo "Creating virtual environment at $VENV_PATH..."
$PYTHON_CMD -m venv "$VENV_PATH"

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Failed to create virtual environment!"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies from pyproject.toml
echo "Installing dependencies from pyproject.toml..."
if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    pip install -e "$PROJECT_ROOT"
else
    echo "ERROR: pyproject.toml not found!"
    exit 1
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || echo "✗ PyTorch not installed"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" || echo "✗ Transformers not installed"
python -c "import peft; print(f'✓ PEFT: {peft.__version__}')" || echo "✗ PEFT not installed"
python -c "import skyrl; print(f'✓ SkyRL: {skyrl.__version__ if hasattr(skyrl, \"__version__\") else \"installed\"}')" || echo "✗ SkyRL not installed"
python -c "import wandb; print(f'✓ WandB: {wandb.__version__}')" || echo "✗ WandB not installed"
python -c "import deepspeed; print(f'✓ DeepSpeed: {deepspeed.__version__}')" || echo "✗ DeepSpeed not installed"

echo ""
echo "=========================================="
echo "Setup completed at: $(date)"
echo "=========================================="
echo "Virtual environment location: $VENV_PATH"
echo ""
echo "To activate the venv manually:"
echo "  source $VENV_PATH/bin/activate"
echo ""

