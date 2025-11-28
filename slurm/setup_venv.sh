#!/bin/bash
# Standalone script to set up rftvenv virtual environment
# Can be run locally or on SLURM login nodes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "Setting up rftvenv virtual environment"
echo -e "==========================================${NC}"
echo "Start Time: $(date)"
echo ""

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Check if Python 3.11+ is available
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=$(which python3)
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}Found Python: $PYTHON_CMD (version $PYTHON_VERSION)${NC}"
    
    # Check if version is 3.11 or higher
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]); then
        echo -e "${RED}ERROR: Python 3.11+ required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}ERROR: python3 not found!${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_PATH="$PROJECT_ROOT/rftvenv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment already exists at $VENV_PATH${NC}"
    read -p "Remove existing venv and create fresh one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf "$VENV_PATH"
    else
        echo "Keeping existing venv. Activating and upgrading packages..."
        source "$VENV_PATH/bin/activate"
        pip install --upgrade pip setuptools wheel
        if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
            pip install -e "$PROJECT_ROOT"
        else
            pip install -r "$PROJECT_ROOT/requirements.txt"
        fi
        exit 0
    fi
fi

echo "Creating virtual environment at $VENV_PATH..."
$PYTHON_CMD -m venv "$VENV_PATH"

if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}ERROR: Failed to create virtual environment!${NC}"
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
    echo -e "${YELLOW}WARNING: pyproject.toml not found, falling back to requirements.txt${NC}"
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        echo -e "${RED}ERROR: Neither pyproject.toml nor requirements.txt found!${NC}"
        exit 1
    fi
fi

# Verify installation
echo ""
echo -e "${GREEN}=========================================="
echo "Verifying installation..."
echo -e "==========================================${NC}"
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || echo -e "${RED}✗ PyTorch not installed${NC}"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" || echo -e "${RED}✗ Transformers not installed${NC}"
python -c "import peft; print(f'✓ PEFT: {peft.__version__}')" || echo -e "${RED}✗ PEFT not installed${NC}"
python -c "import skyrl; print(f'✓ SkyRL: installed')" || echo -e "${RED}✗ SkyRL not installed${NC}"
python -c "import wandb; print(f'✓ WandB: {wandb.__version__}')" || echo -e "${RED}✗ WandB not installed${NC}"
python -c "import deepspeed; print(f'✓ DeepSpeed: {deepspeed.__version__}')" || echo -e "${RED}✗ DeepSpeed not installed${NC}"

echo ""
echo -e "${GREEN}=========================================="
echo "Setup completed at: $(date)"
echo -e "==========================================${NC}"
echo "Virtual environment location: $VENV_PATH"
echo ""
echo "To activate the venv:"
echo -e "${GREEN}  source $VENV_PATH/bin/activate${NC}"
echo ""

