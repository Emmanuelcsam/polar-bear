#!/bin/bash

# Setup script for Fiber Optics Neural Network on HPC
# This script prepares the environment and installs dependencies

echo "==========================================="
echo "Fiber Optics NN Environment Setup"
echo "==========================================="

# Set project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Project directory: $PROJECT_DIR"

# Load required modules
echo "Loading HPC modules..."
module purge
module load cuda/11.8
module load cudnn/8.6
module load python/3.10
module load gcc/11.2

# Create virtual environment
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv $VENV_DIR
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install required packages
echo "Installing required packages..."
cat > /tmp/requirements.txt << EOF
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=8.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
tqdm>=4.60.0
tensorboard>=2.5.0
pyyaml>=5.4.0
pandas>=1.3.0
scipy>=1.7.0
seaborn>=0.11.0
wandb>=0.12.0
pytest>=6.2.0
pytest-cov>=2.12.0
black>=21.0
flake8>=3.9.0
mypy>=0.910
EOF

pip install -r /tmp/requirements.txt

# Install additional packages for HPC optimization
echo "Installing HPC optimization packages..."
pip install \
    apex \
    horovod \
    mpi4py \
    tensorboardX \
    pytorch-lightning

# Create necessary directories
echo "Creating project directories..."
mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/checkpoints
mkdir -p $PROJECT_DIR/results
mkdir -p $PROJECT_DIR/cache
mkdir -p $PROJECT_DIR/temp
mkdir -p $PROJECT_DIR/statistics
mkdir -p $PROJECT_DIR/output

# Set up environment variables
echo "Setting up environment variables..."
cat > $PROJECT_DIR/.env << EOF
# Environment variables for Fiber Optics NN
export PROJECT_ROOT=$PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:\$PYTHONPATH
export OMP_NUM_THREADS=16
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
EOF

# Download sample data if not present
if [ ! -d "$PROJECT_DIR/samples" ]; then
    echo "Creating samples directory..."
    mkdir -p $PROJECT_DIR/samples
    echo "Please add sample fiber optic images to: $PROJECT_DIR/samples"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo ""
echo "==========================================="
echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo "  source $PROJECT_DIR/.env"
echo "==========================================="