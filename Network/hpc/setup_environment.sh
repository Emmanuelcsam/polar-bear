#!/bin/bash
# Setup script for Bora HPC environment

echo "Setting up Fiber Optics Neural Network environment on Bora HPC..."

# Load necessary modules
module purge
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0
module load openmpi/4.1.1

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing additional requirements..."
pip install numpy scipy pandas matplotlib seaborn
pip install opencv-python-headless pillow
pip install scikit-learn scikit-image
pip install tensorboard wandb
pip install tqdm pyyaml h5py
pip install einops timm

# Create necessary directories
echo "Creating project directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
mkdir -p tensorized_data
mkdir -p samples

# Set up environment variables
cat > setup_env.sh << 'EOF'
#!/bin/bash
# Source this file before running the program

# Module loading
module load python/3.9 cuda/11.8 gcc/9.3.0 openmpi/4.1.1

# Activate virtual environment
source venv/bin/activate

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# PyTorch settings
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# OpenMP settings
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# NCCL settings for distributed training
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

echo "Environment configured for Fiber Optics Neural Network"
EOF

chmod +x setup_env.sh

echo "Setup complete! Before running the program, execute:"
echo "  source setup_env.sh"