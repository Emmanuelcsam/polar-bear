#!/bin/bash
#SBATCH --job-name=fiber_optics_nn
#SBATCH --output=logs/fiber_optics_%j.out
#SBATCH --error=logs/fiber_optics_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@wm.edu

# Load required modules (adjust based on Kuro HPC modules)
module purge
module load cuda/11.8
module load cudnn/8.6
module load python/3.10
module load gcc/11.2

# Set up environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# CUDA settings
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
mkdir -p cache

# Activate virtual environment (if using one)
# source /path/to/your/venv/bin/activate

# OR use conda environment
# module load anaconda3
# conda activate fiber_optics

# Navigate to project directory
cd /path/to/your/polar-bear/Network

# Log system information
echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Install dependencies if needed (first run only)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install numpy opencv-python pillow matplotlib scikit-learn tqdm tensorboard

# Run the main training script
echo "Starting training..."
python core/main.py

# For distributed training across multiple GPUs (uncomment if needed)
# python core/main_distributed.py

echo "Job completed on $(date)"