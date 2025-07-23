#!/bin/bash
# Interactive debugging session for HPC

echo "Starting interactive GPU session for debugging..."

# Request interactive session with GPU
salloc --partition=gpu \
       --gres=gpu:1 \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task=8 \
       --mem=32G \
       --time=2:00:00 \
       --job-name=fiber-debug

# Once allocated, this script continues
echo "Interactive session allocated on node: $SLURM_JOB_NODELIST"

# Load modules
module purge
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Activate environment
source venv/bin/activate

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Navigate to project
cd /path/to/polar-bear/Network

echo "Environment ready for debugging"
echo "You can now run:"
echo "  - python main.py --mode train --debug"
echo "  - python -m pdb main.py"
echo "  - nvidia-smi"
echo "  - python test_gpu.py"

# Keep session alive
bash