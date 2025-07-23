#!/bin/bash
#SBATCH --job-name=fiber-optics-distributed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fiber_optics_dist_%j.out
#SBATCH --error=logs/fiber_optics_dist_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@wm.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module purge
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0
module load openmpi/4.1.1

# Activate virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Set environment variables for distributed training
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Get master node information
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Master Node: $MASTER_ADDR"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Navigate to project directory
cd /path/to/polar-bear/Network

# Run distributed training using srun
echo "Starting distributed training on $SLURM_NNODES nodes with $SLURM_NTASKS tasks..."
srun python main_distributed.py \
    --mode train \
    --config config.yaml \
    --distributed

echo "End Time: $(date)"
echo "Distributed training completed"