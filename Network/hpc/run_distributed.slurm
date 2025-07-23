#!/bin/bash
#SBATCH --job-name=fiber_optics_distributed
#SBATCH --output=logs/distributed_%j.out
#SBATCH --error=logs/distributed_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@wm.edu

# Load required modules
module purge
module load cuda/11.8
module load cudnn/8.6
module load python/3.10
module load gcc/11.2
module load openmpi/4.1.4

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Distributed training environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# NCCL settings for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo

# Create directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p results

# Navigate to project directory
cd /path/to/your/polar-bear/Network

# Activate environment
source venv/bin/activate
source .env

# Log job information
echo "========================================="
echo "Distributed Training Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $WORLD_SIZE"
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "========================================="

# Run distributed training
srun python core/main_distributed.py \
    --distributed \
    --config config/config.yaml \
    --epochs 100 \
    --results-dir results/distributed_${SLURM_JOB_ID}

echo "Distributed training completed on $(date)"