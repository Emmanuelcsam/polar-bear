#!/bin/bash
#SBATCH --job-name=fiber-optics-scaling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=2-8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/fiber_optics_scaling_%j.out
#SBATCH --error=logs/fiber_optics_scaling_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@wm.edu
#SBATCH --exclusive

# This script is for testing scaling performance across different node counts

# Create logs directory
mkdir -p logs

# Load modules
module purge
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0
module load openmpi/4.1.1

# Activate environment
source venv/bin/activate

# Environment setup
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL optimizations for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_TIMEOUT=22
export NCCL_IB_CUDA_SUPPORT=1

# Get master node
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# Print configuration
echo "========================================"
echo "FIBER OPTICS NEURAL NETWORK - HPC SCALING TEST"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Total Tasks: $SLURM_NTASKS"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Start Time: $(date)"
echo "========================================"

# Navigate to project
cd /path/to/polar-bear/Network

# Create results directory
RESULTS_DIR="results/scaling_test_${SLURM_JOB_ID}"
mkdir -p $RESULTS_DIR

# Run distributed training with performance monitoring
echo "Starting distributed training..."
srun --cpu-bind=verbose python main_distributed.py \
    --mode train \
    --config config.yaml \
    --distributed \
    --epochs 10 \
    --benchmark \
    --results-dir $RESULTS_DIR

# Collect performance metrics
if [ $SLURM_PROCID -eq 0 ]; then
    echo "========================================"
    echo "Performance Summary:"
    echo "Nodes: $SLURM_NNODES"
    echo "GPUs: $WORLD_SIZE"
    cat $RESULTS_DIR/performance_metrics.json
    echo "========================================"
fi

echo "End Time: $(date)"