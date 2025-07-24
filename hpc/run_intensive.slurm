#!/bin/bash
#SBATCH --job-name=fiber_intensive
#SBATCH --output=logs/intensive_%j.out
#SBATCH --error=logs/intensive_%j.err
#SBATCH --time=7-00:00:00  # 7 days maximum
#SBATCH --partition=gpu-long  # Long-running GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4  # Request 4 GPUs for data parallel
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --mail-user=your-email@wm.edu
#SBATCH --requeue  # Requeue job if preempted

# Signal handling for checkpointing before timeout
trap 'echo "Received signal, saving checkpoint..."; kill -SIGUSR1 $PID' SIGUSR1 SIGTERM

# Load modules
module purge
module load cuda/11.8
module load cudnn/8.6
module load python/3.10
module load gcc/11.2

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Performance optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Create directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
mkdir -p /scratch/$USER/fiber_optics/{dataset,reference,checkpoints,temp}

# Copy data to scratch for better I/O (if not already there)
if [ ! -d "/scratch/$USER/fiber_optics/dataset/chunk_1" ]; then
    echo "Copying dataset to scratch..."
    cp -r dataset/* /scratch/$USER/fiber_optics/dataset/
    cp -r reference/* /scratch/$USER/fiber_optics/reference/
fi

# Navigate to project
cd /path/to/your/polar-bear/Network

# Activate environment
source venv/bin/activate
source .env

# Export scratch paths
export DATA_PATH=/scratch/$USER/fiber_optics/dataset
export REFERENCE_PATH=/scratch/$USER/fiber_optics/reference
export CHECKPOINT_PATH=/scratch/$USER/fiber_optics/checkpoints

# Function to get remaining time
get_remaining_time() {
    # Get remaining time in seconds
    remaining=$(squeue -h -j $SLURM_JOB_ID -o "%L" | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
    echo $remaining
}

# Log start information
echo "========================================="
echo "Intensive Training Job Started"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================="

# Main training loop with automatic continuation
CYCLE=1
MAX_CYCLES=1000  # Practically unlimited

while [ $CYCLE -le $MAX_CYCLES ]; do
    echo -e "\n=== Training Cycle $CYCLE started at $(date) ==="
    
    # Check remaining time
    remaining_time=$(get_remaining_time)
    
    # If less than 10 minutes remaining, prepare for requeue
    if [ $remaining_time -lt 600 ]; then
        echo "Less than 10 minutes remaining, preparing for requeue..."
        
        # Copy checkpoints back to home
        cp -r /scratch/$USER/fiber_optics/checkpoints/* checkpoints/
        
        # Check if we should resubmit
        if [ $CYCLE -lt $MAX_CYCLES ]; then
            echo "Resubmitting job for continuation..."
            sbatch --dependency=afterany:$SLURM_JOB_ID $0
        fi
        
        exit 0
    fi
    
    # Run training with timeout (leave 10 minutes buffer)
    timeout $((remaining_time - 600)) python -u core/main.py \
        --config config/config_hpc_intensive.yaml \
        --resume-latest &
    
    PID=$!
    wait $PID
    EXIT_CODE=$?
    
    # Check exit code
    if [ $EXIT_CODE -eq 124 ]; then
        echo "Training cycle $CYCLE timed out (expected behavior)"
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "Training cycle $CYCLE completed successfully"
    else
        echo "Training cycle $CYCLE failed with exit code $EXIT_CODE"
        
        # Check if it's a recoverable error
        if grep -q "CUDA out of memory" logs/intensive_${SLURM_JOB_ID}.err; then
            echo "OOM error detected, reducing batch size and retrying..."
            # Modify config to reduce batch size
            python -c "
import yaml
with open('config/config_hpc_intensive.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['training']['batch_size'] = int(config['training']['batch_size'] * 0.8)
with open('config/config_hpc_intensive.yaml', 'w') as f:
    yaml.dump(config, f)
print(f\"Reduced batch size to {config['training']['batch_size']}\")
"
        else
            # Non-recoverable error, wait before retry
            echo "Waiting 60 seconds before retry..."
            sleep 60
        fi
    fi
    
    # Increment cycle
    CYCLE=$((CYCLE + 1))
    
    # Periodic maintenance
    if [ $((CYCLE % 10)) -eq 0 ]; then
        echo "Running maintenance tasks..."
        
        # Clear cache
        rm -rf /scratch/$USER/fiber_optics/temp/*
        
        # Copy latest checkpoints to home
        cp -r /scratch/$USER/fiber_optics/checkpoints/* checkpoints/
        
        # Log GPU status
        nvidia-smi
        
        # Log disk usage
        df -h /scratch/$USER
    fi
done

# Final cleanup
echo -e "\n========================================="
echo "Intensive training completed all cycles"
echo "End time: $(date)"
echo "Total cycles: $((CYCLE - 1))"
echo "========================================="

# Copy all results back
cp -r /scratch/$USER/fiber_optics/checkpoints/* checkpoints/
cp -r results/* $HOME/polar-bear/Network/results/

# Generate final report
python -c "
import json
from pathlib import Path

results = {
    'job_id': '$SLURM_JOB_ID',
    'total_cycles': $((CYCLE - 1)),
    'end_time': '$(date)',
    'node': '$(hostname)',
}

Path('results').mkdir(exist_ok=True)
with open('results/intensive_training_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
"