#!/bin/bash
#SBATCH --job-name=fiber-optics-single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fiber_optics_%j.out
#SBATCH --error=logs/fiber_optics_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@wm.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module purge
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Activate virtual environment (create if doesn't exist)
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Navigate to project directory
cd /path/to/polar-bear/Network

# Run training (single GPU mode)
echo "Starting single GPU training..."
python main.py --mode train --config config.yaml

# Run evaluation after training
echo "Running evaluation..."
python main.py --mode evaluate --config config.yaml

echo "End Time: $(date)"
echo "Job completed successfully"