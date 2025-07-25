#!/bin/bash
#
# SBATCH --job-name=fiber_optic_train # A descriptive name for your job
# SBATCH --nodes=1                    # We'll use a single node
# SBATCH --ntasks-per-node=1          # A single task (the python script) will run on the node
# SBATCH --cpus-per-task=20           # Request all 20 CPU cores on a 'bora' node
# SBATCH --mem=120G                   # Request 120GB of memory, leaving a small buffer from the 128GB max
# SBATCH --time=2-00:00:00            # Set a 2-day time limit, adjust as needed for your training duration
# SBATCH --partition=batch            # Submit to the 'batch' partition
# SBATCH --constraint=bora            # Explicitly request a 'bora' node
# SBATCH --output=fiber_optic_%j.out  # Capture output and errors in a file named with the job ID

# --- Environment Setup ---
# This section prepares the environment for your Python application.
# It loads necessary modules like Conda for environment management and CUDA for GPU support.

echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "======================================================"

# It's good practice to purge any modules loaded by default
module purge

# Load the Anaconda module (or the specific Python module you use)
# The exact name might differ on your cluster (e.g., anaconda3, python/3.9.12)
module load anaconda/2021.11

# Activate your dedicated Conda environment.
# Replace 'your_env_name' with the actual name of your Conda environment.
# It is assumed you have an environment with PyTorch, torchvision, yaml, albumentations, etc. installed.
source activate your_env_name

# --- Application Execution ---
# This section runs your main Python script.

echo "Starting the Fiber Optic Analysis training..."

# The main.py script is the entry point to your application.
# It will read the config.yaml file for all settings.
# The script is launched using 'python'.
python main.py

echo "======================================================"
echo "Job finished at $(date)"
echo "======================================================"
