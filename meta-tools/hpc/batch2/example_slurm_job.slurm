#!/bin/bash

# =============================================================================
# SLURM Job Script for Neural Network Training
# =============================================================================
# This is an example Slurm job script for training a neural network on the
# W&M HPC cluster. You can customize this script to meet the specific
# requirements of your training job.
#
# To submit this script, use the command:
# sbatch example_slurm_job.slurm
# =============================================================================

# --- Job Configuration ---

# Set the name of the job
#SBATCH --job-name=nn_training

# Set the partition (queue) to use. Common partitions include 'parallel', 'gpu', etc.
#SBATCH --partition=parallel

# Set the number of nodes to use
#SBATCH --nodes=1

# Set the number of tasks (processes) to run
#SBATCH --ntasks=1

# Set the number of CPUs per task. This should be adjusted based on your
# data loading and processing needs.
#SBATCH --cpus-per-task=8

# Set the amount of memory to allocate for the job.
#SBATCH --mem=32G

# Set the maximum time the job will run.
#SBATCH --time=04:00:00

# --- GPU Configuration (optional) ---
# Uncomment the following lines if you need to use a GPU.
# #SBATCH --gres=gpu:1
# #SBATCH --constraint=gpu

# --- Output and Error Files ---
# Set the path for the job's output and error files.
# %j will be replaced with the job ID.
#SBATCH --output=/sciclone/scr-lst/your_username/my_neural_network/logs/job_%j.out
#SBATCH --error=/sciclone/scr-lst/your_username/my_neural_network/logs/job_%j.err

# --- Email Notifications ---
# Set your email address and the events for which you want to receive notifications.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_username@wm.edu

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working Directory: $(pwd)"

# Load necessary modules
# It's good practice to purge any existing modules first.
module purge
module load miniforge3

# Load the CUDA module if you are using a GPU
# module load cuda

# Activate your conda environment
# Replace 'my_env' with the name of your conda environment.
source activate my_env

# --- Training ---
echo "Starting neural network training..."

# The main command to run your training script.
# Adjust the paths and arguments as needed.
python /sciclone/data10/your_username/my_neural_network/train.py \
    --data_dir /sciclone/data10/your_username/my_neural_network/images \
    --ref_dir /sciclone/data10/your_username/my_neural_network/reference_tensors \
    --output_dir /sciclone/scr-lst/your_username/my_neural_network/output \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --model resnet50

# --- Job Completion ---
echo "Job finished at $(date)"
