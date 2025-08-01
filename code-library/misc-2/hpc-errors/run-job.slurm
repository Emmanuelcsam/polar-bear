#!/bin/tcsh
#
# ==============================================================================
# SLURM Job Script for Fiber Optic Analysis Deep Learning on W&M Bora Cluster
# ==============================================================================
#
#SBATCH --job-name=fiber-optic-dl
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=2                    # Request 2 compute nodes for multi-node training
#SBATCH --ntasks-per-node=2          # 2 tasks per node (2 GPUs per node)
#SBATCH --cpus-per-task=16           # 16 CPU cores per task
#SBATCH --gres=gpu:2                 # Request 2 GPUs per node
#SBATCH --mem-per-gpu=64G            # 64GB memory per GPU
#SBATCH --time=12:00:00              # 12 hour runtime (adjust based on your needs)
#SBATCH --constraint=v100            # Request V100 GPUs (or use p100 for P100s)
#SBATCH --signal=B:USR1@600          # Send signal 10 minutes before timeout
#SBATCH --mail-type=END,FAIL         # Email on job completion or failure
#SBATCH --mail-user=your_userid@wm.edu  # Replace with your W&M email

# --- Environment Setup ---
#
echo "========================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Running on hosts: ${SLURM_JOB_NODELIST}"
echo "Running on ${SLURM_NNODES} nodes"
echo "Running on ${SLURM_NPROCS} total processes"
echo "Job started on: `date`"
echo "========================================================"

# Set job directory and checkpoint directory
set job_dir = $SLURM_SUBMIT_DIR
set checkpoint_dir = /sciclone/scr20/$USER/fiber-optic-checkpoints
set local_data = /local/scr/$USER/data_$SLURM_JOB_ID

# Create necessary directories
mkdir -p $checkpoint_dir
mkdir -p $local_data

# Load required modules
module purge
module load anaconda3/2023.09
module load cuda/12.3

# Set up distributed training environment variables
setenv MASTER_PORT `expr 10000 + $SLURM_JOB_ID % 10000`
set master_addr = `scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1`
setenv MASTER_ADDR $master_addr
setenv WORLD_SIZE $SLURM_NTASKS

# NCCL optimization for InfiniBand
setenv NCCL_DEBUG INFO
setenv NCCL_SOCKET_IFNAME ^docker0,lo
setenv NCCL_IB_DISABLE 0
setenv NCCL_IB_CUDA_SUPPORT 1
setenv NCCL_NET_GDR_LEVEL 2

# Set OpenMP threads
setenv OMP_NUM_THREADS $SLURM_CPUS_PER_TASK

# Signal handler for checkpointing
onintr save_and_resubmit

# Check if this is a restart from checkpoint
if ( -f $checkpoint_dir/latest_checkpoint.pth ) then
    echo "Resuming from checkpoint"
    set resume_flag = "--checkpoint_path $checkpoint_dir/latest_checkpoint.pth"
else
    echo "Starting fresh training"
    set resume_flag = ""
    
    # Copy data to local scratch on first run (if data directory exists)
    if ( -d $job_dir/data ) then
        echo "Copying data to local scratch..."
        rsync -av $job_dir/data/ $local_data/
        setenv DATA_PATH $local_data
    else
        setenv DATA_PATH $job_dir/data
    endif
endif

# Activate conda environment
# Note: Replace with your actual environment path
conda activate /sciclone/scr20/$USER/conda-envs/fiber-optic-env

# Change to job directory
cd $job_dir

# Update config to use checkpoint directory
set config_override = "--config_override system.checkpoints_path=$checkpoint_dir"

# Launch distributed training
echo " "
echo "Starting distributed training on $SLURM_NNODES nodes"
echo "MASTER_ADDR:MASTER_PORT=${MASTER_ADDR}:${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo " "

# Run the training script using srun for proper MPI/distributed setup
srun python main.py \
    --mode train \
    --config config.yaml \
    $config_override \
    $resume_flag \
    --verbose

# Check exit status
if ( $status == 0 ) then
    echo "Training completed successfully"
    # Copy final results back to job directory
    rsync -av $checkpoint_dir/ $job_dir/checkpoints/
else
    echo "Training interrupted or failed"
    goto save_and_resubmit
endif

echo " "
echo "Job finished on: `date`"
echo "========================================================"

exit 0

# Signal handler for graceful shutdown and resubmission
save_and_resubmit:
    echo " "
    echo "Received termination signal, initiating checkpoint save..."
    
    # Send checkpoint signal to Python processes
    killall -USR1 python
    
    # Wait for checkpoint to complete
    sleep 30
    
    # Check if we should resubmit
    if ( $?SLURM_RESTART_COUNT ) then
        @ restart_count = $SLURM_RESTART_COUNT + 1
    else
        @ restart_count = 1
    endif
    
    # Limit automatic resubmissions to prevent infinite loops
    if ( $restart_count < 5 ) then
        # Resubmit job with dependency
        setenv SLURM_RESTART_COUNT $restart_count
        set new_job = `sbatch --parsable --dependency=afterany:$SLURM_JOB_ID $0`
        echo "Resubmitted as job $new_job (restart #$restart_count)"
    else
        echo "Maximum restart count reached, not resubmitting"
    endif
    
    exit 0