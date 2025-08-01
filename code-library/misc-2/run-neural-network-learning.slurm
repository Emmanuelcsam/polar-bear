#!/bin/tcsh
#
# SLURM Job Script for Fiber Optic Analysis on William & Mary Bora Cluster
# Fully tested and verified for W&M HPC environment
#
#SBATCH --job-name=fiber-optic-dl
#SBATCH --output=fiber-dl-%j.out
#SBATCH --error=fiber-dl-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20         # Bora has 20 cores per node
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G                   # Leave buffer from 128GB max
#SBATCH --time=48:00:00              # 48 hours (under 72 hour max)
#SBATCH --partition=bora             # Target Bora partition
#SBATCH --constraint=bora            # Ensure Bora nodes
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ecsampson@wm.edu

# --- Critical W&M tcsh Environment Setup ---
# MUST use tcsh syntax for W&M HPC
echo "========================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: `hostname`"
echo "Job started at: `date`"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "========================================================"

# Set environment variables using tcsh syntax
set job_dir = $SLURM_SUBMIT_DIR
set scratch_dir = /sciclone/scr20/$USER/fiber-$SLURM_JOB_ID
set data_dir = $job_dir/fiber_optic_data
set checkpoint_dir = $job_dir/checkpoints
set conda_env_path = /sciclone/scr-lst/$USER/polar-bear-env

# Create necessary directories
echo "Creating directories..."
mkdir -p $checkpoint_dir
mkdir -p $scratch_dir

# Verify data directory exists
if ( ! -d "$data_dir" ) then
    echo "ERROR: Data directory not found at $data_dir"
    echo "Expected structure:"
    echo "  fiber_optic_data/"
    echo "    core/"
    echo "    cladding/"
    echo "    ferrule/"
    echo "    defects/"
    exit 1
endif

# Load W&M specific modules
echo "Loading modules..."
module purge
module load miniforge3/24.9.2-0
module load cuda/12.3

# Display loaded modules for verification
echo "Currently loaded modules:"
module list

# Setup conda environment
echo "Setting up Python environment..."

# Check if conda environment exists
if ( -d "$conda_env_path" ) then
    echo "Activating existing conda environment at $conda_env_path"
    conda activate $conda_env_path
else
    echo "Creating new conda environment..."
    # Create environment in scratch space as per W&M best practices
    conda create -p $conda_env_path python=3.10 -y
    conda activate $conda_env_path
    
    # Install PyTorch with CUDA support
    echo "Installing PyTorch and dependencies..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install remaining requirements
    pip install -r $job_dir/requirements.txt
endif

# Verify Python installation
echo " "
echo "Python environment check:"
which python
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')" || echo "No CUDA devices found"

# Set OpenMP threads for optimal performance
setenv OMP_NUM_THREADS $SLURM_CPUS_PER_TASK

# For distributed training (if needed in future)
if ( $SLURM_NNODES > 1 ) then
    setenv MASTER_PORT `expr 10000 + $SLURM_JOB_ID % 10000`
    set master_addr = `scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1`
    setenv MASTER_ADDR $master_addr
    setenv WORLD_SIZE $SLURM_NTASKS
    echo "Distributed setup: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
endif

# Copy small datasets to scratch for better I/O
# Check dataset size (in KB)
set data_size = `du -sk $data_dir | cut -f1`
if ( $data_size < 10485760 ) then  # Less than 10GB
    echo "Copying data to scratch for better performance..."
    cp -r $data_dir $scratch_dir/
    set data_path = $scratch_dir/fiber_optic_data
else
    set data_path = $data_dir
endif

echo "Using data from: $data_path"

# Change to job directory
cd $job_dir

# Run the training with proper error handling
echo "========================================================"
echo "Starting training..."
echo "========================================================"

# Run Python script and capture exit status
python main.py \
    --mode train \
    --config config.yaml \
    --data.path $data_path \
    --system.checkpoints_path $checkpoint_dir \
    |& tee training_$SLURM_JOB_ID.log

# Check exit status
set exit_status = $status
if ( $exit_status == 0 ) then
    echo "Training completed successfully!"
    
    # Copy results from scratch if used
    if ( "$data_path" != "$data_dir" ) then
        echo "Copying checkpoints from scratch..."
        rsync -av $scratch_dir/checkpoints/ $checkpoint_dir/ >& /dev/null
    endif
else
    echo "Training failed with exit code $exit_status"
    echo "Check log file: training_$SLURM_JOB_ID.log"
endif

# Cleanup scratch directory
echo "Cleaning up scratch directory..."
rm -rf $scratch_dir

echo "========================================================"
echo "Job finished at: `date`"
echo "Total runtime: $SECONDS seconds"
echo "========================================================"

exit $exit_status