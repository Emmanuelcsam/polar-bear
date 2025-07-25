#!/bin/tcsh
#SBATCH --job-name=fiber_optics_nn
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ecsampson@wm.edu
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Job description
echo "Starting Fiber Optics Neural Network Training on W&M HPC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: `date`"
echo "User: $USER"
set start_time = `date +%s`

# Check what node we got
echo "Running on node: `hostname`"

# Check node load
ckload 0.05

# Navigate to the work directory
cd /sciclone/scr10/$USER/polar-bear/Network

# Verify we're in the right place
if (! -d core) then
    echo "ERROR: Cannot find core directory."
    echo "Current directory: `pwd`"
    exit 1
endif

# Load required modules
module purge
module load miniforge3/24.9.2-0

# Source the SciClone tcsh configuration
source /usr/local/etc/sciclone.cshrc

# Add user's local bin to PATH
setenv PATH ${HOME}/.local/bin:${PATH}

# Activate your environment
echo "Activating polar-bear-env..."
set VENV_PATH = /sciclone/scr-lst/$USER/polar-bear-env

if (-f $VENV_PATH/bin/activate.csh) then
    source $VENV_PATH/bin/activate.csh
else
    echo "WARNING: Could not find activate.csh, trying conda..."
    conda activate $VENV_PATH
endif

# Verify Python
echo "Python location: `which python`"
echo "Python version: `python --version`"

# Check for GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Set environment variables based on GPU availability
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if ($status == 0) then
    echo "GPU detected, using CUDA"
    set device = "cuda"
else
    echo "No GPU detected, using CPU"
    set device = "cpu"
endif

# Set CPU threading
setenv OMP_NUM_THREADS 8
setenv MKL_NUM_THREADS 8

# Create directories
mkdir -p logs checkpoints results output statistics

# Setup data symlinks if needed
if (! -e dataset) then
    ln -s /sciclone/data10/$USER/polar-bear/dataset dataset
endif
if (! -e reference) then
    ln -s /sciclone/data10/$USER/polar-bear/reference reference
endif

# Update config for device
if (-f config/config.yaml) then
    sed -i "s/device: \"[^\"]*\"/device: \"$device\"/" config/config.yaml
endif

# Update results directory for this run
sed -i "s|results_dir: \"../results\"|results_dir: \"results/run_${SLURM_JOB_ID}\"|" config/config.yaml

# Run the main program
echo "Starting training on $device..."
python core/main.py >& logs/training_${SLURM_JOB_ID}.log

# Check exit status
set exit_status = $?
if ($exit_status == 0) then
    echo "Training completed successfully!"
    
    # Copy results back
    mkdir -p $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}
    cp -r results/run_${SLURM_JOB_ID}/* $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}/
    cp logs/training_${SLURM_JOB_ID}.log $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}/
    
    if (-f checkpoints/best_model.pth) then
        cp checkpoints/best_model.pth $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}/
    endif
else
    echo "Training failed with exit code $exit_status"
    echo "Check logs for details"
endif

# Calculate runtime
set end_time = `date +%s`
@ runtime = $end_time - $start_time
echo "Job completed at: `date`"
echo "Total runtime: $runtime seconds"

echo ""
echo "=== Summary ==="
echo "Device used: $device"
echo "Results saved to: $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}/"