#!/bin/tcsh
#SBATCH --job-name=fiber_optics_nn
#SBATCH -N 1                     # Number of nodes
#SBATCH -n 64                    # Total number of cores
#SBATCH --ntasks-per-node=64     # Cores per node (Kuro has 64)
#SBATCH -t 48:00:00             # Walltime (48 hours max for Kuro)
#SBATCH --mem=256G              # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ecsampson@wm.edu   # Update if different
#SBATCH -o slurm-%j.out         # Standard output
#SBATCH -e slurm-%j.err         # Standard error

# Job description
echo "Starting Fiber Optics Neural Network Training on W&M HPC - Kuro"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: `date`"
echo "User: $USER"
set start_time = `date +%s`

# Check node load before starting (optional but recommended)
ckload 0.05

# Your code is already in scr10, so go there directly
cd /sciclone/scr10/$USER/polar-bear

# Verify we're in the right place
if (! -d core) then
    echo "ERROR: Cannot find core directory. Are we in the right place?"
    echo "Current directory: `pwd`"
    exit 1
endif

# Load required modules
module purge
module load miniforge3/24.9.2-0

# Add user's local bin to PATH (where pip installed packages)
setenv PATH ${HOME}/.local/bin:${PATH}

# Activate your environment
echo "Activating polar-bear-env..."
set VENV_PATH = /sciclone/scr-lst/$USER/polar-bear-env

# It appears to be a regular Python venv based on your prompt
if (-f $VENV_PATH/bin/activate.csh) then
    source $VENV_PATH/bin/activate.csh
else
    echo "WARNING: Could not find activate.csh, trying conda..."
    conda activate $VENV_PATH
endif

# Verify Python and dependencies
echo "Python location: `which python`"
echo "Python version: `python --version`"
echo "PATH: $PATH"

# Check if PyTorch is installed
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
if ($status != 0) then
    echo "ERROR: PyTorch still not found after installation!"
    echo "Checking pip list..."
    pip list | grep torch
    exit 1
endif

# Set environment variables for optimal CPU performance
setenv OMP_NUM_THREADS 16      # Use multiple threads per process
setenv MKL_NUM_THREADS 16      # Intel MKL threading
setenv NUMEXPR_NUM_THREADS 16

# Create necessary directories if they don't exist
mkdir -p logs checkpoints results output statistics

# Fix the dataset and reference symlinks
# First remove broken symlinks if they exist
if (-l dataset/dataset) then
    echo "Removing broken dataset symlink..."
    rm dataset/dataset
endif

if (-l reference/reference) then
    echo "Removing broken reference symlink..."
    rm reference/reference
endif

# Check if dataset and reference are already symlinks or directories
if (-d dataset && ! -l dataset) then
    echo "dataset is a directory, not creating symlink"
else if (! -e dataset) then
    echo "Creating symlink to dataset..."
    ln -s /sciclone/data10/$USER/polar-bear/dataset dataset
endif

if (-d reference && ! -l reference) then
    echo "reference is a directory, not creating symlink"
else if (! -e reference) then
    echo "Creating symlink to reference data..."
    ln -s /sciclone/data10/$USER/polar-bear/reference reference
endif

# Ensure config uses CPU (since Kuro has no GPU)
if (-f config/config.yaml) then
    # Make a backup
    cp config/config.yaml config/config.yaml.bak
    # Change device to CPU
    sed -i 's/device: "cuda"/device: "cpu"/' config/config.yaml
    # Update data paths to use local directories
    sed -i 's|data_path: "../dataset"|data_path: "./dataset"|' config/config.yaml
    sed -i 's|reference_data_path: "../reference"|reference_data_path: "./reference"|' config/config.yaml
    sed -i 's|tensorized_data_path: "../reference"|tensorized_data_path: "./reference"|' config/config.yaml
endif

# Run the main program with CPU optimization
echo "Starting training on CPU..."
echo "Working directory: `pwd`"
ls -la dataset reference

python core/main.py \
    --config config/config.yaml \
    --mode production \
    --num-epochs 50 \
    --checkpoint-interval 5 \
    --batch-size 32 \
    --results-dir results/run_${SLURM_JOB_ID} \
    >& logs/training_${SLURM_JOB_ID}.log

# Check exit status
set exit_status = $?
if ($exit_status == 0) then
    echo "Training completed successfully!"
    
    # Copy results back to home directory
    echo "Copying results to home..."
    mkdir -p $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}
    cp -r results/run_${SLURM_JOB_ID}/* $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}/
    cp logs/training_${SLURM_JOB_ID}.log $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}/
    
    # Copy best model checkpoint
    if (-f checkpoints/best_model.pth) then
        cp checkpoints/best_model.pth $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}/
    endif
    
    # Restore original config
    if (-f config/config.yaml.bak) then
        mv config/config.yaml.bak config/config.yaml
    endif
else
    echo "Training failed with exit code $exit_status"
    echo "Check logs for details"
    
    # Restore original config even on failure
    if (-f config/config.yaml.bak) then
        mv config/config.yaml.bak config/config.yaml
    endif
endif

# Calculate runtime
set end_time = `date +%s`
@ runtime = $end_time - $start_time
echo "Job completed at: `date`"
echo "Total runtime: $runtime seconds"

# Summary
echo ""
echo "=== Summary ==="
echo "Working directory: /sciclone/scr10/$USER/polar-bear"
echo "Data directory: /sciclone/data10/$USER/polar-bear"
echo "Environment: /sciclone/scr-lst/$USER/polar-bear-env"
echo "Results saved to: $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}/"
echo "To continue training, use checkpoint: /sciclone/scr10/$USER/polar-bear/checkpoints/best_model.pth"