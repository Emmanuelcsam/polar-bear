#!/bin/tcsh
#SBATCH --job-name=fiber_optics_nn
#SBATCH --partition=hima        # Use hima partition for GPU access
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 16                   # Number of cores (adjust based on GPU node)
#SBATCH --constraint=v100       # Request V100 GPU (hi07) - fastest option
#SBATCH -t 12:00:00            # Walltime (12 hours)
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ecsampson@wm.edu
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Job description
echo "Starting Fiber Optics Neural Network Training on W&M HPC - Hima GPU Node"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: `date`"
echo "User: $USER"
echo "Hostname: `hostname`"
set start_time = `date +%s`

# Check node load before starting
ckload 0.05

# Navigate to the work directory on scratch filesystem
cd /sciclone/scr10/$USER/polar-bear/Network

# Verify we're in the right place
if (! -d core) then
    echo "ERROR: Cannot find core directory. Are we in the right place?"
    echo "Current directory: `pwd`"
    exit 1
endif

# Load required modules
module purge
module load miniforge3/24.9.2-0

# Load CUDA module if available
module load cuda

# Source the SciClone tcsh configuration
source /usr/local/etc/sciclone.cshrc

# Add user's local bin to PATH (where pip installed packages)
setenv PATH ${HOME}/.local/bin:${PATH}

# Activate your environment
echo "Activating polar-bear-env..."
set VENV_PATH = /sciclone/scr-lst/$USER/polar-bear-env

# Activate Python virtual environment for tcsh
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

# Check if PyTorch is installed and can see GPUs
echo "Checking PyTorch and GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
if ($status != 0) then
    echo "ERROR: PyTorch not found!"
    echo "Checking pip list..."
    pip list | grep torch
    exit 1
endif

# Set environment variables for optimal GPU performance
setenv OMP_NUM_THREADS 8       # Use 8 threads per process
setenv MKL_NUM_THREADS 8       # Intel MKL threading
setenv NUMEXPR_NUM_THREADS 8

# CUDA environment variables
setenv CUDA_VISIBLE_DEVICES 0  # Use first GPU

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

# Use the original config.yaml that has CUDA enabled
if (-f config/config.yaml.original) then
    cp config/config.yaml.original config/config.yaml
    echo "Restored original GPU-enabled configuration"
endif

# Ensure config uses CUDA
if (-f config/config.yaml) then
    sed -i 's/device: "cpu"/device: "cuda"/' config/config.yaml
    # Also adjust batch size for GPU memory
    sed -i 's/batch_size: 8/batch_size: 16/' config/config.yaml
endif

# Validate data availability
echo "Validating data setup..."
echo "Working directory: `pwd`"
ls -la dataset reference

# Check for data files
echo "Checking for data files..."
set has_images = `find dataset -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" -o -name "*.bmp" -o -name "*.tiff" -o -name "*.tif" 2>/dev/null | head -1 | wc -l`
set has_tensors = `find reference -name "*.pt" 2>/dev/null | head -1 | wc -l`

if ($has_images == 0 && $has_tensors == 0) then
    echo "ERROR: No data files found!"
    echo "Please ensure either:"
    echo "  1. Image files exist in: /sciclone/data10/$USER/polar-bear/dataset/"
    echo "  2. Tensor files (.pt) exist in: /sciclone/data10/$USER/polar-bear/reference/"
    exit 1
else
    if ($has_images > 0) then
        echo "Found image files in dataset directory"
        find dataset -name "*.jpg" -o -name "*.png" | head -5
    endif
    if ($has_tensors > 0) then
        echo "Found tensor files in reference directory"
        find reference -name "*.pt" | head -5
    endif
endif

# Run the main program with GPU optimization
echo "Starting training on GPU..."

# Update config for this specific run
sed -i "s|results_dir: \"../results\"|results_dir: \"results/run_${SLURM_JOB_ID}\"|" config/config.yaml

# Run without any command line arguments - everything from config.yaml
python core/main.py >& logs/training_${SLURM_JOB_ID}.log

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
else
    echo "Training failed with exit code $exit_status"
    echo "Check logs for details"
endif

# Calculate runtime
set end_time = `date +%s`
@ runtime = $end_time - $start_time
echo "Job completed at: `date`"
echo "Total runtime: $runtime seconds"

# Summary
echo ""
echo "=== Summary ==="
echo "Working directory: /sciclone/scr10/$USER/polar-bear/Network"
echo "Data directory: /sciclone/data10/$USER/polar-bear"
echo "Environment: /sciclone/scr-lst/$USER/polar-bear-env"
echo "Results saved to: $HOME/polar-bear/Network/hpc_results/${SLURM_JOB_ID}/"
echo "To continue training, use checkpoint: /sciclone/scr10/$USER/polar-bear/Network/checkpoints/best_model.pth"