#!/bin/tcsh
# Setup script for William & Mary HPC (Bora/Kuro)
# Run this after logging into the cluster to set up your environment

echo "Setting up Polar Bear Neural Network on W&M HPC"
echo "=============================================="

# Check which cluster we're on
set hostname = `hostname`
echo "Running on: $hostname"

# Create necessary directories
echo "Creating directory structure..."
mkdir -p /sciclone/scr10/$USER/polar-bear
mkdir -p /sciclone/data10/$USER/polar-bear/dataset
mkdir -p /sciclone/data10/$USER/polar-bear/reference
mkdir -p /sciclone/scr-lst/$USER

# Load modules
echo "Loading modules..."
module purge
module load miniforge3/24.9.2-0

# Create Python virtual environment
echo "Creating Python virtual environment..."
cd /sciclone/scr-lst/$USER
python -m venv polar-bear-env

# Activate environment
source polar-bear-env/bin/activate.csh

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
cd /sciclone/scr10/$USER/polar-bear/Network
if (-f requirements.txt) then
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found!"
    echo "Please copy your project files to /sciclone/scr10/$USER/polar-bear/Network first"
    exit 1
endif

# Create symlinks in the Network directory
cd /sciclone/scr10/$USER/polar-bear/Network
if (! -e dataset) then
    ln -s /sciclone/data10/$USER/polar-bear/dataset dataset
endif
if (! -e reference) then
    ln -s /sciclone/data10/$USER/polar-bear/reference reference
endif

echo ""
echo "Setup complete!"
echo "=============================================="
echo "Next steps:"
echo "1. Copy your dataset files to: /sciclone/data10/$USER/polar-bear/dataset/"
echo "2. Copy your reference files to: /sciclone/data10/$USER/polar-bear/reference/"
echo "3. Submit your job:"
echo "   - For Bora (GPU): sbatch run_job_bora.slurm"
echo "   - For Kuro (CPU): sbatch run_job_kuro.slurm"
echo ""
echo "To check job status: squeue -u $USER"
echo "To cancel a job: scancel <job_id>"