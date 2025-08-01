#!/bin/tcsh
#SBATCH -A mylab
#SBATCH --partition=hima         # GPU nodes on Bora
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -J endface_cnn
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Fiber Optic End-Face CNN Training Script
# William & Mary HPC Bora Cluster Ready
# Integrated CNN Pipeline with Statistical Priors

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load required modules
module load miniforge3/24.9.2-0
module load cuda/12.3

# Activate conda environment
source activate fiber-ai

# Set up scratch directory
set SCRATCH_DIR = /sciclone/scr20/$USER/fiber_cnn_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# ============================================================================
# PROJECT STRUCTURE VALIDATION
# ============================================================================

echo "=== Validating Project Structure ==="

# Check for required directories
if (! -d "../dataset") then
    echo "ERROR: dataset/ directory not found"
    exit 1
endif

if (! -d "../reference") then
    echo "ERROR: reference/ directory not found"
    exit 1
endif

if (! -d "../statistics") then
    echo "ERROR: statistics/ directory not found"
    exit 1
endif

# Check for source code
if (! -f "../src/train.py") then
    echo "ERROR: src/train.py not found"
    exit 1
endif

if (! -f "../src/model.py") then
    echo "ERROR: src/model.py not found"
    exit 1
endif

if (! -f "../src/dataset.py") then
    echo "ERROR: src/dataset.py not found"
    exit 1
endif

if (! -f "../configs/bora.yaml") then
    echo "ERROR: configs/bora.yaml not found"
    exit 1
endif

echo "Project structure validation passed"

# ============================================================================
# DATA HANDLING
# ============================================================================

echo "=== Setting up Data ==="

# Copy source code to scratch
cp -r ../src .
cp -r ../configs .
cp -r ../reference .

# Check dataset size and copy if manageable
set DATASET_SIZE = `du -sh ../dataset | cut -f1`
echo "Dataset size: $DATASET_SIZE"

# If dataset is under 20GB, copy to scratch for faster I/O
if (`echo $DATASET_SIZE | sed 's/G//'` < 20) then
    echo "Copying dataset to scratch for faster I/O"
    cp -r ../dataset .
    set DATA_ROOT = "./dataset"
else
    echo "Dataset too large, streaming from original location"
    set DATA_ROOT = "../dataset"
endif

# Copy statistics for reference tensor generation
cp -r ../statistics .

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=== Setting up Environment ==="

# Create conda environment if it doesn't exist
conda create -n fiber-ai python=3.9 -y
source activate fiber-ai

# Install required packages
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install albumentations opencv-python tensorboard pyyaml
pip install segmentation-models-pytorch timm
pip install lmdb  # Optional, for faster data loading

# ============================================================================
# STATISTICAL PRIOR GENERATION
# ============================================================================

echo "=== Generating Statistical Priors ==="

# Generate reference statistics from analysis reports
python src/utils.py

# Verify reference statistics were created
if (! -f "reference/ref_stats.pt") then
    echo "ERROR: Failed to generate reference statistics"
    exit 1
endif

echo "Statistical priors generated successfully"

# ============================================================================
# RUNTIME CONFIGURATION
# ============================================================================

echo "=== Creating Runtime Configuration ==="

# Create runtime config with actual paths
cat > configs/runtime.yaml << EOF
# Runtime configuration for Bora cluster
data_root: "$DATA_ROOT"
lmdb: null  # Disable LMDB for now
bs: 8
w: 8
lr: 3e-4
epochs: 50
ckpt_dir: "./checkpoints"
ref_stats: "./reference/ref_stats.pt"
num_classes: 40
EOF

# ============================================================================
# DISTRIBUTED TRAINING SETUP
# ============================================================================

echo "=== Setting up Distributed Training ==="

# Get node information
set NODE_LIST = `scontrol show hostnames $SLURM_JOB_NODELIST`
set MASTER_NODE = `hostname`
set MASTER_PORT = 12345

# Set environment variables for distributed training
setenv MASTER_ADDR $MASTER_NODE
setenv MASTER_PORT $MASTER_PORT
setenv WORLD_SIZE $SLURM_NTASKS
setenv RANK $SLURM_PROCID
setenv LOCAL_RANK $SLURM_LOCALID

echo "Master node: $MASTER_NODE"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Local rank: $LOCAL_RANK"

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

echo "=== Starting Training ==="

# Create output directories
mkdir -p checkpoints logs

# Launch distributed training
srun torchrun \
    --standalone \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    src/train.py \
    --config configs/runtime.yaml \
    --local_rank $SLURM_LOCALID

# Check training exit status
if ($status != 0) then
    echo "ERROR: Training failed with exit code $status"
    exit 1
endif

# ============================================================================
# POST-TRAINING VALIDATION
# ============================================================================

echo "=== Post-Training Validation ==="

# Check if checkpoints were created
if (! -f "checkpoints/last.pt") then
    echo "ERROR: No checkpoint found after training"
    exit 1
endif

# Copy results back to project directory
echo "Copying results to project directory"
cp -r checkpoints ../checkpoints_${SLURM_JOB_ID}
cp -r logs ../logs_${SLURM_JOB_ID}

# ============================================================================
# CLEANUP
# ============================================================================

echo "=== Cleanup ==="

# Remove scratch directory
cd ..
rm -rf $SCRATCH_DIR

echo "=== Training Complete ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Results saved to: ../checkpoints_${SLURM_JOB_ID}"
echo "Logs saved to: ../logs_${SLURM_JOB_ID}"