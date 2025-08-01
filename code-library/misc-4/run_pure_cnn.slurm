#!/bin/tcsh
#SBATCH --job-name=fiber-pure-cnn
#SBATCH --output=fiber-cnn-%j.out  
#SBATCH --error=fiber-cnn-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8  
#SBATCH --gres=gpu:8         # Request 8 GPUs for multi-GPU training
#SBATCH --cpus-per-task=6
#SBATCH --mem=240G           # Increased memory for larger batches
#SBATCH --time=72:00:00      # Extended time for deep learning training
#SBATCH --constraint=bo      # Bora nodes
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ecsampson@wm.edu

# Environment setup
echo "========================================================"
echo "PURE CNN FIBER OPTIC ANALYSIS - HPC Bora"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start: `date`"
echo "========================================================"

# Load modules
source /usr/local/etc/sciclone.cshrc
module purge
module load miniforge3/24.9.2-0
module load cuda/12.3

# Setup environment
setenv JOB_SCRATCH /sciclone/scr20/$USER/fiber-pure-$SLURM_JOB_ID
setenv CONDA_PKGS_DIRS $JOB_SCRATCH/conda_pkgs  
mkdir -p $JOB_SCRATCH

# Activate conda environment
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.csh
setenv ENV_PREFIX /sciclone/scr20/$USER/fiber-pure-env

# Create or activate environment
if ( ! -d $ENV_PREFIX ) then
    echo ">>> Creating optimized PyTorch environment"
    conda create -y -p $ENV_PREFIX python=3.11
    conda activate $ENV_PREFIX
    
    # Install PyTorch with CUDA support
    mamba install -y -c pytorch -c nvidia -c conda-forge \
        pytorch==2.4 torchvision torchaudio pytorch-cuda=12.1 \
        timm segmentation-models-pytorch \
        numpy scipy scikit-learn pandas \
        opencv albumentations matplotlib seaborn tqdm pyyaml \
        tensorboard wandb
        
    # Additional ML libraries
    pip install --no-cache-dir \
        efficientnet-pytorch \
        focal-loss-torch \
        torchmetrics
else
    conda activate $ENV_PREFIX
endif

# Verify GPU setup
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Setup multi-GPU communication
setenv MASTER_ADDR localhost
setenv MASTER_PORT 29500
setenv NCCL_DEBUG INFO
setenv NCCL_SOCKET_IFNAME ib0

# Copy data to scratch if needed
setenv DATA_SRC $SLURM_SUBMIT_DIR/dataset
setenv REF_SRC $SLURM_SUBMIT_DIR/reference

if ( `du -sk $DATA_SRC | cut -f1` < 20971520 ) then  # < 20GB
    echo ">>> Copying data to scratch for faster I/O"
    rsync -a --info=progress2 $DATA_SRC $JOB_SCRATCH/
    rsync -a --info=progress2 $REF_SRC $JOB_SCRATCH/
    setenv DATA_PATH $JOB_SCRATCH/dataset
    setenv REF_PATH $JOB_SCRATCH/reference
else
    setenv DATA_PATH $DATA_SRC
    setenv REF_PATH $REF_SRC
endif

# Launch multi-GPU training
echo ">>> Starting distributed training with $SLURM_NTASKS_PER_NODE GPUs"

cd $SLURM_SUBMIT_DIR

# Use torchrun for distributed training
torchrun \
    --standalone \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    fiber_cnn_pure.py \
    --data-dir $DATA_PATH \
    --reference-dir $REF_PATH \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-3 \
    --output-dir $SLURM_SUBMIT_DIR/checkpoints

set exit_status = $status

# Cleanup
echo "Training finished with exit code $exit_status"
if ( $exit_status == 0 ) then
    echo ">>> Training completed successfully!"
    # Copy results back if needed
    if ( "$DATA_PATH" != "$DATA_SRC" ) then
        rsync -av $SLURM_SUBMIT_DIR/checkpoints/ $SLURM_SUBMIT_DIR/final_checkpoints/
    endif
endif

rm -rf $JOB_SCRATCH
echo "End: `date`"
exit $exit_status 