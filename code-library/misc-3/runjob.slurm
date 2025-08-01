#!/bin/bash
#SBATCH --job-name=fiber_optic_defect_detection
#SBATCH --output=fiber_optic_%j.out
#SBATCH --error=fiber_optic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# William and Mary HPC Bora specific settings
module purge
module load cuda/11.8
module load python/3.9
module load gcc/9.3.0

# Set environment variables for optimal GPU performance
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    
    # Install required packages optimized for HPC
    pip install --upgrade pip
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
    pip install opencv-python==4.8.0.74
    pip install numpy==1.24.3
    pip install scipy==1.10.1
    pip install scikit-learn==1.3.0
    pip install scikit-image==0.21.0
    pip install albumentations==1.3.1
    pip install PyYAML==6.0
    pip install matplotlib==3.7.1
    pip install seaborn==0.12.2
    pip install pandas==2.0.3
    pip install tqdm==4.65.0
    pip install tensorboard==2.13.0
    pip install efficientnet-pytorch==0.7.1
else
    source venv/bin/activate
fi

# Print system information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "========================"

# Print GPU information
nvidia-smi

# Print dataset information
echo "=== Dataset Information ==="
echo "Number of chunk directories: $(find dataset -maxdepth 1 -name 'chunk_*' -type d | wc -l)"
echo "Total images in dataset: $(find dataset -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' | wc -l)"
echo "Reference directories: $(find reference -maxdepth 1 -type d | wc -l)"
echo "Reference .pt files: $(find reference -name '*.pt' | wc -l)"
echo "============================"

# Set up logging
export PYTHONUNBUFFERED=1
exec > >(tee -a "training_log_${SLURM_JOB_ID}.txt")
exec 2>&1

# Start training with error handling
echo "Starting fiber optic defect detection training..."
echo "Start time: $(date)"

python neuralnetworkscript.py 2>&1 | tee -a training_output.log

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Training completed successfully at $(date)"
    
    # Generate training summary
    echo "=== Training Summary ==="
    echo "Final model saved: $(ls -la final_model.pth 2>/dev/null || echo 'Not found')"
    echo "Checkpoints created: $(ls checkpoint_epoch_*.pth 2>/dev/null | wc -l)"
    echo "Log file size: $(du -h log.txt 2>/dev/null || echo 'Not found')"
    echo "========================"
    
    # Copy results to permanent storage (adjust path as needed)
    # cp -r *.pth *.log *.txt /path/to/permanent/storage/
    
else
    echo "Training failed with exit code ${PIPESTATUS[0]} at $(date)"
    echo "Check training_output.log and log.txt for detailed error information"
    exit 1
fi

echo "Job completed at $(date)"
