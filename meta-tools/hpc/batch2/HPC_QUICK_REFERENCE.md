# HPC Quick Reference Card - Fiber Optics Neural Network

## Pre-Deployment Checklist

### 1. Environment Setup
```bash
# Load required modules
module load python/3.9 cuda/11.8 cudnn/8.6 openmpi/4.1

# Create virtual environment
python -m venv fiber_optics_env
source fiber_optics_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python wandb tensorboard pyyaml
```

### 2. Data Preparation
```bash
# Copy data to HPC scratch (faster I/O)
cp -r reference/ /scratch/$USER/fiber_optics/
cp -r dataset/ /scratch/$USER/fiber_optics/

# Update paths in config.yaml
sed -i 's|reference|/scratch/'$USER'/fiber_optics/reference|g' config.yaml
sed -i 's|dataset|/scratch/'$USER'/fiber_optics/dataset|g' config.yaml
```

### 3. Quick Config Changes for HPC

#### For Maximum Performance:
```yaml
# In config.yaml, update these key settings:
system:
  num_workers: 32  # Half of CPU cores
  device: "cuda"
  
training:
  batch_size: 128  # For V100/A100
  use_amp: true
  amp_opt_level: "O2"
  
optimization:
  use_distributed: true
  num_data_workers: 32
  flash_attention: true
  compile_model: true
```

#### For Maximum Debugging:
```yaml
system:
  log_level: "DEBUG"
  log_every_n_steps: 1
  enable_profiling: true
  profile_memory: true

debug:
  enabled: true
  save_intermediate_features: true
  save_gradient_flow: true
  save_attention_maps: true
  debug_image_frequency: 10

monitoring:
  use_tensorboard: true
  use_wandb: true
  profile_batches: [10, 20, 50, 100]
```

## SLURM Job Scripts

### Single GPU Job (Testing/Debugging)
```bash
#!/bin/bash
#SBATCH --job-name=fiber_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00

source fiber_optics_env/bin/activate
cd /path/to/polar-bear/Network

# Use the HPC config
cp config_hpc_optimal.yaml config.yaml

# Run with debugging
python main.py
```

### Multi-GPU Job (Production Training)
```bash
#!/bin/bash
#SBATCH --job-name=fiber_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00

source fiber_optics_env/bin/activate
cd /path/to/polar-bear/Network

# Use the HPC config
cp config_hpc_optimal.yaml config.yaml

# Distributed training
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    main.py
```

## Performance Tuning Quick Tips

### 1. Batch Size Selection
- **V100 (16GB)**: batch_size = 64-96
- **V100 (32GB)**: batch_size = 128-192
- **A100 (40GB)**: batch_size = 192-256
- **A100 (80GB)**: batch_size = 256-384

### 2. Memory Issues
```yaml
# If OOM, try these in order:
gradient_accumulation_steps: 4  # Simulate larger batches
use_gradient_checkpointing: true  # Trade compute for memory
batch_size: 64  # Reduce batch size
base_channels: 64  # Reduce model size
```

### 3. Speed Optimization Priority
1. Enable mixed precision (use_amp: true)
2. Increase batch size to GPU limit
3. Enable Flash Attention if available
4. Use compiled model (PyTorch 2.0+)
5. Increase num_workers to CPU count/2
6. Use distributed training for multi-GPU

## Monitoring Commands

### During Training
```bash
# GPU utilization
watch -n 1 nvidia-smi

# CPU and memory
htop

# Disk I/O
iotop

# Network (for distributed)
iftop
```

### TensorBoard
```bash
# On HPC node
tensorboard --logdir=runs --port=6006 --bind_all

# On local machine (tunnel)
ssh -L 6006:localhost:6006 user@hpc-login
# Then open http://localhost:6006
```

### Weights & Biases
```bash
# First login
wandb login

# Monitor at https://wandb.ai/your-entity/fiber-optics-hpc
```

## Common Issues & Solutions

### 1. NCCL Errors
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0  # Use InfiniBand
export NCCL_IB_DISABLE=0
```

### 2. CUDA OOM
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Clear cache in code: torch.cuda.empty_cache()
```

### 3. Slow Data Loading
```bash
# Use RAM disk for cache
export CACHE_PATH=/dev/shm/fiber_cache
# Increase workers
num_workers: 64
```

### 4. Distributed Training Hangs
```bash
# Kill zombie processes
pkill -f "python.*main.py"
# Use different port
--master_port=29501
```

## Key Performance Metrics

### Expected Training Speed (8x V100)
- Images/second: 800-1200
- Epoch time (100k images): 2-3 minutes
- Total training (200 epochs): 6-10 hours

### Resource Usage
- GPU Memory: 80-90% per GPU
- GPU Utilization: >95%
- CPU Usage: 50-70%
- Disk I/O: 500MB/s - 2GB/s

### Model Statistics
- Parameters: ~100-200M (advanced)
- FLOPs: ~50-100 GFLOPs
- Checkpoint size: 400-800MB

## Emergency Commands

```bash
# Kill all jobs
scancel -u $USER

# Clear GPU memory
nvidia-smi --gpu-reset

# Check queue
squeue -u $USER

# Job efficiency
seff <job_id>

# Detailed job info
scontrol show job <job_id>
```

## Optimization Workflow

1. **Start Small**: Test with 1 GPU, small batch
2. **Profile**: Enable profiling for 100 batches
3. **Scale Up**: Increase GPUs and batch size
4. **Monitor**: Watch GPU util and memory
5. **Tune**: Adjust based on bottlenecks
6. **Production**: Disable debugging, maximize batch

Remember: Always test configuration changes on a small subset before full training!