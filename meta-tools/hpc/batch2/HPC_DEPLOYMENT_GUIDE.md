# HPC Deployment Guide for Fiber Optics Neural Network

This guide provides detailed instructions for deploying and running the Polar Bear Fiber Optics Neural Network on William & Mary's Bora HPC cluster.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Single GPU Training](#single-gpu-training)
3. [Multi-GPU Distributed Training](#multi-gpu-distributed-training)
4. [Performance Optimization](#performance-optimization)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

## Initial Setup

### 1. Login to Bora
```bash
ssh your_username@bora.wm.edu
```

### 2. Clone the Repository
```bash
cd /your/work/directory
git clone https://github.com/your-repo/polar-bear.git
cd polar-bear/Network
```

### 3. Run Setup Script
```bash
cd hpc
chmod +x setup_environment.sh
./setup_environment.sh
source setup_env.sh
```

### 4. Test GPU Access
```bash
# Request interactive GPU session
salloc --partition=gpu --gres=gpu:1 --time=0:30:00

# Test GPU
python test_gpu.py
```

## Single GPU Training

### Basic Training Job
Edit the SLURM script to set your email and paths:
```bash
nano run_single_gpu.slurm
# Update: --mail-user=your_email@wm.edu
# Update: cd /path/to/polar-bear/Network
```

Submit the job:
```bash
sbatch run_single_gpu.slurm
```

Monitor job status:
```bash
squeue -u $USER
tail -f logs/fiber_optics_*.out
```

### Custom Training Parameters
```bash
# Edit config.yaml for training parameters
nano config.yaml

# Or pass parameters directly
sbatch run_single_gpu.slurm --epochs 100 --batch-size 64
```

## Multi-GPU Distributed Training

### Small Scale (2-4 GPUs)
```bash
# Edit distributed script
nano run_distributed.slurm
# Update paths and email

# Submit for 4 GPUs across 2 nodes
sbatch --nodes=2 --ntasks-per-node=2 run_distributed.slurm
```

### Large Scale (8+ GPUs)
```bash
# Use the scaling script for optimal performance
sbatch run_multinode_scaling.slurm
```

### Monitoring Distributed Training
```bash
# Check all tasks
squeue -u $USER -t RUNNING

# Monitor master node output
tail -f logs/fiber_optics_dist_*.out

# Check GPU utilization across nodes
ssh node001 nvidia-smi
```

## Performance Optimization

### 1. Data Loading Optimization
```yaml
# In config.yaml
training:
  num_workers: 16  # 2x number of GPUs
  pin_memory: true
  prefetch_factor: 2
```

### 2. Mixed Precision Training
```yaml
training:
  use_amp: true  # Automatic Mixed Precision
  gradient_accumulation_steps: 4
```

### 3. NCCL Optimization
Add to SLURM script:
```bash
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_TREE_THRESHOLD=0
```

### 4. Batch Size Scaling
- Single V100: batch_size = 32-64
- Multi-GPU: batch_size = 16-32 per GPU
- Gradient accumulation for larger effective batch sizes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config.yaml
   # Or use gradient accumulation
   training:
     batch_size: 16
     gradient_accumulation_steps: 4
   ```

2. **Distributed Training Hangs**
   ```bash
   # Enable NCCL debugging
   export NCCL_DEBUG=INFO
   
   # Check network connectivity
   srun --nodes=2 hostname
   ```

3. **Slow Data Loading**
   ```bash
   # Check data location
   # Move data to fast scratch space
   cp -r /path/to/data /scratch/$USER/
   ```

4. **Module Not Found**
   ```bash
   # Ensure environment is activated
   source hpc/setup_env.sh
   
   # Reinstall requirements
   pip install -r requirements.txt
   ```

### Performance Debugging
```python
# Add to main_distributed.py for profiling
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    with_stack=True
) as prof:
    # Training code here
    pass
```

## Best Practices

### 1. Resource Management
- Always specify time limits appropriately
- Use --exclusive for benchmarking
- Release resources when done

### 2. Data Management
```bash
# Stage data to scratch for better I/O
mkdir -p /scratch/$USER/fiber_optics_data
rsync -av /path/to/data/ /scratch/$USER/fiber_optics_data/
```

### 3. Checkpointing
```yaml
# Enable frequent checkpointing
training:
  checkpoint_interval: 10  # epochs
  save_best_only: false
```

### 4. Job Arrays for Hyperparameter Search
```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --job-name=fiber_hparam

LEARNING_RATES=(0.001 0.0005 0.0001)
BATCH_SIZES=(16 32 64)

LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID % 3]}
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID / 3]}

srun python main.py --lr $LR --batch-size $BS
```

### 5. Monitoring and Logging
```bash
# Use tensorboard
tensorboard --logdir=logs --host=0.0.0.0 --port=6006

# Or Weights & Biases
wandb login
export WANDB_PROJECT=fiber-optics-hpc
```

## Example Workflow

### Complete Training Pipeline
```bash
# 1. Setup environment
cd /path/to/polar-bear/Network
source hpc/setup_env.sh

# 2. Test single GPU
sbatch --partition=gpu --gres=gpu:1 --wrap="python main.py --epochs 1"

# 3. Run full training
sbatch hpc/run_distributed.slurm

# 4. Monitor progress
watch -n 60 'squeue -u $USER; tail -n 20 logs/fiber_optics_dist_*.out'

# 5. Evaluate results
sbatch --dependency=afterok:$SLURM_JOB_ID --wrap="python main.py --mode evaluate"
```

## Performance Expectations

### Single V100 GPU
- Training speed: ~100-200 images/second
- Memory usage: 12-16 GB
- Typical epoch time: 5-10 minutes

### Multi-GPU Scaling
- 2 GPUs: 1.8x speedup
- 4 GPUs: 3.5x speedup
- 8 GPUs: 6.5x speedup

### Optimization Impact
- Mixed precision: 1.5-2x speedup
- Optimized data loading: 20-30% improvement
- Gradient accumulation: Allows 2-4x larger effective batch size

## Support

For HPC-specific issues:
- W&M HPC Documentation: https://www.wm.edu/it/rc/
- Contact: hpc@wm.edu

For software issues:
- Create issue on GitHub repository
- Check logs in `logs/` directory
- Run diagnostic script: `python hpc/test_gpu.py`