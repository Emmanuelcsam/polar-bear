# William & Mary HPC Distributed PyTorch Training Guide

## System Architecture Overview

William & Mary's Bora/Sciclone HPC cluster provides dedicated GPU computing resources through the **Hima partition**, accessible from the shared login node `bora.sciclone.wm.edu`. The system has fully transitioned to SLURM workload management and offers both NVIDIA Tesla P100 and V100 GPUs for deep learning workloads.

### GPU Resources on Hima Partition
- **GPU Nodes**: hi04-hi07 (4 nodes total)
  - hi04, hi05: NVIDIA Tesla P100 GPUs
  - hi06, hi07: NVIDIA Tesla V100 GPUs
- **Hardware**: 2×16-core Intel Xeon E5-2683 v4 processors, 256 GB RAM per node
- **Network**: QDR InfiniBand interconnect for high-speed communication
- **CUDA Version**: 9.1 (system-installed), newer versions available via modules

### Key Access Details
- **Login**: `ssh username@bora.sciclone.wm.edu`
- **Cross-partition submission**: Jobs submitted from Bora can target Hima using constraints
- **Storage**: Use `/sciclone/scr20/$USER` or `/local/scr` for optimal GPU performance

## Production-Ready tcsh SLURM Script

Here's the complete production-ready SLURM job script in tcsh for your fiber optic analysis deep learning code:

```tcsh
#!/bin/tcsh -l
#SBATCH --job-name=fiber-optic-dl
#SBATCH --output=fiber-dl-%j.out
#SBATCH --error=fiber-dl-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=64G
#SBATCH --time=12:00:00
#SBATCH --constraint=v100
#SBATCH --signal=B:USR1@600

# Set up environment for tcsh
set job_dir = $SLURM_SUBMIT_DIR
set checkpoint_dir = /sciclone/scr20/$USER/checkpoints
set local_data = /local/scr/$USER/data_$SLURM_JOB_ID

# Load required modules
module load anaconda3/2023.09
module load cuda/12.3

# Set up distributed training environment
setenv MASTER_PORT `expr 10000 + $SLURM_JOB_ID % 10000`
set master_addr = `scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1`
setenv MASTER_ADDR $master_addr
setenv WORLD_SIZE $SLURM_NTASKS

# NCCL optimization for InfiniBand
setenv NCCL_DEBUG INFO
setenv NCCL_SOCKET_IFNAME ^docker0,lo
setenv NCCL_IB_DISABLE 0
setenv NCCL_IB_CUDA_SUPPORT 1
setenv NCCL_NET_GDR_LEVEL 2

# Set OpenMP threads (hyperthreading enabled)
setenv OMP_NUM_THREADS $SLURM_CPUS_PER_TASK

# Create directories
mkdir -p $checkpoint_dir
mkdir -p $local_data

# Signal handler for checkpointing before timeout
onintr save_and_resubmit

# Check if this is a restart
if ( -f $checkpoint_dir/checkpoint_latest.pth ) then
    echo "Resuming from checkpoint"
    set resume_flag = "--resume $checkpoint_dir/checkpoint_latest.pth"
else
    echo "Starting fresh training"
    set resume_flag = ""
    
    # Copy data to local scratch on first run
    echo "Copying data to local scratch..."
    rsync -av $job_dir/data/ $local_data/
endif

# Activate conda environment
conda activate /sciclone/scr20/$USER/conda-envs/pytorch-gpu

# Launch distributed training
echo "Starting distributed training on $SLURM_NNODES nodes"
echo "MASTER_ADDR:MASTER_PORT=${MASTER_ADDR}:${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"

srun python train_fiber_optic.py \
    --data-dir $local_data \
    --checkpoint-dir $checkpoint_dir \
    --batch-size 64 \
    --epochs 1000 \
    --checkpoint-freq 1800 \
    $resume_flag

# Check if training completed
if ( $status == 0 ) then
    echo "Training completed successfully"
    # Copy final results back
    rsync -av $checkpoint_dir/ $job_dir/results/
else
    echo "Training interrupted, will resubmit"
    goto save_and_resubmit
endif

exit 0

# Signal handler for graceful shutdown
save_and_resubmit:
    echo "Received termination signal, saving checkpoint..."
    
    # Send checkpoint signal to Python process
    killall -USR1 python
    
    # Wait for checkpoint to complete
    sleep 30
    
    # Resubmit job
    set new_job = `sbatch --parsable --dependency=afterany:$SLURM_JOB_ID $0`
    echo "Resubmitted as job $new_job"
    
    exit 0
```

## Environment Setup Instructions

### 1. Create Python Environment

```tcsh
# Request interactive GPU session
srun -C v100 --gres=gpu:1 --mem=16G --time=1:00:00 --pty tcsh

# Load modules
module load anaconda3/2023.09
module load cuda/12.3

# Create conda environment in scratch space
conda create -p /sciclone/scr20/$USER/conda-envs/pytorch-gpu python=3.11

# Activate environment
conda activate /sciclone/scr20/$USER/conda-envs/pytorch-gpu

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional packages
conda install numpy scipy matplotlib h5py tensorboard -c conda-forge
pip install wandb tensorboardX
```

### 2. Configure Conda

Create `~/.condarc`:
```yaml
envs_dirs:
  - /sciclone/scr20/$USER/conda-envs
pkgs_dirs:
  - /sciclone/scr20/$USER/conda-pkgs
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
```

## Python Training Script Template

Here's a template for `train_fiber_optic.py` with distributed training support:

```python
import os
import argparse
import signal
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed.checkpoint as dcp

class FiberOpticModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model architecture here
        pass

def setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    
    return world_size, rank, local_rank

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    if dist.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state': model.module.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_latest.pth")
        torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth")

def signal_handler(signum, frame):
    # Save checkpoint when receiving signal
    global model, optimizer, epoch, checkpoint_dir
    print(f"Received signal {signum}, saving checkpoint...")
    save_checkpoint(model, optimizer, epoch, checkpoint_dir)
    dist.destroy_process_group()
    exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--checkpoint-freq', type=int, default=1800)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # Setup distributed training
    world_size, rank, local_rank = setup_distributed()
    
    # Setup signal handler
    signal.signal(signal.SIGUSR1, signal_handler)
    
    # Create model
    model = FiberOpticModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 * world_size)
    criterion = nn.MSELoss()
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
        model.module.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Setup data loading
    dataset = YourFiberOpticDataset(args.data_dir)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(local_rank, non_blocking=True)
            target = target.to(local_rank, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Checkpoint at specified frequency
        if epoch % args.checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch, args.checkpoint_dir)
        
        if rank == 0:
            print(f"Epoch {epoch}/{args.epochs}, Loss: {loss.item():.4f}")
    
    # Final checkpoint
    save_checkpoint(model, optimizer, args.epochs, args.checkpoint_dir)
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    # Make globals accessible to signal handler
    global model, optimizer, epoch, checkpoint_dir
    main()
```

## Key Configuration Details

### tcsh-Specific Syntax
- **Environment variables**: Use `setenv VAR value` (not `export`)
- **Local variables**: Use `set var = value` (spaces required)
- **Command substitution**: Use backticks `` `command` ``
- **Conditionals**: Use `if ( condition ) then ... endif`

### GPU Allocation Strategy
- **P100 GPUs**: Use `--constraint=p100` for older but capable GPUs
- **V100 GPUs**: Use `--constraint=v100` for newer, faster GPUs
- **Mixed allocation**: Use `--constraint=gpu` for any available GPU

### Storage Optimization
- **Local scratch**: `/local/scr` provides fastest I/O for training data
- **Parallel scratch**: `/sciclone/scr20/$USER` for checkpoints and results
- **Home directory**: Limited to 50GB, use only for code and small files

### Network Configuration
- **InfiniBand**: Hima uses QDR InfiniBand for inter-node communication
- **NCCL settings**: Optimized for W&M's network topology
- **GPU-Direct**: Enabled for optimal distributed training performance

## Monitoring and Debugging

### Job Monitoring Commands
```tcsh
# Check job status
squeue -u $USER

# Monitor GPU usage on allocated nodes
srun --jobid=$SLURM_JOB_ID --overlap nvidia-smi

# View detailed job information
scontrol show job $SLURM_JOB_ID

# Check job efficiency after completion
sacct -j $SLURM_JOB_ID --format=JobID,State,Elapsed,MaxRSS,CPUTime
```

### Common Issues and Solutions

1. **Module not found**: Add `-l` flag to tcsh shebang to load login environment
2. **CUDA version mismatch**: Load appropriate CUDA module matching PyTorch
3. **Out of memory**: Reduce batch size or use gradient accumulation
4. **Network timeouts**: Increase NCCL_TIMEOUT for large models
5. **Checkpointing failures**: Ensure sufficient disk space in scratch directories

## Best Practices Summary

1. **Always checkpoint** every 30-60 minutes for 12-hour jobs
2. **Use local scratch** for training data to minimize I/O bottlenecks
3. **Monitor GPU utilization** to ensure efficient resource usage
4. **Scale learning rate** with world size for distributed training
5. **Test on single GPU** before scaling to multi-node setups
6. **Clean up old checkpoints** to avoid filling scratch quotas

This comprehensive setup will enable efficient distributed PyTorch training on William & Mary's HPC system with automatic resubmission and robust checkpointing for long-running deep learning experiments.