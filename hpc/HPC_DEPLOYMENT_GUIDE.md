# HPC Deployment Guide for Fiber Optics Neural Network

This guide provides instructions for deploying and running the Fiber Optics Neural Network system on the William & Mary Kuro HPC cluster or similar HPC systems.

## Prerequisites

- Access to the HPC cluster with GPU nodes
- Basic familiarity with SLURM job scheduler
- Storage quota for datasets and model checkpoints

## Quick Start

1. **Clone the repository** to your HPC home directory:
   ```bash
   cd $HOME
   git clone https://github.com/your-repo/polar-bear.git
   cd polar-bear/Network
   ```

2. **Set up the environment**:
   ```bash
   chmod +x hpc/setup_environment.sh
   ./hpc/setup_environment.sh
   ```

3. **Prepare your data**:
   - Copy your fiber optic images to `dataset/` directory
   - Place reference images in `reference/` directory
   - Add sample images to `samples/` for testing

4. **Configure paths** in the SLURM scripts:
   - Edit `hpc/run_job.slurm` and update:
     - Email address for notifications
     - Path to project directory
     - Any module names specific to your HPC

5. **Submit a job**:
   ```bash
   sbatch hpc/run_job.slurm
   ```

## Available SLURM Scripts

### 1. Single GPU Training (`run_job.slurm`)
Basic training job for single GPU:
```bash
sbatch hpc/run_job.slurm
```

**Resources:**
- 1 GPU
- 16 CPU cores
- 64GB RAM
- 24 hours walltime

### 2. Distributed Training (`run_distributed.slurm`)
Multi-node, multi-GPU distributed training:
```bash
sbatch hpc/run_distributed.slurm
```

**Resources:**
- 2 nodes, 4 GPUs per node (8 GPUs total)
- 8 CPU cores per task
- 256GB RAM per node
- 48 hours walltime

### 3. Performance Benchmark (`run_benchmark.slurm`)
Test inference speed and memory usage:
```bash
sbatch hpc/run_benchmark.slurm
```

**Resources:**
- 1 GPU
- 16 CPU cores
- 64GB RAM
- 4 hours walltime

### 4. Interactive Debugging
Get an interactive GPU session:
```bash
./hpc/interactive_debug.sh
```

## Configuration

### Adjusting for Your HPC System

1. **Module names** - Check available modules:
   ```bash
   module avail cuda
   module avail python
   ```
   Update module names in SLURM scripts accordingly.

2. **Partition names** - Check available partitions:
   ```bash
   sinfo
   ```
   Update `--partition` in SLURM scripts.

3. **GPU types** - Check available GPUs:
   ```bash
   sinfo -o "%P %G"
   ```
   Update `--gres` if needed (e.g., `--gres=gpu:v100:1`).

### Optimizing Performance

1. **Batch size** - Edit `config/config.yaml`:
   ```yaml
   training:
     batch_size: 128  # Adjust based on GPU memory
   ```

2. **Number of workers** - Based on CPU cores:
   ```yaml
   system:
     num_workers: 16  # Set to number of CPUs
   ```

3. **Mixed precision training** - Already enabled:
   ```yaml
   training:
     use_amp: true
     amp_opt_level: "O2"
   ```

## Monitoring Jobs

### Check job status:
```bash
squeue -u $USER
```

### View job output:
```bash
tail -f logs/fiber_optics_<job_id>.out
```

### Cancel a job:
```bash
scancel <job_id>
```

### Check GPU utilization:
```bash
ssh <node_name>
nvidia-smi
```

## Data Management

### Using scratch space for better I/O:
```bash
# In your SLURM script, copy data to scratch
cp -r $HOME/polar-bear/Network/dataset $SCRATCH/
export DATA_PATH=$SCRATCH/dataset
```

### Cleaning up:
```bash
# At the end of job, copy results back
cp -r $SCRATCH/results/* $HOME/polar-bear/Network/results/
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size in config.yaml
- Enable gradient checkpointing
- Use mixed precision training

### CUDA errors
1. Check CUDA version compatibility:
   ```bash
   python hpc/test_gpu.py
   ```

2. Ensure correct modules are loaded:
   ```bash
   module list
   ```

### Slow data loading
- Increase num_workers
- Use SSD/scratch space for data
- Enable persistent_workers in config

### Connection issues
- Use tmux or screen for long sessions
- Save checkpoints frequently

## Best Practices

1. **Always test with small data first**:
   ```bash
   # Quick test run
   sbatch --time=00:30:00 hpc/run_job.slurm
   ```

2. **Use checkpointing**:
   - Model saves automatically to `checkpoints/`
   - Resume from checkpoint if job times out

3. **Monitor resource usage**:
   ```bash
   # After job completes
   seff <job_id>
   ```

4. **Request appropriate resources**:
   - Don't over-request memory/time
   - Use job arrays for parameter sweeps

## Advanced Usage

### Custom experiments
Create your own SLURM script based on the templates:
```bash
cp hpc/run_job.slurm hpc/my_experiment.slurm
# Edit as needed
sbatch hpc/my_experiment.slurm
```

### Performance profiling
Enable profiling in config:
```yaml
monitoring:
  enable_profiling: true
  profile_batches: [10, 50, 100]
```

### Using Weights & Biases
Set up W&B credentials:
```bash
wandb login
```

Enable in config:
```yaml
monitoring:
  use_wandb: true
  wandb_project: "fiber-optics-hpc"
```

## Support

For HPC-specific issues:
- Check your institution's HPC documentation
- Contact HPC support team

For software issues:
- Check logs in `logs/` directory
- Review error messages in `.err` files
- Open an issue on the project repository