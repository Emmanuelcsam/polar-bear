# Running Polar Bear Neural Network on William & Mary HPC

This guide provides instructions for running the Fiber Optics Neural Network on William & Mary's HPC clusters (Bora and Kuro).

## Cluster Specifications

### Bora Cluster (GPU-enabled)
- **Purpose**: GPU-accelerated deep learning
- **GPUs**: Available (specific models vary)
- **Usage**: For training neural networks with CUDA support
- **Constraint**: Use `-C bo` in SLURM

### Kuro Cluster (CPU-only)
- **Purpose**: Large-scale CPU parallel computing
- **Nodes**: 47 nodes with AMD EPYC processors
- **Cores**: 64 cores per node
- **Memory**: 384 GB per node
- **Usage**: For CPU-intensive parallel jobs

## Initial Setup

1. **Login to the appropriate cluster**:
   ```bash
   ssh username@bora.sciclone.wm.edu  # For GPU jobs
   ssh username@kuro.sciclone.wm.edu  # For CPU jobs
   ```

2. **Copy project files to scratch**:
   ```bash
   cd /sciclone/scr10/$USER
   git clone <your-repo> polar-bear
   ```

3. **Run the setup script**:
   ```tcsh
   cd /sciclone/scr10/$USER/polar-bear/Network
   chmod +x setup_hpc.sh
   ./setup_hpc.sh
   ```

4. **Copy your data files**:
   - Images: `/sciclone/data10/$USER/polar-bear/dataset/`
   - Reference tensors: `/sciclone/data10/$USER/polar-bear/reference/`

## Submitting Jobs

### For GPU Training (Bora):
```bash
sbatch run_job_bora.slurm
```

### For CPU Training (Kuro):
```bash
sbatch run_job_kuro.slurm
```

## Job Management

- **Check job status**: `squeue -u $USER`
- **View job details**: `scontrol show job <job_id>`
- **Cancel a job**: `scancel <job_id>`
- **Check node load**: `ckload 0.05`

## Configuration Files

- **config.yaml**: Main configuration (GPU-enabled)
- **config_hpc_kuro.yaml**: CPU-specific configuration for Kuro
- **run_job_bora.slurm**: SLURM script for Bora (GPU)
- **run_job_kuro.slurm**: SLURM script for Kuro (CPU)

## Important Notes

1. **Shell Environment**: W&M HPC uses tcsh by default. All scripts are written for tcsh.

2. **Module System**: Always load required modules:
   ```tcsh
   module load miniforge3/24.9.2-0
   ```

3. **Storage Locations**:
   - Working directory: `/sciclone/scr10/$USER/polar-bear`
   - Data storage: `/sciclone/data10/$USER/polar-bear`
   - Virtual environment: `/sciclone/scr-lst/$USER/polar-bear-env`

4. **Time Limits**:
   - Maximum walltime: 48 hours (Kuro), 72 hours (most clusters)
   - Jobs are automatically terminated after walltime

5. **Best Practices**:
   - Always use `ckload` before starting intensive jobs
   - Don't produce output in .cshrc files
   - Use scratch filesystems for better I/O performance
   - Copy results back to home directory after job completion

## Troubleshooting

1. **Module not found**: Make sure to source the SciClone configuration:
   ```tcsh
   source /usr/local/etc/sciclone.cshrc
   ```

2. **Python package issues**: Activate the virtual environment:
   ```tcsh
   source /sciclone/scr-lst/$USER/polar-bear-env/bin/activate.csh
   ```

3. **CUDA not available**: Ensure you're on Bora (not Kuro) and have requested GPUs:
   ```tcsh
   #SBATCH --gpus=2
   ```

4. **Memory errors**: Adjust batch size in config.yaml or request more memory:
   ```tcsh
   #SBATCH --mem=256G
   ```

## Support

For HPC-specific issues, contact: hpc-help@wm.edu

For software issues, check the logs in:
- `logs/training_${SLURM_JOB_ID}.log`
- `slurm-${SLURM_JOB_ID}.out`
- `slurm-${SLURM_JOB_ID}.err`