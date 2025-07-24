# W&M HPC Quick Reference Card

## Essential Commands

### Connection
```bash
# On campus/VPN
ssh username@bora.sciclone.wm.edu

# Off campus  
ssh -J username@bastion.wm.edu username@bora.sciclone.wm.edu

# With graphics
ssh -Y username@bora.sciclone.wm.edu
```

### Job Management
```bash
# Submit job
sbatch script.sh

# Interactive session
salloc -N1 -n8 -t 2:00:00

# View jobs
squeue -u $USER

# Cancel job
scancel JOBID

# Job efficiency  
seff JOBID

# Job details
scontrol show job JOBID
```

### File Locations
```bash
# Home (backed up, 16TB total)
/sciclone/home/username

# Scratch (fast I/O, not backed up)
/sciclone/scr10/username
/sciclone/scr20/username
/sciclone/pscr/username

# Data storage
/sciclone/data10/username

# Local scratch (node-specific)
/local/scr/username
```

### Module Commands
```bash
module avail          # List available
module list           # List loaded
module load name      # Load module
module unload name    # Unload module
module purge          # Unload all
```

## Resource Specification

### Basic Options
```bash
-N, --nodes=2              # Number of nodes
-n, --ntasks=32            # Total cores
--ntasks-per-node=16       # Cores per node
-t, --time=1-12:30:00      # Walltime (d-hh:mm:ss)
--mem=64G                  # Memory per node
-J, --job-name=myjob       # Job name
```

### Cluster-Specific
```bash
# Main campus
-C bo    # Bora (20 cores/node)
-C hi    # Hima (32 cores/node, GPU)

# VIMS campus  
-C pt    # Potomac (12-20 cores/node)
-C pm    # Pamunkey (64 cores/node)
-C jm    # James (20 cores/node, default)
```

### GPU Options
```bash
--gpus=1           # Request 1 GPU
-C hi --gpus=1     # Hima GPU
-C p100            # Specific GPU type
```

## Quick Job Templates

### Serial Job
```bash
#!/bin/tcsh
#SBATCH -N 1 -n 1
#SBATCH -t 1:00:00
#SBATCH -J serial_job

cd /sciclone/scr10/$USER/project
./myprogram
```

### Parallel MPI Job
```bash
#!/bin/tcsh
#SBATCH -N 4 --ntasks-per-node=32
#SBATCH -t 12:00:00
#SBATCH -J mpi_job

cd /sciclone/scr10/$USER/project
module load intel intelmpi
srun ./mpi_program
```

### Python Job
```bash
#!/bin/tcsh
#SBATCH -N 1 -n 16
#SBATCH -t 4:00:00

cd /sciclone/scr10/$USER/project
module load miniforge3/24.9.2-0
conda activate myenv
python script.py
```

### Array Job
```bash
#!/bin/tcsh
#SBATCH -N 1 -n 1
#SBATCH -t 0:30:00
#SBATCH -a 1-100

./program input_$SLURM_ARRAY_TASK_ID.dat
```

## File Transfer

### From Local to HPC
```bash
# Single file
scp file.txt username@bora:/sciclone/scr10/username/

# Directory
scp -r mydir/ username@bora:/sciclone/scr10/username/

# With rsync (better for large transfers)
rsync -avzP local_dir/ username@bora:/remote_dir/
```

### From HPC to Local
```bash
# Single file
scp username@bora:/path/to/file.txt ./

# Directory
scp -r username@bora:/path/to/dir/ ./
```

## Common Software Modules

### Compilers
```bash
intel/2024.0
gcc/13.1.0
```

### MPI
```bash
intelmpi/2024.0
openmpi/4.1.5
```

### Languages
```bash
python/3.12.7
miniforge3/24.9.2-0
R/4.3.0
matlab/R2023b
```

### Applications
```bash
gaussian/16
vasp/6.4.1
lammps/2023.08.02
gromacs/2023.3
```

## Useful Commands

### Check Resources
```bash
# Disk usage
du -hs ~/
df -h /sciclone/scr10

# Node info
sinfo
scontrol show node

# Queue status
squeue -p partition
```

### Process Management
```bash
# Your processes
ps -fu $USER

# Kill process
kill PID
kill -9 PID  # Force

# Interactive monitor
top
htop
```

### Environment
```bash
# Current directory
pwd

# List files
ls -la

# Find files
find . -name "*.txt"

# Disk usage of current dir
du -hs .
```

## Best Practices

1. **Always use scratch for I/O**
   ```bash
   cd /sciclone/scr10/$USER/project
   ```

2. **Request only needed resources**
   - Test with small jobs first
   - Use seff to check efficiency

3. **Clean up scratch regularly**
   ```bash
   find /sciclone/scr* -user $USER -mtime +30
   ```

4. **Use modules for software**
   ```bash
   module load software_name
   ```

5. **Check job before submitting**
   ```bash
   sbatch --test-only script.sh
   ```

## Getting Help

- **Email**: hpc-help@wm.edu
- **Website**: https://www.wm.edu/it/rc
- **Office**: ISC3, 1st floor

## Emergency Commands

```bash
# Cancel all your jobs
scancel -u $USER

# Exit interactive session
exit

# Break out of command
Ctrl+C

# Suspend process
Ctrl+Z

# Check if logged into compute node
hostname
```

---
*Keep this reference handy when working on W&M HPC systems!*