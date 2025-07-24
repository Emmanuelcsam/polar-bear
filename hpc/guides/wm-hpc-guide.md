# Complete Step-by-Step Guide to Running Jobs on William & Mary HPC

## Table of Contents
1. [Initial Setup and Access](#initial-setup)
2. [Understanding the HPC Environment](#understanding-environment)
3. [Basic Linux Commands](#basic-commands)
4. [Job Submission with Slurm](#job-submission)
5. [Template Files](#template-files)
6. [Common Workflows](#common-workflows)
7. [Troubleshooting](#troubleshooting)

---

## 1. Initial Setup and Access {#initial-setup}

### Step 1.1: Obtain HPC Account
1. Request access through W&M IT
2. You'll receive your username (typically your W&M ID)
3. Default shell is tcsh (can request bash if preferred)

### Step 1.2: Configure SSH Access

#### For Off-Campus Access:
Create/edit `~/.ssh/config` on your local machine:
```
Host bora.sciclone.wm.edu
    HostName bora.sciclone.wm.edu
    ProxyJump username@bastion.wm.edu
    User username
```

#### For On-Campus/VPN Access:
Simply use: `ssh username@bora.sciclone.wm.edu`

### Step 1.3: First Login
```bash
# Off-campus
ssh username@bora.sciclone.wm.edu

# With graphics support (for GUI applications)
ssh -Y username@bora.sciclone.wm.edu
```

### Step 1.4: Set Up Your Environment
```bash
# Check your shell
echo $0

# Check for symbolic links (if missing, run restorewmlinks)
ls -la ~/
# You should see links like:
# data10 -> /sciclone/data10/username
# lscr -> /local/scr/username

# If missing, run:
restorewmlinks
```

---

## 2. Understanding the HPC Environment {#understanding-environment}

### Available Clusters and Their Specifications:

**Main Campus:**
- **bora**: General purpose, 20 cores/node, use `-C bo`
- **hima**: GPU nodes, 32 cores/node, use `-C hi`
- **femto**: 32 cores/node
- **kuro**: 64 cores/node, 48hr max walltime
- **astral**: GPU cluster
- **gulf**: GPU cluster

**VIMS Campus:**
- **potomac**: 12-20 cores/node, use `-C pt`, 180hr max
- **pamunkey**: 64 cores/node, use `-C pm`, 180hr max
- **james**: 20 cores/node, use `-C jm` (default)

### File Systems:
```bash
# Home directory (16TB total, backed up)
/sciclone/home/username

# Scratch directories (for job I/O, not backed up)
/sciclone/scr10/username
/sciclone/scr20/username
/sciclone/pscr/username

# Data directory (for large datasets)
/sciclone/data10/username
```

### Important Environment Variables:
```bash
# Check your PATH
echo $PATH

# Check loaded modules
module list

# Available modules
module avail
```

---

## 3. Basic Linux Commands {#basic-commands}

### Essential Commands for HPC:
```bash
# Navigation
pwd                     # Print working directory
cd /path/to/directory   # Change directory
cd ~                    # Go to home
cd -                    # Go to previous directory
ls -la                  # List all files with details

# File operations
cp source dest          # Copy file
cp -r source dest       # Copy directory
mv oldname newname      # Move/rename
rm filename             # Remove file
rm -r directory         # Remove directory (careful!)
mkdir newdir            # Make directory

# File permissions
chmod u+x script.sh     # Make executable for user
chmod go+r file         # Make readable for group/others
chmod 755 file          # rwxr-xr-x permissions

# Viewing files
cat file                # Display entire file
less file               # Page through file
head -n 20 file         # First 20 lines
tail -n 20 file         # Last 20 lines

# Finding files
find . -name "*.txt"    # Find all .txt files
find . -size +100M      # Find files > 100MB

# Disk usage
df -h                   # File system usage
du -hs *                # Directory sizes

# Process management
ps -fu username         # Your processes
top                     # Interactive process viewer
kill PID                # Kill process
kill -9 PID             # Force kill (last resort)
```

---

## 4. Job Submission with Slurm {#job-submission}

### Slurm Commands Reference:
| Action | Slurm Command | Old Torque Command |
|--------|---------------|-------------------|
| Submit batch job | `sbatch script.sh` | `qsub script.sh` |
| Interactive session | `salloc` | `qstat -I` |
| View jobs | `squeue` | `qstat` |
| Cancel job | `scancel JOBID` | `qdel JOBID` |
| Run MPI | `srun` | `mpirun/mpiexec` |
| Job efficiency | `seff JOBID` | N/A |

### Resource Specification Options:
```bash
-N, --nodes            # Number of nodes
-n, --ntasks           # Total number of cores
--ntasks-per-node      # Cores per node (like ppn in Torque)
-c, --cpus-per-task    # Threads per task (for hybrid)
-t, --time             # Walltime (various formats)
--mem=                 # Memory per node (e.g., --mem=64G)
-J, --job-name         # Job name
-C, --constraint       # Node features (bo, hi, pt, pm, etc.)
-G, --gpus             # Number of GPUs
--mail-type=ALL        # Email notifications
--mail-user=email      # Email address (not gmail)
```

### Interactive Session Examples:
```bash
# Basic interactive session (1 node, 20 cores, 1 hour)
salloc -N1 -n20 -t 1:00:00

# Bora cluster specifically
salloc -N1 -n20 -t 30:00 -C bo

# Hima with GPU
salloc -N1 -n32 -t 1-0 -C hi --gpus=1

# VIMS potomac cluster
salloc -N1 -n20 -t 1-0 -C pt

# Multi-node with specific cores per node
salloc -N4 --ntasks-per-node=32 -t 2:00:00
```

---

## 5. Template Files {#template-files}

### 5.1 Basic Serial Job (serial_job.sh)
```bash
#!/bin/tcsh
#SBATCH --job-name=serial_test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=username@wm.edu

# For bash scripts, add after SBATCH lines:
# source /usr/local/etc/sciclone.bashrc

# Change to working directory
cd /sciclone/scr10/$USER/myproject

# Load required modules
module load python/3.12.7

# Run your program
python my_script.py > output.log
```

### 5.2 Parallel MPI Job (mpi_job.sh)
```bash
#!/bin/tcsh
#SBATCH --job-name=mpi_test
#SBATCH -N 4
#SBATCH --ntasks-per-node=32
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=username@wm.edu

# Check node load (optional but recommended)
ckload 0.05

# Change to scratch directory
cd /sciclone/scr10/$USER/mpi_project

# Load MPI modules
module load intel/2024.0
module load intelmpi/2024.0

# Run MPI program
srun ./my_mpi_program > output.log
```

### 5.3 GPU Job (gpu_job.sh)
```bash
#!/bin/tcsh
#SBATCH --job-name=gpu_test
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH -C hi
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=username@wm.edu

cd /sciclone/scr10/$USER/gpu_project

# Load CUDA modules
module load cuda/12.1

# Run GPU program
./my_gpu_program
```

### 5.4 Python with Conda Environment (python_conda.sh)
```bash
#!/bin/tcsh
#SBATCH --job-name=python_conda
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 6:00:00

cd /sciclone/scr10/$USER/python_project

# Load miniforge
module load miniforge3/24.9.2-0

# Activate conda environment
conda activate myenv

# Check python location
which python

# Run python script
python analysis.py --input data.csv --output results.csv
```

### 5.5 Array Job for Parameter Sweep (array_job.sh)
```bash
#!/bin/tcsh
#SBATCH --job-name=param_sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -a 1-100
#SBATCH --mail-type=FAIL

cd /sciclone/scr10/$USER/parameter_study

# Use array task ID for different parameters
./my_program --param $SLURM_ARRAY_TASK_ID > output_$SLURM_ARRAY_TASK_ID.log
```

### 5.6 Jupyter Notebook Job (jupyter_job.sh)
```bash
#!/bin/tcsh
#SBATCH --job-name=jupyter
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 4:00:00

# Get the hostname
set host = `hostname`

# Load python
module load python/3.12.7

# Start jupyter
cd /sciclone/scr10/$USER/notebooks
~/.local/bin/jupyter-notebook --no-browser --ip=* --port=8888

# Instructions will be in slurm-JOBID.out
```

---

## 6. Common Workflows {#common-workflows}

### Workflow 1: First-Time Setup
```bash
# 1. Login
ssh username@bora.sciclone.wm.edu

# 2. Create project structure
mkdir -p /sciclone/scr10/$USER/projects
mkdir -p /sciclone/data10/$USER/datasets
cd /sciclone/scr10/$USER/projects

# 3. Copy files from local machine (run on local)
scp -r myproject/ username@bora.sciclone.wm.edu:/sciclone/scr10/username/projects/

# 4. Set up Python environment
module load miniforge3/24.9.2-0
conda create -n myproject python=3.11 numpy pandas matplotlib
conda activate myproject
pip install -r requirements.txt
```

### Workflow 2: Running a Python Analysis
```bash
# 1. Start interactive session
salloc -N1 -n8 -t 2:00:00

# 2. Load environment
cd /sciclone/scr10/$USER/myproject
module load miniforge3/24.9.2-0
conda activate myproject

# 3. Test your script
python test_script.py

# 4. Exit interactive session
exit

# 5. Submit batch job
sbatch python_job.sh

# 6. Monitor job
squeue -u $USER
```

### Workflow 3: Compiling and Running C/C++ Code
```bash
# 1. Load compiler modules
module load intel/2024.0
module load intelmpi/2024.0

# 2. Compile
mpicc -O3 -o my_program my_program.c

# 3. Test with small run
salloc -N1 -n4 -t 0:30:00
srun -n 4 ./my_program

# 4. Submit full job
sbatch mpi_job.sh
```

### Workflow 4: Using Jupyter Notebooks
```bash
# 1. Install Jupyter (one time)
pip install --user notebook

# 2. Submit Jupyter job
sbatch jupyter_job.sh

# 3. Check output for connection info
cat slurm-JOBID.out

# 4. On local machine, create tunnel
ssh -NL 8888:nodeXX:8888 username@bora.sciclone.wm.edu

# 5. Open browser to http://127.0.0.1:8888
```

---

## 7. Troubleshooting {#troubleshooting}

### Common Issues and Solutions:

**Issue: Job won't start**
```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job JOBID

# Common reasons:
# - Requested resources unavailable
# - Walltime too long
# - Wrong constraint (-C flag)
```

**Issue: Job fails immediately**
```bash
# Check error file
cat slurm-JOBID.out

# Common causes:
# - Module not loaded
# - Wrong path
# - Permission denied
# - Missing input files
```

**Issue: Out of memory**
```bash
# Check memory usage of completed job
seff JOBID

# Request more memory
#SBATCH --mem=128G
```

**Issue: MPI job hangs**
```bash
# Check node load before running
ckload -X 0.05

# Email hpc-help@wm.edu if nodes have rouge processes
```

**Issue: Can't find files**
```bash
# Remember: jobs start in submission directory
# Always use absolute paths or cd to working directory
pwd  # Check current directory
```

### Getting Help:
- Email: hpc-help@wm.edu
- Website: https://www.wm.edu/it/rc
- Office: 1st floor ISC3

### Best Practices:
1. Always test with small/short jobs first
2. Use scratch filesystems for I/O
3. Clean up old files regularly
4. Monitor your disk usage with `du -hs`
5. Use version control (git) for code
6. Document your workflows
7. Request only resources you need