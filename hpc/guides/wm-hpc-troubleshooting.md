# W&M HPC Troubleshooting Guide

## Common Issues and Solutions

### 1. Connection Problems

#### Cannot SSH to cluster
```bash
# Error: "Connection refused" or "Connection timed out"

# Check 1: Are you on campus or using VPN?
# If off-campus without VPN, use bastion:
ssh -J username@bastion.wm.edu username@bora.sciclone.wm.edu

# Check 2: Is your account active?
# Contact hpc-help@wm.edu if unsure

# Check 3: Network issues
ping bora.sciclone.wm.edu
```

#### SSH key issues
```bash
# Generate new SSH key if needed
ssh-keygen -t rsa -b 4096

# Add to ssh-agent
ssh-add ~/.ssh/id_rsa

# For MPI over SSH (required for some applications)
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### 2. Job Submission Issues

#### Job won't start (stays pending)
```bash
# Check why job is pending
scontrol show job JOBID | grep Reason

# Common reasons:
# - Resources not available
# - Priority (other jobs ahead)
# - ReqNodeNotAvail (requested specific unavailable nodes)
# - QOSMaxCpuPerUserLimit (exceeding user limits)

# Solutions:
# 1. Request fewer resources
# 2. Check if specific nodes are down:
sinfo -N

# 3. Remove node constraints if not needed
# Change: #SBATCH -C hi
# To: (remove the line)
```

#### Job fails immediately
```bash
# Check error output
cat slurm-JOBID.out

# Common causes and fixes:

# 1. Module not found
# Solution: Check available modules
module avail software_name

# 2. Command not found
# Solution: Load required module first
module load software_name

# 3. Permission denied
# Solution: Make script executable
chmod u+x script.sh

# 4. Bad interpreter
# Solution: Check first line of script
head -1 script.sh
# Should be: #!/bin/tcsh or #!/bin/bash
```

### 3. Runtime Errors

#### Out of memory
```bash
# Check memory usage of failed job
seff JOBID

# Solutions:
# 1. Request more memory
#SBATCH --mem=128G

# 2. Use higher memory nodes
#SBATCH -C bigmem

# 3. Optimize code to use less memory
```

#### Job runs slowly
```bash
# Check if using all requested cores
# During job run:
srun --jobid=JOBID hostname
ssh node_name
top

# Look for:
# - CPU usage < 100% * cores
# - High wait time (wa)
# - Swapping (si/so > 0)

# Solutions:
# 1. Ensure parallel code is actually parallel
# 2. Check I/O patterns (use local scratch)
# 3. Profile code for bottlenecks
```

#### MPI errors
```bash
# Common MPI error messages:

# "All nodes which are allocated for this job are already filled"
# Solution: Use srun instead of mpirun/mpiexec

# "MPIDI_CH3I_RDMA_Process.ib_send_op_process"
# Solution: Network issue, try resubmitting

# "Fatal error in PMPI_Init"
# Solution: Module conflicts, use:
module purge
module load intel intelmpi
```

### 4. Storage Issues

#### Quota exceeded
```bash
# Check quotas
quota -s

# Find large files
du -hs ~/.* ~/* | sort -h | tail -20
find ~ -size +1G -type f

# Clean up:
# 1. Remove old job outputs
rm -f slurm-*.out

# 2. Clean conda cache
conda clean --all

# 3. Clear pip cache
pip cache purge

# 4. Find and remove old files
find /sciclone/scr*/$USER -mtime +30 -type f
```

#### Cannot write to scratch
```bash
# Check if scratch exists
ls -ld /sciclone/scr10/$USER

# Create if missing
mkdir -p /sciclone/scr10/$USER

# Check permissions
# Should show: drwx------ username groupname
# Fix if needed:
chmod 700 /sciclone/scr10/$USER
```

### 5. Software Issues

#### Python package installation fails
```bash
# Error: "Permission denied"
# Solution: Install to user directory
pip install --user package_name

# Error: "No space left on device"
# Solution: Use different temp directory
export TMPDIR=/sciclone/scr10/$USER/tmp
mkdir -p $TMPDIR
pip install --user package_name

# Better solution: Use conda environment
module load miniforge3/24.9.2-0
conda create -n myenv python=3.11
conda activate myenv
conda install package_name
```

#### Module conflicts
```bash
# Error: "MODULE_NAME cannot be loaded due to a conflict"

# Solution 1: Purge and reload
module purge
module load required_modules

# Solution 2: Check dependencies
module show software_name

# Solution 3: Use compatible versions
module avail software_name
```

### 6. Performance Issues

#### Check job efficiency
```bash
# After job completes
seff JOBID

# Look for:
# - CPU Efficiency < 80%: Code not using all cores
# - Memory Efficiency < 50%: Requested too much memory

# During job run
ssh compute_node
top -u $USER
```

#### Node seems slow
```bash
# Check for other users' leftover processes
ckload 0.05

# If load is high, report to hpc-help@wm.edu
# Include node name and job ID
```

### 7. File Transfer Issues

#### Slow transfers
```bash
# Use compression
rsync -avzP source/ dest/

# For many small files, tar first
tar czf archive.tar.gz directory/
scp archive.tar.gz remote:
# Then on remote:
tar xzf archive.tar.gz
```

#### Connection drops during transfer
```bash
# Use screen or tmux
screen -S transfer
rsync -avP source/ dest/
# Ctrl+A, D to detach
# screen -r transfer to reattach

# Or use nohup
nohup rsync -avP source/ dest/ &
```

### 8. Environment Issues

#### Wrong shell
```bash
# W&M default is tcsh
# To check:
echo $0

# If you prefer bash, ask hpc-help@wm.edu
# Or in scripts, specify interpreter:
#!/bin/bash
#SBATCH ...
source /usr/local/etc/sciclone.bashrc
```

#### Missing commands after login
```bash
# Likely missing from PATH
# Check:
echo $PATH

# Fix in ~/.tcshrc or ~/.bashrc:
setenv PATH ${PATH}:/path/to/bin  # tcsh
export PATH=$PATH:/path/to/bin    # bash
```

### 9. Quick Diagnostic Commands

```bash
# System info
hostname                    # Which node am I on?
whoami                      # Username
groups                      # Group memberships

# Job info
squeue -u $USER            # My jobs
scontrol show job JOBID    # Job details
sacct -j JOBID            # Job history

# Resource info
sinfo                      # Cluster status
scontrol show node NODE    # Node details
module avail              # Available software

# Storage info
quota -s                   # Quotas
df -h .                   # Current filesystem
du -hs .                  # Current directory size
```

### 10. When to Contact Support

Contact hpc-help@wm.edu when:

1. **Account issues**: Cannot login, need password reset
2. **Software requests**: Need new software installed
3. **System issues**: Nodes down, filesystem problems
4. **Quota increases**: Need more storage
5. **Persistent problems**: Tried solutions but still failing

Include in your email:
- Job ID (if applicable)
- Error messages (copy/paste, not screenshot)
- What you've already tried
- Script used (attach file)
- Module list (`module list` output)

### Emergency Recovery

#### Accidentally deleted important files
```bash
# Check if still in memory
lsof | grep filename

# Check backups (home directory only)
# Contact hpc-help@wm.edu immediately
```

#### Runaway job using all resources
```bash
# Cancel immediately
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# If cannot login to cancel
# Email hpc-help@wm.edu urgently
```

---
*Remember: Most issues have been seen before. Don't hesitate to ask for help!*