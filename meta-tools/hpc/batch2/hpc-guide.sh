#!/bin/bash

# =============================================================================
# William & Mary HPC Complete Walkthrough Guide for Neural Network Project
# =============================================================================
# This script guides you through setting up and running your neural network 
# project (e.g., polar-bear) on W&M HPC. Run this in one terminal to follow 
# instructions. In a second terminal, execute the commands as prompted.
#
# Based on official W&M HPC docs (as of July 23, 2025), including Sciclone 
# clusters, Slurm, Miniforge3/Conda, file systems, and best practices.
# Key updates: Miniforge3 replaced Anaconda3 in Feb 2025; use free channels.
# Assumes your project involves Python scripts, dataset images in subfolders,
# reference .pt files in subfolders, and training on Kuro cluster (AMD Zen 4,
# 64 cores/node, up to 980 GB mem, Rocky 9.2, deployed 2024, Slurm, for MPI
# jobs >=64 cores).
#
# Key Resources:
# - Official Guide: https://www.wm.edu/offices/it/services/researchcomputing/using/
# - Slurm: https://www.wm.edu/offices/it/services/researchcomputing/using/running_jobs_slurm/
# - Python/Conda: https://www.wm.edu/offices/it/services/researchcomputing/using/software/python/
# - Clusters: https://www.wm.edu/offices/it/services/researchcomputing/oldjunk/hw/nodes/
# - File Systems: https://www.wm.edu/offices/it/services/researchcomputing/using/filesandfilesystems/
# - Modules: https://www.wm.edu/offices/it/services/researchcomputing/using/modules/
# - Access: https://www.wm.edu/offices/it/services/researchcomputing/using/connecting/
# - Troubleshooting: https://www.wm.edu/offices/it/services/researchcomputing/using/troubleshooting/
#
# Run: bash this_script.sh
# =============================================================================

# Color codes for readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to display a formatted header
display_header() {
    echo -e "${PURPLE}=================================================================${NC}"
    echo -e "${PURPLE} $1 ${NC}"
    echo -e "${PURPLE}=================================================================${NC}"
    echo
}

# Function to pause and wait for user confirmation
wait_for_user() {
    read -p "Press Enter to continue to the next step..." </dev/tty
}

# Start the guide
display_header "Introduction"
echo -e "Welcome to the W&M HPC Walkthrough for your Neural Network Project (e.g., polar-bear)."
echo -e "This guide merges all provided documents and official W&M HPC info, updated for 2025."
echo -e "Follow steps in order. Copy-paste/execute commands in ANOTHER terminal."
echo -e "${YELLOW}Important:${NC} Use Kuro cluster (AMD Zen 4, 64 cores/node, up to 980 GB mem, Rocky 9.2) for training."
echo -e "Your shell is likely tcsh (check with 'echo \$0'). If bash, adjust as noted."
echo -e "Contact hpc-help@wm.edu for help."
echo
wait_for_user

# Step 1: Account Setup and Prerequisites
display_header "Step 1: Obtain HPC Account and Check Prerequisites"
echo -e "1. Request/renew HPC access: https://www.wm.edu/offices/it/services/researchcomputing/acctreq/"
echo -e "   - Accounts expire; renew via form if needed."
echo -e "2. Learn Unix/Linux basics: https://www.lynda.com/Developer-Network-Administration-tutorials/Understanding-SSH/189066-2.html?org=wm.edu"
echo -e "3. Install SSH client (OpenSSH on Linux/Mac, PuTTY on Windows)."
echo -e "4. For off-campus: Use VPN[](https://www.wm.edu/offices/it/services/network/vpn/) or bastion (code.wm.edu/IT/bastion-host-instructions)."
echo -e "5. Username: Typically W&M ID (e.g., ecsampson)."
echo -e "${GREEN}In Terminal 2:${NC} Test SSH: ssh <username>@kuro.sciclone.wm.edu"
echo -e "If issues: See https://www.wm.edu/offices/it/services/researchcomputing/using/connecting/"
echo
wait_for_user

# Step 2: SSH Configuration
display_header "Step 2: Configure SSH Access"
echo -e "Edit ~/.ssh/config for easy access (create if needed)."
echo -e "For off-campus:"
echo "Host kuro.sciclone.wm.edu"
echo "    HostName kuro.sciclone.wm.edu"
echo "    ProxyJump <username>@bastion.wm.edu"
echo "    User <username>"
echo -e "For on-campus/VPN: ssh <username>@kuro.sciclone.wm.edu"
echo -e "With graphics: ssh -Y <username>@kuro.sciclone.wm.edu"
echo -e "${GREEN}In Terminal 2:${NC} nano ~/.ssh/config, paste config, save. Then ssh kuro.sciclone.wm.edu."
echo -e "First login: Check shell (echo \$0), symlinks (ls -la ~/), run restorewmlinks if missing."
echo -e "Troubleshoot: https://www.wm.edu/offices/it/services/researchcomputing/using/connecting/"
echo
wait_for_user

# Step 3: Understand HPC Environment
display_header "Step 3: Understand Clusters and File Systems"
echo -e "Clusters (Sciclone main-campus):"
echo -e "- Kuro: AMD Zen 4, 64 cores/node (ku01-ku47), 980 GB mem, Rocky 9.2, deployed 2024. For MPI/parallel jobs >=64 cores."
echo -e "- Astral: 64 cores/node (as01), 14 TB mem, 8x Nvidia A30 GPUs (24GB), Rocky 9.2, 2024. Shared memory GPU cluster."
echo -e "- Gulf: 16-32 cores/node, 128-512 GB mem, 2x Nvidia A40 GPUs (48GB) on gu03-gu06, Rocky 9.3, 2024. High memory & GPU nodes."
echo -e "- Bora: Xeon Broadwell, 20-128 cores/node, up to 524 GB mem, CentOS 7.3, 2017. Main MPI/parallel cluster."
echo -e "VIMS: Chesapeake, Potomac (Opteron, 12-20 cores/node, 32 GB mem), Pamunkey (64 cores/node), James (20 cores/node)."
echo -e "Use constraints: -C bo (bora), hi (hima, but not listed; use for GPUs), pt (potomac), pm (pamunkey), jm (james)."
echo -e "File Systems:"
echo -e "- Home: /sciclone/home/<user> (small files, backed up weekly, low perf)."
echo -e "- Scratch: /sciclone/scr10/<user>, scr20, pscr (for parallel I/O), scr-lst (for Kuro/post-processing), /local/scr/<user> (node-local, medium-high perf, no backup, purge >90 days inactive)."
echo -e "- Data: /sciclone/data10/<user> (input datasets, weekly backup, not for heavy I/O)."
echo -e "Best Practices: Use scratch for I/O/outputs, data10 for datasets, home for code. Manage usage to avoid impacting others; purge old files."
echo -e "${GREEN}In Terminal 2:${NC} Login to kuro: ssh <user>@kuro.sciclone.wm.edu"
echo -e "Check: echo \$PATH, module list, module avail, df -h, du -hs *"
echo -e "Details: https://www.wm.edu/offices/it/services/researchcomputing/oldjunk/hw/nodes/, https://www.wm.edu/offices/it/services/researchcomputing/using/filesandfilesystems/"
echo
wait_for_user

# Step 4: Set Up Python Environment
display_header "Step 4: Set Up Conda Environment with Miniforge3"
echo -e "Miniforge3 (replaced Anaconda3 Feb 2025; uses conda-forge by default)."
echo -e "1. Load module: module load miniforge3 (latest available)."
echo -e "2. For bash: eval \"\$(conda shell.bash hook)\""
echo -e "3. Create env: conda create -n my_env python=3.12 -y (use conda-forge; in /sciclone/scr10/<user> to avoid quotas)."
echo -e "   - Or: conda create -p /sciclone/scr10/<user>/my_env python=3.12"
echo -e "4. Activate: conda activate my_env"
echo -e "5. Install: conda install numpy pandas torch (conda-forge channel)."
echo -e "   - Pip: conda install pip, python -m pip install opencv-python scikit-image etc. (keeps in env)."
echo -e "6. List: conda list. Export: conda env export > env.yml."
echo -e "7. Deactivate: conda deactivate."
echo -e "Venv alternative: module load python/3.12.7, python -m venv env, source env/bin/activate (bash) or activate.csh (tcsh)."
echo -e "${YELLOW}Note:${NC} Use free channels (conda-forge, bioconda). Test in interactive session. For GPU: Install torch with CUDA support."
echo -e "${GREEN}In Terminal 2:${NC} Run commands above. For your project: Install torch>=2.0.0 (CUDA), torchvision, numpy, opencv-python, wandb, etc."
echo -e "Details: https://www.wm.edu/offices/it/services/researchcomputing/using/software/python/"
echo
wait_for_user

# Step 5: Upload Project Files
display_header "Step 5: Upload Project Files"
echo -e "Project: polar-bear (Python scripts, dataset subfolders, reference .pt subfolders)."
echo -e "Use data10 for datasets/reference, scr-lst/scr10 for outputs/temp (fast I/O)."
echo -e "1. Create dirs: mkdir -p /sciclone/data10/<user>/polar-bear/{dataset,reference} /sciclone/scr-lst/<user>/polar-bear/{outputs,logs}"
echo -e "2. From local: rsync -avz --progress /path/to/local/project/ <user>@kuro.sciclone.wm.edu:/sciclone/data10/<user>/polar-bear/ (excludes .git, etc.)"
echo -e "   - Dataset: rsync -avz --progress /path/to/dataset/ <user>@kuro:/sciclone/data10/<user>/polar-bear/dataset/ (handles subfolders recursively)."
echo -e "   - Reference: rsync -avz --progress /path/to/reference/ <user>@kuro:/sciclone/data10/<user>/polar-bear/reference/ (handles .pt in subfolders)."
echo -e "3. Verify: ssh to HPC, ls -l /sciclone/data10/<user>/polar-bear/, du -hs dataset reference."
echo -e "${YELLOW}Note:${NC} rsync -a preserves structure. For large data, consider Globus[](https://www.wm.edu/offices/it/services/researchcomputing/using/filesandfilesystems/xfers/globus/)."
echo -e "Best Practices: Absolute paths in scripts, clean >90 days old (find /sciclone/scr* -user <user> -mtime +90)."
echo -e "Details: https://www.wm.edu/offices/it/services/researchcomputing/using/filesandfilesystems/"
echo
wait_for_user

# Step 6: Configure Project (config.yaml)
display_header "Step 6: Configure Project with Optimal Settings"
echo -e "Edit config.yaml on HPC for optimal HPC training (use nano config.yaml)."
echo -e "Incorporate these sections for max performance/debugging (from HPC_OPTIMAL_CONFIG_GUIDE.md):"
echo -e "system: device: 'cuda', num_workers: 32, log_level: 'DEBUG', data_path: '/sciclone/data10/<user>/polar-bear/dataset', etc."
echo -e "model: architecture: 'advanced', base_channels: 128, use_se_blocks: true, use_deformable_conv: true, etc."
echo -e "training: num_epochs: 200, batch_size: 128, use_amp: true, augmentation enabled, etc."
echo -e "optimizer: type: 'sam_lookahead', learning_rate: 0.002, etc."
echo -e "loss: weights balanced, focal_loss gamma: 3.0, etc."
echo -e "advanced: use_gradient_checkpointing: true, use_nas: true, use_maml: true, etc."
echo -e "monitoring: track_memory_usage: true, use_tensorboard: true, use_wandb: true, etc."
echo -e "debug: enabled: true, save_intermediate_features: true, check_nan: true, etc."
echo -e "runtime: mode: 'research', etc."
echo -e "optimization: gradient_accumulation_steps: 4, mixed_precision_training: true, use_distributed: true, etc."
echo -e "${YELLOW}Note:${NC} Adjust paths to HPC file systems. For subfolders, ensure data loaders use recursive loading."
echo -e "${GREEN}In Terminal 2:${NC} nano /sciclone/data10/<user>/polar-bear/config.yaml, paste optimal settings."
echo -e "Details: See attached HPC_OPTIMAL_CONFIG_GUIDE.md for full YAML."
echo
wait_for_user

# Step 7: Create Slurm Job Script
display_header "Step 7: Create Slurm Job Script"
echo -e "Customize for training: Kuro (parallel partition), 64 CPUs, 128G mem, 12hr time (max 48hr)."
echo -e "Optimal template (from HPC_QUICK_REFERENCE.md, save as run_training.slurm):"
echo "#!/bin/bash"
echo "#SBATCH --job-name=polar-bear-train"
echo "#SBATCH --partition=parallel"
echo "#SBATCH --nodes=1 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8"  # For multi-GPU
echo "#SBATCH --mem=256G --time=24:00:00"
echo "#SBATCH --output=/sciclone/scr-lst/<user>/polar-bear/logs/%j.out"
echo "#SBATCH --error=/sciclone/scr-lst/<user>/polar-bear/logs/%j.err"
echo "#SBATCH --mail-type=ALL --mail-user=<user>@wm.edu"
echo ""
echo "module load miniforge3 cuda/11.8 cudnn/8.6"
echo "conda activate my_env"
echo "cd /sciclone/data10/<user>/polar-bear"
echo "export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=INFO PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
echo "python -m torch.distributed.launch --nproc_per_node=8 main.py  # Or train.py"
echo "echo 'Job finished at $(date)'"
echo -e "Make executable: chmod +x run_training.slurm"
echo -e "${YELLOW}Note:${NC} For single GPU debug: --gres=gpu:1, --cpus-per-task=16, --mem=64G, --time=4:00:00, no distributed."
echo -e "Test: sbatch --test-only run_training.slurm"
echo -e "Details: https://www.wm.edu/offices/it/services/researchcomputing/using/running_jobs_slurm/"
echo
wait_for_user

# Step 8: Submit and Monitor Job
display_header "Step 8: Submit and Monitor Job"
echo -e "1. Submit: sbatch run_training.slurm (get Job ID)."
echo -e "2. Monitor: squeue -u <user> (status), tail -f logs/<job_id>.out (output)."
echo -e "   - Details: scontrol show job <job_id>"
echo -e "   - Efficiency: seff <job_id> (after completion)."
echo -e "   - Account: sacct -u <user> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS"
echo -e "   - Cancel: scancel <job_id>"
echo -e "Interactive test: salloc --nodes=1 --cpus-per-task=8 --time=1:00:00 --gres=gpu:1, then run python manually."
echo -e "GPU monitoring: watch -n 1 nvidia-smi"
echo -e "Troubleshoot: Check logs, OOM (reduce batch_size), NCCL errors (export NCCL_DEBUG=INFO), slow loading (increase num_workers)."
echo -e "Details: https://www.wm.edu/offices/it/services/researchcomputing/using/running_jobs_slurm/, https://www.wm.edu/offices/it/services/researchcomputing/using/troubleshooting/"
echo
wait_for_user

# Step 9: Retrieve Results
display_header "Step 9: Retrieve Results"
echo -e "After job: rsync -avz --progress <user>@kuro:/sciclone/scr-lst/<user>/polar-bear/outputs/ /local/path/results/"
echo -e "   - Logs: rsync -avz --progress <user>@kuro:/sciclone/scr-lst/<user>/polar-bear/logs/ /local/path/logs/"
echo -e "Clean up: rm -rf /sciclone/scr-lst/<user>/polar-bear/outputs/* (after download; purge >90 days)."
echo -e "For large transfers: Use Globus[](https://www.wm.edu/offices/it/services/researchcomputing/using/filesandfilesystems/xfers/globus/)."
echo -e "Best Practices: Archive in data10 if needed, monitor usage (du -hs)."
echo -e "Details: https://www.wm.edu/offices/it/services/researchcomputing/using/filesandfilesystems/"
echo
wait_for_user

# Step 10: Troubleshooting and Best Practices
display_header "Step 10: Troubleshooting and Best Practices"
echo -e "Common Issues:"
echo -e "- Job won't start: Wrong partition/constraint, resources unavailable (squeue -p parallel)."
echo -e "- Fails immediately: Module not loaded, wrong path (use absolute), permissions (chmod 755 script)."
echo -e "- OOM: Check seff <job_id>, request more --mem, reduce batch_size, enable gradient_checkpointing."
echo -e "- NCCL hangs: export NCCL_DEBUG=INFO, check network."
echo -e "- Files not found: Use absolute paths, cd in script."
echo -e "Best Practices: Test small jobs, use scratch for I/O, git for code, document workflows, clean regularly."
echo -e "Performance Tips: Batch size 128-256 for A100, enable amp/mixed precision, distributed for multi-GPU."
echo -e "Help: hpc-help@wm.edu, https://www.wm.edu/offices/it/services/researchcomputing/using/troubleshooting/"
echo
wait_for_user

# End
display_header "End of Walkthrough"
echo -e "You've completed the guide! Good luck with your polar-bear project."
echo -e "For full training: Use optimal config.yaml (200 epochs, advanced arch, debugging on)."
echo -e "If subfolders not handled, modify train.py with glob '**/*.pt' recursive=True."
echo
