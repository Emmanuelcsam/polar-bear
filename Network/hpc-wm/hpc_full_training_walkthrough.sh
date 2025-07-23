#!/bin/bash

# =============================================================================
# Fiber Optics Neural Network - HPC Full Training Walkthrough
# =============================================================================
# This script provides a complete walkthrough for deploying and running
# the fiber optics neural network on W&M HPC for 12-hour training sessions
# with full dataset processing.
# =============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration variables (modify these as needed)
HPC_USERNAME="${HPC_USERNAME:-ecsampson}"
HPC_HOST="${HPC_HOST:-kuro.sciclone.wm.edu}"
LOCAL_PROJECT_DIR="$(pwd)"
REMOTE_PROJECT_DIR="/sciclone/data10/$HPC_USERNAME/fiber_optics_nn"
REMOTE_SCRATCH_DIR="/sciclone/scr-lst/$HPC_USERNAME/fiber_optics"
DATASET_LOCAL_DIR="$LOCAL_PROJECT_DIR/dataset"
REFERENCE_LOCAL_DIR="$LOCAL_PROJECT_DIR/reference"
WALLTIME="12:00:00"
MEMORY="128G"
CPUS="64"
PARTITION="parallel"

# Function to display header
display_header() {
    clear
    echo -e "${PURPLE}=================================================${NC}"
    echo -e "${PURPLE}  Fiber Optics Neural Network HPC Deployment    ${NC}"
    echo -e "${PURPLE}  Full Dataset 12-Hour Training Walkthrough      ${NC}"
    echo -e "${PURPLE}=================================================${NC}"
    echo
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${CYAN}Checking prerequisites...${NC}"
    
    # Check SSH key
    if [ ! -f "$HOME/.ssh/id_rsa" ] && [ ! -f "$HOME/.ssh/id_ed25519" ]; then
        echo -e "${RED}✗ SSH key not found${NC}"
        echo "  Please generate an SSH key: ssh-keygen -t ed25519"
        exit 1
    else
        echo -e "${GREEN}✓ SSH key found${NC}"
    fi
    
    # Check local files
    if [ ! -f "config.yaml" ]; then
        echo -e "${RED}✗ config.yaml not found${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ config.yaml found${NC}"
    fi
    
    if [ ! -f "main.py" ]; then
        echo -e "${RED}✗ main.py not found${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ main.py found${NC}"
    fi
    
    # Check dataset
    if [ ! -d "$DATASET_LOCAL_DIR" ]; then
        echo -e "${RED}✗ Dataset directory not found${NC}"
        exit 1
    else
        DATASET_SIZE=$(du -sh "$DATASET_LOCAL_DIR" 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓ Dataset found (Size: $DATASET_SIZE)${NC}"
    fi
    
    # Check reference
    if [ ! -d "$REFERENCE_LOCAL_DIR" ]; then
        echo -e "${YELLOW}⚠ Reference directory not found (optional)${NC}"
    else
        echo -e "${GREEN}✓ Reference directory found${NC}"
    fi
    
    echo
}

# Function to create requirements file
create_requirements() {
    echo -e "${CYAN}Creating requirements.txt...${NC}"
    cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.7.0
einops>=0.6.0
matplotlib>=3.6.0
pyyaml>=6.0
scikit-learn>=1.2.0
tqdm>=4.65.0
wandb>=0.15.0
psutil>=5.9.0
h5py>=3.8.0
pillow>=9.4.0
scipy>=1.10.0
pandas>=1.5.0
EOF
    echo -e "${GREEN}✓ requirements.txt created${NC}"
    echo
}

# Function to create training script with checkpointing
create_training_script() {
    echo -e "${CYAN}Creating enhanced training script...${NC}"
    cat > train_with_checkpoint.py << 'EOF'
#!/usr/bin/env python3
"""Enhanced training script with checkpointing for long runs"""

import os
import sys
import signal
import argparse
import torch
import logging
import time
from pathlib import Path
from datetime import datetime
import psutil

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import create_parser, run_fiber_optics_network
from logger import FiberOpticsLogger

class CheckpointTrainer:
    def __init__(self, args):
        self.args = args
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = FiberOpticsLogger().logger
        self.start_time = time.time()
        self.checkpoint_interval = 3600  # Save every hour
        self.last_checkpoint = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle termination signals gracefully"""
        self.logger.info(f"Received signal {signum}, saving checkpoint...")
        self.save_checkpoint("emergency")
        sys.exit(0)
        
    def save_checkpoint(self, tag="regular"):
        """Save checkpoint with metadata"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        # Get current training state (you'll need to modify main.py to expose this)
        state = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
            'tag': tag,
            'args': vars(self.args),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024 / 1024,  # GB
            'cpu_percent': psutil.cpu_percent(interval=1),
        }
        
        torch.save(state, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Keep only last 5 regular checkpoints
        if tag == "regular":
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_regular_*.pt"))
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
                
    def find_latest_checkpoint(self):
        """Find the most recent checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            return checkpoints[-1]
        return None
        
    def run(self):
        """Run training with periodic checkpointing"""
        # Check for existing checkpoint
        if self.args.resume_latest:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                self.logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                # Load checkpoint and update args as needed
                checkpoint = torch.load(latest_checkpoint)
                self.start_time -= checkpoint['elapsed_time']
                
        # Modify args for long training
        self.args.max_epochs = 1000  # Set high, will be limited by time
        self.args.checkpoint_interval = self.checkpoint_interval
        
        # Run main training loop with checkpoint callback
        def checkpoint_callback():
            current_time = time.time()
            if current_time - self.last_checkpoint > self.checkpoint_interval:
                self.save_checkpoint("regular")
                self.last_checkpoint = current_time
                
        # You'll need to modify main.py to accept and call this callback
        self.args.checkpoint_callback = checkpoint_callback
        
        # Run the main training
        run_fiber_optics_network(self.args)
        
        # Save final checkpoint
        self.save_checkpoint("final")

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument('--resume-latest', action='store_true', 
                       help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint-interval', type=int, default=3600,
                       help='Checkpoint interval in seconds')
    args = parser.parse_args()
    
    trainer = CheckpointTrainer(args)
    trainer.run()
EOF
    echo -e "${GREEN}✓ Enhanced training script created${NC}"
    echo
}

# Function to create resource monitor
create_resource_monitor() {
    echo -e "${CYAN}Creating resource monitoring script...${NC}"
    cat > monitor_resources.py << 'EOF'
#!/usr/bin/env python3
"""Resource monitoring for HPC training runs"""

import psutil
import time
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/resource_monitor.log'),
        logging.StreamHandler()
    ]
)

class ResourceMonitor:
    def __init__(self, interval=300):  # 5 minute intervals
        self.interval = interval
        self.stats_file = Path('logs/resource_stats.json')
        self.stats = []
        
    def collect_stats(self):
        """Collect current resource statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'memory': {
                'total': psutil.virtual_memory().total / (1024**3),
                'used': psutil.virtual_memory().used / (1024**3),
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'read_bytes': psutil.disk_io_counters().read_bytes / (1024**3),
                'write_bytes': psutil.disk_io_counters().write_bytes / (1024**3)
            },
            'process': {
                'memory_gb': psutil.Process().memory_info().rss / (1024**3),
                'num_threads': psutil.Process().num_threads()
            }
        }
        
        return stats
        
    def run(self):
        """Run continuous monitoring"""
        logging.info("Starting resource monitoring...")
        
        try:
            while True:
                stats = self.collect_stats()
                self.stats.append(stats)
                
                # Log current stats
                logging.info(f"CPU: {stats['cpu']['percent']}%, "
                           f"Memory: {stats['memory']['percent']}% "
                           f"({stats['memory']['used']:.1f}/{stats['memory']['total']:.1f} GB)")
                
                # Save stats to file
                with open(self.stats_file, 'w') as f:
                    json.dump(self.stats, f, indent=2)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")
        except Exception as e:
            logging.error(f"Monitoring error: {e}")

if __name__ == "__main__":
    monitor = ResourceMonitor()
    monitor.run()
EOF
    echo -e "${GREEN}✓ Resource monitoring script created${NC}"
    echo
}

# Function to create SLURM job script
create_slurm_script() {
    echo -e "${CYAN}Creating SLURM job script...${NC}"
    cat > train_full_dataset.slurm << EOF
#!/bin/bash
#SBATCH --job-name=fiber_optics_full
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --time=$WALLTIME
#SBATCH --mem=$MEMORY
#SBATCH --exclusive
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$HPC_USERNAME@wm.edu

echo "=========================================="
echo "Job started on: \$(date)"
echo "Node: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
echo "=========================================="

# Load environment modules
module load miniforge3/24.1.2
module list

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate fiber_optics_env || {
    echo "Creating new conda environment..."
    conda create -n fiber_optics_env python=3.10 -y
    conda activate fiber_optics_env
}

# Install requirements
pip install -r requirements.txt

# Set environment variables for optimal performance
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Use local scratch for temporary files
export TMPDIR=/local/scr/\$USER
mkdir -p \$TMPDIR

# Log system information
echo "CPU Info:"
lscpu | grep -E "Model name|Socket|Core|Thread"
echo
echo "Memory Info:"
free -h
echo
echo "GPU Info (if available):"
nvidia-smi 2>/dev/null || echo "No GPU available"
echo

# Create necessary directories
mkdir -p logs checkpoints statistics

# Start resource monitoring in background
python monitor_resources.py &
MONITOR_PID=\$!

# Main training command with all optimizations
echo "Starting training at: \$(date)"
python train_with_checkpoint.py \\
    --mode automatic \\
    --data-dir "$REMOTE_PROJECT_DIR/dataset" \\
    --reference-dir "$REMOTE_PROJECT_DIR/reference" \\
    --output-dir "$REMOTE_SCRATCH_DIR/outputs" \\
    --batch-size 32 \\
    --num-workers \$SLURM_CPUS_PER_TASK \\
    --device cuda \\
    --mixed-precision \\
    --gradient-accumulation 4 \\
    --checkpoint-interval 3600 \\
    --resume-latest \\
    --enable-wandb \\
    --wandb-project fiber_optics_hpc \\
    --save-best \\
    --save-last

# Check exit status
TRAIN_EXIT_CODE=\$?

# Stop monitoring
kill \$MONITOR_PID 2>/dev/null

# Copy results back to data directory
if [ \$TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
    cp -r "$REMOTE_SCRATCH_DIR/outputs/"* "$REMOTE_PROJECT_DIR/results/"
    cp checkpoints/checkpoint_final_*.pt "$REMOTE_PROJECT_DIR/checkpoints/"
else
    echo "Training failed with exit code: \$TRAIN_EXIT_CODE"
    # Save emergency checkpoint
    cp checkpoints/checkpoint_emergency_*.pt "$REMOTE_PROJECT_DIR/checkpoints/" 2>/dev/null
fi

# Generate summary report
python -c "
import json
from pathlib import Path

stats_file = Path('logs/resource_stats.json')
if stats_file.exists():
    with open(stats_file) as f:
        stats = json.load(f)
    
    if stats:
        avg_cpu = sum(s['cpu']['percent'] for s in stats) / len(stats)
        max_mem = max(s['memory']['used'] for s in stats)
        total_io = stats[-1]['disk']['read_bytes'] + stats[-1]['disk']['write_bytes']
        
        print(f'\\nResource Usage Summary:')
        print(f'Average CPU Usage: {avg_cpu:.1f}%')
        print(f'Peak Memory Usage: {max_mem:.1f} GB')
        print(f'Total I/O: {total_io:.1f} GB')
"

echo "=========================================="
echo "Job finished on: \$(date)"
echo "=========================================="
EOF
    echo -e "${GREEN}✓ SLURM job script created${NC}"
    echo
}

# Function to setup remote directories
setup_remote_dirs() {
    echo -e "${CYAN}Setting up remote directories on HPC...${NC}"
    
    ssh $HPC_USERNAME@$HPC_HOST << EOF
        # Create project directories
        mkdir -p $REMOTE_PROJECT_DIR/{dataset,reference,checkpoints,results,logs}
        mkdir -p $REMOTE_SCRATCH_DIR/{outputs,temp}
        
        # Set proper permissions
        chmod 750 $REMOTE_PROJECT_DIR
        chmod 750 $REMOTE_SCRATCH_DIR
        
        echo "Remote directories created successfully"
        
        # Check available space
        echo
        echo "Storage quotas:"
        echo "Data directory:"
        df -h $REMOTE_PROJECT_DIR | tail -1
        echo
        echo "Scratch directory:"
        df -h $REMOTE_SCRATCH_DIR | tail -1
EOF
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Remote directories set up${NC}"
    else
        echo -e "${RED}✗ Failed to setup remote directories${NC}"
        exit 1
    fi
    echo
}

# Function to upload project files
upload_project() {
    echo -e "${CYAN}Uploading project files to HPC...${NC}"
    echo "This may take a while for large datasets..."
    
    # Create tar archive excluding unnecessary files
    echo "Creating archive..."
    tar --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='logs/*' \
        --exclude='checkpoints/*' \
        --exclude='wandb' \
        --exclude='*.pt' \
        -czf fiber_optics_project.tar.gz \
        *.py *.yaml *.txt *.sh *.slurm
    
    # Upload project files
    echo "Uploading project files..."
    scp fiber_optics_project.tar.gz $HPC_USERNAME@$HPC_HOST:$REMOTE_PROJECT_DIR/
    
    # Extract on remote
    ssh $HPC_USERNAME@$HPC_HOST "cd $REMOTE_PROJECT_DIR && tar -xzf fiber_optics_project.tar.gz && rm fiber_optics_project.tar.gz"
    
    # Upload dataset with progress
    echo "Uploading dataset (this may take a long time)..."
    rsync -avz --progress \
        --exclude='*.log' \
        --exclude='*.tmp' \
        "$DATASET_LOCAL_DIR/" \
        "$HPC_USERNAME@$HPC_HOST:$REMOTE_PROJECT_DIR/dataset/"
    
    # Upload reference if exists
    if [ -d "$REFERENCE_LOCAL_DIR" ]; then
        echo "Uploading reference data..."
        rsync -avz --progress \
            "$REFERENCE_LOCAL_DIR/" \
            "$HPC_USERNAME@$HPC_HOST:$REMOTE_PROJECT_DIR/reference/"
    fi
    
    # Cleanup local archive
    rm -f fiber_optics_project.tar.gz
    
    echo -e "${GREEN}✓ Project uploaded successfully${NC}"
    echo
}

# Function to submit job
submit_job() {
    echo -e "${CYAN}Submitting job to HPC queue...${NC}"
    
    # Submit job and capture job ID
    JOB_OUTPUT=$(ssh $HPC_USERNAME@$HPC_HOST "cd $REMOTE_PROJECT_DIR && sbatch train_full_dataset.slurm")
    JOB_ID=$(echo $JOB_OUTPUT | grep -oE '[0-9]+')
    
    if [ -n "$JOB_ID" ]; then
        echo -e "${GREEN}✓ Job submitted successfully with ID: $JOB_ID${NC}"
        echo
        echo -e "${YELLOW}Useful commands:${NC}"
        echo "  Check job status:  ssh $HPC_USERNAME@$HPC_HOST 'squeue -u $HPC_USERNAME'"
        echo "  View job output:   ssh $HPC_USERNAME@$HPC_HOST 'tail -f $REMOTE_PROJECT_DIR/logs/train_$JOB_ID.out'"
        echo "  Cancel job:        ssh $HPC_USERNAME@$HPC_HOST 'scancel $JOB_ID'"
        echo "  Job efficiency:    ssh $HPC_USERNAME@$HPC_HOST 'seff $JOB_ID'"
    else
        echo -e "${RED}✗ Failed to submit job${NC}"
        exit 1
    fi
    echo
}

# Function to monitor job
monitor_job() {
    if [ -z "$1" ]; then
        echo "Please provide job ID"
        return
    fi
    
    echo -e "${CYAN}Monitoring job $1...${NC}"
    echo "Press Ctrl+C to stop monitoring"
    echo
    
    while true; do
        clear
        echo -e "${PURPLE}=== Job Status ===${NC}"
        ssh $HPC_USERNAME@$HPC_HOST "squeue -j $1 2>/dev/null || echo 'Job not in queue'"
        echo
        echo -e "${PURPLE}=== Latest Output ===${NC}"
        ssh $HPC_USERNAME@$HPC_HOST "tail -20 $REMOTE_PROJECT_DIR/logs/train_$1.out 2>/dev/null || echo 'No output yet'"
        sleep 30
    done
}

# Function to download results
download_results() {
    echo -e "${CYAN}Downloading results from HPC...${NC}"
    
    # Create local results directory
    mkdir -p results checkpoints statistics
    
    # Download results
    rsync -avz --progress \
        "$HPC_USERNAME@$HPC_HOST:$REMOTE_PROJECT_DIR/results/" \
        "results/"
    
    # Download checkpoints
    rsync -avz --progress \
        "$HPC_USERNAME@$HPC_HOST:$REMOTE_PROJECT_DIR/checkpoints/" \
        "checkpoints/"
    
    # Download logs
    rsync -avz --progress \
        "$HPC_USERNAME@$HPC_HOST:$REMOTE_PROJECT_DIR/logs/" \
        "logs/"
    
    echo -e "${GREEN}✓ Results downloaded successfully${NC}"
    echo
}

# Main menu
main_menu() {
    while true; do
        display_header
        echo -e "${YELLOW}Main Menu:${NC}"
        echo "1) Full deployment (steps 1-5)"
        echo "2) Check prerequisites"
        echo "3) Setup remote directories"
        echo "4) Upload project files"
        echo "5) Submit training job"
        echo "6) Monitor job"
        echo "7) Download results"
        echo "8) Advanced options"
        echo "9) Exit"
        echo
        read -p "Select option: " choice
        
        case $choice in
            1)
                display_header
                check_prerequisites
                create_requirements
                create_training_script
                create_resource_monitor
                create_slurm_script
                setup_remote_dirs
                upload_project
                submit_job
                read -p "Press Enter to continue..."
                ;;
            2)
                display_header
                check_prerequisites
                read -p "Press Enter to continue..."
                ;;
            3)
                display_header
                setup_remote_dirs
                read -p "Press Enter to continue..."
                ;;
            4)
                display_header
                upload_project
                read -p "Press Enter to continue..."
                ;;
            5)
                display_header
                submit_job
                read -p "Press Enter to continue..."
                ;;
            6)
                display_header
                read -p "Enter job ID to monitor: " job_id
                monitor_job $job_id
                ;;
            7)
                display_header
                download_results
                read -p "Press Enter to continue..."
                ;;
            8)
                advanced_menu
                ;;
            9)
                echo -e "${GREEN}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                sleep 2
                ;;
        esac
    done
}

# Advanced options menu
advanced_menu() {
    while true; do
        display_header
        echo -e "${YELLOW}Advanced Options:${NC}"
        echo "1) Interactive session on compute node"
        echo "2) Test with small dataset"
        echo "3) Profile code performance"
        echo "4) Setup distributed training"
        echo "5) Clean remote directories"
        echo "6) Back to main menu"
        echo
        read -p "Select option: " choice
        
        case $choice in
            1)
                echo -e "${CYAN}Starting interactive session...${NC}"
                ssh -t $HPC_USERNAME@$HPC_HOST "salloc --partition=$PARTITION --nodes=1 --cpus-per-task=16 --mem=32G --time=2:00:00"
                ;;
            2)
                echo -e "${CYAN}Creating test job script...${NC}"
                # Create smaller test script
                sed 's/--time=12:00:00/--time=1:00:00/g; s/--mem=128G/--mem=32G/g' train_full_dataset.slurm > test_training.slurm
                echo -e "${GREEN}✓ Test script created as test_training.slurm${NC}"
                read -p "Press Enter to continue..."
                ;;
            3)
                echo -e "${CYAN}Profiling information will be collected during next run${NC}"
                echo "Add --profile flag to training command"
                read -p "Press Enter to continue..."
                ;;
            4)
                echo -e "${CYAN}Distributed training setup...${NC}"
                echo "This requires multi-node allocation"
                echo "Contact HPC support for assistance"
                read -p "Press Enter to continue..."
                ;;
            5)
                echo -e "${YELLOW}Warning: This will delete all remote files!${NC}"
                read -p "Are you sure? (yes/no): " confirm
                if [ "$confirm" = "yes" ]; then
                    ssh $HPC_USERNAME@$HPC_HOST "rm -rf $REMOTE_PROJECT_DIR/* $REMOTE_SCRATCH_DIR/*"
                    echo -e "${GREEN}✓ Remote directories cleaned${NC}"
                fi
                read -p "Press Enter to continue..."
                ;;
            6)
                break
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                sleep 2
                ;;
        esac
    done
}

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -u, --username USER  Set HPC username (default: $HPC_USERNAME)"
    echo "  -H, --host HOST      Set HPC host (default: $HPC_HOST)"
    echo "  -t, --time TIME      Set walltime (default: $WALLTIME)"
    echo "  -m, --memory MEM     Set memory allocation (default: $MEMORY)"
    echo "  -c, --cpus CPUS      Set CPU count (default: $CPUS)"
    echo "  --quick              Run full deployment without prompts"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -u|--username)
            HPC_USERNAME="$2"
            shift 2
            ;;
        -H|--host)
            HPC_HOST="$2"
            shift 2
            ;;
        -t|--time)
            WALLTIME="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -c|--cpus)
            CPUS="$2"
            shift 2
            ;;
        --quick)
            # Run full deployment
            check_prerequisites
            create_requirements
            create_training_script
            create_resource_monitor
            create_slurm_script
            setup_remote_dirs
            upload_project
            submit_job
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Start main menu
main_menu