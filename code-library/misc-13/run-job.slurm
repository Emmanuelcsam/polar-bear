#!/bin/tcsh
# ======================================================================
# SLURM SUBMISSION SCRIPT FOR THE W&M BORA SUBCLUSTER 
# ======================================================================

# --- SLURM Resource Directives ---
#SBATCH --job-name=fiber-optic-training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=12:30:00
#SBATCH --mem=100G
#SBATCH -C bo
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ecsampson@wm.edu

# --- Job Environment Setup and Diagnostics ---
echo "========================================================"
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_JOB_NODELIST"
echo "Running on $SLURM_NNODES nodes"
echo "Running on $SLURM_NTASKS total processes"
echo "Working directory is $SLURM_SUBMIT_DIR"
echo "Job started on: `date`"
echo "========================================================"

# --- Software Environment Configuration ---
# Source the standard W&M tcsh environment
source /usr/local/etc/sciclone.cshrc

# Load required modules
echo "Loading modules..."
module purge
module load miniforge3/24.9.2-0
module load cuda/12.3
echo "Modules loaded."
module list

# Activate conda environment
echo "Activating conda environment..."
conda activate /sciclone/scr-lst/ecsampson/polar-bear-env
echo "Conda environment activated."

# --- Install missing Python packages ---
echo "Checking for required Python packages..."
python -c "import skimage" >& /dev/null
if ( $status != 0 ) then
    echo "Installing scikit-image..."
    pip install scikit-image
endif

# --- Application Execution ---
cd $SLURM_SUBMIT_DIR

# --- Extract master node from SLURM_JOB_NODELIST ---
# CRITICAL FIX: Don't rely on scontrol which doesn't exist on compute nodes
# Parse the nodelist manually

set nodelist = "$SLURM_JOB_NODELIST"
echo "Original nodelist: $nodelist"

# Check if nodelist contains brackets (compressed format like "bo[15-16]")
echo "$nodelist" | grep -q '\['
if ( $status == 0 ) then
    # Extract prefix (e.g., "bo" from "bo[15-16]")
    set prefix = `echo "$nodelist" | sed 's/\[.*//'`
    # Extract first number (e.g., "15" from "bo[15-16]")
    set first_num = `echo "$nodelist" | sed 's/.*\[//' | sed 's/-.*//' | sed 's/\].*//'`
    # Construct first node name
    setenv MASTER_ADDR "${prefix}${first_num}"
else
    # Check if nodelist contains comma (expanded format like "bo15,bo16")
    echo "$nodelist" | grep -q ','
    if ( $status == 0 ) then
        # Take the first node from comma-separated list
        setenv MASTER_ADDR `echo "$nodelist" | cut -d',' -f1`
    else
        # Single node case (e.g., "bo15")
        setenv MASTER_ADDR "$nodelist"
    endif
endif

setenv MASTER_PORT 29500

echo "--- Distributed Training Environment ---"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "----------------------------------------"

# Verify MASTER_ADDR is set
if ( "$MASTER_ADDR" == "" ) then
    echo "ERROR: Failed to determine MASTER_ADDR from nodelist"
    exit 1
endif

# --- Launch distributed training ---
echo "Starting training with torchrun..."

# Use direct torchrun without srun since srun is not available
# Set up hostfile for distributed training
set hostfile = "/tmp/hostfile.$SLURM_JOB_ID"
echo "Creating hostfile at $hostfile"

# Parse nodelist to create hostfile
if ( "$SLURM_NNODES" == "1" ) then
    # Single node case
    echo "$MASTER_ADDR slots=$SLURM_NTASKS_PER_NODE" > $hostfile
else
    # Multi-node case - expand nodelist
    # For "bo[15-16]", create entries for bo15 and bo16
    echo "$nodelist" | grep -q '\['
    if ( $status == 0 ) then
        set prefix = `echo "$nodelist" | sed 's/\[.*//'`
        set range = `echo "$nodelist" | sed 's/.*\[//' | sed 's/\].*//'`
        set start = `echo "$range" | cut -d'-' -f1`
        set end = `echo "$range" | cut -d'-' -f2`
        
        @ node_num = $start
        while ( $node_num <= $end )
            echo "${prefix}${node_num} slots=$SLURM_NTASKS_PER_NODE" >> $hostfile
            @ node_num++
        end
    else
        # Already expanded format
        foreach node (`echo "$nodelist" | tr ',' ' '`)
            echo "$node slots=$SLURM_NTASKS_PER_NODE" >> $hostfile
        end
    endif
endif

echo "Hostfile contents:"
cat $hostfile

# Launch torchrun on each node
if ( "$SLURM_NNODES" == "1" ) then
    # Single node - run directly
    torchrun \
        --nproc_per_node=$SLURM_NTASKS_PER_NODE \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        core/main_distributed.py
else
    # Multi-node - use SSH to launch on each node
    echo "Multi-node training detected. Using SSH to launch on each node..."
    
    # Create launch script
    set launch_script = "/tmp/launch_torchrun.$SLURM_JOB_ID.sh"
    cat > $launch_script << 'EOF'
#!/bin/bash
source /usr/local/etc/sciclone.bashrc
module load miniforge3/24.9.2-0
module load cuda/12.3
conda activate /sciclone/scr-lst/ecsampson/polar-bear-env
cd $SLURM_SUBMIT_DIR

NODE_RANK=$1
torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    core/main_distributed.py
EOF
    chmod +x $launch_script
    
    # Launch on each node
    @ rank = 0
    foreach node (`cat $hostfile | awk '{print $1}'`)
        if ( $rank == 0 ) then
            # Master node - run in foreground
            echo "Launching on master node $node (rank $rank)"
            ssh $node $launch_script $rank &
            set master_pid = $!
        else
            # Worker nodes - run in background
            echo "Launching on worker node $node (rank $rank)"
            ssh $node $launch_script $rank &
        endif
        @ rank++
    end
    
    # Wait for master process
    wait $master_pid
    
    # Clean up
    rm -f $launch_script
endif

# Clean up hostfile
rm -f $hostfile

# --- Job Completion ---
echo "========================================================"
echo "Job finished with exit code $? on: `date`"
echo "========================================================"
