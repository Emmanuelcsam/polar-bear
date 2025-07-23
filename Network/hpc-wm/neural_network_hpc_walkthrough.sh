#!/bin/bash

# =============================================================================
# William & Mary HPC Walkthrough for Neural Network Training
# =============================================================================
# This script provides a detailed, step-by-step guide for uploading and
# running a neural network project on the W&M HPC cluster. It is specifically
# designed for projects with Python scripts, a directory of images, and a
# directory of reference tensors.
#
# Instructions:
# 1. Make this script executable: chmod +x neural_network_hpc_walkthrough.sh
# 2. Run the script: ./neural_network_hpc_walkthrough.sh
# 3. Follow the on-screen prompts and instructions.
#
# For more information on the W&M HPC, visit:
# https://www.wm.edu/it/rc
# =============================================================================

# --- Configuration ---
# These variables can be modified to match your project's specific needs.

# Your W&M HPC username
HPC_USERNAME="your_username"

# The HPC cluster you want to use (e.g., "kuro.sciclone.wm.edu")
HPC_HOST="kuro.sciclone.wm.edu"

# The name of your project. This will be used to create directories on the HPC.
PROJECT_NAME="my_neural_network"

# The local path to your Python scripts and project files.
# Assumes the script is run from the project's root directory.
LOCAL_PROJECT_DIR=$(pwd)

# The local path to your directory of images.
LOCAL_IMAGE_DIR="${LOCAL_PROJECT_DIR}/images"

# The local path to your directory of reference tensors (.pt files).
LOCAL_REFERENCE_DIR="${LOCAL_PROJECT_DIR}/reference_tensors"

# The remote directory on the HPC where your project will be stored.
# It's recommended to use the /sciclone/data10 filesystem for large datasets.
REMOTE_PROJECT_DIR="/sciclone/data10/${HPC_USERNAME}/${PROJECT_NAME}"

# The remote directory on the HPC for job outputs and temporary files.
# The /sciclone/scr-lst filesystem is ideal for this.
REMOTE_SCRATCH_DIR="/sciclone/scr-lst/${HPC_USERNAME}/${PROJECT_NAME}"

# --- Script ---

# Color codes for better readability
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
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}=================================================================${NC}"
    echo
}

# Function to pause and wait for user confirmation
wait_for_user() {
    read -p "Press Enter to continue to the next step..."
}

# --- Walkthrough Steps ---

# Step 1: Introduction and Configuration
display_header "Step 1: Introduction and Configuration"
echo -e "Welcome to the W&M HPC walkthrough for neural network training."
echo -e "This script will guide you through the process of uploading your"
echo -e "project and running a training job on the HPC cluster."
echo
echo -e "${YELLOW}Please review the configuration variables at the beginning of this script"
echo -e "and ensure they are set correctly for your project.${NC}"
echo
echo -e "Your current configuration is:"
echo -e "--------------------------------"
echo -e "HPC Username:      ${CYAN}${HPC_USERNAME}${NC}"
echo -e "HPC Host:          ${CYAN}${HPC_HOST}${NC}"
echo -e "Project Name:      ${CYAN}${PROJECT_NAME}${NC}"
echo -e "Local Project Dir: ${CYAN}${LOCAL_PROJECT_DIR}${NC}"
echo -e "Local Image Dir:   ${CYAN}${LOCAL_IMAGE_DIR}${NC}"
echo -e "Local Ref. Dir:    ${CYAN}${LOCAL_REFERENCE_DIR}${NC}"
echo -e "Remote Project Dir:${CYAN}${REMOTE_PROJECT_DIR}${NC}"
echo -e "Remote Scratch Dir:${CYAN}${REMOTE_SCRATCH_DIR}${NC}"
echo -e "--------------------------------"
echo
echo -e "${RED}Important:${NC} Before proceeding, make sure you have a W&M HPC account"
echo -e "and can connect to the cluster via SSH."
echo
wait_for_user

# Step 2: Local File Check
display_header "Step 2: Checking Local Files and Directories"
echo -e "This step will check for the necessary local files and directories."
echo
if [ -d "${LOCAL_IMAGE_DIR}" ]; then
    echo -e "${GREEN}✓ Image directory found at: ${LOCAL_IMAGE_DIR}${NC}"
else
    echo -e "${RED}✗ Image directory not found at: ${LOCAL_IMAGE_DIR}${NC}"
    echo -e "Please make sure your image directory exists and the path is correct."
    exit 1
fi
if [ -d "${LOCAL_REFERENCE_DIR}" ]; then
    echo -e "${GREEN}✓ Reference tensor directory found at: ${LOCAL_REFERENCE_DIR}${NC}"
else
    echo -e "${RED}✗ Reference tensor directory not found at: ${LOCAL_REFERENCE_DIR}${NC}"
    echo -e "Please make sure your reference tensor directory exists and the path is correct."
    exit 1
fi
echo
echo -e "Local file check complete."
echo
wait_for_user

# Step 3: Setting up Remote Directories
display_header "Step 3: Setting up Remote Directories on the HPC"
echo -e "This step will connect to the HPC via SSH and create the necessary"
echo -e "directories for your project."
echo
echo -e "The following command will be executed:"
echo -e "${CYAN}ssh ${HPC_USERNAME}@${HPC_HOST} \"mkdir -p ${REMOTE_PROJECT_DIR} ${REMOTE_SCRATCH_DIR}\"${NC}"
echo
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh ${HPC_USERNAME}@${HPC_HOST} "mkdir -p ${REMOTE_PROJECT_DIR} ${REMOTE_SCRATCH_DIR}"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Remote directories created successfully.${NC}"
    else
        echo -e "${RED}✗ Failed to create remote directories.${NC}"
        echo -e "Please check your SSH connection and permissions."
        exit 1
    fi
else
    echo -e "Skipping remote directory setup."
fi
echo
wait_for_user

# Step 4: Uploading Your Project Files
display_header "Step 4: Uploading Your Project Files to the HPC"
echo -e "This step will upload your Python scripts, image data, and reference"
echo -e "tensors to the HPC. We will use 'rsync' for efficient file transfer."
echo
echo -e "${YELLOW}Note:${NC} This may take a long time depending on the size of your data."
echo
echo -e "The following commands will be executed:"
echo -e "1. Upload Python scripts and other project files:"
echo -e "   ${CYAN}rsync -avz --exclude='images' --exclude='reference_tensors' ${LOCAL_PROJECT_DIR}/ ${HPC_USERNAME}@${HPC_HOST}:${REMOTE_PROJECT_DIR}/${NC}"
echo -e "2. Upload image directory:"
echo -e "   ${CYAN}rsync -avz ${LOCAL_IMAGE_DIR}/ ${HPC_USERNAME}@${HPC_HOST}:${REMOTE_PROJECT_DIR}/images/${NC}"
echo -e "3. Upload reference tensor directory:"
echo -e "   ${CYAN}rsync -avz ${LOCAL_REFERENCE_DIR}/ ${HPC_USERNAME}@${HPC_HOST}:${REMOTE_PROJECT_DIR}/reference_tensors/${NC}"
echo
read -p "Do you want to proceed with the upload? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "Uploading project files..."
    rsync -avz --exclude='images' --exclude='reference_tensors' ${LOCAL_PROJECT_DIR}/ ${HPC_USERNAME}@${HPC_HOST}:${REMOTE_PROJECT_DIR}/
    echo -e "Uploading image directory..."
    rsync -avz ${LOCAL_IMAGE_DIR}/ ${HPC_USERNAME}@${HPC_HOST}:${REMOTE_PROJECT_DIR}/images/
    echo -e "Uploading reference tensor directory..."
    rsync -avz ${LOCAL_REFERENCE_DIR}/ ${HPC_USERNAME}@${HPC_HOST}:${REMOTE_PROJECT_DIR}/reference_tensors/
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Project files uploaded successfully.${NC}"
    else
        echo -e "${RED}✗ Failed to upload project files.${NC}"
        echo -e "Please check your SSH connection and file paths."
        exit 1
    fi
else
    echo -e "Skipping file upload."
fi
echo
wait_for_user

# Step 5: Creating the Slurm Job Script
display_header "Step 5: Creating the Slurm Job Script"
echo -e "This step will create a Slurm job script named 'run_training.slurm'"
echo -e "in your local project directory. This script is a template that you"
echo -e "can customize for your specific training needs."
echo
# Create the Slurm job script
cat > run_training.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=${PROJECT_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=${REMOTE_SCRATCH_DIR}/logs/job_%j.out
#SBATCH --error=${REMOTE_SCRATCH_DIR}/logs/job_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${HPC_USERNAME}@wm.edu

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working Directory: $(pwd)"

# Load necessary modules (e.g., for Python, CUDA)
module load miniforge3
module load cuda

# Activate your conda environment
# Replace 'my_env' with the name of your conda environment
source activate my_env

# --- Training ---
echo "Starting training..."
python ${REMOTE_PROJECT_DIR}/train.py \
    --data_dir ${REMOTE_PROJECT_DIR}/images \
    --ref_dir ${REMOTE_PROJECT_DIR}/reference_tensors \
    --output_dir ${REMOTE_SCRATCH_DIR}/output \
    --epochs 100 \
    --batch_size 32

echo "Job finished at $(date)"
EOF
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Slurm job script 'run_training.slurm' created successfully.${NC}"
    echo
    echo -e "${YELLOW}Important:${NC} Please review the 'run_training.slurm' script and"
    echo -e "customize it as needed. You may need to change:"
    echo -e "- The number of CPUs, memory, and walltime"
    echo -e "- The name of your conda environment"
    echo -e "- The path to your main Python training script"
    echo -e "- The command-line arguments for your training script"
else
    echo -e "${RED}✗ Failed to create Slurm job script.${NC}"
    exit 1
fi
echo
wait_for_user

# Step 6: Submitting the Job
display_header "Step 6: Submitting the Job to the HPC"
echo -e "This step will upload the Slurm job script and submit it to the"
echo -e "HPC queue using the 'sbatch' command."
echo
echo -e "The following commands will be executed:"
echo -e "1. Upload the Slurm script:"
echo -e "   ${CYAN}scp run_training.slurm ${HPC_USERNAME}@${HPC_HOST}:${REMOTE_PROJECT_DIR}/${NC}"
echo -e "2. Submit the job:"
echo -e "   ${CYAN}ssh ${HPC_USERNAME}@${HPC_HOST} \"sbatch ${REMOTE_PROJECT_DIR}/run_training.slurm\"${NC}"
echo
read -p "Do you want to proceed with submitting the job? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "Uploading Slurm script..."
    scp run_training.slurm ${HPC_USERNAME}@${HPC_HOST}:${REMOTE_PROJECT_DIR}/
    echo -e "Submitting job..."
    ssh ${HPC_USERNAME}@${HPC_HOST} "sbatch ${REMOTE_PROJECT_DIR}/run_training.slurm"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Job submitted successfully.${NC}"
    else
        echo -e "${RED}✗ Failed to submit job.${NC}"
        echo -e "Please check the Slurm script and your HPC account status."
        exit 1
    fi
else
    echo -e "Skipping job submission."
fi
echo
wait_for_user

# Step 7: Monitoring the Job
display_header "Step 7: Monitoring the Job"
echo -e "Once your job is submitted, you can monitor its status using the"
echo -e "following commands on the HPC:"
echo
echo -e "To check the queue and see your job's status:"
echo -e "${CYAN}squeue -u ${HPC_USERNAME}${NC}"
echo
echo -e "To view the output of your job in real-time:"
echo -e "${CYAN}tail -f ${REMOTE_SCRATCH_DIR}/logs/job_<job_id>.out${NC}"
echo
echo -e "To view any errors from your job:"
echo -e "${CYAN}tail -f ${REMOTE_SCRATCH_DIR}/logs/job_<job_id>.err${NC}"
echo
echo -e "To get detailed information about your job:"
echo -e "${CYAN}scontrol show job <job_id>${NC}"
echo
echo -e "To cancel your job:"
echo -e "${CYAN}scancel <job_id>${NC}"
echo
wait_for_user

# Step 8: Retrieving Your Results
display_header "Step 8: Retrieving Your Results"
echo -e "After your job is complete, your results will be in the"
echo -e "'${REMOTE_SCRATCH_DIR}/output' directory on the HPC."
echo
echo -e "To download your results to your local machine, you can use 'rsync':"
echo -e "${CYAN}rsync -avz ${HPC_USERNAME}@${HPC_HOST}:${REMOTE_SCRATCH_DIR}/output/ ${LOCAL_PROJECT_DIR}/results/${NC}"
echo
echo -e "Make sure you have a 'results' directory in your local project folder."
echo
read -p "Do you want to create a 'results' directory locally? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p results
    echo -e "${GREEN}✓ 'results' directory created.${NC}"
else
    echo -e "Skipping directory creation."
fi
echo
wait_for_user

# --- End of Walkthrough ---
display_header "End of Walkthrough"
echo -e "You have completed the W&M HPC walkthrough for neural network training."
echo -e "If you have any questions or need further assistance, please contact"
echo -e "the W&M IT High-Performance Computing team."
echo
echo -e "Good luck with your research!"
echo
