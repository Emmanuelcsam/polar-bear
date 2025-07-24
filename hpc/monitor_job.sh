#!/bin/bash
# Script to monitor intensive training job progress

# Check if job ID is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <job_id>"
    echo "Example: $0 12345"
    exit 1
fi

JOB_ID=$1
LOG_FILE="logs/intensive_${JOB_ID}.out"
ERR_FILE="logs/intensive_${JOB_ID}.err"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Monitoring Job: $JOB_ID"
echo "========================================="

# Function to check job status
check_job_status() {
    status=$(squeue -h -j $JOB_ID -o "%T" 2>/dev/null)
    if [ -z "$status" ]; then
        echo -e "${RED}Job $JOB_ID not found or completed${NC}"
        return 1
    else
        echo -e "Job Status: ${GREEN}$status${NC}"
        return 0
    fi
}

# Function to show job info
show_job_info() {
    echo -e "\n${YELLOW}Job Information:${NC}"
    scontrol show job $JOB_ID | grep -E "JobId|JobName|UserId|Partition|NumNodes|NumCPUs|mem=|RunTime|TimeLimit|NodeList|Reason"
}

# Function to check GPU usage
check_gpu_usage() {
    node=$(squeue -h -j $JOB_ID -o "%N" 2>/dev/null)
    if [ ! -z "$node" ]; then
        echo -e "\n${YELLOW}GPU Usage on $node:${NC}"
        ssh -o ConnectTimeout=5 $node "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv"
    fi
}

# Function to check training progress
check_training_progress() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "\n${YELLOW}Training Progress:${NC}"
        
        # Get current cycle
        current_cycle=$(grep -c "Training Cycle" $LOG_FILE)
        echo "Current Cycle: $current_cycle"
        
        # Get latest epoch
        latest_epoch=$(grep "Epoch" $LOG_FILE | tail -1)
        if [ ! -z "$latest_epoch" ]; then
            echo "Latest: $latest_epoch"
        fi
        
        # Get latest loss
        latest_loss=$(grep -E "Loss:|loss:" $LOG_FILE | tail -1)
        if [ ! -z "$latest_loss" ]; then
            echo "Latest: $latest_loss"
        fi
        
        # Get latest accuracy/similarity
        latest_acc=$(grep -E "Accuracy:|Similarity:" $LOG_FILE | tail -1)
        if [ ! -z "$latest_acc" ]; then
            echo "$latest_acc"
        fi
        
        # Check for checkpoints
        checkpoint_count=$(grep -c "Saving checkpoint" $LOG_FILE)
        echo "Checkpoints saved: $checkpoint_count"
    fi
}

# Function to check for errors
check_errors() {
    if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
        echo -e "\n${YELLOW}Recent Errors:${NC}"
        tail -n 10 $ERR_FILE | while read line; do
            if [[ $line == *"ERROR"* ]] || [[ $line == *"Error"* ]]; then
                echo -e "${RED}$line${NC}"
            elif [[ $line == *"WARNING"* ]] || [[ $line == *"Warning"* ]]; then
                echo -e "${YELLOW}$line${NC}"
            else
                echo "$line"
            fi
        done
    fi
}

# Function to show resource usage
show_resource_usage() {
    echo -e "\n${YELLOW}Resource Usage:${NC}"
    
    # Get efficiency stats if job is completed
    if ! check_job_status > /dev/null 2>&1; then
        seff $JOB_ID 2>/dev/null || echo "Efficiency stats not available yet"
    else
        # Show current usage
        sacct -j $JOB_ID --format=JobID,Elapsed,CPUTime,MaxRSS,MaxVMSize,State -n
    fi
}

# Main monitoring loop
while true; do
    clear
    
    # Check if job still exists
    if ! check_job_status; then
        echo -e "\n${RED}Job has completed or was cancelled${NC}"
        
        # Show final stats
        show_resource_usage
        
        # Check if job was requeued
        new_job=$(squeue -u $USER -h -o "%i %j" | grep "fiber_intensive" | awk '{print $1}')
        if [ ! -z "$new_job" ] && [ "$new_job" != "$JOB_ID" ]; then
            echo -e "\n${GREEN}Job was requeued with new ID: $new_job${NC}"
            echo "Run: $0 $new_job"
        fi
        
        exit 0
    fi
    
    # Show monitoring info
    show_job_info
    check_gpu_usage
    check_training_progress
    check_errors
    show_resource_usage
    
    # Show last few lines of output
    if [ -f "$LOG_FILE" ]; then
        echo -e "\n${YELLOW}Recent Output:${NC}"
        tail -n 5 $LOG_FILE
    fi
    
    echo -e "\n${YELLOW}Press Ctrl+C to stop monitoring${NC}"
    echo "Refreshing in 30 seconds..."
    
    # Wait 30 seconds before refresh
    sleep 30
done