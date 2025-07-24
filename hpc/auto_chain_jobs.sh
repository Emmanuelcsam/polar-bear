#!/bin/bash
# Automatic job chaining script for continuous training
# This script submits a chain of dependent jobs to maximize HPC usage

# Configuration
TOTAL_JOBS=10  # Number of jobs to chain
TIME_PER_JOB="7-00:00:00"  # 7 days per job
JOB_SCRIPT="hpc/run_intensive.slurm"
LOG_DIR="logs/job_chain"

# Create log directory
mkdir -p $LOG_DIR

echo "========================================="
echo "Automatic Job Chaining Setup"
echo "========================================="
echo "Total jobs to chain: $TOTAL_JOBS"
echo "Time per job: $TIME_PER_JOB"
echo "Total training time: $((TOTAL_JOBS * 7)) days"
echo "========================================="

# Submit first job
echo "Submitting job 1/$TOTAL_JOBS..."
job1=$(sbatch --parsable $JOB_SCRIPT)
echo "Job 1 submitted with ID: $job1"
echo $job1 > $LOG_DIR/job_chain.txt

# Submit remaining jobs with dependencies
prev_job=$job1
for i in $(seq 2 $TOTAL_JOBS); do
    echo "Submitting job $i/$TOTAL_JOBS (depends on job $prev_job)..."
    
    # Submit with dependency on previous job
    job=$(sbatch --parsable --dependency=afterany:$prev_job $JOB_SCRIPT)
    echo "Job $i submitted with ID: $job"
    echo $job >> $LOG_DIR/job_chain.txt
    
    prev_job=$job
    sleep 1  # Small delay to avoid overwhelming scheduler
done

echo ""
echo "========================================="
echo "Job chain submitted successfully!"
echo "========================================="
echo ""
echo "To monitor the job chain:"
echo "  squeue -u $USER"
echo ""
echo "To monitor a specific job:"
echo "  ./hpc/monitor_job.sh <job_id>"
echo ""
echo "Job IDs saved to: $LOG_DIR/job_chain.txt"
echo ""
echo "To cancel all jobs in the chain:"
echo "  cat $LOG_DIR/job_chain.txt | xargs scancel"
echo "========================================="

# Create monitoring script for this chain
cat > $LOG_DIR/monitor_chain.sh << 'EOF'
#!/bin/bash
echo "Job Chain Status:"
echo "================="
while read job_id; do
    status=$(squeue -h -j $job_id -o "%i %T %L" 2>/dev/null)
    if [ ! -z "$status" ]; then
        echo "$status"
    else
        echo "$job_id COMPLETED"
    fi
done < $(dirname $0)/job_chain.txt
EOF

chmod +x $LOG_DIR/monitor_chain.sh
echo ""
echo "Monitor the entire chain with:"
echo "  $LOG_DIR/monitor_chain.sh"