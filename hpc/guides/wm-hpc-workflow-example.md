# Complete W&M HPC Workflow Example: From Start to Finish

This guide walks through a complete real-world example of using W&M HPC to run a Python data analysis project.

## Project Overview
We'll analyze a large dataset using parallel processing on the HPC cluster. This example demonstrates:
- Setting up the environment
- Transferring data
- Testing code interactively
- Submitting batch jobs
- Monitoring progress
- Retrieving results

---

## Step 1: Initial Setup (One-time)

### 1.1 Configure SSH access (on your local machine)
```bash
# Create SSH config file
mkdir -p ~/.ssh
cat >> ~/.ssh/config << EOF
Host wm-hpc
    HostName bora.sciclone.wm.edu
    User YOUR_USERNAME
    ProxyJump YOUR_USERNAME@bastion.wm.edu
EOF
```

### 1.2 Connect to HPC
```bash
# Now you can simply use:
ssh wm-hpc

# First time login checklist:
echo $USER                 # Verify username
echo $0                    # Check shell (should be -tcsh)
ls -la ~/                  # Check for symlinks
pwd                        # Should be /sciclone/home/username
```

### 1.3 Set up project structure
```bash
# Create project directories
cd /sciclone/scr10/$USER
mkdir -p my_analysis/{data,src,results,logs,scripts}

# Create data directory for large files
mkdir -p /sciclone/data10/$USER/my_analysis

# Set up convenient alias (add to ~/.tcshrc)
echo 'alias proj="cd /sciclone/scr10/$USER/my_analysis"' >> ~/.tcshrc
source ~/.tcshrc
```

---

## Step 2: Prepare Your Code and Data

### 2.1 Transfer code from local machine
```bash
# On your local machine:
cd ~/my_local_project
scp -r *.py requirements.txt wm-hpc:/sciclone/scr10/YOUR_USERNAME/my_analysis/src/
```

### 2.2 Transfer data files
```bash
# For large data files, use rsync with compression
rsync -avzP --stats data/*.csv wm-hpc:/sciclone/data10/YOUR_USERNAME/my_analysis/
```

### 2.3 Create main analysis script
On HPC, create `/sciclone/scr10/$USER/my_analysis/src/analyze_data.py`:

```python
#!/usr/bin/env python
"""
Main analysis script for HPC processing
"""
import pandas as pd
import numpy as np
import argparse
import logging
import os
from pathlib import Path
import multiprocessing as mp
from datetime import datetime

def setup_logging(log_dir):
    """Configure logging"""
    log_file = log_dir / f"analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_chunk(args):
    """Process a single data chunk"""
    chunk_id, file_path, start_row, end_row = args
    
    # Read chunk
    df = pd.read_csv(file_path, skiprows=range(1, start_row), nrows=end_row-start_row)
    
    # Perform analysis (example: calculate statistics)
    results = {
        'chunk_id': chunk_id,
        'rows': len(df),
        'mean': df.select_dtypes(include=[np.number]).mean().to_dict(),
        'std': df.select_dtypes(include=[np.number]).std().to_dict()
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Parallel data analysis')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--chunks', type=int, default=None, help='Number of chunks')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logger = setup_logging(log_dir)
    logger.info(f"Starting analysis of {args.input_file}")
    
    # Determine number of processes
    n_cores = int(os.environ.get('SLURM_NTASKS', mp.cpu_count()))
    n_chunks = args.chunks or n_cores
    
    # Count total rows
    with open(args.input_file, 'r') as f:
        total_rows = sum(1 for line in f) - 1  # Subtract header
    
    # Create chunks
    chunk_size = total_rows // n_chunks
    chunks = []
    for i in range(n_chunks):
        start_row = i * chunk_size + 1  # +1 to skip header
        end_row = start_row + chunk_size if i < n_chunks - 1 else total_rows + 1
        chunks.append((i, args.input_file, start_row, end_row))
    
    logger.info(f"Processing {total_rows} rows in {n_chunks} chunks using {n_cores} cores")
    
    # Process in parallel
    with mp.Pool(n_cores) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combine results
    combined_df = pd.DataFrame(results)
    output_file = output_dir / f"analysis_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    combined_df.to_csv(output_file, index=False)
    
    logger.info(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
```

---

## Step 3: Set Up Python Environment

### 3.1 Load miniforge and create environment
```bash
proj  # Use our alias to go to project directory

# Load miniforge module
module load miniforge3/24.9.2-0

# Create conda environment
conda create -n analysis python=3.11 -y
conda activate analysis

# Install packages
conda install pandas numpy matplotlib scipy -y
pip install -r src/requirements.txt

# Save environment
conda env export > environment.yml
```

---

## Step 4: Test Interactively

### 4.1 Request interactive session
```bash
# Request 4 cores for 30 minutes
salloc -N1 -n4 -t 0:30:00

# After allocation, check where you are
hostname
pwd
```

### 4.2 Test the script with small data
```bash
# Go to project directory
cd /sciclone/scr10/$USER/my_analysis

# Activate environment
module load miniforge3/24.9.2-0
conda activate analysis

# Create small test file
head -1000 /sciclone/data10/$USER/my_analysis/big_data.csv > test_data.csv

# Run test
python src/analyze_data.py test_data.csv --chunks 4

# Check output
ls -la results/
cat logs/*.log
```

### 4.3 Exit interactive session
```bash
exit  # Returns you to login node
```

---

## Step 5: Create Batch Job Script

Create `/sciclone/scr10/$USER/my_analysis/scripts/run_analysis.sh`:

```bash
#!/bin/tcsh
#SBATCH --job-name=data_analysis
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH -t 4:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@wm.edu
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

# Record job info
echo "Job started on `hostname` at `date`"
echo "Job ID: $SLURM_JOB_ID"
echo "Cores: $SLURM_NTASKS"

# Set up environment
cd /sciclone/scr10/$USER/my_analysis
module load miniforge3/24.9.2-0
conda activate analysis

# Check environment
which python
python --version

# Run analysis on full dataset
set datafile = /sciclone/data10/$USER/my_analysis/big_data.csv

if (! -f $datafile) then
    echo "Error: Data file not found: $datafile"
    exit 1
endif

# Create timestamped output directory
set timestamp = `date +%Y%m%d_%H%M%S`
set outdir = results/run_$timestamp
mkdir -p $outdir

# Run the analysis
echo "Starting analysis at `date`"
python src/analyze_data.py $datafile --output-dir $outdir --chunks 64

# Check if successful
if ($status == 0) then
    echo "Analysis completed successfully at `date`"
    
    # Create summary
    echo "Creating summary report..."
    ls -la $outdir > $outdir/file_list.txt
    
    # Copy results to data directory for safety
    cp -r $outdir /sciclone/data10/$USER/my_analysis/
else
    echo "Analysis failed with exit code $status"
    exit $status
endif

echo "Job completed at `date`"
```

---

## Step 6: Submit and Monitor Job

### 6.1 Submit the job
```bash
cd /sciclone/scr10/$USER/my_analysis
sbatch scripts/run_analysis.sh

# You'll see something like:
# Submitted batch job 12345
```

### 6.2 Monitor job progress
```bash
# Check job status
squeue -u $USER

# Watch status (updates every 2 seconds)
watch -n 2 squeue -u $USER

# Check job details
scontrol show job 12345

# Monitor output file
tail -f logs/slurm-12345.out
```

### 6.3 Check resource usage during run
```bash
# Get node where job is running
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# SSH to that node (e.g., bo15)
ssh bo15

# Check processes
top -u $USER

# Exit back to login node
exit
```

---

## Step 7: Post-Processing and Results

### 7.1 Check job completion
```bash
# After job finishes, check efficiency
seff 12345

# Review logs
cat logs/slurm-12345.out
grep ERROR logs/*.log
```

### 7.2 Verify results
```bash
# List results
ls -la results/run_*/

# Quick validation
cd results/run_*
head analysis_results_*.csv
wc -l *.csv
```

### 7.3 Create visualization script
Create `src/plot_results.py`:

```python
#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_results(csv_file):
    df = pd.read_csv(csv_file)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    df['chunk_id'].plot(kind='bar', ax=ax)
    ax.set_title('Analysis Results by Chunk')
    ax.set_xlabel('Chunk ID')
    ax.set_ylabel('Value')
    
    # Save plot
    output_file = csv_file.replace('.csv', '.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_results(sys.argv[1])
```

### 7.4 Generate plots (interactive session)
```bash
salloc -N1 -n1 -t 0:30:00

module load miniforge3/24.9.2-0
conda activate analysis

cd /sciclone/scr10/$USER/my_analysis
python src/plot_results.py results/run_*/analysis_results_*.csv

exit
```

---

## Step 8: Transfer Results Back

### 8.1 Package results
```bash
cd /sciclone/scr10/$USER/my_analysis
tar czf results_20240206.tar.gz results/ logs/
```

### 8.2 Download to local machine
```bash
# On your local machine:
scp wm-hpc:/sciclone/scr10/YOUR_USERNAME/my_analysis/results_20240206.tar.gz ./
tar xzf results_20240206.tar.gz
```

---

## Step 9: Clean Up

### 9.1 Remove temporary files
```bash
# On HPC:
cd /sciclone/scr10/$USER/my_analysis

# Remove test files
rm -f test_data.csv

# Clean old logs (keep recent ones)
find logs/ -name "*.log" -mtime +7 -delete

# Remove old slurm output files
find logs/ -name "slurm-*.out" -mtime +7 -delete
```

### 9.2 Archive project
```bash
# Create archive of entire project
cd /sciclone/scr10/$USER
tar czf my_analysis_archive.tar.gz my_analysis/

# Move to data directory for long-term storage
mv my_analysis_archive.tar.gz /sciclone/data10/$USER/
```

---

## Summary Checklist

- [ ] Set up SSH configuration
- [ ] Create project directory structure
- [ ] Transfer code and data
- [ ] Set up conda environment
- [ ] Test code interactively
- [ ] Create batch job script
- [ ] Submit job
- [ ] Monitor progress
- [ ] Check results
- [ ] Transfer results back
- [ ] Clean up temporary files
- [ ] Archive project

## Tips for Success

1. **Always test with small data first**
2. **Use scratch filesystems for active I/O**
3. **Monitor job efficiency with `seff`**
4. **Keep logs for debugging**
5. **Document your workflow**
6. **Clean up regularly**

## Common Next Steps

- Scale up to larger datasets
- Optimize code for better performance
- Submit array jobs for parameter sweeps
- Set up automated pipelines
- Share results with collaborators

---

*This workflow can be adapted for different types of analyses and software packages. The key principles remain the same: test small, scale up, monitor progress, and clean up after yourself.*