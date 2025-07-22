#!/bin/bash
# ============================================
# Collection of W&M HPC Template Files
# Save each section as a separate file
# ============================================

# ============================================
# FILE: setup_project.sh
# Purpose: Initialize a new project structure
# ============================================
#!/bin/tcsh
# Usage: ./setup_project.sh project_name

if ($#argv != 1) then
    echo "Usage: $0 project_name"
    exit 1
endif

set project = $1
set base = /sciclone/scr10/$USER

# Create project structure
mkdir -p $base/$project/{src,data,results,logs,scripts}
mkdir -p /sciclone/data10/$USER/$project

# Create README
cat << EOF > $base/$project/README.md
# $project

Created: `date`
User: $USER

## Directory Structure:
- src/     : Source code
- data/    : Input data (small files only)
- results/ : Output files
- logs/    : Job logs
- scripts/ : Job submission scripts

## Large Data:
Large datasets should go in: /sciclone/data10/$USER/$project
EOF

echo "Project $project created at $base/$project"

# ============================================
# FILE: monitor_job.sh
# Purpose: Monitor running job in real-time
# ============================================
#!/bin/tcsh
# Usage: ./monitor_job.sh JOBID

if ($#argv != 1) then
    echo "Usage: $0 JOBID"
    exit 1
endif

set jobid = $1

while (1)
    clear
    echo "=== Job Monitor for $jobid ==="
    echo "Time: `date`"
    echo ""
    
    # Show job info
    scontrol show job $jobid | grep -E "JobState|RunTime|NodeList"
    
    # Show queue position if pending
    squeue -j $jobid
    
    # Show efficiency if running
    seff $jobid >& /tmp/seff_$$.tmp
    if ($status == 0) then
        cat /tmp/seff_$$.tmp
    endif
    rm -f /tmp/seff_$$.tmp
    
    sleep 30
end

# ============================================
# FILE: matlab_job.sh
# Purpose: MATLAB batch job template
# ============================================
#!/bin/tcsh
#SBATCH --job-name=matlab_job
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=username@wm.edu

cd /sciclone/scr10/$USER/matlab_project

# Load MATLAB module
module load matlab/R2023b

# Run MATLAB script
matlab -nodisplay -nosplash -r "run('my_analysis.m'); exit"

# ============================================
# FILE: r_job.sh
# Purpose: R batch job template
# ============================================
#!/bin/tcsh
#SBATCH --job-name=r_analysis
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 2:00:00
#SBATCH --mem=32G

cd /sciclone/scr10/$USER/r_project

# Load R module
module load R/4.3.0

# Run R script
Rscript analysis.R

# ============================================
# FILE: gaussian_job.sh
# Purpose: Gaussian 16 quantum chemistry job
# ============================================
#!/bin/tcsh
#SBATCH --job-name=gaussian
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH --mem=64G

cd /sciclone/scr10/$USER/gaussian

# Load Gaussian
module load gaussian/16

# Set scratch directory
setenv GAUSS_SCRDIR /local/scr/$USER

# Run Gaussian
g16 < input.gjf > output.log

# ============================================
# FILE: compile_and_test.sh
# Purpose: Compile and test MPI program
# ============================================
#!/bin/tcsh
# Usage: ./compile_and_test.sh program.c

if ($#argv != 1) then
    echo "Usage: $0 program.c"
    exit 1
endif

set program = $1:r  # Remove extension

# Load modules
module load intel/2024.0
module load intelmpi/2024.0

# Compile
echo "Compiling $1..."
mpicc -O3 -Wall -o $program $1

if ($status != 0) then
    echo "Compilation failed!"
    exit 1
endif

echo "Compilation successful!"
echo ""
echo "To test interactively:"
echo "  salloc -N1 -n4 -t 0:30:00"
echo "  srun -n 4 ./$program"
echo ""
echo "To submit batch job:"
echo "  sbatch mpi_job.sh"

# ============================================
# FILE: cleanup_scratch.sh
# Purpose: Clean old files from scratch
# ============================================
#!/bin/tcsh
# Usage: ./cleanup_scratch.sh [days]

set days = 30
if ($#argv >= 1) then
    set days = $1
endif

echo "Finding files older than $days days in scratch..."
echo ""

foreach dir (/sciclone/scr*/$USER /sciclone/pscr/$USER)
    if (-d $dir) then
        echo "=== $dir ==="
        find $dir -type f -mtime +$days -ls | head -20
        set count = `find $dir -type f -mtime +$days | wc -l`
        echo "Total files older than $days days: $count"
        echo ""
    endif
end

echo "To delete these files, run:"
echo "find /sciclone/scr*/$USER -type f -mtime +$days -delete"

# ============================================
# FILE: job_stats.sh
# Purpose: Show job statistics for user
# ============================================
#!/bin/tcsh
# Usage: ./job_stats.sh [username]

set user = $USER
if ($#argv >= 1) then
    set user = $1
endif

echo "=== Job Statistics for $user ==="
echo "Date: `date`"
echo ""

# Current jobs
echo "Current Jobs:"
squeue -u $user --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

# Recent completed jobs (last 7 days)
echo ""
echo "Recently Completed Jobs (last 7 days):"
sacct -u $user -S `date -d '7 days ago' +%Y-%m-%d` \
    --format=JobID,JobName,Partition,State,Elapsed,ExitCode \
    | head -20

# ============================================
# FILE: interactive_session.sh
# Purpose: Quick interactive session launcher
# ============================================
#!/bin/tcsh
# Usage: ./interactive_session.sh [cores] [hours]

set cores = 8
set hours = 2

if ($#argv >= 1) set cores = $1
if ($#argv >= 2) set hours = $2

echo "Requesting interactive session:"
echo "  Cores: $cores"
echo "  Time:  $hours hours"
echo ""

salloc -N1 -n$cores -t ${hours}:00:00

# ============================================
# FILE: conda_setup.sh
# Purpose: Set up conda environment for project
# ============================================
#!/bin/tcsh
# Usage: ./conda_setup.sh env_name

if ($#argv != 1) then
    echo "Usage: $0 environment_name"
    exit 1
endif

set envname = $1

# Load miniforge
module load miniforge3/24.9.2-0

# Create environment
echo "Creating conda environment: $envname"
conda create -n $envname python=3.11 -y

# Activate and install common packages
conda activate $envname
conda install -y numpy pandas matplotlib scipy jupyter scikit-learn

# Save environment
conda env export > ${envname}_environment.yml

echo ""
echo "Environment $envname created!"
echo "To activate: conda activate $envname"
echo "Environment saved to: ${envname}_environment.yml"

# ============================================
# FILE: transfer_files.sh
# Purpose: Transfer files to/from HPC
# ============================================
#!/bin/bash
# Run this on your LOCAL machine
# Usage: ./transfer_files.sh [to|from] local_path remote_path

if [ $# -ne 3 ]; then
    echo "Usage: $0 [to|from] local_path remote_path"
    echo "Example: $0 to ./mydata/ /sciclone/scr10/username/project/"
    exit 1
fi

direction=$1
local_path=$2
remote_path=$3
username="YOUR_USERNAME"  # Change this!
host="bora.sciclone.wm.edu"

if [ "$direction" = "to" ]; then
    echo "Uploading $local_path to $username@$host:$remote_path"
    rsync -avzP --stats "$local_path" "$username@$host:$remote_path"
elif [ "$direction" = "from" ]; then
    echo "Downloading $username@$host:$remote_path to $local_path"
    rsync -avzP --stats "$username@$host:$remote_path" "$local_path"
else
    echo "Error: direction must be 'to' or 'from'"
    exit 1
fi

# ============================================
# FILE: node_check.sh
# Purpose: Check available nodes and resources
# ============================================
#!/bin/tcsh
# Usage: ./node_check.sh [partition]

echo "=== W&M HPC Node Status ==="
echo "Time: `date`"
echo ""

# Show partition summary
echo "Partition Summary:"
sinfo -s
echo ""

# Show detailed node info for specific partition
if ($#argv >= 1) then
    echo "Nodes in partition $1:"
    sinfo -p $1 -N -o "%.12N %.6t %.4c %.8m %.8d %.6w %.8f"
else
    echo "All nodes:"
    sinfo -N -o "%.12N %.9P %.6t %.4c %.8m %.8d %.6w %.8f" | head -30
endif

echo ""
echo "State codes: idle=available, alloc=in use, down=offline"