#!/bin/tcsh
#SBATCH --job-name=check_resources
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH -o check-%j.out

echo "Checking available resources..."
echo "================================"
echo "Hostname: `hostname`"
echo "Date: `date`"
echo ""

echo "SLURM Info:"
echo "-----------"
sinfo -o "%20N %10c %10m %25f %10G"

echo ""
echo "Partition Info:"
echo "---------------"
sinfo -s

echo ""
echo "Available Constraints:"
echo "---------------------"
sinfo -o "%20N %30f"

echo ""
echo "GPU Info:"
echo "---------"
sinfo -o "%20N %10G %10g %20b"