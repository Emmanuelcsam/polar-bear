#!/bin/bash
#SBATCH --job-name=fiber_optics_benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Load modules
module purge
module load cuda/11.8
module load cudnn/8.6
module load python/3.10
module load gcc/11.2

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Navigate to project
cd /path/to/your/polar-bear/Network

# Activate environment
source venv/bin/activate
source .env

# Create results directory
RESULTS_DIR="results/benchmark_${SLURM_JOB_ID}"
mkdir -p $RESULTS_DIR

echo "========================================="
echo "Performance Benchmark"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

# Run benchmarks
echo "1. Testing GPU..."
python hpc/test_gpu.py | tee $RESULTS_DIR/gpu_test.log

echo -e "\n2. Running inference benchmark..."
python -c "
import sys
sys.path.append('.')
from core.main import UnifiedFiberOpticsSystem
import time
import torch

system = UnifiedFiberOpticsSystem(mode='production')
system.network.eval()

# Warm-up
for _ in range(10):
    dummy = torch.randn(1, 3, 1024, 1024).cuda()
    with torch.no_grad():
        _ = system.network(dummy)

# Benchmark
batch_sizes = [1, 4, 8, 16, 32]
for bs in batch_sizes:
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        dummy = torch.randn(bs, 3, 1024, 1024).cuda()
        with torch.no_grad():
            _ = system.network(dummy)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    throughput = (bs * 100) / elapsed
    
    print(f'Batch size {bs}: {throughput:.2f} images/sec')
" | tee $RESULTS_DIR/inference_benchmark.log

echo -e "\n3. Memory profiling..."
python -c "
import torch
import sys
sys.path.append('.')
from core.main import UnifiedFiberOpticsSystem

# Test memory usage
system = UnifiedFiberOpticsSystem(mode='production')
system.network.eval()

print('Model parameters:', sum(p.numel() for p in system.network.parameters()))
print('Model size (MB):', sum(p.numel() * p.element_size() for p in system.network.parameters()) / 1024 / 1024)

# Test different batch sizes
for bs in [1, 2, 4, 8, 16, 32]:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        dummy = torch.randn(bs, 3, 1024, 1024).cuda()
        with torch.no_grad():
            _ = system.network(dummy)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f'Batch size {bs}: Peak memory {peak_memory:.2f} GB')
    except RuntimeError as e:
        print(f'Batch size {bs}: Out of memory')
        break
" | tee $RESULTS_DIR/memory_profile.log

echo -e "\nBenchmark completed. Results saved to $RESULTS_DIR"