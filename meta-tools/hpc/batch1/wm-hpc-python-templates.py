#!/usr/bin/env python
"""
Collection of Python templates and utilities for W&M HPC
Save each section as a separate .py file
"""

# ============================================
# FILE: parallel_processing.py
# Purpose: Parallel processing with multiprocessing
# ============================================
import multiprocessing as mp
import numpy as np
import time
import os

def process_chunk(args):
    """Process a chunk of data"""
    chunk_id, data_chunk = args
    print(f"Process {os.getpid()} handling chunk {chunk_id}")
    
    # Simulate some computation
    result = np.sum(data_chunk ** 2)
    time.sleep(0.1)
    
    return chunk_id, result

def main():
    # Get number of cores from SLURM or default
    n_cores = int(os.environ.get('SLURM_NTASKS', mp.cpu_count()))
    print(f"Using {n_cores} cores")
    
    # Generate sample data
    data_size = 1000000
    data = np.random.rand(data_size)
    
    # Split data into chunks
    chunk_size = data_size // n_cores
    chunks = [(i, data[i*chunk_size:(i+1)*chunk_size]) 
              for i in range(n_cores)]
    
    # Process in parallel
    start_time = time.time()
    with mp.Pool(n_cores) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combine results
    total = sum(result for _, result in results)
    
    print(f"Total: {total}")
    print(f"Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

# ============================================
# FILE: mpi4py_example.py
# Purpose: MPI parallel processing example
# ============================================
from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"Process {rank} of {size} on {MPI.Get_processor_name()}")
    
    # Master process creates data
    if rank == 0:
        data = np.random.rand(size * 100)
        chunks = np.array_split(data, size)
    else:
        chunks = None
    
    # Scatter data to all processes
    local_data = comm.scatter(chunks, root=0)
    
    # Each process works on its chunk
    local_result = np.sum(local_data ** 2)
    
    # Gather results
    results = comm.gather(local_result, root=0)
    
    if rank == 0:
        total = sum(results)
        print(f"Total sum: {total}")

if __name__ == "__main__":
    main()

# ============================================
# FILE: gpu_computation.py
# Purpose: GPU computation with CuPy
# ============================================
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("CuPy not available, falling back to NumPy")

import time

def matrix_multiplication_benchmark(size=5000):
    """Compare CPU vs GPU matrix multiplication"""
    
    # Create random matrices
    if GPU_AVAILABLE:
        # GPU version
        print(f"Running on GPU...")
        a_gpu = cp.random.rand(size, size)
        b_gpu = cp.random.rand(size, size)
        
        # Warm up
        _ = cp.dot(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start = time.time()
        c_gpu = cp.dot(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start
        
        print(f"GPU time: {gpu_time:.3f} seconds")
        
        # CPU version for comparison
        print(f"Running on CPU...")
        a_cpu = cp.asnumpy(a_gpu)
        b_cpu = cp.asnumpy(b_gpu)
    else:
        print(f"Running on CPU only...")
        a_cpu = cp.random.rand(size, size)
        b_cpu = cp.random.rand(size, size)
    
    start = time.time()
    c_cpu = cp.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start
    
    print(f"CPU time: {cpu_time:.3f} seconds")
    
    if GPU_AVAILABLE:
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")

if __name__ == "__main__":
    matrix_multiplication_benchmark()

# ============================================
# FILE: data_pipeline.py
# Purpose: Efficient data processing pipeline
# ============================================
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import logging

def setup_logging(log_file='pipeline.log'):
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def process_data_file(file_path):
    """Process a single data file"""
    logging.info(f"Processing {file_path}")
    
    # Read data
    df = pd.read_csv(file_path)
    
    # Example processing
    df['processed'] = df.select_dtypes(include=[np.number]).mean(axis=1)
    
    # Save results
    output_path = file_path.parent / 'processed' / f"processed_{file_path.name}"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logging.info(f"Saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Data processing pipeline')
    parser.add_argument('input_dir', help='Input directory')
    parser.add_argument('--pattern', default='*.csv', help='File pattern')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    args = parser.parse_args()
    
    setup_logging()
    
    # Find all files
    input_path = Path(args.input_dir)
    files = list(input_path.glob(args.pattern))
    logging.info(f"Found {len(files)} files to process")
    
    if args.parallel:
        # Parallel processing
        import multiprocessing as mp
        n_cores = int(os.environ.get('SLURM_NTASKS', mp.cpu_count()))
        
        with mp.Pool(n_cores) as pool:
            results = pool.map(process_data_file, files)
    else:
        # Sequential processing
        results = [process_data_file(f) for f in files]
    
    logging.info(f"Processed {len(results)} files")

if __name__ == "__main__":
    main()

# ============================================
# FILE: checkpoint_restart.py
# Purpose: Job with checkpoint/restart capability
# ============================================
import pickle
import signal
import sys
import time
import numpy as np
from pathlib import Path

class CheckpointedJob:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / 'checkpoint.pkl'
        self.state = None
        self.should_exit = False
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, saving checkpoint...")
        self.should_exit = True
    
    def save_checkpoint(self):
        """Save current state"""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self.state, f)
        print(f"Checkpoint saved to {self.checkpoint_file}")
    
    def load_checkpoint(self):
        """Load previous state if exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                self.state = pickle.load(f)
            print(f"Checkpoint loaded from {self.checkpoint_file}")
            return True
        return False
    
    def run(self):
        """Main computation loop"""
        # Try to load checkpoint
        if not self.load_checkpoint():
            # Initialize new state
            print("Starting new computation...")
            self.state = {
                'iteration': 0,
                'results': [],
                'data': np.random.rand(1000000)
            }
        else:
            print(f"Resuming from iteration {self.state['iteration']}")
        
        # Main loop
        max_iterations = 10000
        checkpoint_interval = 100
        
        while self.state['iteration'] < max_iterations:
            if self.should_exit:
                self.save_checkpoint()
                sys.exit(0)
            
            # Do some work
            result = np.sum(self.state['data'] ** 2) / (self.state['iteration'] + 1)
            self.state['results'].append(result)
            self.state['iteration'] += 1
            
            # Periodic checkpoint
            if self.state['iteration'] % checkpoint_interval == 0:
                self.save_checkpoint()
                print(f"Iteration {self.state['iteration']}/{max_iterations}")
            
            # Simulate computation time
            time.sleep(0.01)
        
        print("Computation complete!")
        return self.state['results']

if __name__ == "__main__":
    job = CheckpointedJob()
    results = job.run()
    print(f"Final result: {results[-1]}")

# ============================================
# FILE: slurm_array_processor.py
# Purpose: Process files using SLURM array jobs
# ============================================
import os
import sys
import argparse
from pathlib import Path

def process_file(file_path):
    """Process a single file"""
    print(f"Processing: {file_path}")
    
    # Your processing code here
    # Example: count lines
    with open(file_path, 'r') as f:
        line_count = sum(1 for line in f)
    
    # Save result
    result_file = f"result_{file_path.stem}.txt"
    with open(result_file, 'w') as f:
        f.write(f"File: {file_path}\n")
        f.write(f"Lines: {line_count}\n")
    
    print(f"Result saved to: {result_file}")
    return result_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_list', help='Text file containing list of files')
    args = parser.parse_args()
    
    # Get array task ID from environment
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))
    
    # Read file list
    with open(args.file_list, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    # Process the file corresponding to this task ID
    if task_id <= len(files):
        file_to_process = files[task_id - 1]  # 1-indexed
        process_file(file_to_process)
    else:
        print(f"Task ID {task_id} exceeds number of files ({len(files)})")

if __name__ == "__main__":
    main()

# ============================================
# FILE: memory_monitor.py
# Purpose: Monitor memory usage during execution
# ============================================
import psutil
import os
import time
import threading
import numpy as np

class MemoryMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.monitoring = False
        self.max_memory = 0
        self.memory_history = []
        self.process = psutil.Process(os.getpid())
    
    def _monitor(self):
        """Monitor memory in background thread"""
        while self.monitoring:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_history.append(memory_mb)
            self.max_memory = max(self.max_memory, memory_mb)
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return stats"""
        self.monitoring = False
        self.thread.join()
        
        return {
            'max_memory_mb': self.max_memory,
            'avg_memory_mb': np.mean(self.memory_history),
            'samples': len(self.memory_history)
        }

def memory_intensive_task():
    """Example memory-intensive computation"""
    print("Starting memory-intensive task...")
    
    # Allocate large arrays
    arrays = []
    for i in range(5):
        size = 100_000_000  # 100M elements
        arr = np.random.rand(size)
        arrays.append(arr)
        print(f"Allocated array {i+1}: {arr.nbytes / 1024 / 1024:.1f} MB")
        time.sleep(1)
    
    # Do some computation
    result = sum(np.sum(arr) for arr in arrays)
    
    return result

if __name__ == "__main__":
    # Start monitoring
    monitor = MemoryMonitor()
    monitor.start()
    
    # Run task
    result = memory_intensive_task()
    
    # Stop monitoring and get stats
    stats = monitor.stop()
    
    print(f"\nMemory Statistics:")
    print(f"  Max memory: {stats['max_memory_mb']:.1f} MB")
    print(f"  Avg memory: {stats['avg_memory_mb']:.1f} MB")
    print(f"  Result: {result}")

# ============================================
# FILE: hpc_utils.py
# Purpose: Utility functions for HPC environment
# ============================================
import os
import subprocess
import socket

def get_job_info():
    """Get SLURM job information"""
    info = {
        'job_id': os.environ.get('SLURM_JOB_ID', 'N/A'),
        'job_name': os.environ.get('SLURM_JOB_NAME', 'N/A'),
        'nodes': os.environ.get('SLURM_JOB_NODELIST', 'N/A'),
        'ntasks': os.environ.get('SLURM_NTASKS', '1'),
        'cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK', '1'),
        'hostname': socket.gethostname(),
        'scratch_dir': f"/sciclone/scr10/{os.environ.get('USER', 'unknown')}"
    }
    return info

def print_environment():
    """Print HPC environment information"""
    info = get_job_info()
    print("=== HPC Environment ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 20)

def check_scratch_space():
    """Check available scratch space"""
    scratch_dirs = [
        f"/sciclone/scr10/{os.environ.get('USER')}",
        f"/sciclone/scr20/{os.environ.get('USER')}",
        f"/local/scr/{os.environ.get('USER')}"
    ]
    
    print("=== Scratch Space ===")
    for dir_path in scratch_dirs:
        if os.path.exists(dir_path):
            result = subprocess.run(
                ['df', '-h', dir_path], 
                capture_output=True, 
                text=True
            )
            print(f"{dir_path}:")
            print(result.stdout)
    print("=" * 20)

if __name__ == "__main__":
    print_environment()
    check_scratch_space()