#!/usr/bin/env python3
"""
Distributed Training Entry Point for Fiber Optics Neural Network
Supports multi-node/multi-GPU training on HPC clusters
"""

import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main import UnifiedFiberOpticsSystem
from config_loader import get_config
from distributed_utils import (
    init_distributed, cleanup_distributed, is_main_process,
    distributed_print, synchronize, get_rank, get_world_size
)
from trainer import EnhancedTrainer

def get_runtime_config(config):
    """Get runtime configuration from config file"""
    runtime = config.runtime
    
    # Create args-like object for compatibility
    class Args:
        def __init__(self):
            self.mode = runtime.mode if runtime.mode in ['train', 'evaluate', 'benchmark'] else 'train'
            self.config = runtime.config_path
            self.distributed = runtime.distributed
            self.epochs = config.training.num_epochs  # Use training.num_epochs
            self.checkpoint = runtime.checkpoint
            self.benchmark = runtime.benchmark
            self.results_dir = runtime.results_dir
    
    return Args()

def setup_environment():
    """Setup environment for optimal performance"""
    # Set threading optimizations
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() // 2)
    
    # PyTorch optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def train_distributed(args):
    """Main distributed training function"""
    # Initialize distributed training
    rank, local_rank, world_size = init_distributed()
    
    distributed_print(f"Initialized distributed training: Rank {rank}/{world_size}")
    
    # Load configuration
    config = get_config(args.config)
    
    # Override distributed settings
    config.training.distributed = True
    config.training.world_size = world_size
    config.training.rank = rank
    
    # Adjust batch size for distributed training
    if hasattr(config.training, 'batch_size'):
        config.training.batch_size = config.training.batch_size // world_size
        distributed_print(f"Adjusted batch size to {config.training.batch_size} per GPU")
    
    # Initialize system
    distributed_print("Initializing UnifiedFiberOpticsSystem...")
    system = UnifiedFiberOpticsSystem(mode="production", config_path=args.config)
    
    # Create distributed trainer
    distributed_print("Creating distributed trainer...")
    trainer = EnhancedTrainer(
        model=system.network,
        distributed=True
    )
    
    # Load checkpoint if provided
    if args.checkpoint and is_main_process():
        distributed_print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Synchronize before training
    synchronize()
    
    # Training
    if args.mode == 'train':
        distributed_print("Starting distributed training...")
        
        # Track timing
        start_time = time.time()
        
        # Get distributed data loaders
        train_loader, val_loader = system.data_loader.get_data_loaders(
            distributed=True,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers
        )
        
        # Train
        num_epochs = args.epochs if args.epochs else config.training.num_epochs
        trainer.train(
            num_epochs=num_epochs,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        training_time = time.time() - start_time
        
        # Save performance metrics
        if is_main_process() and args.benchmark:
            metrics = {
                'world_size': world_size,
                'epochs': num_epochs,
                'total_time': training_time,
                'time_per_epoch': training_time / num_epochs,
                'final_loss': trainer.best_val_loss,
                'samples_per_second': len(train_loader.dataset) * num_epochs / training_time
            }
            
            results_dir = Path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            with open(results_dir / 'performance_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            distributed_print(f"\nTraining completed in {training_time:.2f} seconds")
            distributed_print(f"Samples/second: {metrics['samples_per_second']:.2f}")
    
    elif args.mode == 'evaluate':
        distributed_print("Running distributed evaluation...")
        system.evaluate_performance()
    
    elif args.mode == 'benchmark':
        distributed_print("Running performance benchmark...")
        benchmark_distributed(system, trainer, args)
    
    # Cleanup
    cleanup_distributed()

def benchmark_distributed(system, trainer, args):
    """Run performance benchmarking"""
    distributed_print("Starting distributed performance benchmark...")
    
    # Create dummy data for consistent benchmarking
    batch_size = 32
    num_batches = 100
    
    # Warm-up
    distributed_print("Warming up...")
    for _ in range(10):
        dummy_input = torch.randn(batch_size, 3, 1024, 1024).cuda()
        with torch.no_grad():
            _ = system.network(dummy_input)
    
    # Synchronize before benchmark
    synchronize()
    torch.cuda.synchronize()
    
    # Benchmark forward pass
    distributed_print("Benchmarking forward pass...")
    start_time = time.time()
    
    for _ in range(num_batches):
        dummy_input = torch.randn(batch_size, 3, 1024, 1024).cuda()
        with torch.no_grad():
            _ = system.network(dummy_input)
    
    torch.cuda.synchronize()
    synchronize()
    
    total_time = time.time() - start_time
    throughput = (batch_size * num_batches * get_world_size()) / total_time
    
    if is_main_process():
        distributed_print(f"\nBenchmark Results:")
        distributed_print(f"World Size: {get_world_size()}")
        distributed_print(f"Total Time: {total_time:.2f}s")
        distributed_print(f"Throughput: {throughput:.2f} images/second")
        distributed_print(f"Time per batch: {total_time/num_batches*1000:.2f}ms")
        
        # Save benchmark results
        results = {
            'world_size': get_world_size(),
            'batch_size': batch_size,
            'num_batches': num_batches,
            'total_time': total_time,
            'throughput': throughput,
            'time_per_batch_ms': total_time/num_batches*1000
        }
        
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f'benchmark_gpu{get_world_size()}.json', 'w') as f:
            json.dump(results, f, indent=2)

def main():
    """Main entry point"""
    # Load configuration
    config = get_config()
    args = get_runtime_config(config)
    
    # Setup environment
    setup_environment()
    
    if args.distributed:
        # Distributed training
        train_distributed(args)
    else:
        # Single GPU training (fallback to main.py logic)
        distributed_print("Running in single GPU mode...")
        from main import main as single_gpu_main
        single_gpu_main()

if __name__ == "__main__":
    main()