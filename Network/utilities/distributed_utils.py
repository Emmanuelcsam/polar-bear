#!/usr/bin/env python3
"""
Distributed Training Utilities for HPC Deployment
Provides initialization and utilities for multi-node/multi-GPU training
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Tuple, Any
import subprocess
import logging

def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed training environment
    
    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    # Check if running under SLURM
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        
        # Set master address from SLURM
        if 'MASTER_ADDR' not in os.environ:
            node_list = os.environ['SLURM_JOB_NODELIST']
            master_node = subprocess.check_output(
                f'scontrol show hostname {node_list} | head -n1', 
                shell=True
            ).decode().strip()
            os.environ['MASTER_ADDR'] = master_node
            
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
    
    # Check if running under torchrun/torch.distributed.launch
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    
    # Single GPU fallback
    else:
        print("No distributed environment detected. Running on single GPU.")
        rank = 0
        local_rank = 0
        world_size = 1
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
    
    return rank, local_rank, world_size

def init_distributed(backend: str = 'nccl') -> Tuple[int, int, int]:
    """
    Initialize PyTorch distributed process group
    
    Args:
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU)
        
    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    rank, local_rank, world_size = setup_distributed()
    
    # Initialize process group if multi-GPU
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                rank=rank,
                world_size=world_size
            )
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
        # Synchronize
        dist.barrier()
        
        if rank == 0:
            print(f"Distributed training initialized:")
            print(f"  Backend: {backend}")
            print(f"  World size: {world_size}")
            print(f"  Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    """Check if current process is the main process"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_world_size() -> int:
    """Get total number of processes"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    """Get current process rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_local_rank() -> int:
    """Get local rank on current node"""
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    return 0

def synchronize():
    """Synchronize all processes"""
    if dist.is_initialized():
        dist.barrier()

def reduce_tensor(tensor: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Reduce tensor across all processes
    
    Args:
        tensor: Tensor to reduce
        reduction: Type of reduction ('mean', 'sum')
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    # Clone to avoid modifying original
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if reduction == 'mean':
        rt /= get_world_size()
        
    return rt

def gather_tensors(tensor: torch.Tensor) -> list:
    """
    Gather tensors from all processes
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all processes (only on rank 0)
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    
    # Ensure tensor is on correct device
    if tensor.is_cuda:
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
    else:
        # For CPU tensors
        tensor_list = [None for _ in range(world_size)]
        dist.all_gather_object(tensor_list, tensor)
    
    return tensor_list

def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast object from source rank to all other ranks
    
    Args:
        obj: Object to broadcast
        src: Source rank
        
    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj
    
    object_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]

def wrap_model_ddp(model: torch.nn.Module, 
                   device_ids: Optional[list] = None,
                   find_unused_parameters: bool = False) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel
    
    Args:
        model: Model to wrap
        device_ids: List of GPU device IDs
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        return model
    
    if device_ids is None and torch.cuda.is_available():
        device_ids = [get_local_rank()]
    
    model = DDP(
        model,
        device_ids=device_ids,
        output_device=device_ids[0] if device_ids else None,
        find_unused_parameters=find_unused_parameters
    )
    
    return model

def save_checkpoint_distributed(state_dict: dict, filepath: str, is_best: bool = False):
    """
    Save checkpoint in distributed setting (only on main process)
    
    Args:
        state_dict: State dictionary to save
        filepath: Path to save checkpoint
        is_best: Whether this is the best model
    """
    if is_main_process():
        torch.save(state_dict, filepath)
        
        if is_best:
            import shutil
            best_path = filepath.replace('.pth', '_best.pth')
            shutil.copyfile(filepath, best_path)

def setup_distributed_logging(rank: int, world_size: int) -> logging.Logger:
    """
    Setup logging for distributed training
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(f'rank_{rank}')
    logger.setLevel(logging.INFO)
    
    # Only log from main process by default
    if rank == 0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[%(asctime)s][Rank {rank}/{world_size}][%(name)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Suppress output from non-main processes
        logger.addHandler(logging.NullHandler())
    
    return logger

class DistributedMetricTracker:
    """Track and synchronize metrics across distributed processes"""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, name: str, value: torch.Tensor, count: int = 1):
        """Update metric with new value"""
        if name not in self.metrics:
            self.metrics[name] = torch.zeros(1, device=value.device)
            self.counts[name] = 0
        
        self.metrics[name] += value.detach() * count
        self.counts[name] += count
    
    def get_average(self, name: str, sync: bool = True) -> float:
        """Get average value of metric across all processes"""
        if name not in self.metrics:
            return 0.0
        
        if sync and dist.is_initialized():
            # Synchronize across processes
            metric_sum = reduce_tensor(self.metrics[name], 'sum')
            count_sum = reduce_tensor(
                torch.tensor(self.counts[name], device=self.metrics[name].device), 
                'sum'
            )
            return (metric_sum / count_sum).item()
        else:
            return (self.metrics[name] / self.counts[name]).item()
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counts.clear()

def distributed_print(*args, **kwargs):
    """Print only from main process"""
    if is_main_process():
        print(*args, **kwargs)