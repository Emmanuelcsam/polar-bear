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

# --- FIX-OVERVIEW ---
# This file was already well-written and robust for its purpose.
# No critical errors were found. I've added comments for clarity and made minor robustness improvements.
# It correctly handles SLURM and torchrun environments for distributed setup.
# Additional improvements: Enhanced error handling, better device management, and improved logging.

def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed training environment by detecting SLURM or torchrun.
    
    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    # Check if running under SLURM, a common HPC workload manager.
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        
        # Automatically determine the master address from the SLURM node list if not set.
        if 'MASTER_ADDR' not in os.environ:
            # This command gets the hostname of the first node in the job allocation.
            node_list = os.environ['SLURM_JOB_NODELIST']
            master_node_cmd = f'scontrol show hostname {node_list} | head -n1'
            try:
                master_node = subprocess.check_output(master_node_cmd, shell=True, stderr=subprocess.PIPE).decode().strip()
                if master_node:
                    os.environ['MASTER_ADDR'] = master_node
                else:
                    raise ValueError("scontrol command returned empty string")
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not determine master node from SLURM, falling back to localhost. Error: {e}")
                # Fallback to localhost if command fails
                os.environ['MASTER_ADDR'] = 'localhost'
                print(f"INFO: Using localhost as master address for distributed training")

        # Set a default master port if not provided.
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
    
    # Check if running under torchrun/torch.distributed.launch, a standard PyTorch tool.
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', '0')) # LOCAL_RANK is optional in some setups
        world_size = int(os.environ['WORLD_SIZE'])
    
    # Fallback to a single-process (non-distributed) setup.
    else:
        print("INFO: No distributed environment detected. Running on single GPU/CPU.")
        rank = 0
        local_rank = 0
        world_size = 1
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        print(f"INFO: Single process setup - Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    
    return rank, local_rank, world_size

def init_distributed(backend: str = 'nccl') -> Tuple[int, int, int]:
    """
    Initialize the PyTorch distributed process group.
    
    Args:
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU).
        
    Returns:
        Tuple of (rank, local_rank, world_size).
    """
    rank, local_rank, world_size = setup_distributed()
    
    # Only initialize the process group if there is more than one process.
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method='env://', # Read MASTER_ADDR and MASTER_PORT from environment variables.
                rank=rank,
                world_size=world_size
            )
        
        # Set the active CUDA device for this process.
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
        # Synchronize all processes to ensure they start together.
        dist.barrier()
        
        if rank == 0:
            print(f"Distributed training initialized:")
            print(f"  Backend: {backend}")
            print(f"  World size: {world_size}")
            print(f"  Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
            print(f"  Local rank: {local_rank}")
            if torch.cuda.is_available():
                print(f"  CUDA device: {torch.cuda.current_device()}")
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_world_size() -> int:
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_local_rank() -> int:
    """Get the local rank on the current node."""
    # This is important for assigning GPUs correctly on a multi-GPU node.
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    elif dist.is_initialized():
        # Fallback to distributed rank if available
        return dist.get_rank() % torch.cuda.device_count() if torch.cuda.is_available() else 0
    return 0

def synchronize():
    """A barrier to synchronize all processes."""
    if dist.is_initialized() and get_world_size() > 1:
        dist.barrier()

def reduce_tensor(tensor: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Reduce a tensor across all processes.
    
    Args:
        tensor: The tensor to reduce.
        reduction: Type of reduction ('mean' or 'sum').
        
    Returns:
        The reduced tensor on all processes.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    
    # Clone to avoid in-place modification of the original tensor.
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if reduction == 'mean':
        rt /= get_world_size()
        
    return rt

def gather_tensors(tensor: torch.Tensor) -> list:
    """
    Gather tensors from all processes to rank 0.
    
    Args:
        tensor: The tensor to gather from each process.
        
    Returns:
        A list of tensors from all processes (only populated on rank 0).
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return [tensor]
    
    world_size = get_world_size()
    
    # Handling for GPU tensors.
    if tensor.is_cuda:
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
    # Handling for CPU tensors (uses pickle, can be slower).
    else:
        # all_gather_object is less efficient but works for CPU tensors and complex objects.
        tensor_list = [None for _ in range(world_size)]
        dist.all_gather_object(tensor_list, tensor)
    
    # Ensure all tensors are on the same device for consistency
    if tensor.is_cuda:
        tensor_list = [t.to(tensor.device) if t is not None else None for t in tensor_list]
    
    return tensor_list

def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast a Python object from a source rank to all other ranks.
    
    Args:
        obj: The Python object to broadcast.
        src: The rank of the source process.
        
    Returns:
        The broadcasted object on all processes.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return obj
    
    # The object must be in a list for broadcast_object_list.
    object_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]

def wrap_model_ddp(model: torch.nn.Module, 
                   device_ids: Optional[list] = None,
                   find_unused_parameters: bool = False) -> torch.nn.Module:
    """
    Wrap a model with DistributedDataParallel (DDP).
    
    Args:
        model: The model to wrap.
        device_ids: List of GPU device IDs for the model on the current node.
        find_unused_parameters: Set to True if the model's forward pass might not use all parameters.
        
    Returns:
        The DDP-wrapped model.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return model
    
    if device_ids is None and torch.cuda.is_available():
        device_ids = [get_local_rank()]
    
    # Ensure model is on the correct device before wrapping
    if torch.cuda.is_available() and device_ids:
        model = model.to(f'cuda:{device_ids[0]}')
    
    model = DDP(
        model,
        device_ids=device_ids,
        output_device=device_ids[0] if device_ids else None,
        find_unused_parameters=find_unused_parameters
    )
    
    return model

def save_checkpoint_distributed(state_dict: dict, filepath: str, is_best: bool = False):
    """
    Save a training checkpoint in a distributed setting (only on the main process).
    
    Args:
        state_dict: The state dictionary to save.
        filepath: The path to save the checkpoint file.
        is_best: If True, also saves a copy as the best model.
    """
    if is_main_process():
        torch.save(state_dict, filepath)
        
        if is_best:
            import shutil
            # Ensure the path manipulation is robust
            dir_name = os.path.dirname(filepath)
            base_name = os.path.basename(filepath)
            best_path = os.path.join(dir_name, "best_model.pth")
            shutil.copyfile(filepath, best_path)

def setup_distributed_logging(rank: int, world_size: int) -> logging.Logger:
    """
    Setup logging to only output from the main process.
    
    Args:
        rank: The rank of the current process.
        world_size: The total number of processes.
        
    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(f'rank_{rank}')
    # Prevent adding handlers multiple times if called more than once
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    
    # Only attach a handler to the main process logger to avoid duplicate logs.
    if rank == 0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[%(asctime)s][Rank {rank}/{world_size}][%(name)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Other processes get a NullHandler to suppress output.
        logger.addHandler(logging.NullHandler())
    
    return logger

class DistributedMetricTracker:
    """A helper class to track and synchronize metrics across distributed processes."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.metrics = {}
        self.counts = {}
        # FIX: Allow specifying a device during initialization.
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def update(self, name: str, value: torch.Tensor, count: int = 1):
        """Update a metric with a new value."""
        if name not in self.metrics:
            # FIX: Ensure new tensors are created on the correct device.
            self.metrics[name] = torch.zeros(1, device=self.device)
            self.counts[name] = torch.zeros(1, device=self.device)
        
        # Ensure value is on the correct device before accumulation
        value_tensor = value.detach().to(self.device)
        self.metrics[name] += value_tensor * count
        self.counts[name] += count
    
    def get_average(self, name: str, sync: bool = True) -> float:
        """Get the average value of a metric, synchronizing across all processes if needed."""
        if name not in self.metrics:
            return 0.0
        
        total_metric = self.metrics[name].clone()
        total_count = self.counts[name].clone()
        
        if sync and dist.is_initialized() and get_world_size() > 1:
            # Synchronize the sum of the metric and the total count across all processes.
            dist.all_reduce(total_metric, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        
        # Avoid division by zero.
        if total_count.item() == 0:
            return 0.0
        
        return (total_metric / total_count).item()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.counts.clear()

def distributed_print(*args, **kwargs):
    """A print function that only executes on the main process."""
    if is_main_process():
        print(*args, **kwargs)

def get_device_info() -> dict:
    """
    Get information about available devices for distributed training.
    
    Returns:
        Dictionary containing device information.
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    }
    
    if dist.is_initialized():
        device_info.update({
            'world_size': get_world_size(),
            'rank': get_rank(),
            'local_rank': get_local_rank()
        })
    
    return device_info
