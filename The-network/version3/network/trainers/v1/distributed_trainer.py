import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os
from typing import Optional
import time

from ..utils.logger import get_logger
from ..config.config import get_config
from .trainer import FiberOpticsTrainer, FiberOpticsDataset


class DistributedFiberOpticsTrainer(FiberOpticsTrainer):
    """
    Distributed training for HPC environments.
    "the program will run entirely in hpc meaning that it has to be able to run 
    on gpu and in parallel"
    """
    
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        # Initialize process group
        self._init_distributed(rank, world_size)
        
        # Initialize parent class
        super().__init__(model)
        
        self.rank = rank
        self.world_size = world_size
        
        # Wrap model for distributed training
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
        )
        
        self.logger.info(f"Initialized DistributedTrainer on rank {rank}/{world_size}")
    
    def _init_distributed(self, rank: int, world_size: int):
        """Initialize distributed process group"""
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        # Initialize the process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Distributed training loop.
        "but this is only for the training sequence so that it can be trained fast 
        and effectively with a large database and multiple trials over and over again"
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        if self.rank == 0:
            self.logger.info(f"Starting distributed training on {self.world_size} GPUs")
        
        # Load reference data (only on rank 0 to save memory)
        if self.rank == 0:
            self.logger.info("Loading reference tensors...")
            self.data_loader.load_all_references(preload=True)
        
        # Synchronize all processes
        dist.barrier()
        
        # Create distributed dataset
        train_dataset = self._create_training_dataset()
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE // self.world_size,  # Split batch across GPUs
            sampler=train_sampler,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        
        # Training loop
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)  # Ensure different shuffling each epoch
            
            epoch_start = time.time()
            self.model.train()
            
            epoch_losses = []
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.cuda(self.rank)
                labels = labels.cuda(self.rank)
                
                # Train step
                loss, metrics = self._train_step(images, labels)
                epoch_losses.append(loss.item())
                
                # Log on main process
                if self.rank == 0 and batch_idx % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}, Batch {batch_idx}: "
                        f"Loss={loss.item():.4f}, Similarity={metrics['similarity']:.4f}"
                    )
            
            # Aggregate metrics across all processes
            avg_loss = self._reduce_metric(epoch_losses)
            
            if self.rank == 0:
                self.training_history['loss'].append(avg_loss)
                
                # Adjust learning rate
                self.scheduler.step(avg_loss)
                
                # Log epoch time
                epoch_time = time.time() - epoch_start
                self.logger.log_performance_metric(
                    f"distributed_epoch_{epoch+1}_time", 
                    epoch_time, 
                    "s"
                )
                
                # Save checkpoint
                if epoch % 10 == 0:
                    self._save_checkpoint(epoch)
            
            # Synchronize all processes
            dist.barrier()
        
        if self.rank == 0:
            self.logger.info("Distributed training completed")
        
        # Cleanup
        self.cleanup()
    
    def _reduce_metric(self, metric_list: list) -> float:
        """Reduce metrics across all processes"""
        # Convert to tensor
        metric_tensor = torch.tensor(metric_list).cuda(self.rank)
        
        # Get average within this process
        local_avg = metric_tensor.mean()
        
        # Reduce across all processes
        dist.all_reduce(local_avg, op=dist.ReduceOp.SUM)
        global_avg = local_avg / self.world_size
        
        return global_avg.item()
    
    def _save_checkpoint(self, epoch: int):
        """Save checkpoint (only on rank 0)"""
        if self.rank != 0:
            return
        
        # Get the underlying model (not DDP wrapper)
        model_to_save = self.model.module
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"distributed_fiber_optics_nn_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'world_size': self.world_size
        }, checkpoint_path)
        
        self.logger.info(f"Saved distributed checkpoint to {checkpoint_path}")
    
    def cleanup(self):
        """Clean up distributed process group"""
        dist.destroy_process_group()


def launch_distributed_training(num_gpus: Optional[int] = None):
    """
    Launch distributed training across multiple GPUs.
    "the program will run entirely in hpc"
    """
    from ..models.fiber_optics_nn import FiberOpticsNeuralNetwork
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        print("Not enough GPUs for distributed training. Use regular trainer instead.")
        return
    
    print(f"Launching distributed training on {num_gpus} GPUs")
    
    # Spawn processes
    import torch.multiprocessing as mp
    mp.spawn(
        _train_worker,
        args=(num_gpus,),
        nprocs=num_gpus,
        join=True
    )


def _train_worker(rank: int, world_size: int):
    """Worker function for distributed training"""
    # Create model
    from ..models.fiber_optics_nn import FiberOpticsNeuralNetwork
    model = FiberOpticsNeuralNetwork()
    
    # Create distributed trainer
    trainer = DistributedFiberOpticsTrainer(model, rank, world_size)
    
    # Run training
    trainer.train()


# HPC-specific optimizations
class HPCOptimizedDataLoader:
    """
    Optimized data loading for HPC environments.
    "multiple trials over and over again"
    """
    
    def __init__(self, dataset, batch_size: int, num_workers: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Enable various optimizations
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = True
        self.prefetch_factor = 2
        
    def get_dataloader(self, distributed: bool = False, rank: int = 0, 
                      world_size: int = 1) -> DataLoader:
        """Get optimized dataloader"""
        
        if distributed:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True
        )
        
        return dataloader


class MixedPrecisionTrainer(DistributedFiberOpticsTrainer):
    """
    Trainer with mixed precision support for faster training.
    "the overall program follows this equation I=Ax1+Bx2+Cx3"
    """
    
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        super().__init__(model, rank, world_size)
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
    def _train_step(self, images: torch.Tensor, labels: torch.Tensor):
        """Training step with mixed precision"""
        self.optimizer.zero_grad()
        
        # Use automatic mixed precision
        with torch.cuda.amp.autocast():
            # Run normal training step
            loss, metrics = super()._train_step(images, labels)
        
        # Scale loss and backward
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Step optimizer
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss, metrics