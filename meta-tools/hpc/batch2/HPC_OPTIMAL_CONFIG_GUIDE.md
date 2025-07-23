# HPC Optimal Configuration Guide for Fiber Optics Neural Network

## Overview
This guide provides the best configuration settings for training the fiber optics neural network on HPC systems with maximum performance, comprehensive statistics, and debugging capabilities.

## Recommended Config.yaml Settings for HPC

### 1. System Settings (Critical for HPC)
```yaml
system:
  # Device configuration
  device: "cuda"  # Force CUDA for HPC GPU nodes
  gpu_id: 0  # Adjust based on allocated GPU
  num_workers: 16  # Increase for HPC (typical HPC nodes have 32-128 cores)
  seed: 42  # Keep deterministic for reproducibility
  
  # Logging configuration - MAXIMUM verbosity for debugging
  log_level: "DEBUG"  # Full debugging information
  log_to_file: true
  log_file_path: "logs/hpc_fiber_optics.log"
  logs_path: "logs"
  verbose_logging: true
  log_every_n_steps: 1  # Log every step for debugging
  save_logs_to_wandb: true  # Enable for remote monitoring
  
  # Paths - Use HPC scratch for performance
  data_path: "/scratch/$USER/fiber_optics/dataset"  # HPC scratch
  tensorized_data_path: "/scratch/$USER/fiber_optics/reference"
  reference_data_path: "/scratch/$USER/fiber_optics/reference"
  checkpoints_path: "/scratch/$USER/fiber_optics/checkpoints"
  results_path: "/scratch/$USER/fiber_optics/results"
  cache_path: "/dev/shm/fiber_optics_cache"  # RAM disk for speed
  temp_path: "/tmp/fiber_optics"
  
  # System behavior
  auto_cleanup_temp: false  # Keep for debugging
  max_cache_size_gb: 50.0  # Increase for HPC
  enable_profiling: true  # ENABLE for performance analysis
  profile_memory: true  # Track memory usage
  deterministic_mode: true  # Reproducible results
```

### 2. Model Architecture (Maximum Capability)
```yaml
model:
  # Architecture type
  architecture: "advanced"  # Use most advanced architecture
  
  # Increase capacity for HPC
  input_channels: 3
  base_channels: 128  # Doubled from default
  num_blocks: [3, 4, 6, 3]  # Deeper network like ResNet-50
  
  # Enable ALL advanced components
  use_se_blocks: true
  se_reduction: 16
  use_deformable_conv: true
  use_cbam: true
  use_efficient_channel_attention: true  # Enable ECA too
  
  # Adaptive computation
  use_adaptive_computation: true
  adaptive_threshold: 0.98  # Higher threshold
  max_computation_steps: 20  # More computation steps
  
  # Increase embeddings for better representation
  num_classes: 3
  num_reference_embeddings: 5000  # 5x increase
  embedding_dim: 512  # Doubled dimension
```

### 3. Training Configuration (HPC Optimized)
```yaml
training:
  # Larger batch sizes for HPC
  num_epochs: 200  # More epochs for convergence
  batch_size: 128  # Maximum for V100/A100 GPUs
  validation_split: 0.2
  early_stopping_patience: 20  # More patience
  
  # Full augmentation
  augmentation:
    enabled: true
    random_rotation: 30  # More aggressive
    random_scale: [0.8, 1.2]
    random_brightness: 0.2
    random_contrast: 0.2
    horizontal_flip: true
    vertical_flip: true
  
  # Gradient clipping for stability
  gradient_clip_value: 1.0
  gradient_clip_norm: 10.0  # Increased for larger models
  
  # Mixed precision - CRITICAL for HPC performance
  use_amp: true
  amp_opt_level: "O2"  # O2 for better performance
  
  # Knowledge distillation
  use_distillation: true  # Enable if teacher available
  distillation_alpha: 0.7
  distillation_temperature: 4.0
  teacher_model_path: "/scratch/$USER/fiber_optics/teacher_model.pth"
```

### 4. Optimizer Configuration (Advanced)
```yaml
optimizer:
  # Use most advanced optimizer
  type: "sam_lookahead"  # Best generalization
  
  # Tuned learning rates for HPC
  learning_rate: 0.002  # Slightly higher for larger batches
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1.0e-8
  
  # SAM parameters - tuned for stability
  sam_rho: 0.1  # Increased for better generalization
  sam_adaptive: true
  
  # Lookahead parameters
  lookahead_k: 10  # More lookahead steps
  lookahead_alpha: 0.5
  lookahead_pullback: "pullback"  # Enable pullback
  
  # Advanced scheduler
  scheduler:
    type: "cosine"  # Better for long training
    patience: 10
    factor: 0.5
    min_lr: 1.0e-7
  
  # Parameter-specific learning rates
  param_groups:
    feature_extractor: 1.0
    segmentation: 0.5
    reference_embeddings: 0.1
    decoder: 0.5
    equation_parameters: 0.05  # Slower for equation params
```

### 5. Loss Function Configuration (Comprehensive)
```yaml
loss:
  # Balanced weights for all losses
  weights:
    segmentation: 0.25
    anomaly: 0.25  # Increased for better anomaly detection
    contrastive: 0.15
    perceptual: 0.15
    wasserstein: 0.10
    reconstruction: 0.10
  
  # Tuned focal loss for imbalanced data
  focal_loss:
    segmentation_alpha: 0.25
    segmentation_gamma: 3.0  # Increased gamma
    anomaly_alpha: 0.75
    anomaly_gamma: 5.0  # Very high for rare defects
  
  # Enhanced contrastive loss
  contrastive_loss:
    temperature: 0.05  # Lower temperature
    normalize: true
  
  # Wasserstein loss parameters
  wasserstein_loss:
    p: 2  # 2-Wasserstein for better gradients
    epsilon: 0.01  # Smaller epsilon
    max_iter: 200  # More iterations
  
  # Perceptual loss - use deeper network
  perceptual_loss:
    network: "vgg19"  # Deeper network
    layers: [3, 8, 15, 22, 29, 36]  # More layers
    use_spatial: true  # Enable spatial
```

### 6. Advanced Features (Research Mode)
```yaml
advanced:
  # Enable gradient checkpointing for memory efficiency
  use_gradient_checkpointing: true
  
  # Neural Architecture Search
  use_nas: true  # Enable for architecture optimization
  nas_search_space: "darts"
  
  # Meta-learning
  use_maml: true  # Enable for few-shot learning
  maml_inner_steps: 10  # More inner steps
  maml_inner_lr: 0.01
  
  # Self-supervised pretraining
  use_self_supervised: true  # Beneficial for limited data
  ssl_method: "simclr"
  
  # Uncertainty estimation - CRITICAL for production
  use_uncertainty: true
  uncertainty_method: "ensemble"  # Best but expensive
  dropout_samples: 20  # More samples
  
  # Continual learning
  use_continual_learning: true
  replay_buffer_size: 5000  # Large buffer
  
  # Neural ODE for smooth dynamics
  use_neural_ode: true
  ode_solver: "dopri5"
  ode_rtol: 1e-4  # Tighter tolerance
  ode_atol: 1e-5
```

### 7. Performance Monitoring (Maximum Statistics)
```yaml
monitoring:
  # Track everything
  track_memory_usage: true
  track_computation_time: true
  track_gradient_norms: true
  
  # Tensorboard - essential for HPC
  use_tensorboard: true
  tensorboard_dir: "runs/hpc_experiment"
  
  # Weights & Biases for remote monitoring
  use_wandb: true
  wandb_project: "fiber-optics-hpc"
  wandb_entity: "your-team"
  
  # Profiling - critical for optimization
  enable_profiling: true
  profile_batches: [10, 20, 50, 100, 200]  # Multiple checkpoints
```

### 8. Debug Configuration (Maximum Information)
```yaml
debug:
  # Full debugging
  enabled: true
  
  # Save everything for analysis
  save_intermediate_features: true
  save_gradient_flow: true
  save_attention_maps: true
  save_feature_evolution: true
  save_loss_landscape: true
  
  # Numerical checks
  check_nan: true
  check_inf: true
  check_gradient_explosion: true
  gradient_explosion_threshold: 1000.0  # Higher threshold
  
  # Deterministic mode
  deterministic: true
  benchmark: false  # Disable for reproducibility
  
  # Debug outputs
  print_model_summary: true
  print_parameter_counts: true
  save_debug_images: true
  debug_image_frequency: 10  # Every 10 batches
```

### 9. Runtime Configuration (HPC Specific)
```yaml
runtime:
  # Research mode for maximum statistics
  mode: "research"  # Full statistical analysis
  
  # Batch processing settings
  batch_process: true
  max_batch_images: 10000  # Process more images
  
  # Real-time settings (for testing)
  realtime_source: null
  realtime_fps: 60  # Higher FPS target
  
  # Analysis options
  save_visualizations: true
  export_metrics: true
  generate_report: true
```

### 10. Performance Optimization (HPC Critical)
```yaml
optimization:
  # Memory optimization
  gradient_accumulation_steps: 4  # Simulate larger batches
  mixed_precision_training: true
  memory_efficient_attention: true  # For large models
  flash_attention: true  # If available
  
  # Computation optimization
  compile_model: true  # PyTorch 2.0 compile
  use_channels_last: true  # Better GPU utilization
  fuse_batch_norm: true
  
  # Data loading optimization
  num_data_workers: 32  # More workers for HPC
  data_loader_pin_memory: true
  non_blocking_transfer: true
  
  # Distributed training - ESSENTIAL for multi-GPU
  use_distributed: true
  distributed_backend: "nccl"  # Best for NVIDIA GPUs
  world_size: 8  # Adjust to number of GPUs
  rank: 0  # Set via environment variable
```

### 11. HPC-Specific Environment Variables
```yaml
# Add these to your SLURM script:
# export OMP_NUM_THREADS=8
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_DEBUG=INFO
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## SLURM Job Script Example

```bash
#!/bin/bash
#SBATCH --job-name=fiber_optics_hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_large
#SBATCH --output=fiber_optics_%j.out
#SBATCH --error=fiber_optics_%j.err

# Load modules
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

# Set environment
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Activate virtual environment
source /path/to/venv/bin/activate

# Change to working directory
cd /path/to/polar-bear/Network

# Run distributed training
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    main.py
```

## Performance Tips

1. **Data Loading**: 
   - Pre-process and cache tensors in HPC scratch space
   - Use RAM disk (/dev/shm) for frequently accessed data
   - Increase num_workers to CPU count / 2

2. **Memory Management**:
   - Enable gradient checkpointing for very deep models
   - Use gradient accumulation for effective larger batches
   - Clear cache periodically with torch.cuda.empty_cache()

3. **Multi-GPU Training**:
   - Always use DistributedDataParallel (DDP)
   - Sync batch norm across GPUs
   - Use NCCL backend for NVIDIA GPUs

4. **Monitoring**:
   - Set up TensorBoard server on HPC login node
   - Use WandB for remote monitoring
   - Save checkpoints frequently to scratch

5. **Debugging**:
   - Start with smaller batch sizes
   - Enable all numerical checks
   - Profile first 100 batches
   - Save gradient flow visualizations

## Expected Performance

With these settings on a typical HPC node (8x V100/A100 GPUs):
- Training throughput: 800-1200 images/second
- Memory usage: 40-50GB per GPU
- Training time: 4-8 hours for 200 epochs
- Model size: ~500MB (advanced architecture)

## Troubleshooting

1. **OOM Errors**: Reduce batch_size or enable gradient_checkpointing
2. **Slow Training**: Check data loading pipeline, increase num_workers
3. **NaN Loss**: Enable gradient clipping, reduce learning rate
4. **Poor Convergence**: Adjust loss weights, try different optimizer
5. **NCCL Errors**: Check network connectivity between GPUs

This configuration maximizes performance while maintaining comprehensive debugging and statistics collection for robust HPC training.