# main.py
# Main entry point for the fiber optic analysis system

import torch
import logging
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import ConfigManager, setup_logging, setup_distributed, ensure_directories
from dataset import create_dataloaders
from trainer import create_trainer
from evaluator import create_evaluator
from optimizer import create_optimizer
from utils import setup_seed, get_device_info, log_model_summary, MetricsTracker

def setup_environment(config):
    """Setup the training/evaluation environment."""
    # Setup logging
    log_level = logging.DEBUG if config.system.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    # Setup random seeds
    setup_seed(config.system.seed)
    
    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()
    
    # Ensure directories exist
    ensure_directories(config)
    
    # Log environment info
    if rank == 0:
        device_info = get_device_info()
        logger.info("=== Fiber Optic Analysis System ===")
        logger.info(f"Mode: {config.system.mode}")
        logger.info(f"Config: {config.system.config_path}")
        logger.info(f"CUDA available: {device_info['cuda_available']}")
        logger.info(f"GPU count: {device_info['cuda_device_count']}")
        logger.info(f"Distributed: {is_distributed}")
        if is_distributed:
            logger.info(f"Rank: {rank}/{world_size}")
    
    return logger, rank, world_size, local_rank, is_distributed

def train_mode(config, logger, rank, world_size, local_rank, is_distributed):
    """Execute training mode."""
    logger.info("Starting training mode...")
    
    # Create data loaders
    try:
        train_loader, val_loader, train_sampler = create_dataloaders(
            config, world_size, rank
        )
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # Create trainer
    trainer = create_trainer(config, rank, world_size, local_rank, is_distributed)
    
    # Log model summary (only on main process)
    if rank == 0:
        log_model_summary(trainer.model, input_size=(1, 3, config.data.image_size, config.data.image_size))
    
    # Load checkpoint if specified
    if config.system.checkpoint_path:
        if trainer.load_checkpoint(config.system.checkpoint_path):
            logger.info(f"Resumed training from: {config.system.checkpoint_path}")
        else:
            logger.warning(f"Failed to load checkpoint: {config.system.checkpoint_path}")
    
    # Start training
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def eval_mode(config, logger, rank, world_size, local_rank, is_distributed):
    """Execute evaluation mode."""
    logger.info("Starting evaluation mode...")
    
    if not config.system.checkpoint_path:
        logger.error("Checkpoint path required for evaluation mode")
        return
    
    # Create data loaders
    try:
        _, val_loader, _ = create_dataloaders(config, world_size, rank)
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # Create evaluator
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    evaluator = create_evaluator(config, device)
    
    # Load model
    from model import load_model
    try:
        model = load_model(config, config.system.checkpoint_path, device)
        model.eval()
        logger.info(f"Model loaded from: {config.system.checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Evaluate model
    try:
        metrics = evaluator.evaluate(model, val_loader)
        
        # Log results
        logger.info("=== Evaluation Results ===")
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Average Similarity: {metrics['avg_similarity']:.4f}")
        logger.info(f"Pass Rate: {metrics['pass_rate']:.4f}")
        
        # Per-class accuracies
        logger.info("Per-class Accuracies:")
        for class_name, accuracy in metrics['class_accuracies'].items():
            logger.info(f"  {class_name}: {accuracy:.4f}")
        
        # Analyze failure cases
        failure_analysis = evaluator.analyze_failure_cases(model, val_loader)
        logger.info(f"Failure Rate: {failure_analysis['failure_rate']:.4f}")
        
        if 'failure_statistics' in failure_analysis:
            stats = failure_analysis['failure_statistics']
            logger.info("Failure Statistics:")
            logger.info(f"  Avg Similarity: {stats['avg_similarity']:.4f}")
            logger.info(f"  Avg Anomaly Score: {stats['avg_anomaly_score']:.4f}")
            logger.info(f"  Avg Classification Confidence: {stats['avg_classification_confidence']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def optimize_mode(config, logger, rank, world_size, local_rank, is_distributed):
    """Execute optimization mode (pruning, distillation)."""
    logger.info("Starting optimization mode...")
    
    if not config.system.checkpoint_path:
        logger.error("Checkpoint path required for optimization mode")
        return
    
    # Create data loaders
    try:
        train_loader, val_loader, _ = create_dataloaders(config, world_size, rank)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # Load model
    from model import load_model
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    try:
        model = load_model(config, config.system.checkpoint_path, device)
        logger.info(f"Model loaded from: {config.system.checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Create optimizer
    model_optimizer = create_optimizer(config, device)
    
    # Apply optimizations
    try:
        optimized_model = model_optimizer.optimize_model(
            model, train_loader, val_loader, config.system.checkpoints_path
        )
        logger.info("Model optimization completed")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

def main():
    """Main function."""
    # Load configuration
    try:
        config_manager = ConfigManager("config.yaml")
        config = config_manager.config
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup environment
    logger, rank, world_size, local_rank, is_distributed = setup_environment(config)
    
    try:
        # Execute based on mode
        if config.system.mode == 'train':
            train_mode(config, logger, rank, world_size, local_rank, is_distributed)
        elif config.system.mode == 'eval':
            eval_mode(config, logger, rank, world_size, local_rank, is_distributed)
        elif config.system.mode == 'optimize':
            optimize_mode(config, logger, rank, world_size, local_rank, is_distributed)
        else:
            logger.error(f"Unknown mode: {config.system.mode}")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        if config.system.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    logger.info("Execution completed successfully")

if __name__ == '__main__':
    main()
