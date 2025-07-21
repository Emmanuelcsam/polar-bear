#!/usr/bin/env python3
"""
Main training script for fiber optics neural network.
Supports both single GPU and distributed training.
"""

import torch
import argparse
from pathlib import Path
import sys

# Add the module to path
sys.path.append(str(Path(__file__).parent))

from fiber_optics_nn import (
    get_logger,
    get_config,
    FiberOpticsNeuralNetwork,
    FiberOpticsTrainer,
    launch_distributed_training
)


def main():
    """
    Main training entry point.
    "the program will run entirely in hpc meaning that it has to be able to run 
    on gpu and in parallel, but this is only for the training sequence"
    """
    # Parse command line arguments (without using argparse flags as requested)
    # Usage: python main_training.py [distributed] [num_epochs]
    
    distributed = False
    num_epochs = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "distributed":
            distributed = True
        else:
            try:
                num_epochs = int(sys.argv[1])
            except ValueError:
                pass
    
    if len(sys.argv) > 2 and not distributed:
        try:
            num_epochs = int(sys.argv[2])
        except ValueError:
            pass
    
    # Initialize logger and config
    logger = get_logger()
    config = get_config()
    
    logger.info("=" * 80)
    logger.info("FIBER OPTICS NEURAL NETWORK - TRAINING MODE")
    logger.info("=" * 80)
    
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s)")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"  GPU {i}: {gpu_name}")
    else:
        logger.warning("No GPUs found. Training will be slower on CPU.")
    
    if distributed and torch.cuda.device_count() > 1:
        # Launch distributed training
        logger.info("Launching distributed training across multiple GPUs...")
        launch_distributed_training(num_gpus=torch.cuda.device_count())
    else:
        # Single GPU or CPU training
        logger.info("Starting single device training...")
        
        # Create model
        model = FiberOpticsNeuralNetwork()
        
        # Create trainer
        trainer = FiberOpticsTrainer(model)
        
        # Start training
        trainer.train(num_epochs=num_epochs)
        
        # Save final model
        save_final_model(model, trainer, logger)
    
    logger.info("Training completed successfully!")


def save_final_model(model, trainer, logger):
    """Save the final trained model"""
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    final_path = checkpoint_dir / "fiber_optics_nn_final.pth"
    best_path = checkpoint_dir / "fiber_optics_nn_best.pth"
    
    # Save final checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'training_history': trainer.training_history,
        'config': trainer.config.__dict__
    }, final_path)
    
    logger.info(f"Saved final model to {final_path}")
    
    # Also save as best model if it's the only one
    if not best_path.exists():
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'training_history': trainer.training_history,
            'config': trainer.config.__dict__
        }, best_path)
        logger.info(f"Saved best model to {best_path}")


def resume_training():
    """
    Resume training from a checkpoint.
    "the program will allow me to see and tweak the parameters and weights of 
    these fitting equations and over time the program will get more accurate"
    """
    logger = get_logger()
    config = get_config()
    
    # Find latest checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoints = list(checkpoint_dir.glob("fiber_optics_nn_epoch_*.pth"))
    
    if not checkpoints:
        logger.error("No checkpoints found to resume from")
        return
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    latest_checkpoint = checkpoints[-1]
    
    logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
    
    # Create model and trainer
    model = FiberOpticsNeuralNetwork()
    trainer = FiberOpticsTrainer(model)
    
    # Load checkpoint
    start_epoch = trainer.load_checkpoint(latest_checkpoint)
    
    # Continue training
    remaining_epochs = config.NUM_EPOCHS - start_epoch
    if remaining_epochs > 0:
        logger.info(f"Continuing training for {remaining_epochs} more epochs...")
        trainer.train(num_epochs=remaining_epochs)
    else:
        logger.info("Training already completed!")


def adjust_parameters():
    """
    Interactive parameter adjustment interface.
    "the program will allow me to see and tweak the parameters and weights"
    """
    logger = get_logger()
    config = get_config()
    
    logger.info("=" * 60)
    logger.info("PARAMETER ADJUSTMENT INTERFACE")
    logger.info("=" * 60)
    
    # Display current parameters
    logger.info("Current equation parameters (I=Ax1+Bx2+Cx3...):")
    params = config.get_equation_parameters()
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nAdjustable parameters:")
    logger.info("1. GRADIENT_WEIGHT_FACTOR (A)")
    logger.info("2. PIXEL_POSITION_WEIGHT_FACTOR (B)")
    logger.info("3. SIMILARITY_THRESHOLD")
    logger.info("4. LEARNING_RATE")
    logger.info("5. LOSS_ADJUSTMENT_RATE")
    
    # This is a placeholder for an interactive interface
    # In practice, you would implement a GUI or web interface
    logger.info("\nTo adjust parameters, modify config/config.py and restart training")


def analyze_training_history():
    """Analyze and visualize training history"""
    import json
    import numpy as np
    
    logger = get_logger()
    
    # Find latest checkpoint with history
    checkpoint_dir = Path("checkpoints")
    final_checkpoint = checkpoint_dir / "fiber_optics_nn_final.pth"
    
    if not final_checkpoint.exists():
        logger.error("No final checkpoint found")
        return
    
    # Load training history
    checkpoint = torch.load(final_checkpoint, map_location='cpu')
    history = checkpoint.get('training_history', {})
    
    if not history:
        logger.error("No training history found in checkpoint")
        return
    
    logger.info("=" * 60)
    logger.info("TRAINING HISTORY ANALYSIS")
    logger.info("=" * 60)
    
    # Analyze loss progression
    if 'loss' in history:
        losses = history['loss']
        logger.info(f"\nLoss progression over {len(losses)} epochs:")
        logger.info(f"  Initial loss: {losses[0]:.4f}")
        logger.info(f"  Final loss: {losses[-1]:.4f}")
        logger.info(f"  Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
        logger.info(f"  Best loss: {min(losses):.4f} at epoch {np.argmin(losses) + 1}")
    
    # Analyze similarity scores
    if 'similarity_scores' in history:
        similarities = history['similarity_scores']
        logger.info(f"\nSimilarity score progression:")
        logger.info(f"  Initial similarity: {similarities[0]:.4f}")
        logger.info(f"  Final similarity: {similarities[-1]:.4f}")
        logger.info(f"  Average similarity: {np.mean(similarities):.4f}")
        
        # Check how many meet threshold
        threshold = get_config().SIMILARITY_THRESHOLD
        above_threshold = sum(1 for s in similarities if s >= threshold)
        logger.info(f"  Epochs above threshold ({threshold}): {above_threshold}/{len(similarities)}")
    
    # Save analysis to file
    analysis_path = checkpoint_dir / "training_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump({
            'loss_stats': {
                'initial': float(losses[0]) if 'loss' in history else None,
                'final': float(losses[-1]) if 'loss' in history else None,
                'best': float(min(losses)) if 'loss' in history else None
            },
            'similarity_stats': {
                'initial': float(similarities[0]) if 'similarity_scores' in history else None,
                'final': float(similarities[-1]) if 'similarity_scores' in history else None,
                'average': float(np.mean(similarities)) if 'similarity_scores' in history else None
            }
        }, f, indent=2)
    
    logger.info(f"\nAnalysis saved to: {analysis_path}")


if __name__ == "__main__":
    # Check for special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "resume":
            resume_training()
        elif sys.argv[1] == "adjust":
            adjust_parameters()
        elif sys.argv[1] == "analyze":
            analyze_training_history()
        else:
            main()
    else:
        main()