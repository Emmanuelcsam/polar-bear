#!/usr/bin/env python3
"""
Visualization module for Fiber Optics Neural Network System
Visualizes analysis results including segmentation, anomalies, and similarities
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
import cv2
from typing import Dict, Optional, List, Tuple, Union
from datetime import datetime

from config_loader import get_config
from logger import get_logger

class FiberOpticsVisualizer:
    """
    Visualizer for fiber optics analysis results
    "the program will spit out an anomaly or defect map"
    """
    
    def __init__(self, use_interactive: bool = False):
        print(f"[{datetime.now()}] Initializing FiberOpticsVisualizer")
        
        self.config = get_config()
        self.logger = get_logger("FiberOpticsVisualizer")
        
        # Set matplotlib backend
        if use_interactive:
            try:
                matplotlib.use('TkAgg')  # Interactive backend
                self.interactive = True
            except:
                self.logger.warning("Failed to set interactive backend, using non-interactive")
                self.interactive = False
        else:
            self.interactive = False
        
        self.logger.log_class_init("FiberOpticsVisualizer")
        
        # Color maps for different visualizations
        self.region_colors = {
            'core': [1.0, 0.0, 0.0],      # Red
            'cladding': [0.0, 1.0, 0.0],   # Green
            'ferrule': [0.0, 0.0, 1.0]     # Blue
        }
        
        self.anomaly_cmap = plt.cm.hot
        self.quality_cmap = plt.cm.viridis
        
        print(f"[{datetime.now()}] FiberOpticsVisualizer initialized")
    
    def visualize_config_overview(self, save_path: Optional[str] = None):
        """
        Visualize comprehensive configuration overview
        """
        self.logger.log_process_start("Configuration Overview Visualization")
        
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Fiber Optics Neural Network Configuration Overview', fontsize=20, fontweight='bold')
        
        # 1. System Configuration
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        system_text = f"System Configuration\n"
        system_text += f"{'='*50}\n"
        system_text += f"Device: {self.config.system.device}\n"
        system_text += f"GPU ID: {self.config.system.gpu_id}\n"
        system_text += f"Workers: {self.config.system.num_workers}\n"
        system_text += f"Log Level: {self.config.system.log_level}\n"
        system_text += f"Data Path: {self.config.system.data_path}\n"
        system_text += f"Checkpoints: {self.config.system.checkpoints_path}\n"
        system_text += f"Results: {self.config.system.results_path}\n"
        ax1.text(0.05, 0.95, system_text, transform=ax1.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 2. Model Architecture
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        model_text = f"Model Architecture\n"
        model_text += f"{'='*30}\n"
        model_text += f"Architecture: {self.config.model.architecture}\n"
        model_text += f"Input Channels: {self.config.model.input_channels}\n"
        model_text += f"Base Channels: {self.config.model.base_channels}\n"
        model_text += f"Blocks: {self.config.model.num_blocks}\n"
        model_text += f"SE Blocks: {self.config.model.use_se_blocks}\n"
        model_text += f"CBAM: {self.config.model.use_cbam}\n"
        model_text += f"Deformable Conv: {self.config.model.use_deformable_conv}\n"
        model_text += f"Embedding Dim: {self.config.model.embedding_dim}\n"
        ax2.text(0.05, 0.95, model_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 3. Equation Coefficients
        ax3 = fig.add_subplot(gs[1, 1])
        coeffs = self.config.equation.coefficients
        labels = list(coeffs.keys())
        values = list(coeffs.values())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
        
        bars = ax3.bar(labels, values, color=colors)
        ax3.set_title('Equation Coefficients', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Value')
        ax3.set_ylim(min(0, min(values) - 0.2), max(values) + 0.2)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 4. Training Configuration
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        train_text = f"Training Configuration\n"
        train_text += f"{'='*30}\n"
        train_text += f"Epochs: {self.config.training.num_epochs}\n"
        train_text += f"Batch Size: {self.config.training.batch_size}\n"
        train_text += f"Validation Split: {self.config.training.validation_split}\n"
        train_text += f"Early Stopping: {self.config.training.early_stopping_patience}\n"
        train_text += f"Use AMP: {self.config.training.use_amp}\n"
        train_text += f"Gradient Clip: {self.config.training.gradient_clip_value}\n"
        ax4.text(0.05, 0.95, train_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # 5. Loss Weights
        ax5 = fig.add_subplot(gs[2, :2])
        loss_weights = self.config.loss.weights
        labels = list(loss_weights.keys())
        values = list(loss_weights.values())
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax5.pie(values, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        ax5.set_title('Loss Component Weights', fontsize=12, fontweight='bold')
        
        # 6. Optimizer Configuration
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        opt_text = f"Optimizer Configuration\n"
        opt_text += f"{'='*30}\n"
        opt_text += f"Type: {self.config.optimizer.type}\n"
        opt_text += f"Learning Rate: {self.config.optimizer.learning_rate}\n"
        opt_text += f"Weight Decay: {self.config.optimizer.weight_decay}\n"
        opt_text += f"Betas: {self.config.optimizer.betas}\n"
        opt_text += f"Scheduler: {self.config.optimizer.scheduler.type}\n"
        opt_text += f"Min LR: {self.config.optimizer.scheduler.min_lr}\n"
        ax6.text(0.05, 0.95, opt_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # 7. Runtime Configuration
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.axis('off')
        runtime_text = f"Runtime Configuration\n"
        runtime_text += f"{'='*30}\n"
        runtime_text += f"Mode: {self.config.runtime.mode}\n"
        runtime_text += f"Distributed: {self.config.runtime.distributed}\n"
        runtime_text += f"Benchmark: {self.config.runtime.benchmark}\n"
        runtime_text += f"Results Dir: {self.config.runtime.results_dir}\n"
        runtime_text += f"Max Batch: {self.config.runtime.max_batch_images}\n"
        ax7.text(0.05, 0.95, runtime_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        # 8. Anomaly Detection Thresholds
        ax8 = fig.add_subplot(gs[3, 1])
        anomaly_params = {
            'Threshold': self.config.anomaly.threshold,
            'Min Size': self.config.anomaly.min_defect_size,
            'Max Size': self.config.anomaly.max_defect_size,
            'Confidence': self.config.anomaly.confidence_threshold
        }
        
        bars = ax8.bar(anomaly_params.keys(), anomaly_params.values(), 
                       color=['red', 'orange', 'yellow', 'green'])
        ax8.set_title('Anomaly Detection Parameters', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Value')
        ax8.set_yscale('log')
        ax8.grid(True, alpha=0.3)
        
        # 9. Similarity Metrics Configuration
        ax9 = fig.add_subplot(gs[3, 2])
        sim_weights = self.config.similarity.combination_weights
        labels = list(sim_weights.keys())
        sizes = list(sim_weights.values())
        
        colors = ['gold', 'lightcoral', 'lightskyblue']
        ax9.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=140)
        ax9.set_title('Similarity Metric Weights', fontsize=12, fontweight='bold')
        
        # 10. Feature Extraction Settings
        ax10 = fig.add_subplot(gs[4, :])
        ax10.axis('off')
        feature_text = f"Feature Extraction Configuration\n"
        feature_text += f"{'='*60}\n"
        feature_text += f"Scales: {self.config.features.scales} (Weights: {self.config.features.scale_weights})\n"
        feature_text += f"Gradient Kernel Size: {self.config.features.gradient_kernel_size}\n"
        feature_text += f"Position Encoding: {self.config.features.use_position_encoding} (Dim: {self.config.features.position_encoding_dim})\n"
        feature_text += f"Trend Window: {self.config.features.trend_window_size} (Poly Degree: {self.config.features.trend_polynomial_degree})\n"
        ax10.text(0.05, 0.95, feature_text, transform=ax10.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender", alpha=0.8))
        
        # 11. Performance Settings
        ax11 = fig.add_subplot(gs[5, :])
        ax11.axis('off')
        perf_text = f"Performance & Optimization Settings\n"
        perf_text += f"{'='*60}\n"
        perf_text += f"Target FPS: {self.config.realtime.target_fps} | "
        perf_text += f"Max Batch Size: {self.config.realtime.max_batch_size} | "
        perf_text += f"Dynamic Batching: {self.config.realtime.dynamic_batching}\n"
        perf_text += f"Gradient Accumulation: {self.config.optimization.gradient_accumulation_steps} | "
        perf_text += f"Mixed Precision: {self.config.optimization.mixed_precision_training}\n"
        perf_text += f"Flash Attention: {self.config.optimization.flash_attention} | "
        perf_text += f"Compile Model: {self.config.optimization.compile_model} | "
        perf_text += f"Distributed Backend: {self.config.optimization.distributed_backend}\n"
        ax11.text(0.05, 0.95, perf_text, transform=ax11.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose", alpha=0.8))
        
        plt.tight_layout()
        self._save_or_show(save_path)
        plt.close()
        
        self.logger.log_process_end("Configuration Overview Visualization")
    
    def visualize_equation_components(self, results: Dict, save_path: Optional[str] = None):
        """
        Visualize equation components and their contributions
        """
        self.logger.log_process_start("Equation Components Visualization")
        
        if 'equation_info' not in results:
            self.logger.warning("No equation info in results")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Component values
        components = results['equation_info']['components']
        labels = list(components.keys())
        values = list(components.values())
        
        # Bar chart
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        bars = ax1.barh(labels, values, color=colors)
        ax1.set_xlabel('Component Value')
        ax1.set_title('Equation Component Values', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{value:.3f}', ha='left', va='center')
        
        # Contribution pie chart
        if 'weighted_components' in results['equation_info']:
            weighted = results['equation_info']['weighted_components']
            labels2 = list(weighted.keys())
            sizes = list(weighted.values())
            
            # Normalize to percentages
            total = sum(sizes)
            percentages = [s/total * 100 for s in sizes]
            
            colors2 = plt.cm.tab10(np.linspace(0, 1, len(labels2)))
            wedges, texts, autotexts = ax2.pie(percentages, labels=labels2, 
                                               colors=colors2, autopct='%1.1f%%',
                                               startangle=90)
            ax2.set_title('Component Contributions', fontweight='bold')
        
        # Overall equation
        if 'final_score' in results['equation_info']:
            equation_text = f"Final Score = {results['equation_info']['final_score']:.4f}"
            if 'equation_string' in results['equation_info']:
                equation_text = f"{results['equation_info']['equation_string']}\n{equation_text}"
            fig.suptitle(equation_text, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_or_show(save_path)
        plt.close()
        
        self.logger.log_process_end("Equation Components Visualization")
    
    def _save_or_show(self, save_path: Optional[str] = None, dpi: int = 300):
        """Helper method to save or show plot depending on backend and save_path"""
        if save_path:
            try:
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                self.logger.info(f"Visualization saved to: {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save visualization: {e}")
        elif self.interactive:
            try:
                plt.show()
            except Exception as e:
                self.logger.warning(f"Failed to show plot interactively: {e}")
        else:
            self.logger.warning("Non-interactive mode and no save path provided - visualization not displayed")
    
    def visualize_complete_analysis(self, image_path: Union[str, np.ndarray], results: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of analysis results
        """
        self.logger.log_process_start("Complete Analysis Visualization")
        
        try:
            # Load original image
            if isinstance(image_path, str):
                if not Path(image_path).exists():
                    self.logger.error(f"Image file not found: {image_path}")
                    return
                image = cv2.imread(image_path)
                if image is None:
                    self.logger.error(f"Failed to load image: {image_path}")
                    return
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path
                if not isinstance(image, np.ndarray):
                    self.logger.error(f"Invalid image type: {type(image)}")
                    return
        
            # Create figure with subplots
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Fiber Optics Analysis Results', fontsize=16)
        
            # 1. Original Image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
        
            # 2. Segmentation
            if 'region_probs' in results and len(results['region_probs']) > 0:
                seg_vis = self._create_segmentation_overlay(image, results['region_probs'][0])
                axes[0, 1].imshow(seg_vis)
                axes[0, 1].set_title('Region Segmentation')
                axes[0, 1].axis('off')
            else:
                axes[0, 1].text(0.5, 0.5, 'No segmentation data', ha='center', va='center')
                axes[0, 1].axis('off')
        
            # 3. Anomaly Map
            if 'anomaly_map' in results and len(results['anomaly_map']) > 0:
                anomaly_data = results['anomaly_map'][0]
                if isinstance(anomaly_data, torch.Tensor):
                    anomaly_data = anomaly_data.cpu().numpy()
                im3 = axes[0, 2].imshow(anomaly_data, cmap=self.anomaly_cmap)
                axes[0, 2].set_title(f'Anomaly Map (max: {anomaly_data.max():.3f})')
                axes[0, 2].axis('off')
                plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
            else:
                axes[0, 2].text(0.5, 0.5, 'No anomaly data', ha='center', va='center')
                axes[0, 2].axis('off')
        
            # 4. Reconstruction
            if 'reconstruction' in results and len(results['reconstruction']) > 0:
                recon = results['reconstruction'][0]
                if isinstance(recon, torch.Tensor):
                    recon = recon.cpu().numpy()
                if isinstance(recon, np.ndarray) and recon.ndim == 3:
                    if recon.shape[0] == 3:  # CHW format
                        recon = np.transpose(recon, (1, 2, 0))
                    elif recon.shape[2] != 3:  # Ensure it's HWC with 3 channels
                        self.logger.warning(f"Unexpected reconstruction shape: {recon.shape}")
                axes[0, 3].imshow(np.clip(recon, 0, 1))
                axes[0, 3].set_title('Reconstruction')
                axes[0, 3].axis('off')
            else:
                axes[0, 3].text(0.5, 0.5, 'No reconstruction', ha='center', va='center')
                axes[0, 3].axis('off')
        
            # 5. Trend Adherence
            if 'trend_adherence' in results and len(results['trend_adherence']) > 0:
                trend_data = results['trend_adherence'][0]
                if isinstance(trend_data, torch.Tensor):
                    trend_data = trend_data.cpu().numpy()
                im5 = axes[1, 0].imshow(trend_data, cmap=self.quality_cmap)
                axes[1, 0].set_title('Trend Adherence')
                axes[1, 0].axis('off')
                plt.colorbar(im5, ax=axes[1, 0], fraction=0.046)
        
            # 6. Quality Map
            if 'quality_map' in results and len(results['quality_map']) > 0:
                quality_data = results['quality_map'][0]
                if isinstance(quality_data, torch.Tensor):
                    quality_data = quality_data.cpu().numpy()
                im6 = axes[1, 1].imshow(quality_data, cmap=self.quality_cmap)
                axes[1, 1].set_title('Quality Map')
                axes[1, 1].axis('off')
                plt.colorbar(im6, ax=axes[1, 1], fraction=0.046)
        
            # 7. Reconstruction Error
            if 'reconstruction_error' in results and len(results['reconstruction_error']) > 0:
                error_data = results['reconstruction_error'][0]
                if isinstance(error_data, torch.Tensor):
                    error_data = error_data.cpu().numpy()
                im7 = axes[1, 2].imshow(error_data, cmap='gray')
                axes[1, 2].set_title('Reconstruction Error')
                axes[1, 2].axis('off')
                plt.colorbar(im7, ax=axes[1, 2], fraction=0.046)
        
            # 8. Summary Statistics
            axes[1, 3].text(0.1, 0.9, 'Analysis Summary:', fontsize=14, weight='bold', 
                           transform=axes[1, 3].transAxes)
            
            summary_text = ""
            if 'summary' in results:
                summary = results['summary']
                summary_text += f"Final Similarity: {summary['final_similarity_score']:.4f}\n"
                summary_text += f"Meets Threshold: {'Yes' if summary['meets_threshold'] else 'No'}\n"
                summary_text += f"Primary Region: {summary['primary_region']}\n"
                summary_text += f"Anomaly Score: {summary['anomaly_score']:.4f}\n"
                summary_text += f"Max Anomaly: {summary['max_anomaly_score']:.4f}\n"
                summary_text += f"Reconstruction Error: {summary['reconstruction_error']:.4f}\n"
            
            if 'equation_info' in results:
                summary_text += "\nEquation Components:\n"
                for comp, value in results['equation_info']['components'].items():
                    summary_text += f"  {comp}: {value:.4f}\n"
            
            axes[1, 3].text(0.1, 0.8, summary_text, fontsize=10, 
                           transform=axes[1, 3].transAxes, verticalalignment='top')
            axes[1, 3].axis('off')
        
            plt.tight_layout()
            
            # Save or show
            if save_path:
                try:
                    save_dir = Path(save_path).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"Visualization saved to: {save_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save visualization: {e}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error in visualization: {e}")
            if 'fig' in locals():
                plt.close(fig)
        
        self.logger.log_process_end("Complete Analysis Visualization")
    
    def _create_segmentation_overlay(self, image: np.ndarray, 
                                   region_probs: np.ndarray) -> np.ndarray:
        """Create colored overlay for segmentation"""
        try:
            # Get region predictions
            if isinstance(region_probs, torch.Tensor):
                region_probs = region_probs.cpu().numpy()
            
            if region_probs.ndim != 3:
                self.logger.error(f"Invalid region_probs shape: {region_probs.shape}")
                return image / 255.0 if image.dtype == np.uint8 else image
            
            regions = np.argmax(region_probs, axis=0)
            
            # Create colored overlay
            overlay = np.zeros_like(image, dtype=np.float32)
            
            # Create consistent region index mapping
            region_names = list(self.region_colors.keys())
            
            # Apply colors for each region
            for region_idx, region_name in enumerate(region_names):
                if region_idx < region_probs.shape[0]:  # Ensure index is valid
                    mask = regions == region_idx
                    overlay[mask] = self.region_colors[region_name]
            
            # Normalize image to [0, 1] range
            if image.dtype == np.uint8:
                image_norm = image / 255.0
            else:
                image_norm = np.clip(image, 0, 1)
            
            # Blend with original
            alpha = 0.4
            blended = (1 - alpha) * image_norm + alpha * overlay
            
            return np.clip(blended, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error creating segmentation overlay: {e}")
            return image / 255.0 if image.dtype == np.uint8 else image
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """Plot training history"""
        self.logger.log_process_start("Plotting Training History")
        
        # Validate required keys
        required_keys = ['train_loss', 'val_loss', 'train_similarity', 'val_similarity', 
                        'learning_rates', 'epoch_times']
        missing_keys = [k for k in required_keys if k not in history]
        if missing_keys:
            self.logger.error(f"Missing required history keys: {missing_keys}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # 1. Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Similarity curves
        axes[0, 1].plot(history['train_similarity'], label='Train Similarity')
        axes[0, 1].plot(history['val_similarity'], label='Val Similarity')
        axes[0, 1].axhline(y=0.7, color='r', linestyle='--', label='Threshold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Similarity')
        axes[0, 1].set_title('Similarity Scores')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Learning rate
        axes[1, 0].plot(history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # 4. Epoch times
        axes[1, 1].plot(history['epoch_times'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Epoch Training Time')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        self.logger.log_process_end("Plotting Training History")
    
    def visualize_anomaly_details(self, image: np.ndarray, anomaly_map: np.ndarray,
                                 defects: List[Dict], save_path: Optional[str] = None):
        """Visualize detailed anomaly information"""
        try:
            # Validate inputs
            if not isinstance(image, np.ndarray):
                self.logger.error(f"Invalid image type: {type(image)}")
                return
            if not isinstance(anomaly_map, np.ndarray):
                self.logger.error(f"Invalid anomaly_map type: {type(anomaly_map)}")
                return
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
            # Original with defect boxes
            ax1.imshow(image)
            ax1.set_title('Detected Defects')
        
            # Draw bounding boxes
            for defect in defects:
                bbox = defect['bounding_box']
                rect = patches.Rectangle((bbox[0], bbox[1]), 
                                       bbox[2] - bbox[0], 
                                       bbox[3] - bbox[1],
                                       linewidth=2, 
                                       edgecolor='r', 
                                       facecolor='none')
                ax1.add_patch(rect)
                
                # Add label
                ax1.text(bbox[0], bbox[1] - 5, 
                        f"{defect['type']}\n{defect['confidence']:.2f}",
                        color='white', 
                        backgroundcolor='red',
                        fontsize=8)
            
            ax1.axis('off')
        
            # Anomaly heatmap
            im = ax2.imshow(anomaly_map, cmap=self.anomaly_cmap)
            ax2.set_title('Anomaly Heatmap')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046)
            
            plt.tight_layout()
            
            if save_path:
                try:
                    save_dir = Path(save_path).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"Anomaly visualization saved to: {save_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save anomaly visualization: {e}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error in anomaly visualization: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    def create_comparison_grid(self, results_list: List[Dict], 
                              image_names: List[str],
                              save_path: Optional[str] = None):
        """Create grid comparing multiple analysis results"""
        n_images = len(results_list)
        fig, axes = plt.subplots(n_images, 4, figsize=(16, 4 * n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for i, (results, name) in enumerate(zip(results_list, image_names)):
            # Image name
            axes[i, 0].text(0.5, 0.5, name, fontsize=12, ha='center', va='center')
            axes[i, 0].axis('off')
            
            # Segmentation
            if 'region_probs' in results and len(results['region_probs']) > 0:
                region_data = results['region_probs'][0]
                if isinstance(region_data, torch.Tensor):
                    region_data = region_data.cpu().numpy()
                regions = np.argmax(region_data, axis=0)
                axes[i, 1].imshow(regions, cmap='tab10')
                axes[i, 1].set_title('Segmentation')
                axes[i, 1].axis('off')
            
            # Anomaly map
            if 'anomaly_map' in results and len(results['anomaly_map']) > 0:
                anomaly_data = results['anomaly_map'][0]
                if isinstance(anomaly_data, torch.Tensor):
                    anomaly_data = anomaly_data.cpu().numpy()
                axes[i, 2].imshow(anomaly_data, cmap=self.anomaly_cmap)
                axes[i, 2].set_title('Anomalies')
                axes[i, 2].axis('off')
            
            # Similarity score
            if 'summary' in results:
                score = results['summary']['final_similarity_score']
                color = 'green' if score > 0.7 else 'red'
                axes[i, 3].text(0.5, 0.5, f'{score:.4f}', 
                              fontsize=24, ha='center', va='center', color=color)
                axes[i, 3].set_title('Similarity')
                axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_config_diff(self, config_dict: Dict, reference_dict: Dict = None, 
                             save_path: Optional[str] = None):
        """
        Visualize configuration differences or highlights
        """
        self.logger.log_process_start("Configuration Diff Visualization")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Title
        title = "Configuration Parameters"
        if reference_dict:
            title = "Configuration Differences"
        ax.text(0.5, 0.98, title, fontsize=16, fontweight='bold', 
                ha='center', va='top', transform=ax.transAxes)
        
        # Flatten config dictionary
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_config = flatten_dict(config_dict)
        flat_ref = flatten_dict(reference_dict) if reference_dict else {}
        
        # Create text display
        y_pos = 0.95
        line_height = 0.02
        
        for key in sorted(flat_config.keys()):
            value = flat_config[key]
            
            # Determine color based on diff
            color = 'black'
            marker = ''
            if reference_dict:
                if key not in flat_ref:
                    color = 'green'
                    marker = '+ '  # New parameter
                elif flat_ref[key] != value:
                    color = 'orange'
                    marker = '* '  # Modified parameter
                    value = f"{value} (was: {flat_ref[key]})"
            
            # Display parameter
            text = f"{marker}{key}: {value}"
            ax.text(0.05, y_pos, text, fontsize=9, color=color,
                   fontfamily='monospace', transform=ax.transAxes)
            
            y_pos -= line_height
            if y_pos < 0.05:
                break
        
        # Add legend if showing diff
        if reference_dict:
            legend_text = "Legend: + New | * Modified | Unchanged"
            ax.text(0.95, 0.02, legend_text, fontsize=10, 
                   ha='right', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        self._save_or_show(save_path)
        plt.close()
        
        self.logger.log_process_end("Configuration Diff Visualization")
    
    def generate_config_report(self, save_path: str = "config_report.html"):
        """
        Generate comprehensive HTML report of configuration
        """
        self.logger.log_process_start("Configuration Report Generation")
        
        html_content = """
        <html>
        <head>
            <title>Fiber Optics Neural Network Configuration Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: bold; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .section { margin: 20px 0; }
                .highlight { background-color: #ffffcc; }
                .warning { color: #ff6600; }
                .info { color: #0066cc; }
            </style>
        </head>
        <body>
            <h1>Fiber Optics Neural Network Configuration Report</h1>
            <p>Generated: {}</p>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # System Configuration
        html_content += """
            <div class="section">
                <h2>System Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
        """
        
        system_params = {
            'Device': (self.config.system.device, 'Compute device (cuda/cpu)'),
            'GPU ID': (self.config.system.gpu_id, 'GPU device ID'),
            'Workers': (self.config.system.num_workers, 'Number of data loading workers'),
            'Log Level': (self.config.system.log_level, 'Logging verbosity'),
            'Data Path': (self.config.system.data_path, 'Dataset location'),
            'Checkpoints': (self.config.system.checkpoints_path, 'Model checkpoint directory'),
            'Results': (self.config.system.results_path, 'Results output directory'),
        }
        
        for param, (value, desc) in system_params.items():
            html_content += f"<tr><td>{param}</td><td>{value}</td><td>{desc}</td></tr>"
        
        html_content += "</table></div>"
        
        # Model Architecture
        html_content += """
            <div class="section">
                <h2>Model Architecture</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Impact</th></tr>
        """
        
        model_params = {
            'Architecture': (self.config.model.architecture, 'Model complexity level'),
            'Base Channels': (self.config.model.base_channels, 'Model capacity (higher = more parameters)'),
            'Blocks': (str(self.config.model.num_blocks), 'Network depth configuration'),
            'SE Blocks': (self.config.model.use_se_blocks, 'Squeeze-and-Excitation attention'),
            'CBAM': (self.config.model.use_cbam, 'Convolutional Block Attention Module'),
            'Deformable Conv': (self.config.model.use_deformable_conv, 'Adaptive receptive fields'),
            'Embedding Dim': (self.config.model.embedding_dim, 'Feature representation size'),
        }
        
        for param, (value, impact) in model_params.items():
            html_content += f"<tr><td>{param}</td><td>{value}</td><td>{impact}</td></tr>"
        
        html_content += "</table></div>"
        
        # Training Configuration
        html_content += """
            <div class="section">
                <h2>Training Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Recommendation</th></tr>
        """
        
        train_params = {
            'Epochs': (self.config.training.num_epochs, 'Increase for better convergence'),
            'Batch Size': (self.config.training.batch_size, 'Larger = faster but more memory'),
            'Learning Rate': (self.config.optimizer.learning_rate, 'Start high, decrease if unstable'),
            'Optimizer': (self.config.optimizer.type, 'AdamW recommended for stability'),
            'Mixed Precision': (self.config.training.use_amp, 'Enable for 2x speedup'),
            'Gradient Accumulation': (self.config.optimization.gradient_accumulation_steps, 'Simulate larger batches'),
        }
        
        for param, (value, rec) in train_params.items():
            html_content += f"<tr><td>{param}</td><td>{value}</td><td>{rec}</td></tr>"
        
        html_content += "</table></div>"
        
        # Performance Analysis
        html_content += """
            <div class="section">
                <h2>Performance Analysis</h2>
                <ul>
        """
        
        # Calculate estimated parameters
        base_ch = self.config.model.base_channels
        blocks = sum(self.config.model.num_blocks)
        est_params = base_ch * blocks * 1000  # Rough estimate
        
        html_content += f"<li>Estimated Parameters: ~{est_params:,}</li>"
        html_content += f"<li>Memory Usage (BS={self.config.training.batch_size}): ~{est_params * self.config.training.batch_size / 1e9:.1f}GB</li>"
        html_content += f"<li>Target FPS: {self.config.realtime.target_fps}</li>"
        
        if self.config.optimization.mixed_precision_training:
            html_content += '<li class="info">Mixed Precision Training: ENABLED (2x speedup expected)</li>'
        else:
            html_content += '<li class="warning">Mixed Precision Training: DISABLED (enable for better performance)</li>'
        
        html_content += "</ul></div>"
        
        # Equation Configuration
        html_content += """
            <div class="section">
                <h2>Equation Configuration</h2>
                <p>Master Equation: S(R) = A×x₁ + B×x₂ + C×x₃ + D×x₄ + E×x₅</p>
                <table>
                    <tr><th>Component</th><th>Weight</th><th>Meaning</th></tr>
        """
        
        equation_components = {
            'A': ('Reference Similarity', 'Comparison with reference fibers'),
            'B': ('Trend Adherence', 'Consistency with expected patterns'),
            'C': ('Inverse Anomaly Score', 'Quality/defect detection'),
            'D': ('Segmentation Confidence', 'Region detection accuracy'),
            'E': ('Reconstruction Similarity', 'Feature preservation'),
        }
        
        for coef, (name, meaning) in equation_components.items():
            weight = self.config.equation.coefficients[coef]
            html_content += f"<tr><td>{coef} - {name}</td><td>{weight}</td><td>{meaning}</td></tr>"
        
        html_content += "</table></div>"
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        try:
            with open(save_path, 'w') as f:
                f.write(html_content)
            self.logger.info(f"Configuration report saved to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration report: {e}")
        
        self.logger.log_process_end("Configuration Report Generation")

# Example usage
if __name__ == "__main__":
    visualizer = FiberOpticsVisualizer()
    logger = get_logger("VisualizerTest")
    
    logger.log_process_start("Visualizer Test")
    
    # Create dummy results for testing
    dummy_results = {
        'region_probs': np.random.rand(1, 3, 256, 256),
        'anomaly_map': np.random.rand(1, 256, 256) * 0.5,
        'trend_adherence': np.random.rand(1, 256, 256),
        'quality_map': np.random.rand(1, 256, 256),
        'reconstruction': np.random.rand(1, 3, 256, 256),
        'reconstruction_error': np.random.rand(1, 256, 256),
        'summary': {
            'final_similarity_score': 0.85,
            'meets_threshold': True,
            'primary_region': 'core',
            'anomaly_score': 0.12,
            'max_anomaly_score': 0.45,
            'reconstruction_error': 0.08
        },
        'equation_info': {
            'components': {
                'reference_similarity': 0.88,
                'trend_adherence': 0.91,
                'anomaly_inverse': 0.88,
                'segmentation_confidence': 0.95,
                'reconstruction_similarity': 0.92
            }
        }
    }
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Visualize complete analysis
    visualizer.visualize_complete_analysis(dummy_image, dummy_results, 'test_visualization.png')
    
    # Test configuration overview visualization
    logger.info("Testing configuration overview visualization...")
    visualizer.visualize_config_overview('config_overview.png')
    
    # Test equation components visualization
    logger.info("Testing equation components visualization...")
    visualizer.visualize_equation_components(dummy_results, 'equation_components.png')
    
    # Test configuration report generation
    logger.info("Testing configuration report generation...")
    visualizer.generate_config_report('config_report.html')
    
    # Test configuration diff visualization
    logger.info("Testing configuration diff visualization...")
    test_config = {'model': {'base_channels': 128}, 'training': {'epochs': 200}}
    ref_config = {'model': {'base_channels': 64}, 'training': {'epochs': 100}}
    visualizer.visualize_config_diff(test_config, ref_config, 'config_diff.png')
    
    logger.info("All visualization tests completed")
    logger.log_process_end("Visualizer Test")
    
    print(f"[{datetime.now()}] Visualizer test completed - check generated files")
