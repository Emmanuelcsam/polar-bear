#!/usr/bin/env python3
"""
Visualization module for Fiber Optics Neural Network System
Visualizes analysis results including segmentation, anomalies, and similarities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
import cv2
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from config_loader import get_config
from logger import get_logger

class FiberOpticsVisualizer:
    """
    Visualizer for fiber optics analysis results
    "the program will spit out an anomaly or defect map"
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing FiberOpticsVisualizer")
        
        self.config = get_config()
        self.logger = get_logger("FiberOpticsVisualizer")
        
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
    
    def visualize_complete_analysis(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of analysis results
        """
        self.logger.log_process_start("Complete Analysis Visualization")
        
        # Load original image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Fiber Optics Analysis Results', fontsize=16)
        
        # 1. Original Image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. Segmentation
        if 'region_probs' in results:
            seg_vis = self._create_segmentation_overlay(image, results['region_probs'][0])
            axes[0, 1].imshow(seg_vis)
            axes[0, 1].set_title('Region Segmentation')
            axes[0, 1].axis('off')
        
        # 3. Anomaly Map
        if 'anomaly_map' in results:
            im3 = axes[0, 2].imshow(results['anomaly_map'][0], cmap=self.anomaly_cmap)
            axes[0, 2].set_title(f'Anomaly Map (max: {results["anomaly_map"][0].max():.3f})')
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
        
        # 4. Reconstruction
        if 'reconstruction' in results:
            recon = results['reconstruction'][0]
            if isinstance(recon, np.ndarray) and recon.shape[0] == 3:
                recon = np.transpose(recon, (1, 2, 0))
            axes[0, 3].imshow(np.clip(recon, 0, 1))
            axes[0, 3].set_title('Reconstruction')
            axes[0, 3].axis('off')
        
        # 5. Trend Adherence
        if 'trend_adherence' in results:
            im5 = axes[1, 0].imshow(results['trend_adherence'][0], cmap=self.quality_cmap)
            axes[1, 0].set_title('Trend Adherence')
            axes[1, 0].axis('off')
            plt.colorbar(im5, ax=axes[1, 0], fraction=0.046)
        
        # 6. Quality Map
        if 'quality_map' in results:
            im6 = axes[1, 1].imshow(results['quality_map'][0], cmap=self.quality_cmap)
            axes[1, 1].set_title('Quality Map')
            axes[1, 1].axis('off')
            plt.colorbar(im6, ax=axes[1, 1], fraction=0.046)
        
        # 7. Reconstruction Error
        if 'reconstruction_error' in results:
            im7 = axes[1, 2].imshow(results['reconstruction_error'][0], cmap='gray')
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
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        self.logger.log_process_end("Complete Analysis Visualization")
    
    def _create_segmentation_overlay(self, image: np.ndarray, 
                                   region_probs: np.ndarray) -> np.ndarray:
        """Create colored overlay for segmentation"""
        # Get region predictions
        if isinstance(region_probs, torch.Tensor):
            region_probs = region_probs.cpu().numpy()
        
        regions = np.argmax(region_probs, axis=0)
        
        # Create colored overlay
        overlay = np.zeros_like(image, dtype=np.float32)
        
        # Apply colors for each region
        for region_idx, (region_name, color) in enumerate(self.region_colors.items()):
            mask = regions == region_idx
            overlay[mask] = color
        
        # Blend with original
        alpha = 0.4
        blended = (1 - alpha) * image / 255.0 + alpha * overlay
        
        return np.clip(blended, 0, 1)
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """Plot training history"""
        self.logger.log_process_start("Plotting Training History")
        
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
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
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
            if 'region_probs' in results:
                regions = np.argmax(results['region_probs'][0], axis=0)
                axes[i, 1].imshow(regions, cmap='tab10')
                axes[i, 1].set_title('Segmentation')
                axes[i, 1].axis('off')
            
            # Anomaly map
            if 'anomaly_map' in results:
                axes[i, 2].imshow(results['anomaly_map'][0], cmap=self.anomaly_cmap)
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
    
    # Visualize
    visualizer.visualize_complete_analysis(dummy_image, dummy_results, 'test_visualization.png')
    
    logger.info("Test visualization saved to test_visualization.png")
    logger.log_process_end("Visualizer Test")
    
    print(f"[{datetime.now()}] Visualizer test completed")
