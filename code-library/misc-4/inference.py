#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference Script for Fiber Optic Quality Assurance CNN
Load trained model and perform inference on new images
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging
import argparse
import matplotlib.pyplot as plt
from PIL import Image

# Import the main architecture components
from fiber_cnn_pure import (
    AttentionGate, MBConvBlock, FiberEncoder, FiberDecoder,
    CombinedLoss, FiberAnalysisNet
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FiberInference:
    """Inference class for fiber optic quality assessment"""
    
    def __init__(self, model_path, device='cuda', image_size=512):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup transforms
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Quality labels
        self.quality_labels = ['pass', 'warning', 'fail']
        
        # Zone labels
        self.zone_labels = ['core', 'cladding', 'ferrule']
        
        # Defect labels
        self.defect_labels = ['scratches', 'pits', 'contamination', 'edge_defects']
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
        
        # Load state dict
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return model.to(self.device)
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return image_tensor, image
    
    def predict(self, image_path):
        """Perform inference on a single image"""
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Process outputs
        results = self._process_outputs(outputs, original_image)
        
        return results
    
    def _process_outputs(self, outputs, original_image):
        """Process model outputs into interpretable results"""
        # Get predictions
        zones_logits = outputs['zones']
        defects_logits = outputs['defects']
        quality_logits = outputs['quality']
        
        # Convert to probabilities
        zones_probs = F.softmax(zones_logits, dim=1)
        defects_probs = torch.sigmoid(defects_logits)
        quality_probs = F.softmax(quality_logits, dim=1)
        
        # Get predictions
        zones_pred = torch.argmax(zones_probs, dim=1)
        quality_pred = torch.argmax(quality_probs, dim=1)
        defects_pred = (defects_probs > 0.5).float()
        
        # Convert to numpy for processing
        zones_pred = zones_pred.cpu().numpy()[0]  # Remove batch dimension
        defects_pred = defects_pred.cpu().numpy()[0]
        quality_pred = quality_pred.cpu().numpy()[0]
        
        # Create results dictionary
        results = {
            'quality': {
                'prediction': self.quality_labels[quality_pred],
                'confidence': quality_probs[0, quality_pred].item(),
                'probabilities': quality_probs[0].cpu().numpy()
            },
            'zones': {
                'prediction': zones_pred,
                'probabilities': zones_probs[0].cpu().numpy(),
                'labels': self.zone_labels
            },
            'defects': {
                'predictions': defects_pred,
                'probabilities': defects_probs[0].cpu().numpy(),
                'labels': self.defect_labels,
                'defect_count': int(defects_pred.sum())
            },
            'original_image': original_image
        }
        
        return results
    
    def visualize_results(self, results, save_path=None):
        """Visualize inference results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Fiber Optic Quality Assessment Results', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(results['original_image'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Zone segmentation
        zones_pred = results['zones']['prediction']
        axes[0, 1].imshow(zones_pred, cmap='tab10')
        axes[0, 1].set_title('Zone Segmentation')
        axes[0, 1].axis('off')
        
        # Defect detection
        defects_pred = results['defects']['predictions']
        defect_map = np.sum(defects_pred, axis=0)
        axes[0, 2].imshow(defect_map, cmap='hot')
        axes[0, 2].set_title('Defect Detection')
        axes[0, 2].axis('off')
        
        # Quality assessment
        quality_probs = results['quality']['probabilities']
        axes[1, 0].bar(self.quality_labels, quality_probs)
        axes[1, 0].set_title('Quality Assessment')
        axes[1, 0].set_ylabel('Probability')
        
        # Zone probabilities
        zone_probs = results['zones']['probabilities']
        axes[1, 1].bar(self.zone_labels, zone_probs)
        axes[1, 1].set_title('Zone Probabilities')
        axes[1, 1].set_ylabel('Probability')
        
        # Defect probabilities
        defect_probs = results['defects']['probabilities']
        axes[1, 2].bar(self.defect_labels, defect_probs)
        axes[1, 2].set_title('Defect Probabilities')
        axes[1, 2].set_ylabel('Probability')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, results):
        """Generate a text report of the analysis"""
        report = []
        report.append("=" * 50)
        report.append("FIBER OPTIC QUALITY ASSESSMENT REPORT")
        report.append("=" * 50)
        
        # Quality assessment
        quality = results['quality']
        report.append(f"\nQUALITY ASSESSMENT:")
        report.append(f"  Prediction: {quality['prediction'].upper()}")
        report.append(f"  Confidence: {quality['confidence']:.3f}")
        
        # Zone analysis
        zones = results['zones']
        report.append(f"\nZONE ANALYSIS:")
        for i, (label, prob) in enumerate(zip(zones['labels'], zones['probabilities'])):
            report.append(f"  {label.capitalize()}: {prob:.3f}")
        
        # Defect analysis
        defects = results['defects']
        report.append(f"\nDEFECT ANALYSIS:")
        report.append(f"  Total Defects Detected: {defects['defect_count']}")
        for i, (label, prob) in enumerate(zip(defects['labels'], defects['probabilities'])):
            status = "DETECTED" if defects['predictions'][i] else "CLEAN"
            report.append(f"  {label.capitalize()}: {status} (confidence: {prob:.3f})")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        if quality['prediction'] == 'pass':
            report.append("  ✓ Fiber optic meets quality standards")
        elif quality['prediction'] == 'warning':
            report.append("  ⚠ Minor issues detected - recommend inspection")
        else:
            report.append("  ✗ Quality issues detected - recommend rejection")
        
        report.append("=" * 50)
        
        return "\n".join(report)

def main():
    """Main inference function"""
    
    parser = argparse.ArgumentParser(description='Fiber Optic Quality Assurance Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--image-size', type=int, default=512, help='Input image size')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--save-report', action='store_true', help='Save text report')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    inference = FiberInference(
        model_path=args.model_path,
        device=args.device,
        image_size=args.image_size
    )
    
    # Perform inference
    logger.info(f"Processing image: {args.image_path}")
    results = inference.predict(args.image_path)
    
    # Generate report
    report = inference.generate_report(results)
    print(report)
    
    # Save report if requested
    if args.save_report:
        report_path = os.path.join(args.output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
    
    # Generate visualization if requested
    if args.visualize:
        viz_path = os.path.join(args.output_dir, 'visualization.png')
        inference.visualize_results(results, save_path=viz_path)
    
    # Save results as JSON
    json_path = os.path.join(args.output_dir, 'results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'quality': results['quality'],
        'zones': {
            'prediction': results['zones']['prediction'].tolist(),
            'probabilities': results['zones']['probabilities'].tolist(),
            'labels': results['zones']['labels']
        },
        'defects': {
            'predictions': results['defects']['predictions'].tolist(),
            'probabilities': results['defects']['probabilities'].tolist(),
            'labels': results['defects']['labels'],
            'defect_count': results['defects']['defect_count']
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {json_path}")

if __name__ == "__main__":
    main() 