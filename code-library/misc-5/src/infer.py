# infer.py
# Production Inference Script for Fiber Optic End-Face CNN
# Processes images and outputs region masks + defect classifications

import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from PIL import Image

from src.model import EndfaceNet
from src.dataset import build_default_transforms

def load_model(checkpoint_path: str, num_classes: int = 40, device: str = 'cuda') -> EndfaceNet:
    """Load trained model from checkpoint."""
    model = EndfaceNet(num_classes=num_classes)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    return model

def process_image(model: EndfaceNet, image_path: str, device: str = 'cuda') -> Dict:
    """Process a single image and return predictions."""
    
    # Load and preprocess image
    transforms = build_default_transforms(train=False, img_size=256)
    
    # Load image
    if image_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image format: {image_path}")
    
    # Apply transforms
    sample = transforms(image=img)
    tensor = sample["image"].unsqueeze(0).to(device)  # Add batch dimension
    
    # Inference
    with torch.no_grad():
        mask_logits, defect_logits, stat_feats = model(tensor)
    
    # Process predictions
    mask_probs = torch.sigmoid(mask_logits)  # [1, 3, H, W]
    defect_probs = torch.sigmoid(defect_logits)  # [1, num_classes]
    
    # Convert to numpy
    mask_probs = mask_probs.cpu().numpy()[0]  # [3, H, W]
    defect_probs = defect_probs.cpu().numpy()[0]  # [num_classes]
    stat_feats = stat_feats.cpu().numpy()[0]  # [88]
    
    # Create results dictionary
    results = {
        'image_path': image_path,
        'region_masks': {
            'core': mask_probs[0].tolist(),
            'cladding': mask_probs[1].tolist(),
            'ferrule': mask_probs[2].tolist()
        },
        'defect_probabilities': defect_probs.tolist(),
        'statistical_features': stat_feats.tolist(),
        'predictions': {
            'defects_detected': [],
            'confidence_scores': []
        }
    }
    
    # Identify defects (threshold-based)
    defect_threshold = 0.5
    defect_names = [
        'scratch', 'dig', 'blob', 'contamination', 'crack',
        'chip', 'pit', 'discoloration', 'roughness', 'waviness',
        'eccentricity', 'concentricity', 'roundness', 'surface_finish',
        'edge_defect', 'center_defect', 'peripheral_defect',
        'structural_defect', 'optical_defect', 'mechanical_defect',
        'thermal_defect', 'chemical_defect', 'environmental_defect',
        'manufacturing_defect', 'handling_defect', 'storage_defect',
        'transport_defect', 'installation_defect', 'operation_defect',
        'maintenance_defect', 'inspection_defect', 'calibration_defect',
        'alignment_defect', 'focus_defect', 'illumination_defect',
        'imaging_defect', 'processing_defect', 'analysis_defect',
        'reporting_defect', 'documentation_defect', 'quality_defect'
    ]
    
    for i, prob in enumerate(defect_probs):
        if prob > defect_threshold:
            results['predictions']['defects_detected'].append(defect_names[i])
            results['predictions']['confidence_scores'].append(float(prob))
    
    return results

def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

def create_visualization(results: Dict, image_path: str, output_path: str):
    """Create visualization of predictions."""
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Core mask
    core_mask = np.array(results['region_masks']['core'])
    axes[0, 1].imshow(core_mask, cmap='Reds')
    axes[0, 1].set_title('Core Region')
    axes[0, 1].axis('off')
    
    # Cladding mask
    cladding_mask = np.array(results['region_masks']['cladding'])
    axes[1, 0].imshow(cladding_mask, cmap='Blues')
    axes[1, 0].set_title('Cladding Region')
    axes[1, 0].axis('off')
    
    # Ferrule mask
    ferrule_mask = np.array(results['region_masks']['ferrule'])
    axes[1, 1].imshow(ferrule_mask, cmap='Greens')
    axes[1, 1].set_title('Ferrule Region')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fiber Optic End-Face Inference")
    parser.add_argument('--weights', required=True, help='Path to trained model weights')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--outdir', default='results', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_classes', type=int, default=40, help='Number of defect classes')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load model
    model = load_model(args.weights, args.num_classes, args.device)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        results = process_image(model, str(input_path), args.device)
        
        # Save results
        output_file = Path(args.outdir) / f"{input_path.stem}_results.json"
        save_results(results, str(output_file))
        
        # Create visualization if requested
        if args.visualize:
            viz_file = Path(args.outdir) / f"{input_path.stem}_visualization.png"
            create_visualization(results, str(input_path), str(viz_file))
    
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Processing {len(image_files)} images...")
        
        all_results = []
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                results = process_image(model, str(image_file), args.device)
                all_results.append(results)
                
                # Save individual results
                output_file = Path(args.outdir) / f"{image_file.stem}_results.json"
                save_results(results, str(output_file))
                
                # Create visualization if requested
                if args.visualize:
                    viz_file = Path(args.outdir) / f"{image_file.stem}_visualization.png"
                    create_visualization(results, str(image_file), str(viz_file))
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        # Save summary results
        summary_file = Path(args.outdir) / "summary_results.json"
        summary = {
            'total_images': len(image_files),
            'processed_images': len(all_results),
            'results': all_results
        }
        save_results(summary, str(summary_file))
        
        print(f"Processing complete. Results saved to {args.outdir}")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main() 