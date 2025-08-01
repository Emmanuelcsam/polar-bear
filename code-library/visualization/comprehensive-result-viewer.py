import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path
from common_data_and_utils import load_json_data, load_single_image, log_message

def _save_simple_anomaly_image(results: Dict[str, Any], test_image: np.ndarray, output_path: str):
    """Save a simple image with just anomalies highlighted in blue."""
    img_to_draw = test_image.copy()
    if len(img_to_draw.shape) == 2:
        img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_GRAY2BGR)

    # Draw anomaly regions
    for region in results['local_analysis']['anomaly_regions']:
        x, y, w, h = region['bbox']
        cv2.rectangle(img_to_draw, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue rectangle
    
    # Draw specific defects
    defects = results['specific_defects']
    for scratch in defects['scratches']:
        x1, y1, x2, y2 = scratch['line']
        cv2.line(img_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    cv2.imwrite(output_path, img_to_draw)
    print(f"✓ Simple anomaly image saved to: {output_path}")

def visualize_comprehensive_results(results: Dict[str, Any], reference_model: Dict[str, Any], output_path: str):
    """Create comprehensive visualization of all anomaly detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
    fig.suptitle(f"Anomaly Analysis: {results['metadata'].get('filename', 'Unknown')}", fontsize=16)

    test_img_rgb = cv2.cvtColor(results['test_image'], cv2.COLOR_BGR2RGB) if len(results['test_image'].shape) == 3 else cv2.cvtColor(results['test_image'], cv2.COLOR_GRAY2RGB)
    archetype_rgb = cv2.cvtColor(reference_model['archetype_image'], cv2.COLOR_GRAY2RGB)
    
    # Panel 1: Original vs Archetype
    ax = axes[0, 0]
    comparison_img = np.hstack((test_img_rgb, cv2.resize(archetype_rgb, (test_img_rgb.shape[1], test_img_rgb.shape[0]))))
    ax.imshow(comparison_img)
    ax.set_title('Test Image (Left) vs. Reference Archetype (Right)')
    ax.axis('off')

    # Panel 2: Local Anomaly Heatmap
    ax = axes[0, 1]
    anomaly_map = results['local_analysis']['anomaly_map']
    ax.imshow(test_img_rgb, alpha=0.7)
    im = ax.imshow(cv2.resize(anomaly_map, (test_img_rgb.shape[1], test_img_rgb.shape[0])), cmap='hot', alpha=0.5)
    ax.set_title('Local Anomaly Heatmap')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Detected Anomalies
    ax = axes[1, 0]
    overlay = test_img_rgb.copy()
    for region in results['local_analysis']['anomaly_regions']:
        x, y, w, h = region['bbox']
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(overlay, f"{region['confidence']:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    ax.imshow(overlay)
    ax.set_title(f"Detected Anomaly Regions ({len(results['local_analysis']['anomaly_regions'])})")
    ax.axis('off')

    # Panel 4: Summary Text
    ax = axes[1, 1]
    ax.axis('off')
    verdict = results['verdict']
    summary = (f"Overall Verdict: {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'}\n"
               f"Confidence: {verdict['confidence']:.1%}\n\n"
               f"Global Mahalanobis Distance: {results['global_analysis']['mahalanobis_distance']:.2f}\n"
               f"Structural Similarity (SSIM): {results['structural_analysis']['ssim']:.3f}\n"
               f"Local Anomaly Regions: {len(results['local_analysis']['anomaly_regions'])}\n"
               f"Specific Defects Found:\n"
               f"  - Scratches: {len(results['specific_defects']['scratches'])}\n"
               f"  - Digs: {len(results['specific_defects']['digs'])}\n"
               f"  - Blobs: {len(results['specific_defects']['blobs'])}")
    ax.text(0.05, 0.95, summary, va='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log_message(f"✓ Comprehensive visualization saved to: {output_path}")
    
    _save_simple_anomaly_image(results, results['test_image'], output_path.replace('.png', '_simple.png'))

def main(results_path: str, reference_model_path: str, output_path: str):
    log_message(f"Starting comprehensive result visualization for {results_path}")

    results_data = load_json_data(Path(results_path))
    if results_data is None:
        log_message(f"Failed to load results from {results_path}", level="ERROR")
        return

    reference_model_data = load_json_data(Path(reference_model_path))
    if reference_model_data is None:
        log_message(f"Failed to load reference model from {reference_model_path}", level="ERROR")
        return

    # Load images if paths are provided in JSON
    if 'test_image_path' in results_data:
        test_image = load_single_image(Path(results_data['test_image_path']))
        if test_image is None:
            log_message(f"Failed to load test image from {results_data['test_image_path']}", level="ERROR")
            return
        results_data['test_image'] = test_image
    else:
        log_message("'test_image_path' not found in results JSON. Visualization might fail.", level="WARNING")

    if 'archetype_image_path' in reference_model_data:
        archetype_image = load_single_image(Path(reference_model_data['archetype_image_path']))
        if archetype_image is None:
            log_message(f"Failed to load archetype image from {reference_model_data['archetype_image_path']}", level="ERROR")
            return
        reference_model_data['archetype_image'] = archetype_image
    else:
        log_message("'archetype_image_path' not found in reference model JSON. Visualization might fail.", level="WARNING")

    if 'anomaly_map' in results_data['local_analysis']:
        results_data['local_analysis']['anomaly_map'] = np.array(results_data['local_analysis']['anomaly_map'], dtype=np.uint8)
    else:
        log_message("'anomaly_map' not found in results JSON. Visualization might fail.", level="WARNING")

    visualize_comprehensive_results(results_data, reference_model_data, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive Result Viewer")
    parser.add_argument("--results_path", required=True, help="Path to JSON file with analysis results.")
    parser.add_argument("--reference_model_path", required=True, help="Path to JSON file with reference model data.")
    parser.add_argument("--output_path", required=True, help="Path to save the comprehensive visualization.")
    
    args = parser.parse_args()
    main(args.results_path, args.reference_model_path, args.output_path)
