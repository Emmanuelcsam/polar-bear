import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

def visualize_results(image_path: str, results: Dict[str, Any], save_path: Optional[str] = None):
    """Visualize detection results"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image for visualization: {image_path}")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Fiber Optic Defect Detection Results - {results['pass_fail']['status']}", fontsize=16)
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Zone masks overlay
    zone_overlay = np.zeros_like(image)
    zone_colors = {'core': [255,0,0], 'cladding': [0,255,0], 'ferrule': [0,0,255], 'adhesive': [255,255,0]}
    for name, mask in results['zone_masks'].items():
        if name in zone_colors:
            zone_overlay[mask > 0] = zone_colors[name]
    axes[0, 1].imshow(zone_overlay)
    axes[0, 1].set_title('Fiber Zones')
    axes[0, 1].axis('off')
    
    # All defects overlay
    defect_overlay = image.copy()
    for defect in results['defects']:
        x, y, w, h = defect.bounding_box
        color = {'scratch': (255,0,255), 'dig': (0,255,255)}.get(defect.defect_type, (255,128,0))
        cv2.rectangle(defect_overlay, (x,y), (x+w,y+h), color, 2)
        cv2.putText(defect_overlay, f"{defect.defect_id}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    axes[0, 2].imshow(cv2.cvtColor(defect_overlay, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Detected Defects ({len(results["defects"])})')
    axes[0, 2].axis('off')
    
    # Detection masks
    if 'detection_masks' in results:
        scratch_combined = np.zeros(image.shape[:2], dtype=np.uint8)
        for name, mask in results['detection_masks'].items():
            if 'scratches' in name: scratch_combined = cv2.bitwise_or(scratch_combined, mask)
        axes[1, 0].imshow(scratch_combined, cmap='hot'); axes[1, 0].set_title('Scratch Detections'); axes[1, 0].axis('off')
        
        region_combined = np.zeros(image.shape[:2], dtype=np.uint8)
        for name, mask in results['detection_masks'].items():
            if 'regions' in name: region_combined = cv2.bitwise_or(region_combined, mask)
        axes[1, 1].imshow(region_combined, cmap='hot'); axes[1, 1].set_title('Region Detections'); axes[1, 1].axis('off')

    # Summary
    summary = f"Status: {results['pass_fail']['status']}\nTotal Defects: {len(results['defects'])}\n\n"
    summary += "Defects by Zone:\n" + "\n".join([f"  {k}: {v}" for k,v in results['pass_fail']['defects_by_zone'].items()])
    if results['pass_fail']['failures']:
        summary += "\n\nFailures:\n" + "\n".join([f"- {f}" for f in results['pass_fail']['failures'][:5]])
    axes[1, 2].text(0.1, 0.9, summary, va='top', fontsize=10, family='monospace'); axes[1, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {save_path}")
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':
    print("This script contains the 'visualize_results' function.")
    print("It is intended to be used as part of the unified defect detection system.")
    print("To run a full detection and visualization, use 'jill_main.py'.")
