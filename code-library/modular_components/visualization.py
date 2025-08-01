#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging


def visualize_comprehensive_results(results, output_path):
    """Create comprehensive visualization of all anomaly detection results."""
    # Create large figure with 3x4 grid layout
    fig = plt.figure(figsize=(24, 16))
    
    # Create grid specification
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Get test image
    test_img = results['test_image']
    # Convert BGR to RGB for matplotlib
    if len(test_img.shape) == 3:
        test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    else:
        test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
    
    # Get archetype image
    archetype = results['reference_model']['archetype_image']
    archetype_rgb = cv2.cvtColor(archetype, cv2.COLOR_GRAY2RGB)
    
    # Panel 1: Original Test Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(test_img_rgb)
    ax1.set_title('Test Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Reference Archetype
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(archetype_rgb)
    ax2.set_title('Reference Archetype', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: SSIM Map
    ax3 = fig.add_subplot(gs[0, 2])
    ssim_map = results['structural_analysis']['ssim_map']
    im3 = ax3.imshow(ssim_map, cmap='RdYlBu', vmin=0, vmax=1)
    ax3.set_title(f'SSIM Map (Index: {results["structural_analysis"]["ssim"]:.3f})', 
                 fontsize=14, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Panel 4: Local Anomaly Heatmap
    ax4 = fig.add_subplot(gs[0, 3])
    anomaly_map = results['local_analysis']['anomaly_map']
    
    # Resize anomaly map to match test image if needed
    if anomaly_map.shape != test_img_rgb.shape[:2]:
        anomaly_map_resized = cv2.resize(anomaly_map, 
                                        (test_img_rgb.shape[1], test_img_rgb.shape[0]))
    else:
        anomaly_map_resized = anomaly_map
    
    ax4.imshow(test_img_rgb, alpha=0.7)
    im4 = ax4.imshow(anomaly_map_resized, cmap='hot', alpha=0.5, vmin=0)
    ax4.set_title('Local Anomaly Heatmap', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Panel 5: Detected Anomalies (Blue Highlights)
    ax5 = fig.add_subplot(gs[1, :2])
    overlay = test_img_rgb.copy()
    
    # Draw anomaly regions in blue
    for region in results['local_analysis']['anomaly_regions']:
        x, y, w, h = region['bbox']
        # Draw blue rectangle outline
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 3)
        
        # Fill with semi-transparent blue
        roi = overlay[y:y+h, x:x+w]
        blue_overlay = np.zeros_like(roi)
        blue_overlay[:, :] = [0, 0, 255]
        cv2.addWeighted(roi, 0.7, blue_overlay, 0.3, 0, roi)
        
        # Add confidence text
        cv2.putText(overlay, f'{region["confidence"]:.2f}', 
                   (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    ax5.imshow(overlay)
    ax5.set_title(f'Detected Anomalies ({len(results["local_analysis"]["anomaly_regions"])} regions)', 
                 fontsize=16, fontweight='bold', color='blue')
    ax5.axis('off')
    
    # Panel 6: Specific Defects
    ax6 = fig.add_subplot(gs[1, 2:])
    defect_overlay = test_img_rgb.copy()
    
    # Draw specific defects with different colors
    defects = results['specific_defects']
    
    # Scratches - cyan lines
    for scratch in defects['scratches']:
        x1, y1, x2, y2 = scratch['line']
        cv2.line(defect_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # Digs - magenta circles
    for dig in defects['digs']:
        cx, cy = dig['center']
        radius = int(np.sqrt(dig['area'] / np.pi))
        cv2.circle(defect_overlay, (cx, cy), max(3, radius), (255, 0, 255), -1)
    
    # Blobs - yellow contours
    cv2.drawContours(defect_overlay, [b['contour'] for b in defects['blobs']], 
                    -1, (255, 255, 0), 2)
    
    # Edges - green contours
    cv2.drawContours(defect_overlay, [e['contour'] for e in defects['edges']], 
                    -1, (0, 255, 0), 1)
    
    ax6.imshow(defect_overlay)
    # Create defect count string
    defect_counts = (f"Scratches: {len(defects['scratches'])}, " 
                    f"Digs: {len(defects['digs'])}, "
                    f"Blobs: {len(defects['blobs'])}, "
                    f"Edges: {len(defects['edges'])}")
    ax6.set_title(f'Specific Defects\n{defect_counts}', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # Panel 7: Feature Deviation Chart
    ax7 = fig.add_subplot(gs[2, :2])
    
    # Get top deviating features
    deviations = results['global_analysis']['deviant_features'][:8]
    names = [d[0].replace('_', '\n') for d in deviations]
    z_scores = [d[1] for d in deviations]
    
    # Color code by severity
    colors = ['red' if z > 3 else 'orange' if z > 2 else 'yellow' for z in z_scores]
    
    # Create horizontal bar chart
    bars = ax7.barh(names, z_scores, color=colors)
    ax7.set_xlabel('Z-Score (Standard Deviations from Reference)', fontsize=12)
    ax7.set_title('Most Deviant Features', fontsize=14, fontweight='bold')
    ax7.axvline(x=2, color='orange', linestyle='--', alpha=0.5, label='2σ threshold')
    ax7.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='3σ threshold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, z in zip(bars, z_scores):
        width = bar.get_width()
        ax7.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{z:.1f}', va='center', fontsize=10)
    
    # Panel 8: Analysis Summary
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.axis('off')
    
    # Prepare summary text
    verdict = results['verdict']
    global_stats = results['global_analysis']
    structural = results['structural_analysis']
    
    # Create formatted summary
    summary_text = f"""COMPREHENSIVE ANALYSIS SUMMARY
    
Overall Verdict: {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'}
Confidence: {verdict['confidence']:.1%}

Global Analysis:
• Mahalanobis Distance: {global_stats['mahalanobis_distance']:.2f}
• Max Comparison Score: {global_stats['comparison_stats']['max']:.3f}
• Mean Comparison Score: {global_stats['comparison_stats']['mean']:.3f}

Structural Analysis:
• SSIM Index: {structural['ssim']:.3f}
• Mean Luminance Similarity: {structural['mean_luminance']:.3f}
• Mean Contrast Similarity: {structural['mean_contrast']:.3f}
• Mean Structure Similarity: {structural['mean_structure']:.3f}

Local Analysis:
• Anomaly Regions Found: {len(results['local_analysis']['anomaly_regions'])}
• Max Region Confidence: {max([r['confidence'] for r in results['local_analysis']['anomaly_regions']], default=0):.3f}

Criteria Triggered:
• Mahalanobis: {'✓' if verdict['criteria_triggered']['mahalanobis'] else '✗'}
• Comparison: {'✓' if verdict['criteria_triggered']['comparison'] else '✗'}
• Structural: {'✓' if verdict['criteria_triggered']['structural'] else '✗'}
• Local: {'✓' if verdict['criteria_triggered']['local'] else '✗'}"""
    
    # Add text with box
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    source_name = results['metadata'].get('filename', 'Unknown')
    fig.suptitle(f'Ultra-Comprehensive Anomaly Analysis\nTest: {source_name}', 
                fontsize=20, fontweight='bold')
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Log save location
    logging.info(f"Visualization saved to: {output_path}")
    
    # Also save simplified version
    save_simple_anomaly_image(results, output_path.replace('.png', '_simple.png'))


def save_simple_anomaly_image(results, output_path):
    """Save a simple image with just anomalies highlighted in blue."""
    # Copy test image
    test_img = results['test_image'].copy()
    
    # Draw anomaly regions
    for region in results['local_analysis']['anomaly_regions']:
        x, y, w, h = region['bbox']
        
        # Draw blue rectangle
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        # Fill with semi-transparent blue
        overlay = test_img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, test_img, 0.7, 0, test_img)
    
    # Draw specific defects in blue
    defects = results['specific_defects']
    
    # All defects in blue
    for scratch in defects['scratches']:
        x1, y1, x2, y2 = scratch['line']
        cv2.line(test_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    for dig in defects['digs']:
        cx, cy = dig['center']
        radius = max(3, int(np.sqrt(dig['area'] / np.pi)))
        cv2.circle(test_img, (cx, cy), radius, (255, 0, 0), -1)
    
    cv2.drawContours(test_img, [b['contour'] for b in defects['blobs']], 
                    -1, (255, 0, 0), 2)
    
    # Add verdict text
    verdict = "ANOMALOUS" if results['verdict']['is_anomalous'] else "NORMAL"
    confidence = results['verdict']['confidence']
    
    cv2.putText(test_img, f"{verdict} ({confidence:.1%})", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Save image
    cv2.imwrite(output_path, test_img)
    logging.info(f"Simple anomaly image saved to: {output_path}") 