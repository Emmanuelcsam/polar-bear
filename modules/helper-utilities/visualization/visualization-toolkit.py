#!/usr/bin/env python3
"""
Modular Visualization Functions
==============================
Standalone visualization functions for fiber inspection results,
including interactive viewers and static plots.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False
    logger.warning("Napari not available. Interactive visualization disabled.")

def create_side_by_side_comparison(
    original_image: np.ndarray,
    processed_image: np.ndarray,
    titles: Tuple[str, str] = ("Original", "Processed"),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Create a side-by-side comparison of two images.
    
    Args:
        original_image: First image to display
        processed_image: Second image to display
        titles: Titles for the images
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object or None if error
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Handle grayscale vs color images
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            # BGR to RGB for matplotlib
            original_display = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_display = original_image
        
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            processed_display = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        else:
            processed_display = processed_image
        
        axes[0].imshow(original_display, cmap='gray' if len(original_display.shape) == 2 else None)
        axes[0].set_title(titles[0])
        axes[0].axis('off')
        
        axes[1].imshow(processed_display, cmap='gray' if len(processed_display.shape) == 2 else None)
        axes[1].set_title(titles[1])
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison saved to {save_path}")
        
        return fig
    except Exception as e:
        logger.error(f"Error creating side-by-side comparison: {e}")
        return None

def visualize_defect_overlays(
    base_image: np.ndarray,
    defect_masks: Dict[str, np.ndarray],
    zone_masks: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Create a visualization with defect and zone overlays.
    
    Args:
        base_image: Base image to overlay on
        defect_masks: Dictionary of defect masks
        zone_masks: Optional dictionary of zone masks
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object or None if error
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display base image
        if len(base_image.shape) == 3 and base_image.shape[2] == 3:
            display_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        else:
            display_image = base_image
        
        ax.imshow(display_image, cmap='gray' if len(display_image.shape) == 2 else None)
        
        # Overlay zone masks with transparency
        if zone_masks:
            zone_colors = {
                'Core': 'red',
                'Cladding': 'green', 
                'Adhesive': 'yellow',
                'Contact': 'magenta'
            }
            
            for zone_name, mask in zone_masks.items():
                if np.any(mask):
                    color = zone_colors.get(zone_name, 'gray')
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        # Convert contour to plot coordinates
                        contour_points = contour.squeeze()
                        if len(contour_points.shape) == 2:
                            ax.plot(contour_points[:, 0], contour_points[:, 1], 
                                   color=color, linewidth=2, label=f'{zone_name} Zone')
        
        # Overlay defect masks
        defect_colors = ['red', 'orange', 'purple', 'cyan', 'pink']
        color_idx = 0
        
        for defect_name, mask in defect_masks.items():
            if np.any(mask):
                color = defect_colors[color_idx % len(defect_colors)]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_points = contour.squeeze()
                    if len(contour_points.shape) == 2:
                        ax.plot(contour_points[:, 0], contour_points[:, 1], 
                               color=color, linewidth=3, label=f'{defect_name} Defects')
                color_idx += 1
        
        ax.set_title('Inspection Results Overlay')
        ax.axis('off')
        
        # Add legend if there are overlays
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Overlay visualization saved to {save_path}")
        
        return fig
    except Exception as e:
        logger.error(f"Error creating defect overlay visualization: {e}")
        return None

def create_defect_statistics_plot(
    analysis_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Create statistical plots of defect analysis results.
    
    Args:
        analysis_results: Dictionary containing defect analysis results
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object or None if error
    """
    try:
        defects = analysis_results.get('characterized_defects', [])
        
        if not defects:
            logger.warning("No defects found for statistics plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        zones = [d.get('zone', 'Unknown') for d in defects]
        types = [d.get('type', 'Unknown') for d in defects]
        areas = [d.get('area_px', 0) for d in defects]
        confidences = [d.get('confidence_score', 0) for d in defects]
        
        # Zone distribution
        zone_counts = {}
        for zone in zones:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        if zone_counts:
            axes[0, 0].pie(zone_counts.values(), labels=zone_counts.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Defects by Zone')
        
        # Type distribution
        type_counts = {}
        for defect_type in types:
            type_counts[defect_type] = type_counts.get(defect_type, 0) + 1
        
        if type_counts:
            axes[0, 1].bar(type_counts.keys(), type_counts.values())
            axes[0, 1].set_title('Defects by Type')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Area distribution
        if areas and any(a > 0 for a in areas):
            axes[1, 0].hist(areas, bins=10, alpha=0.7)
            axes[1, 0].set_title('Defect Area Distribution')
            axes[1, 0].set_xlabel('Area (pixels)')
            axes[1, 0].set_ylabel('Count')
        
        # Confidence distribution
        if confidences and any(c > 0 for c in confidences):
            axes[1, 1].hist(confidences, bins=10, alpha=0.7)
            axes[1, 1].set_title('Confidence Score Distribution')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Statistics plot saved to {save_path}")
        
        return fig
    except Exception as e:
        logger.error(f"Error creating statistics plot: {e}")
        return None

def show_interactive_inspection_results(
    original_image: np.ndarray,
    defect_masks: Dict[str, np.ndarray],
    zone_masks: Dict[str, np.ndarray],
    analysis_results: Dict[str, Any]
) -> Optional[Any]:
    """
    Display inspection results in an interactive Napari viewer.
    
    Args:
        original_image: Original inspection image
        defect_masks: Dictionary of defect masks
        zone_masks: Dictionary of zone masks
        analysis_results: Analysis results dictionary
        
    Returns:
        Napari viewer instance or None if not available
    """
    if not NAPARI_AVAILABLE:
        logger.warning("Napari not available. Use install_napari() first.")
        return None
    
    try:
        viewer = napari.Viewer(title='Fiber Inspection Results')
        
        # Add original image
        viewer.add_image(original_image, name='Original Image')
        
        # Add zone masks
        zone_colors = {
            'Core': 'red',
            'Cladding': 'green',
            'Adhesive': 'yellow', 
            'Contact': 'magenta'
        }
        
        for zone_name, mask in zone_masks.items():
            if np.any(mask):
                viewer.add_labels(
                    mask.astype(int),
                    name=f'Zone: {zone_name}',
                    opacity=0.3,
                    color={1: zone_colors.get(zone_name, 'gray')}
                )
        
        # Add combined defect mask
        if original_image.ndim == 3:
            shape_2d = original_image.shape[:2]
        else:
            shape_2d = original_image.shape
        
        all_defects = np.zeros(shape_2d, dtype=int)
        defect_id_counter = 1
        
        for defect in analysis_results.get('characterized_defects', []):
            if 'contour_points_px' in defect:
                defect_mask = np.zeros_like(all_defects)
                contour_points = np.array(defect['contour_points_px']).astype(np.int32)
                cv2.fillPoly(defect_mask, [contour_points], defect_id_counter)
                all_defects[defect_mask > 0] = defect_id_counter
                defect_id_counter += 1
        
        if np.any(all_defects):
            viewer.add_labels(all_defects, name='Detected Defects', opacity=0.7)
        
        # Add text annotations for defects
        text_data = []
        for defect in analysis_results.get('characterized_defects', []):
            cx = defect.get('centroid_x_px', 0)
            cy = defect.get('centroid_y_px', 0)
            defect_id = defect.get('defect_id', '')
            text_data.append([cy, cx, f"ID: {defect_id}"])  # Note: napari uses (y, x) order
        
        if text_data:
            viewer.add_points(
                np.array([[point[0], point[1]] for point in text_data]),
                text=[point[2] for point in text_data],
                name='Defect Labels',
                size=5,
                face_color='yellow'
            )
        
        logger.info("Interactive viewer launched")
        return viewer
    
    except Exception as e:
        logger.error(f"Error creating interactive viewer: {e}")
        return None

def install_napari() -> bool:
    """
    Helper function to install napari (for development environments).
    
    Returns:
        True if installation attempt was made, False otherwise
    """
    try:
        import subprocess
        import sys
        
        logger.info("Attempting to install napari...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "napari[all]"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Napari installation completed successfully")
            return True
        else:
            logger.error(f"Napari installation failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error installing napari: {e}")
        return False

def create_processing_pipeline_visualization(
    original_image: np.ndarray,
    processing_steps: List[Tuple[str, np.ndarray]],
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Create a visualization showing the processing pipeline steps.
    
    Args:
        original_image: Original input image
        processing_steps: List of (step_name, result_image) tuples
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object or None if error
    """
    try:
        num_steps = len(processing_steps) + 1  # +1 for original
        cols = min(4, num_steps)
        rows = (num_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Handle single row case
        if rows == 1:
            axes = [axes] if num_steps == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        # Show original image
        display_original = original_image
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            display_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(display_original, cmap='gray' if len(display_original.shape) == 2 else None)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Show processing steps
        for i, (step_name, step_image) in enumerate(processing_steps):
            ax_idx = i + 1
            if ax_idx < len(axes):
                display_image = step_image
                if len(step_image.shape) == 3 and step_image.shape[2] == 3:
                    display_image = cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)
                
                axes[ax_idx].imshow(display_image, cmap='gray' if len(display_image.shape) == 2 else None)
                axes[ax_idx].set_title(step_name)
                axes[ax_idx].axis('off')
        
        # Hide unused subplots
        for i in range(num_steps, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Pipeline visualization saved to {save_path}")
        
        return fig
    except Exception as e:
        logger.error(f"Error creating pipeline visualization: {e}")
        return None

# Test function
def test_visualization_functions():
    """Test the visualization functions with synthetic data."""
    logger.info("Testing visualization functions...")
    
    # Create synthetic test data
    test_image = np.ones((300, 300, 3), dtype=np.uint8) * 128
    cv2.circle(test_image, (150, 150), 80, (200, 200, 200), -1)
    
    # Create processed version
    processed_image = cv2.GaussianBlur(test_image, (15, 15), 0)
    
    # Test side-by-side comparison
    fig1 = create_side_by_side_comparison(test_image, processed_image, 
                                         ("Original", "Blurred"), "test_comparison.png")
    if fig1:
        plt.close(fig1)
        logger.info("Side-by-side comparison test passed")
    
    # Create synthetic masks
    zone_masks = {}
    for zone_name, radius in [('Core', 40), ('Cladding', 80)]:
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), radius, 255, -1)
        if zone_name != 'Core':
            cv2.circle(mask, (150, 150), 40, 0, -1)
        zone_masks[zone_name] = mask
    
    defect_masks = {
        'scratches': np.zeros((300, 300), dtype=np.uint8),
        'pits': np.zeros((300, 300), dtype=np.uint8)
    }
    # Add some synthetic defects
    cv2.rectangle(defect_masks['scratches'], (100, 140), (200, 145), 255, -1)
    cv2.circle(defect_masks['pits'], (120, 160), 5, 255, -1)
    
    # Test overlay visualization
    fig2 = visualize_defect_overlays(test_image, defect_masks, zone_masks, "test_overlay.png")
    if fig2:
        plt.close(fig2)
        logger.info("Overlay visualization test passed")
    
    # Create synthetic analysis results
    analysis_results = {
        'characterized_defects': [
            {'zone': 'Core', 'type': 'scratch', 'area_px': 25, 'confidence_score': 0.85},
            {'zone': 'Cladding', 'type': 'pit', 'area_px': 15, 'confidence_score': 0.75},
            {'zone': 'Core', 'type': 'pit', 'area_px': 8, 'confidence_score': 0.65}
        ]
    }
    
    # Test statistics plot
    fig3 = create_defect_statistics_plot(analysis_results, "test_statistics.png")
    if fig3:
        plt.close(fig3)
        logger.info("Statistics plot test passed")
    
    # Test pipeline visualization
    processing_steps = [
        ("Blurred", processed_image),
        ("Grayscale", cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)),
    ]
    fig4 = create_processing_pipeline_visualization(test_image, processing_steps, "test_pipeline.png")
    if fig4:
        plt.close(fig4)
        logger.info("Pipeline visualization test passed")
    
    logger.info("Visualization function tests completed")

if __name__ == "__main__":
    test_visualization_functions()
