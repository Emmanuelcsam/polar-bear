#!/usr/bin/env python3
"""
Interactive Visualization Module
==============================
Standalone implementation for interactive visualization of fiber inspection results.
Supports both Napari-based interactive viewing and OpenCV-based static visualization.

Features:
- Interactive Napari viewer with layer management
- Zone mask visualization with color coding
- Defect overlay with classification-based coloring
- Text annotations for defect IDs and status
- Fallback OpenCV visualization for environments without Napari
- Export capabilities for static images
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import argparse
import json

# Try to import napari for interactive visualization
try:
    import napari
    NAPARI_AVAILABLE = True
    logging.info("Napari available for interactive visualization")
except ImportError:
    NAPARI_AVAILABLE = False
    logging.warning("Napari not available. Using fallback visualization.")


class InteractiveVisualizer:
    """Class for visualizing fiber inspection results."""
    
    def __init__(self, use_napari: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            use_napari: Whether to use Napari for interactive visualization
        """
        self.use_napari = use_napari and NAPARI_AVAILABLE
        self.viewer = None
        
        if self.use_napari:
            logging.info("Using Napari for interactive visualization")
        else:
            logging.info("Using OpenCV for static visualization")
    
    def show_inspection_results(self,
                              original_image: np.ndarray,
                              defect_masks: Optional[Dict[str, np.ndarray]] = None,
                              zone_masks: Optional[Dict[str, np.ndarray]] = None,
                              analysis_results: Optional[Dict[str, Any]] = None,
                              interactive: bool = True,
                              save_path: Optional[str] = None) -> Optional[Any]:
        """
        Display comprehensive inspection results.
        
        Args:
            original_image: Original fiber optic image
            defect_masks: Dictionary of defect masks by type
            zone_masks: Dictionary of zone masks (core, cladding, etc.)
            analysis_results: Analysis results including defect characterizations
            interactive: Whether to run interactively
            save_path: Path to save static visualization
            
        Returns:
            Viewer instance if Napari used, image array otherwise
        """
        if self.use_napari:
            return self._show_with_napari(
                original_image, defect_masks, zone_masks, 
                analysis_results, interactive
            )
        else:
            return self._show_with_opencv(
                original_image, defect_masks, zone_masks,
                analysis_results, save_path
            )
    
    def _show_with_napari(self,
                         original_image: np.ndarray,
                         defect_masks: Optional[Dict[str, np.ndarray]],
                         zone_masks: Optional[Dict[str, np.ndarray]],
                         analysis_results: Optional[Dict[str, Any]],
                         interactive: bool) -> Optional[Any]:
        """Show results using Napari interactive viewer."""
        if not NAPARI_AVAILABLE:
            logging.error("Napari not available")
            return None
        
        try:
            # Create viewer
            self.viewer = napari.Viewer(title='Fiber Inspection Results')
            
            # Add original image
            self.viewer.add_image(original_image, name='Original Image')
            
            # Add zone masks with different colors
            if zone_masks:
                zone_colors = {
                    'Core': 'red',
                    'Cladding': 'green', 
                    'Adhesive': 'yellow',
                    'Contact': 'magenta'
                }
                
                for zone_name, mask in zone_masks.items():
                    if np.any(mask):
                        self.viewer.add_labels(
                            mask.astype(int),
                            name=f'Zone: {zone_name}',
                            opacity=0.3,
                            color={1: zone_colors.get(zone_name, 'gray')}
                        )
            
            # Add defect masks
            if defect_masks:
                for defect_type, mask in defect_masks.items():
                    if np.any(mask):
                        self.viewer.add_labels(
                            mask.astype(int),
                            name=f'Defects: {defect_type}',
                            opacity=0.6
                        )
            
            # Add characterized defects from analysis results
            if analysis_results and 'characterized_defects' in analysis_results:
                self._add_defect_annotations_napari(analysis_results['characterized_defects'])
            
            # Add overall status
            if analysis_results and 'overall_status' in analysis_results:
                self._add_status_annotation_napari(analysis_results['overall_status'])
            
            # Run interactively if requested
            if interactive:
                napari.run()
            
            return self.viewer
            
        except Exception as e:
            logging.error(f"Napari visualization failed: {e}")
            return None
    
    def _show_with_opencv(self,
                         original_image: np.ndarray,
                         defect_masks: Optional[Dict[str, np.ndarray]],
                         zone_masks: Optional[Dict[str, np.ndarray]],
                         analysis_results: Optional[Dict[str, Any]],
                         save_path: Optional[str]) -> np.ndarray:
        """Show results using OpenCV static visualization."""
        
        # Create color version of original image
        if len(original_image.shape) == 2:
            result = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            result = original_image.copy()
        
        # Draw zone boundaries
        if zone_masks:
            result = self._draw_zones_opencv(result, zone_masks)
        
        # Draw defect masks
        if defect_masks:
            result = self._draw_defect_masks_opencv(result, defect_masks)
        
        # Draw characterized defects
        if analysis_results and 'characterized_defects' in analysis_results:
            result = self._draw_characterized_defects_opencv(
                result, analysis_results['characterized_defects']
            )
        
        # Add status overlay
        if analysis_results and 'overall_status' in analysis_results:
            result = self._add_status_overlay_opencv(
                result, analysis_results['overall_status']
            )
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, result)
            logging.info(f"Visualization saved to: {save_path}")
        
        return result
    
    def _add_defect_annotations_napari(self, characterized_defects: List[Dict[str, Any]]):
        """Add defect annotations to Napari viewer."""
        if not characterized_defects:
            return
        
        # Determine image shape for defect mask
        if self.viewer and len(self.viewer.layers) > 0:
            image_shape = self.viewer.layers[0].data.shape[-2:]  # Get last 2 dimensions
        else:
            return
        
        # Create combined defect mask
        all_defects = np.zeros(image_shape, dtype=int)
        defect_id_counter = 1
        
        for defect in characterized_defects:
            if 'contour_points_px' in defect:
                contour_points = np.array(defect['contour_points_px']).astype(np.int32)
                cv2.fillPoly(all_defects, [contour_points], defect_id_counter)
                defect_id_counter += 1
        
        # Add defects layer
        if np.any(all_defects):
            self.viewer.add_labels(
                all_defects,
                name='Characterized Defects',
                opacity=0.7
            )
        
        # Add text annotations
        for defect in characterized_defects:
            cx = defect.get('centroid_x_px', 0)
            cy = defect.get('centroid_y_px', 0)
            defect_id = defect.get('defect_id', '')
            classification = defect.get('classification', '')
            
            # Add text annotation
            self.viewer.add_text(
                text=f"{defect_id}: {classification}",
                position=(cy, cx),
                face_color='yellow',
                size=12,
                anchor='center',
                name=f"Label {defect_id}"
            )
    
    def _add_status_annotation_napari(self, status: str):
        """Add overall status annotation to Napari viewer."""
        color = 'green' if status == 'PASS' else 'red'
        
        self.viewer.add_text(
            text=f"Status: {status}",
            position=(10, 10),
            face_color=color,
            size=20,
            anchor='upper_left',
            name='Overall Status'
        )
    
    def _draw_zones_opencv(self, image: np.ndarray, zone_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Draw zone boundaries using OpenCV."""
        zone_colors = {
            'Core': (0, 0, 255),      # Red
            'Cladding': (0, 255, 0),  # Green
            'Adhesive': (0, 255, 255), # Yellow
            'Contact': (255, 0, 255)   # Magenta
        }
        
        result = image.copy()
        
        for zone_name, mask in zone_masks.items():
            if np.any(mask):
                color = zone_colors.get(zone_name, (128, 128, 128))
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours
                cv2.drawContours(result, contours, -1, color, 2)
                
                # Add zone label
                if contours:
                    # Find the largest contour for labeling
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(result, zone_name, (cx-20, cy), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def _draw_defect_masks_opencv(self, image: np.ndarray, defect_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Draw defect masks using OpenCV."""
        result = image.copy()
        
        mask_colors = {
            'scratches': (0, 255, 255),  # Yellow
            'pits': (255, 0, 0),         # Blue
            'general': (0, 165, 255)     # Orange
        }
        
        for mask_type, mask in defect_masks.items():
            if np.any(mask):
                color = mask_colors.get(mask_type.lower(), (255, 255, 255))
                
                # Create colored overlay
                colored_mask = np.zeros_like(result)
                colored_mask[mask > 0] = color
                
                # Blend with original image
                result = cv2.addWeighted(result, 0.7, colored_mask, 0.3, 0)
        
        return result
    
    def _draw_characterized_defects_opencv(self, image: np.ndarray, defects: List[Dict[str, Any]]) -> np.ndarray:
        """Draw characterized defects using OpenCV."""
        result = image.copy()
        
        for defect in defects:
            # Get defect properties
            cx = int(defect.get('centroid_x_px', 0))
            cy = int(defect.get('centroid_y_px', 0))
            classification = defect.get('classification', 'Unknown')
            defect_id = defect.get('defect_id', '')
            
            # Choose color based on classification
            if classification == 'Scratch':
                color = (0, 255, 255)  # Yellow
            elif classification == 'Pit/Dig':
                color = (255, 0, 0)    # Blue
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw contour if available
            if 'contour_points_px' in defect:
                contour_points = np.array(defect['contour_points_px'], dtype=np.int32)
                cv2.drawContours(result, [contour_points], -1, color, 2)
            
            # Draw centroid
            cv2.circle(result, (cx, cy), 3, color, -1)
            
            # Add label
            label = f"{defect_id.split('_')[-1]}:{classification[0]}"
            cv2.putText(result, label, (cx + 5, cy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result
    
    def _add_status_overlay_opencv(self, image: np.ndarray, status: str) -> np.ndarray:
        """Add status overlay using OpenCV."""
        result = image.copy()
        
        # Choose color based on status
        color = (0, 255, 0) if status == 'PASS' else (0, 0, 255)  # Green for PASS, Red for FAIL
        
        # Add status text
        cv2.putText(result, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Add background rectangle for better visibility
        text_size = cv2.getTextSize(f"Status: {status}", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(result, (5, 5), (text_size[0] + 15, 40), (255, 255, 255), -1)
        cv2.putText(result, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return result
    
    def close(self):
        """Close the viewer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def create_multi_panel_visualization(
    original_image: np.ndarray,
    processed_image: Optional[np.ndarray] = None,
    defect_mask: Optional[np.ndarray] = None,
    zone_masks: Optional[Dict[str, np.ndarray]] = None,
    characterized_defects: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a multi-panel visualization showing all stages of processing.
    
    Args:
        original_image: Original fiber image
        processed_image: Preprocessed image
        defect_mask: Binary defect mask
        zone_masks: Dictionary of zone masks
        characterized_defects: List of characterized defects
        save_path: Optional path to save the visualization
        
    Returns:
        Multi-panel visualization image
    """
    # Ensure all images are the same size and format
    h, w = original_image.shape[:2]
    
    # Convert to color if needed
    if len(original_image.shape) == 2:
        orig_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        orig_color = original_image.copy()
    
    # Prepare processed image panel
    if processed_image is not None:
        if len(processed_image.shape) == 2:
            proc_color = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        else:
            proc_color = processed_image.copy()
    else:
        proc_color = orig_color.copy()
    
    # Prepare defect mask panel
    if defect_mask is not None:
        defect_color = cv2.cvtColor(defect_mask, cv2.COLOR_GRAY2BGR)
    else:
        defect_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Prepare final result panel
    result_panel = orig_color.copy()
    
    # Draw zones on result panel
    if zone_masks:
        visualizer = InteractiveVisualizer(use_napari=False)
        result_panel = visualizer._draw_zones_opencv(result_panel, zone_masks)
    
    # Draw characterized defects
    if characterized_defects:
        visualizer = InteractiveVisualizer(use_napari=False)
        result_panel = visualizer._draw_characterized_defects_opencv(result_panel, characterized_defects)
    
    # Add labels to each panel
    panels = [
        (orig_color, "Original"),
        (proc_color, "Processed"),
        (defect_color, "Defects"),
        (result_panel, "Final Result")
    ]
    
    labeled_panels = []
    for panel, label in panels:
        labeled_panel = panel.copy()
        cv2.putText(labeled_panel, label, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(labeled_panel, (5, 5), (len(label) * 15, 35), (0, 0, 0), -1)
        cv2.putText(labeled_panel, label, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        labeled_panels.append(labeled_panel)
    
    # Arrange panels in 2x2 grid
    top_row = np.hstack([labeled_panels[0], labeled_panels[1]])
    bottom_row = np.hstack([labeled_panels[2], labeled_panels[3]])
    multi_panel = np.vstack([top_row, bottom_row])
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, multi_panel)
        logging.info(f"Multi-panel visualization saved to: {save_path}")
    
    return multi_panel


def load_inspection_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Load inspection data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Inspection data loaded from: {file_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load inspection data: {e}")
        return None


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Interactive Visualization")
    parser.add_argument("image", help="Path to original fiber image")
    parser.add_argument("--data", help="Path to JSON file with inspection results")
    parser.add_argument("--defect-mask", help="Path to binary defect mask image")
    parser.add_argument("--use-opencv", action="store_true", 
                       help="Force use of OpenCV instead of Napari")
    parser.add_argument("--multi-panel", action="store_true",
                       help="Create multi-panel visualization")
    parser.add_argument("--output", "-o", help="Path to save visualization")
    parser.add_argument("--no-interactive", action="store_true",
                       help="Don't run interactively (save only)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    try:
        # Load original image
        logging.info(f"Loading image: {args.image}")
        original_image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            raise ValueError(f"Failed to load image: {args.image}")
        
        # Load inspection data if provided
        inspection_data = None
        if args.data:
            inspection_data = load_inspection_data(args.data)
        
        # Load defect mask if provided
        defect_mask = None
        if args.defect_mask:
            defect_mask = cv2.imread(args.defect_mask, cv2.IMREAD_GRAYSCALE)
            if defect_mask is None:
                logging.warning(f"Failed to load defect mask: {args.defect_mask}")
        
        if args.multi_panel:
            # Create multi-panel visualization
            characterized_defects = None
            if inspection_data:
                characterized_defects = inspection_data.get('characterized_defects')
            
            result = create_multi_panel_visualization(
                original_image,
                defect_mask=defect_mask,
                characterized_defects=characterized_defects,
                save_path=args.output
            )
            
            if not args.no_interactive:
                cv2.imshow("Multi-Panel Visualization", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        else:
            # Create interactive or static visualization
            use_napari = not args.use_opencv and NAPARI_AVAILABLE
            visualizer = InteractiveVisualizer(use_napari=use_napari)
            
            # Prepare data for visualization
            defect_masks = {}
            if defect_mask is not None:
                defect_masks['general'] = defect_mask
            
            zone_masks = None
            analysis_results = None
            
            if inspection_data:
                zone_masks = inspection_data.get('zone_masks')
                analysis_results = inspection_data
            
            # Show results
            result = visualizer.show_inspection_results(
                original_image,
                defect_masks=defect_masks if defect_masks else None,
                zone_masks=zone_masks,
                analysis_results=analysis_results,
                interactive=not args.no_interactive,
                save_path=args.output
            )
            
            if not use_napari and not args.no_interactive and result is not None:
                cv2.imshow("Inspection Results", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        logging.info("Visualization complete")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
