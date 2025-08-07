#!/usr/bin/env python3
"""
Visualization module for creating visual reports and overlays.
Provides functions for visualizing defects, comparisons, and analysis results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


def draw_defects_overlay(image: np.ndarray,
                         defects: List[Dict],
                         colors: Optional[Dict[str, Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    Draw defects as overlays on image.
    
    Args:
        image: Input image (BGR or grayscale)
        defects: List of defect dictionaries with 'type' and 'location'/'line' keys
        colors: Optional color mapping for defect types (BGR format)
        
    Returns:
        Image with defects drawn
    """
    # Default colors if not provided
    if colors is None:
        colors = {
            'Blob': (0, 255, 0),        # Green
            'Scratch': (0, 255, 255),   # Cyan
            'LineScratch': (255, 0, 255), # Magenta
            'Dig': (255, 0, 0),         # Blue
            'ANOMALY': (0, 0, 255),     # Red
            'CONTAMINATION': (255, 255, 0)  # Yellow
        }
    
    result = image.copy()
    
    # Convert grayscale to BGR if needed
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    for defect in defects:
        defect_type = defect.get('type', 'UNKNOWN')
        color = colors.get(defect_type, (128, 128, 128))  # Gray for unknown
        
        if 'location' in defect:
            # Bounding box format
            x, y, w, h = defect['location']
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"{defect_type}"
            if 'confidence' in defect:
                label += f": {defect['confidence']:.2f}"
            cv2.putText(result, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        elif 'line' in defect:
            # Line format
            x1, y1, x2, y2 = defect['line']
            cv2.line(result, (x1, y1), (x2, y2), color, 2)
            
        elif 'center' in defect:
            # Circle format
            cx, cy = defect['center']
            radius = int(np.sqrt(defect.get('area', 100) / np.pi))
            cv2.circle(result, (cx, cy), radius, color, 2)
    
    return result


def create_comparison_grid(images: List[np.ndarray],
                          titles: List[str],
                          grid_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a grid visualization of multiple images.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        grid_shape: Optional (rows, cols) for grid layout
        
    Returns:
        Combined grid image
    """
    n_images = len(images)
    
    # Determine grid shape if not provided
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_shape
    
    # Get max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    # Create output grid
    grid_h = max_h * rows
    grid_w = max_w * cols
    
    # Determine if color or grayscale
    is_color = any(len(img.shape) == 3 for img in images)
    
    if is_color:
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    else:
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    
    # Place images in grid
    for idx, (img, title) in enumerate(zip(images, titles)):
        if idx >= rows * cols:
            break
            
        row = idx // cols
        col = idx % cols
        
        y_start = row * max_h
        x_start = col * max_w
        
        # Convert grayscale to BGR if needed for consistency
        if is_color and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif not is_color and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize if needed
        if img.shape[0] != max_h or img.shape[1] != max_w:
            img = cv2.resize(img, (max_w, max_h))
        
        # Place image
        if is_color:
            grid[y_start:y_start+max_h, x_start:x_start+max_w] = img
        else:
            grid[y_start:y_start+max_h, x_start:x_start+max_w] = img
        
        # Add title
        if is_color:
            cv2.putText(grid, title, (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(grid, title, (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    
    return grid


def create_heatmap(data: np.ndarray,
                   colormap: str = 'hot',
                   normalize: bool = True) -> np.ndarray:
    """
    Create a heatmap visualization from data.
    
    Args:
        data: 2D array of values
        colormap: OpenCV colormap name or ID
        normalize: Whether to normalize data to 0-255
        
    Returns:
        Heatmap image (BGR)
    """
    # Normalize if requested
    if normalize:
        data_norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    else:
        data_norm = np.clip(data, 0, 255)
    
    data_uint8 = data_norm.astype(np.uint8)
    
    # Apply colormap
    colormaps = {
        'hot': cv2.COLORMAP_HOT,
        'jet': cv2.COLORMAP_JET,
        'cool': cv2.COLORMAP_COOL,
        'hsv': cv2.COLORMAP_HSV,
        'rainbow': cv2.COLORMAP_RAINBOW
    }
    
    if isinstance(colormap, str):
        colormap_id = colormaps.get(colormap, cv2.COLORMAP_HOT)
    else:
        colormap_id = colormap
    
    heatmap = cv2.applyColorMap(data_uint8, colormap_id)
    
    return heatmap


def overlay_heatmap(image: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    
    Args:
        image: Base image
        heatmap: Heatmap to overlay
        alpha: Transparency of heatmap (0=transparent, 1=opaque)
        
    Returns:
        Combined image
    """
    # Convert grayscale to BGR if needed
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()
    
    # Ensure heatmap is BGR
    if len(heatmap.shape) == 2:
        heatmap_bgr = create_heatmap(heatmap)
    else:
        heatmap_bgr = heatmap
    
    # Resize if needed
    if image_bgr.shape[:2] != heatmap_bgr.shape[:2]:
        heatmap_bgr = cv2.resize(heatmap_bgr, 
                                 (image_bgr.shape[1], image_bgr.shape[0]))
    
    # Blend
    result = cv2.addWeighted(image_bgr, 1-alpha, heatmap_bgr, alpha, 0)
    
    return result


def plot_feature_histogram(features: Dict[str, float],
                          title: str = "Feature Distribution",
                          save_path: Optional[str] = None):
    """
    Create histogram plot of features.
    
    Args:
        features: Dictionary of feature names and values
        title: Plot title
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort features by value
    sorted_features = sorted(features.items(), key=lambda x: x[1])
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    # Create bar plot
    bars = ax.bar(range(len(names)), values)
    
    # Color bars by value
    norm = plt.Normalize(min(values), max(values))
    colors = plt.cm.viridis(norm(values))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Labels and title
    ax.set_xlabel('Feature')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def create_detection_report_image(image: np.ndarray,
                                 detections: List[Dict],
                                 statistics: Dict[str, float]) -> np.ndarray:
    """
    Create a comprehensive detection report image.
    
    Args:
        image: Original image
        detections: List of detections
        statistics: Dictionary of statistics to display
        
    Returns:
        Report image
    """
    # Create larger canvas for report
    h, w = image.shape[:2]
    report_h = h + 200  # Extra space for text
    report_w = max(w, 600)  # Minimum width for text
    
    # Create white background
    if len(image.shape) == 3:
        report = np.ones((report_h, report_w, 3), dtype=np.uint8) * 255
    else:
        report = np.ones((report_h, report_w), dtype=np.uint8) * 255
    
    # Place original image with detections
    img_with_detections = draw_defects_overlay(image, detections)
    
    # Center image if canvas is wider
    x_offset = (report_w - w) // 2
    if len(image.shape) == 3:
        report[0:h, x_offset:x_offset+w] = img_with_detections
    else:
        report[0:h, x_offset:x_offset+w] = cv2.cvtColor(img_with_detections, 
                                                        cv2.COLOR_BGR2GRAY)
    
    # Add statistics text
    text_y = h + 30
    text_color = (0, 0, 0) if len(report.shape) == 3 else 0
    
    # Title
    cv2.putText(report, "DETECTION REPORT", (20, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    text_y += 40
    
    # Detection counts by type
    type_counts = {}
    for det in detections:
        det_type = det.get('type', 'Unknown')
        type_counts[det_type] = type_counts.get(det_type, 0) + 1
    
    cv2.putText(report, f"Total Detections: {len(detections)}", (20, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
    text_y += 25
    
    for det_type, count in type_counts.items():
        cv2.putText(report, f"  {det_type}: {count}", (40, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        text_y += 20
    
    # Statistics
    if statistics:
        text_y += 10
        cv2.putText(report, "Statistics:", (20, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        text_y += 25
        
        for key, value in list(statistics.items())[:5]:  # Show first 5
            text = f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}"
            cv2.putText(report, text, (40, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            text_y += 20
    
    return report


def main():
    """Standalone test function."""
    print("Visualization Module - Standalone Test")
    print("-" * 40)
    
    # Create test image
    test_image = np.ones((300, 400), dtype=np.uint8) * 128
    
    # Create some test defects
    test_defects = [
        {'type': 'Blob', 'location': (50, 50, 40, 40), 'confidence': 0.85},
        {'type': 'Scratch', 'line': (100, 100, 200, 150)},
        {'type': 'Dig', 'center': (250, 100), 'area': 200},
        {'type': 'ANOMALY', 'location': (150, 200, 60, 50), 'confidence': 0.92}
    ]
    
    print("Drawing defects overlay...")
    overlay = draw_defects_overlay(test_image, test_defects)
    cv2.imwrite("viz_overlay_test.png", overlay)
    print("  Overlay saved")
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    images = [
        test_image,
        cv2.GaussianBlur(test_image, (15, 15), 0),
        cv2.Canny(test_image, 50, 150),
        overlay
    ]
    titles = ["Original", "Blurred", "Edges", "Detections"]
    grid = create_comparison_grid(images, titles, (2, 2))
    cv2.imwrite("viz_grid_test.png", grid)
    print("  Grid saved")
    
    # Create heatmap
    print("\nCreating heatmap...")
    heatmap_data = np.random.randn(300, 400) * 50 + 128
    heatmap = create_heatmap(heatmap_data)
    cv2.imwrite("viz_heatmap_test.png", heatmap)
    print("  Heatmap saved")
    
    # Overlay heatmap
    print("\nCreating heatmap overlay...")
    overlay_heat = overlay_heatmap(test_image, heatmap_data, alpha=0.4)
    cv2.imwrite("viz_overlay_heat_test.png", overlay_heat)
    print("  Heatmap overlay saved")
    
    # Create detection report
    print("\nCreating detection report...")
    stats = {
        'mean_confidence': 0.885,
        'total_area': 350,
        'quality_score': 75.5
    }
    report = create_detection_report_image(test_image, test_defects, stats)
    cv2.imwrite("viz_report_test.png", report)
    print("  Report saved")
    
    print("\nAll visualization tests completed!")


if __name__ == "__main__":
    main()
