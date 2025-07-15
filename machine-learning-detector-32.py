#!/usr/bin/env python3
"""
ML Anomaly Detection - Standalone Module
Extracted from fiber optic defect detection system
Uses machine learning for anomaly and defect detection
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from scipy import stats, ndimage
from dataclasses import dataclass


@dataclass
class Defect:
    """Represents a detected defect"""
    type: str
    location: Tuple[int, int]  # (x, y) center
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    severity: float  # 0-1
    confidence: float  # 0-1
    area: int
    properties: Dict[str, Any]
    detection_method: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type,
            'location': self.location,
            'bbox': self.bbox,
            'severity': self.severity,
            'confidence': self.confidence,
            'area': self.area,
            'properties': self.properties,
            'detection_method': self.detection_method
        }


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def statistical_anomaly_detection(image, mask=None, contamination=0.1, n_features=10):
    """
    Detect anomalies using statistical methods.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Region of interest mask
        contamination (float): Expected proportion of outliers
        n_features (int): Number of statistical features to extract
        
    Returns:
        List[Defect]: List of detected anomalies
    """
    defects = []
    
    if mask is not None:
        # Apply mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        roi_coords = np.where(mask > 0)
        if len(roi_coords[0]) == 0:
            return defects
    else:
        masked_image = image
        roi_coords = None
    
    try:
        # Extract features for each pixel in ROI
        features = []
        pixel_locations = []
        
        h, w = image.shape[:2]
        
        # Define neighborhood size for feature extraction
        neighborhood_size = 5
        half_size = neighborhood_size // 2
        
        if roi_coords is not None:
            # Only process pixels in ROI
            for i, j in zip(roi_coords[0], roi_coords[1]):
                if (half_size <= i < h - half_size and 
                    half_size <= j < w - half_size):
                    
                    # Extract neighborhood
                    neighborhood = image[i-half_size:i+half_size+1, 
                                       j-half_size:j+half_size+1]
                    
                    # Compute features
                    feature_vector = extract_pixel_features(neighborhood, i, j, image)
                    features.append(feature_vector)
                    pixel_locations.append((j, i))  # (x, y) format
        else:
            # Process entire image
            for i in range(half_size, h - half_size):
                for j in range(half_size, w - half_size):
                    neighborhood = image[i-half_size:i+half_size+1, 
                                       j-half_size:j+half_size+1]
                    
                    feature_vector = extract_pixel_features(neighborhood, i, j, image)
                    features.append(feature_vector)
                    pixel_locations.append((j, i))  # (x, y) format
        
        if len(features) < 10:
            return defects
        
        features = np.array(features)
        
        # Detect outliers using Elliptic Envelope
        detector = EllipticEnvelope(contamination=contamination, random_state=42)
        outlier_labels = detector.fit_predict(features)
        
        # Find anomalous pixels
        anomaly_pixels = []
        for idx, label in enumerate(outlier_labels):
            if label == -1:  # Outlier
                anomaly_pixels.append(pixel_locations[idx])
        
        if not anomaly_pixels:
            return defects
        
        # Cluster anomalous pixels into defects
        if len(anomaly_pixels) > 1:
            clustering = DBSCAN(eps=10, min_samples=3)
            cluster_labels = clustering.fit_predict(anomaly_pixels)
            
            # Group pixels by cluster
            clusters = {}
            for idx, cluster_id in enumerate(cluster_labels):
                if cluster_id >= 0:  # Valid cluster
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(anomaly_pixels[idx])
            
            # Create defects from clusters
            for cluster_id, pixels in clusters.items():
                if len(pixels) >= 3:  # Minimum defect size
                    defect = create_defect_from_pixels(pixels, image, 'statistical_anomaly')
                    defects.append(defect)
        else:
            # Single anomalous pixel
            defect = create_defect_from_pixels(anomaly_pixels, image, 'statistical_anomaly')
            defects.append(defect)
            
    except Exception as e:
        print(f"Statistical anomaly detection failed: {e}")
    
    return defects


def extract_pixel_features(neighborhood, center_i, center_j, full_image):
    """
    Extract statistical features from a pixel neighborhood.
    
    Args:
        neighborhood (np.ndarray): Local neighborhood around pixel
        center_i (int): Y coordinate in full image
        center_j (int): X coordinate in full image
        full_image (np.ndarray): Full image
        
    Returns:
        np.ndarray: Feature vector
    """
    features = []
    
    # Basic statistics
    features.append(np.mean(neighborhood))
    features.append(np.std(neighborhood))
    features.append(np.median(neighborhood))
    features.append(np.min(neighborhood))
    features.append(np.max(neighborhood))
    
    # Texture features
    features.append(np.var(neighborhood))
    features.append(stats.skew(neighborhood.flatten()))
    features.append(stats.kurtosis(neighborhood.flatten()))
    
    # Gradient features
    grad_x = cv2.Sobel(neighborhood, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(neighborhood, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features.append(np.mean(gradient_magnitude))
    features.append(np.max(gradient_magnitude))
    
    # Local binary pattern (simplified)
    center_pixel = neighborhood[neighborhood.shape[0]//2, neighborhood.shape[1]//2]
    binary_pattern = (neighborhood > center_pixel).astype(int)
    features.append(np.sum(binary_pattern))
    
    return np.array(features)


def morphological_defect_detection(image, mask=None, min_area=5, max_area=1000):
    """
    Detect defects using morphological operations.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Region of interest mask
        min_area (int): Minimum defect area
        max_area (int): Maximum defect area
        
    Returns:
        List[Defect]: List of detected defects
    """
    defects = []
    
    try:
        # Apply mask if provided
        if mask is not None:
            image = cv2.bitwise_and(image, image, mask=mask)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate properties
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Estimate severity based on size and intensity
                roi = image[y:y+h, x:x+w]
                avg_intensity = np.mean(roi)
                severity = min(1.0, area / max_area + (255 - avg_intensity) / 255)
                
                # Classify defect type based on shape and intensity
                aspect_ratio = w / h if h > 0 else 1
                circularity = 4 * np.pi * area / (cv2.arcLength(contour, True)**2) if cv2.arcLength(contour, True) > 0 else 0
                
                if circularity > 0.7:
                    defect_type = 'pit' if avg_intensity < 100 else 'contamination'
                elif aspect_ratio > 2 or aspect_ratio < 0.5:
                    defect_type = 'scratch'
                else:
                    defect_type = 'other_defect'
                
                defect = Defect(
                    type=defect_type,
                    location=(center_x, center_y),
                    bbox=(x, y, w, h),
                    severity=severity,
                    confidence=min(1.0, circularity + 0.5),
                    area=int(area),
                    properties={
                        'aspect_ratio': aspect_ratio,
                        'circularity': circularity,
                        'avg_intensity': avg_intensity
                    },
                    detection_method='morphological'
                )
                defects.append(defect)
        
    except Exception as e:
        print(f"Morphological defect detection failed: {e}")
    
    return defects


def edge_discontinuity_detection(image, mask=None, canny_thresh1=50, canny_thresh2=150, min_gap_size=5):
    """
    Detect defects by finding discontinuities in edges.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Region of interest mask
        canny_thresh1 (int): Lower Canny threshold
        canny_thresh2 (int): Upper Canny threshold
        min_gap_size (int): Minimum gap size to be considered a defect
        
    Returns:
        List[Defect]: List of detected edge discontinuities
    """
    defects = []
    
    try:
        # Apply mask if provided
        if mask is not None:
            image = cv2.bitwise_and(image, image, mask=mask)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
        
        # Apply mask to edges
        if mask is not None:
            edges = cv2.bitwise_and(edges, mask)
        
        # Find contours of edge regions
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for gaps in what should be continuous edges
        for contour in contours:
            if cv2.arcLength(contour, False) > 20:  # Only consider longer edges
                # Analyze contour for gaps
                gaps = find_edge_gaps(contour, min_gap_size)
                
                for gap in gaps:
                    center_x, center_y = gap['center']
                    gap_size = gap['size']
                    
                    # Create bounding box around gap
                    bbox_size = max(gap_size, 10)
                    x = max(0, center_x - bbox_size // 2)
                    y = max(0, center_y - bbox_size // 2)
                    w = min(bbox_size, image.shape[1] - x)
                    h = min(bbox_size, image.shape[0] - y)
                    
                    severity = min(1.0, gap_size / 50.0)  # Normalize gap size
                    
                    defect = Defect(
                        type='edge_discontinuity',
                        location=(center_x, center_y),
                        bbox=(x, y, w, h),
                        severity=severity,
                        confidence=0.7,
                        area=gap_size,
                        properties={'gap_size': gap_size},
                        detection_method='edge_discontinuity'
                    )
                    defects.append(defect)
        
    except Exception as e:
        print(f"Edge discontinuity detection failed: {e}")
    
    return defects


def find_edge_gaps(contour, min_gap_size):
    """
    Find gaps in an edge contour.
    
    Args:
        contour (np.ndarray): Edge contour
        min_gap_size (int): Minimum gap size to detect
        
    Returns:
        List[Dict]: List of gap information
    """
    gaps = []
    
    if len(contour) < 3:
        return gaps
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        distances.append(distance)
    
    # Find unusually large gaps
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    threshold = mean_distance + 2 * std_distance
    
    for i, distance in enumerate(distances):
        if distance > threshold and distance > min_gap_size:
            p1 = contour[i][0]
            p2 = contour[(i + 1) % len(contour)][0]
            center_x = (p1[0] + p2[0]) // 2
            center_y = (p1[1] + p2[1]) // 2
            
            gaps.append({
                'center': (center_x, center_y),
                'size': int(distance)
            })
    
    return gaps


def create_defect_from_pixels(pixels, image, detection_method):
    """
    Create a Defect object from a list of anomalous pixels.
    
    Args:
        pixels (List[Tuple[int, int]]): List of (x, y) pixel coordinates
        image (np.ndarray): Source image
        detection_method (str): Name of detection method
        
    Returns:
        Defect: Defect object
    """
    if not pixels:
        return None
    
    # Calculate bounding box
    x_coords = [p[0] for p in pixels]
    y_coords = [p[1] for p in pixels]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    
    # Calculate severity based on intensity variation
    intensities = [image[y, x] for x, y in pixels if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]]
    if intensities:
        intensity_std = np.std(intensities)
        avg_intensity = np.mean(intensities)
        severity = min(1.0, intensity_std / 50.0 + (255 - avg_intensity) / 255.0)
    else:
        severity = 0.5
    
    # Determine defect type based on properties
    area = len(pixels)
    aspect_ratio = w / h if h > 0 else 1
    
    if area < 10:
        defect_type = 'small_defect'
    elif aspect_ratio > 3:
        defect_type = 'scratch'
    elif avg_intensity < 50 if intensities else False:
        defect_type = 'pit'
    else:
        defect_type = 'anomaly'
    
    return Defect(
        type=defect_type,
        location=(center_x, center_y),
        bbox=(min_x, min_y, w, h),
        severity=severity,
        confidence=0.8,
        area=area,
        properties={
            'pixel_count': len(pixels),
            'intensity_std': intensity_std if intensities else 0,
            'avg_intensity': avg_intensity if intensities else 0
        },
        detection_method=detection_method
    )


def comprehensive_defect_detection(image_path, mask_path=None, output_dir='ml_defect_output'):
    """
    Run comprehensive defect detection using multiple ML methods.
    
    Args:
        image_path (str): Path to input image
        mask_path (str, optional): Path to mask image
        output_dir (str): Output directory for results
        
    Returns:
        dict: Detection results with all found defects
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result
    result = {
        'method': 'comprehensive_ml_detection',
        'image_path': image_path,
        'mask_path': mask_path,
        'success': False,
        'defects': [],
        'detection_methods': []
    }
    
    try:
        # Load image
        if not os.path.exists(image_path):
            result['error'] = f"Image not found: {image_path}"
            return result
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            result['error'] = f"Could not read image: {image_path}"
            return result
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        all_defects = []
        
        # Run statistical anomaly detection
        print("Running statistical anomaly detection...")
        statistical_defects = statistical_anomaly_detection(image, mask)
        all_defects.extend(statistical_defects)
        result['detection_methods'].append('statistical_anomaly')
        
        # Run morphological defect detection
        print("Running morphological defect detection...")
        morphological_defects = morphological_defect_detection(image, mask)
        all_defects.extend(morphological_defects)
        result['detection_methods'].append('morphological')
        
        # Run edge discontinuity detection
        print("Running edge discontinuity detection...")
        edge_defects = edge_discontinuity_detection(image, mask)
        all_defects.extend(edge_defects)
        result['detection_methods'].append('edge_discontinuity')
        
        # Convert defects to dictionaries
        result['defects'] = [defect.to_dict() for defect in all_defects]
        result['total_defects'] = len(all_defects)
        result['success'] = True
        
        # Create visualization
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw all defects
        colors = {
            'statistical_anomaly': (0, 255, 0),      # Green
            'morphological': (255, 0, 0),            # Blue
            'edge_discontinuity': (0, 0, 255),       # Red
            'pit': (255, 255, 0),                    # Cyan
            'scratch': (255, 0, 255),                # Magenta
            'contamination': (0, 255, 255),          # Yellow
            'other_defect': (128, 128, 128),         # Gray
            'small_defect': (128, 0, 128),           # Purple
            'anomaly': (0, 128, 255)                 # Orange
        }
        
        for defect_dict in result['defects']:
            x, y, w, h = defect_dict['bbox']
            defect_type = defect_dict['type']
            detection_method = defect_dict['detection_method']
            
            # Choose color based on type, fallback to detection method
            color = colors.get(defect_type, colors.get(detection_method, (255, 255, 255)))
            
            # Draw bounding box
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{defect_type} ({defect_dict['severity']:.2f})"
            cv2.putText(visualization, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save visualization
        vis_path = os.path.join(output_dir, f'{base_filename}_ml_defects.jpg')
        cv2.imwrite(vis_path, visualization)
        
        # Save results
        result_path = os.path.join(output_dir, f'{base_filename}_ml_results.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4, cls=NumpyEncoder)
        
        print(f"Detection complete. Found {len(all_defects)} defects.")
        
    except Exception as e:
        result['error'] = f"Detection failed: {str(e)}"
        print(f"Error: {e}")
    
    return result


def main():
    """Command line interface for ML defect detection"""
    parser = argparse.ArgumentParser(description='ML-based Defect Detection for Images')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--mask-path', help='Path to mask image (optional)')
    parser.add_argument('--output-dir', default='ml_defect_output',
                       help='Output directory (default: ml_defect_output)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run defect detection
    result = comprehensive_defect_detection(
        image_path=args.image_path,
        mask_path=args.mask_path,
        output_dir=args.output_dir
    )
    
    # Print results
    if args.verbose:
        print(json.dumps(result, indent=2, cls=NumpyEncoder))
    else:
        if result['success']:
            print(f"✓ ML defect detection successful!")
            print(f"  Total defects found: {result['total_defects']}")
            print(f"  Detection methods used: {', '.join(result['detection_methods'])}")
            
            # Group by type
            defect_types = {}
            for defect in result['defects']:
                defect_type = defect['type']
                defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
            
            print("  Defect breakdown:")
            for defect_type, count in defect_types.items():
                print(f"    {defect_type}: {count}")
        else:
            print(f"✗ ML defect detection failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
