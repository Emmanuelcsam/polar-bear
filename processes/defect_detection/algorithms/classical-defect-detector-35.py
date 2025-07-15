#!/usr/bin/env python3
"""
Traditional Defect Detection Module
Computer vision based defect detection for fiber optic images.
Includes scratch, pit, contamination, and crack detection algorithms.
"""

import cv2
import numpy as np
import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, NamedTuple
from dataclasses import dataclass, field
# scipy and sklearn imports are optional - they will be imported if available
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
import warnings

warnings.filterwarnings('ignore')


@dataclass
class Defect:
    """Represents a detected defect"""
    type: str
    location: Tuple[int, int]  # (x, y) center
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    severity: float  # 0-1
    confidence: float  # 0-1
    zone: str  # core, cladding, ferrule
    area: int
    properties: Dict[str, Any] = field(default_factory=dict)
    detection_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type,
            'location': self.location,
            'bbox': self.bbox,
            'severity': self.severity,
            'confidence': self.confidence,
            'zone': self.zone,
            'area': self.area,
            'properties': self.properties,
            'detection_method': self.detection_method
        }


class TraditionalDefectDetector:
    """
    Traditional computer vision defect detection for fiber optic images
    """
    
    def __init__(self,
                 min_defect_size: int = 25,
                 scratch_aspect_ratio_threshold: float = 3.0,
                 pit_circularity_threshold: float = 0.7,
                 contamination_area_threshold: int = 100,
                 noise_filter_kernel_size: int = 3,
                 morphology_kernel_size: int = 5):
        """
        Initialize the traditional defect detector
        
        Args:
            min_defect_size: Minimum area for defect consideration
            scratch_aspect_ratio_threshold: Minimum aspect ratio for scratch detection
            pit_circularity_threshold: Minimum circularity for pit detection
            contamination_area_threshold: Minimum area for contamination detection
            noise_filter_kernel_size: Kernel size for noise filtering
            morphology_kernel_size: Kernel size for morphological operations
        """
        self.min_defect_size = min_defect_size
        self.scratch_aspect_ratio_threshold = scratch_aspect_ratio_threshold
        self.pit_circularity_threshold = pit_circularity_threshold
        self.contamination_area_threshold = contamination_area_threshold
        self.noise_filter_kernel_size = noise_filter_kernel_size
        self.morphology_kernel_size = morphology_kernel_size
    
    def preprocess_zone(self, image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
        """
        Preprocess image region for defect detection
        
        Args:
            image: Input image (BGR or grayscale)
            zone_mask: Binary mask for the zone of interest
            
        Returns:
            Preprocessed grayscale image
        """
        # Apply zone mask
        if len(image.shape) == 3:
            masked = cv2.bitwise_and(image, image, mask=zone_mask)
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.bitwise_and(image, image, mask=zone_mask)
        
        # Apply noise filtering
        if self.noise_filter_kernel_size > 1:
            gray = cv2.medianBlur(gray, self.noise_filter_kernel_size)
        
        return gray
    
    def detect_scratches(self, image: np.ndarray, zone_mask: np.ndarray, 
                        zone_name: str) -> List[Defect]:
        """
        Detect scratch-type defects using oriented line detection
        
        Args:
            image: Input image
            zone_mask: Binary mask for the zone
            zone_name: Name of the zone
            
        Returns:
            List of detected scratch defects
        """
        defects = []
        
        try:
            # Preprocess
            gray = self.preprocess_zone(image, zone_mask)
            
            if np.sum(zone_mask) == 0:  # No valid zone
                return defects
            
            # Line detection using multiple orientations
            angles = np.linspace(0, 180, 12, endpoint=False)
            line_responses = []
            
            for angle in angles:
                # Create oriented kernel for line detection
                kernel_size = 15
                kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
                cv2.line(kernel, (0, kernel_size//2), (kernel_size-1, kernel_size//2), (1.0,), 1)
                
                # Rotate kernel
                center = (kernel_size//2, kernel_size//2)
                M = cv2.getRotationMatrix2D(center, angle, 1)
                kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
                
                # Normalize kernel
                if kernel.sum() > 0:
                    kernel = kernel / kernel.sum()
                
                # Convolve with image
                response = cv2.filter2D(gray, -1, kernel)
                line_responses.append(response)
            
            # Maximum response across orientations
            max_response = np.max(line_responses, axis=0)
            
            # Threshold using Otsu's method
            _, binary = cv2.threshold(max_response.astype(np.uint8), 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to connect line segments
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                             (self.morphology_kernel_size, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_defect_size:
                    continue
                
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's within the zone
                center_x, center_y = x + w//2, y + h//2
                if zone_mask[center_y, center_x] == 0:
                    continue
                
                # Check aspect ratio (scratches are elongated)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                if aspect_ratio > self.scratch_aspect_ratio_threshold:
                    # Calculate severity based on length and area
                    length = max(w, h)
                    severity = min(1.0, length / 100)
                    
                    # Calculate confidence based on response strength
                    mask_contour = np.zeros_like(binary)
                    cv2.fillPoly(mask_contour, [contour], (255,))
                    avg_response = np.mean(max_response[mask_contour > 0])
                    confidence = min(1.0, avg_response / 50)
                    
                    defect = Defect(
                        type="scratch",
                        location=(center_x, center_y),
                        bbox=(x, y, w, h),
                        severity=severity,
                        confidence=confidence,
                        zone=zone_name,
                        area=int(area),
                        properties={
                            'length': length,
                            'aspect_ratio': aspect_ratio,
                            'avg_response': float(avg_response)
                        },
                        detection_method="traditional_line_detection"
                    )
                    defects.append(defect)
            
        except Exception as e:
            print(f"Warning: Scratch detection failed: {e}")
        
        return defects
    
    def detect_pits(self, image: np.ndarray, zone_mask: np.ndarray, 
                   zone_name: str) -> List[Defect]:
        """
        Detect pit-type defects using circular feature detection
        
        Args:
            image: Input image
            zone_mask: Binary mask for the zone
            zone_name: Name of the zone
            
        Returns:
            List of detected pit defects
        """
        defects = []
        
        try:
            # Preprocess
            gray = self.preprocess_zone(image, zone_mask)
            
            if np.sum(zone_mask) == 0:
                return defects
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Find dark regions (pits appear as dark spots)
            # Use adaptive threshold to handle varying illumination
            adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, 
                                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 11, 2)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.morphology_kernel_size, 
                                              self.morphology_kernel_size))
            cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_defect_size:
                    continue
                
                # Calculate properties
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2
                
                # Check if it's within the zone
                if zone_mask[center_y, center_x] == 0:
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    continue
                
                # Check if it's circular enough to be a pit
                if circularity > self.pit_circularity_threshold:
                    # Calculate severity based on size
                    severity = min(1.0, area / 500)
                    
                    # Calculate confidence based on circularity and darkness
                    mask_contour = np.zeros_like(gray)
                    cv2.fillPoly(mask_contour, [contour], (255,))
                    avg_intensity = np.mean(gray[mask_contour > 0])
                    darkness_factor = 1.0 - (avg_intensity / 255.0)
                    confidence = min(1.0, float(circularity * darkness_factor))
                    
                    defect = Defect(
                        type="pit",
                        location=(center_x, center_y),
                        bbox=(x, y, w, h),
                        severity=severity,
                        confidence=confidence,
                        zone=zone_name,
                        area=int(area),
                        properties={
                            'circularity': float(circularity),
                            'avg_intensity': float(avg_intensity),
                            'darkness_factor': float(darkness_factor)
                        },
                        detection_method="traditional_circular_detection"
                    )
                    defects.append(defect)
            
        except Exception as e:
            print(f"Warning: Pit detection failed: {e}")
        
        return defects
    
    def detect_contamination(self, image: np.ndarray, zone_mask: np.ndarray, 
                           zone_name: str) -> List[Defect]:
        """
        Detect contamination using texture and color analysis
        
        Args:
            image: Input image
            zone_mask: Binary mask for the zone
            zone_name: Name of the zone
            
        Returns:
            List of detected contamination defects
        """
        defects = []
        
        try:
            # Preprocess
            gray = self.preprocess_zone(image, zone_mask)
            
            if np.sum(zone_mask) == 0:
                return defects
            
            # Calculate local statistics for anomaly detection
            # Use a sliding window approach
            window_size = 15
            
            # Calculate local mean and standard deviation
            kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size**2)
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            
            # Calculate local variance
            gray_squared = (gray.astype(np.float32))**2
            local_mean_squared = cv2.filter2D(gray_squared, -1, kernel)
            local_variance = local_mean_squared - local_mean**2
            local_std = np.sqrt(np.maximum(local_variance, 0))
            
            # Find areas with unusual statistics
            global_mean = np.mean(gray[zone_mask > 0])
            global_std = np.std(gray[zone_mask > 0])
            
            # Anomaly score based on deviation from local statistics
            anomaly_score = np.abs(gray.astype(np.float32) - local_mean) / (local_std + 1e-8)
            
            # Threshold anomaly score
            threshold = np.percentile(anomaly_score[zone_mask > 0], 95)
            anomaly_mask = (anomaly_score > threshold).astype(np.uint8) * 255
            
            # Apply zone mask
            anomaly_mask = cv2.bitwise_and(anomaly_mask, zone_mask)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
            anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.contamination_area_threshold:
                    continue
                
                # Calculate properties
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2
                
                # Calculate severity based on area and anomaly strength
                mask_contour = np.zeros_like(gray)
                cv2.fillPoly(mask_contour, [contour], (255,))
                avg_anomaly = np.mean(anomaly_score[mask_contour > 0])
                
                severity = min(1.0, float((area / 1000) * (avg_anomaly / threshold)))
                confidence = min(1.0, float(avg_anomaly / threshold))
                
                defect = Defect(
                    type="contamination",
                    location=(center_x, center_y),
                    bbox=(x, y, w, h),
                    severity=severity,
                    confidence=confidence,
                    zone=zone_name,
                    area=int(area),
                    properties={
                        'avg_anomaly_score': float(avg_anomaly),
                        'anomaly_threshold': float(threshold)
                    },
                    detection_method="traditional_anomaly_detection"
                )
                defects.append(defect)
            
        except Exception as e:
            print(f"Warning: Contamination detection failed: {e}")
        
        return defects
    
    def detect_cracks(self, image: np.ndarray, zone_mask: np.ndarray, 
                     zone_name: str) -> List[Defect]:
        """
        Detect crack-type defects using edge detection and morphology
        
        Args:
            image: Input image
            zone_mask: Binary mask for the zone
            zone_name: Name of the zone
            
        Returns:
            List of detected crack defects
        """
        defects = []
        
        try:
            # Preprocess
            gray = self.preprocess_zone(image, zone_mask)
            
            if np.sum(zone_mask) == 0:
                return defects
            
            # Apply edge detection with multiple scales
            edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)
            edges2 = cv2.Canny(gray, 30, 100, apertureSize=3)
            edges3 = cv2.Canny(gray, 100, 200, apertureSize=3)
            
            # Combine edge responses
            combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
            
            # Apply zone mask
            combined_edges = cv2.bitwise_and(combined_edges, zone_mask)
            
            # Morphological operations to connect crack segments
            # Use different kernels for different crack orientations
            kernels = [
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)),  # Vertical
                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),  # Horizontal
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),  # Diagonal
            ]
            
            crack_candidates = []
            for kernel in kernels:
                processed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
                processed = cv2.morphologyEx(processed, cv2.MORPH_DILATE, kernel, iterations=1)
                crack_candidates.append(processed)
            
            # Combine all candidates
            final_cracks = np.zeros_like(combined_edges)
            for candidate in crack_candidates:
                final_cracks = cv2.bitwise_or(final_cracks, candidate)
            
            # Find contours
            contours, _ = cv2.findContours(final_cracks, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_defect_size:
                    continue
                
                # Calculate properties
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2
                
                # Check if it's within the zone
                if zone_mask[center_y, center_x] == 0:
                    continue
                
                # Calculate crack-like properties
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                length = max(w, h)
                
                # Cracks should be elongated
                if aspect_ratio > 2.0:
                    # Calculate severity based on length and area
                    severity = min(1.0, length / 200)
                    
                    # Calculate confidence based on aspect ratio and edge strength
                    mask_contour = np.zeros_like(gray)
                    cv2.fillPoly(mask_contour, [contour], (255,))
                    edge_strength = np.mean(combined_edges[mask_contour > 0])
                    confidence = min(1.0, float((aspect_ratio / 10) * (edge_strength / 255)))
                    
                    defect = Defect(
                        type="crack",
                        location=(center_x, center_y),
                        bbox=(x, y, w, h),
                        severity=severity,
                        confidence=confidence,
                        zone=zone_name,
                        area=int(area),
                        properties={
                            'length': length,
                            'aspect_ratio': float(aspect_ratio),
                            'edge_strength': float(edge_strength)
                        },
                        detection_method="traditional_edge_detection"
                    )
                    defects.append(defect)
            
        except Exception as e:
            print(f"Warning: Crack detection failed: {e}")
        
        return defects
    
    def detect_all_defects(self, image: np.ndarray, zone_masks: Dict[str, np.ndarray]) -> Dict[str, List[Defect]]:
        """
        Detect all types of defects in all zones
        
        Args:
            image: Input image
            zone_masks: Dictionary of zone masks {zone_name: mask}
            
        Returns:
            Dictionary of detected defects by zone
        """
        all_defects = {}
        
        for zone_name, zone_mask in zone_masks.items():
            zone_defects = []
            
            # Detect different types of defects
            zone_defects.extend(self.detect_scratches(image, zone_mask, zone_name))
            zone_defects.extend(self.detect_pits(image, zone_mask, zone_name))
            zone_defects.extend(self.detect_contamination(image, zone_mask, zone_name))
            zone_defects.extend(self.detect_cracks(image, zone_mask, zone_name))
            
            all_defects[zone_name] = zone_defects
        
        return all_defects
    
    def create_zone_masks(self, image_shape: Tuple[int, int], 
                         center: Tuple[int, int],
                         core_radius: Optional[int] = None,
                         cladding_radius: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Create zone masks for defect detection
        
        Args:
            image_shape: Shape of the image (height, width)
            center: Center coordinates
            core_radius: Core radius (optional)
            cladding_radius: Cladding radius (optional)
            
        Returns:
            Dictionary of zone masks
        """
        height, width = image_shape
        masks = {}
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:height, :width]
        cx, cy = center
        
        # Calculate distances from center
        distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        
        # Core mask
        if core_radius:
            masks['core'] = (distances <= core_radius).astype(np.uint8) * 255
        else:
            masks['core'] = np.zeros((height, width), dtype=np.uint8)
        
        # Cladding mask (annular region)
        if cladding_radius:
            cladding_mask = distances <= cladding_radius
            if core_radius:
                core_mask = distances <= core_radius
                cladding_mask = cladding_mask & ~core_mask
            masks['cladding'] = cladding_mask.astype(np.uint8) * 255
        else:
            masks['cladding'] = np.zeros((height, width), dtype=np.uint8)
        
        # Ferrule mask (outside cladding)
        if cladding_radius:
            masks['ferrule'] = (distances > cladding_radius).astype(np.uint8) * 255
        else:
            # Use entire image as ferrule if no cladding defined
            masks['ferrule'] = np.ones((height, width), dtype=np.uint8) * 255
        
        return masks


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Traditional Defect Detection')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--center', nargs=2, type=int, metavar=('X', 'Y'),
                       help='Center coordinates (x y)')
    parser.add_argument('--core-radius', type=int, help='Core radius in pixels')
    parser.add_argument('--cladding-radius', type=int, help='Cladding radius in pixels')
    parser.add_argument('--output-dir', default='defect_detection_output',
                       help='Output directory for results')
    parser.add_argument('--min-defect-size', type=int, default=25,
                       help='Minimum defect size in pixels')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save defect visualization images')
    
    args = parser.parse_args()
    
    # Load image
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not read image: {args.image_path}")
        return
    
    # Set up center and radii
    if args.center:
        center = tuple(args.center)
    else:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
    
    core_radius = args.core_radius
    cladding_radius = args.cladding_radius
    
    # Create detector
    detector = TraditionalDefectDetector(min_defect_size=args.min_defect_size)
    
    # Create zone masks
    zone_masks = detector.create_zone_masks(
        image.shape[:2], center, core_radius, cladding_radius
    )
    
    # Detect defects
    all_defects = detector.detect_all_defects(image, zone_masks)
    
    # Print results
    total_defects = sum(len(defects) for defects in all_defects.values())
    print(f"Total defects detected: {total_defects}")
    
    for zone_name, defects in all_defects.items():
        if defects:
            print(f"\n{zone_name.upper()} zone - {len(defects)} defects:")
            for defect in defects:
                print(f"  {defect.type}: severity={defect.severity:.3f}, "
                      f"confidence={defect.confidence:.3f}, area={defect.area}")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save JSON results
        results_data = {}
        for zone_name, defects in all_defects.items():
            results_data[zone_name] = [defect.to_dict() for defect in defects]
        
        json_path = os.path.join(args.output_dir, f"{Path(args.image_path).stem}_defects.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        
        print(f"\nResults saved to: {json_path}")
        
        # Save visualizations if requested
        if args.save_visualizations:
            vis_image = image.copy()
            
            colors = {
                'scratch': (0, 255, 255),    # Yellow
                'pit': (255, 0, 0),          # Blue
                'contamination': (0, 165, 255),  # Orange
                'crack': (0, 0, 255)         # Red
            }
            
            for zone_name, defects in all_defects.items():
                for defect in defects:
                    color = colors.get(defect.type, (255, 255, 255))
                    x, y, w, h = defect.bbox
                    
                    # Draw bounding box
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Add label
                    label = f"{defect.type} ({defect.confidence:.2f})"
                    cv2.putText(vis_image, label, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            vis_path = os.path.join(args.output_dir, f"{Path(args.image_path).stem}_defects_vis.png")
            cv2.imwrite(vis_path, vis_image)
            print(f"Visualization saved to: {vis_path}")


if __name__ == "__main__":
    main()
