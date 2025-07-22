#!/usr/bin/env python3
"""
Anomaly Detection Module
Advanced anomaly detection using morphological operations and statistical methods.
Extracted and optimized from separation.py and detection.py.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

try:
    from scipy.ndimage import binary_opening, binary_closing
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AnomalyDetector:
    """
    Advanced anomaly detection system using multiple methods.
    Detects defects, scratches, digs, and other anomalies in images.
    """
    
    def __init__(self, 
                 blackhat_threshold: int = 30,
                 morphology_kernel_size: int = 15,
                 min_defect_area: int = 10,
                 max_defect_area: int = 5000,
                 use_scipy: bool = True):
        """
        Initialize the anomaly detector.
        
        Args:
            blackhat_threshold: Threshold for blackhat morphology
            morphology_kernel_size: Size of morphological kernel
            min_defect_area: Minimum area for defect regions
            max_defect_area: Maximum area for defect regions
            use_scipy: Whether to use scipy for advanced morphology
        """
        self.blackhat_threshold = blackhat_threshold
        self.morphology_kernel_size = morphology_kernel_size
        self.min_defect_area = min_defect_area
        self.max_defect_area = max_defect_area
        self.use_scipy = use_scipy and HAS_SCIPY
        self.logger = logging.getLogger(__name__)
    
    def detect_and_inpaint_anomalies(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies and create inpainted version of the image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (inpainted_image, defect_mask)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()
                
            # Create morphological kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.morphology_kernel_size, self.morphology_kernel_size))
            
            # Blackhat morphology to detect dark spots/defects
            blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
            
            # Threshold to create binary mask
            _, defect_mask = cv2.threshold(blackhat, self.blackhat_threshold, 255, cv2.THRESH_BINARY)
            
            # Clean up mask with morphological operations
            if self.use_scipy:
                # Use scipy for more precise morphological operations
                defect_mask = binary_opening(defect_mask, structure=np.ones((3, 3)), iterations=2).astype(np.uint8)
                defect_mask = binary_closing(defect_mask, structure=np.ones((3, 3)), iterations=1).astype(np.uint8)
            else:
                # Use OpenCV morphological operations
                small_kernel = np.ones((3, 3), np.uint8)
                defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, small_kernel, iterations=2)
                defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, small_kernel, iterations=1)
            
            # Inpaint the original image
            inpainted_image = cv2.inpaint(image, defect_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            
            return inpainted_image, defect_mask
            
        except Exception as e:
            self.logger.error(f"Error in detect_and_inpaint_anomalies: {e}")
            return image, np.zeros(image.shape[:2], dtype=np.uint8)
    
    def detect_defect_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect defect regions and extract their properties.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of defect region dictionaries
        """
        try:
            # Get defect mask
            _, defect_mask = self.detect_and_inpaint_anomalies(image)
            
            # Find contours
            contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            defect_regions = []
            
            for i, contour in enumerate(contours):
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area < self.min_defect_area or area > self.max_defect_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                # Calculate additional properties
                perimeter = cv2.arcLength(contour, True)
                
                # Circularity (4π × Area) / (Perimeter²)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Aspect ratio
                aspect_ratio = w / h if h > 0 else 1
                
                # Extent (area / bounding box area)
                extent = area / (w * h) if (w * h) > 0 else 0
                
                # Solidity (area / convex hull area)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Calculate intensity statistics in the region
                mask = np.zeros(defect_mask.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], (255,))
                
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                region_pixels = gray[mask > 0]
                
                if len(region_pixels) > 0:
                    mean_intensity = np.mean(region_pixels)
                    std_intensity = np.std(region_pixels)
                    min_intensity = np.min(region_pixels)
                    max_intensity = np.max(region_pixels)
                else:
                    mean_intensity = std_intensity = min_intensity = max_intensity = 0
                
                # Create defect region dictionary
                defect_region = {
                    'id': i,
                    'contour': contour,
                    'area': float(area),
                    'perimeter': float(perimeter),
                    'centroid': (int(cx), int(cy)),
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'circularity': float(circularity),
                    'aspect_ratio': float(aspect_ratio),
                    'extent': float(extent),
                    'solidity': float(solidity),
                    'mean_intensity': float(mean_intensity),
                    'std_intensity': float(std_intensity),
                    'min_intensity': float(min_intensity),
                    'max_intensity': float(max_intensity),
                    'confidence': self._calculate_defect_confidence(area, circularity, solidity)
                }
                
                defect_regions.append(defect_region)
            
            self.logger.info(f"Detected {len(defect_regions)} defect regions")
            return defect_regions
            
        except Exception as e:
            self.logger.error(f"Error in detect_defect_regions: {e}")
            return []
    
    def _calculate_defect_confidence(self, area: float, circularity: float, solidity: float) -> float:
        """Calculate confidence score for a defect based on its properties."""
        # Base confidence on area (larger defects are more confident)
        area_score = min(area / 100.0, 1.0)  # Normalize to [0, 1]
        
        # Shape score (regular shapes are more likely to be real defects)
        shape_score = (circularity + solidity) / 2.0
        
        # Combine scores
        confidence = (area_score * 0.6 + shape_score * 0.4)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def detect_scratches(self, image: np.ndarray, 
                        min_line_length: int = 30, 
                        max_line_gap: int = 10) -> List[Dict]:
        """
        Detect scratch-like defects using Hough line detection.
        
        Args:
            image: Input image
            min_line_length: Minimum length for detected lines
            max_line_gap: Maximum gap between line segments
            
        Returns:
            List of scratch dictionaries
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=min_line_length, maxLineGap=max_line_gap)
            
            scratches = []
            
            if lines is not None:
                for i, line in enumerate(lines):
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line properties
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    # Center point
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    scratch = {
                        'id': i,
                        'line': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int(cx), int(cy)),
                        'length': float(length),
                        'angle': float(angle),
                        'confidence': min(length / 100.0, 1.0)  # Longer lines = higher confidence
                    }
                    
                    scratches.append(scratch)
            
            self.logger.info(f"Detected {len(scratches)} potential scratches")
            return scratches
            
        except Exception as e:
            self.logger.error(f"Error in detect_scratches: {e}")
            return []
    
    def detect_digs(self, image: np.ndarray) -> List[Dict]:
        """
        Detect dig/pit defects using morphological blackhat.
        
        Args:
            image: Input image
            
        Returns:
            List of dig dictionaries
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Use smaller kernel for digs (typically small and round)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            # Threshold for dark spots
            _, dig_mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(dig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            digs = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter small noise and very large regions
                if area < 5 or area > 500:
                    continue
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    continue
                
                # Calculate circularity (digs should be roughly circular)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Only keep roughly circular defects
                if circularity < 0.3:
                    continue
                
                dig = {
                    'id': i,
                    'center': (int(cx), int(cy)),
                    'area': float(area),
                    'circularity': float(circularity),
                    'confidence': float(circularity * min(area / 50.0, 1.0))
                }
                
                digs.append(dig)
            
            self.logger.info(f"Detected {len(digs)} potential digs")
            return digs
            
        except Exception as e:
            self.logger.error(f"Error in detect_digs: {e}")
            return []
    
    def detect_blobs(self, image: np.ndarray) -> List[Dict]:
        """
        Detect blob-like contamination using blob detection.
        
        Args:
            image: Input image
            
        Returns:
            List of blob dictionaries
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Use contour-based blob detection (more reliable across OpenCV versions)
            self.logger.info("Using contour-based blob detection")
            
            # Apply threshold to find blob-like regions
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            blobs = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter by area (equivalent to blob detector area filter)
                if area < 20 or area > 1000:
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Filter by circularity (equivalent to blob detector)
                if circularity < 0.1:
                    continue
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio (for convexity-like filter)
                aspect_ratio = w / h if h > 0 else 1
                
                # Filter extreme aspect ratios
                if aspect_ratio > 3 or aspect_ratio < 0.33:
                    continue
                
                blob = {
                    'id': i,
                    'center': (int(cx), int(cy)),
                    'area': float(area),
                    'bbox': (x, y, w, h),
                    'aspect_ratio': float(aspect_ratio),
                    'circularity': float(circularity),
                    'confidence': float(circularity * min(area / 100.0, 1.0))
                }
                
                blobs.append(blob)
            
            self.logger.info(f"Detected {len(blobs)} potential blobs")
            return blobs
            
        except Exception as e:
            self.logger.error(f"Error in detect_blobs: {e}")
            return []
    
    def comprehensive_anomaly_detection(self, image: np.ndarray) -> Dict:
        """
        Perform comprehensive anomaly detection using all methods.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing all detection results
        """
        try:
            results = {
                'image_shape': image.shape,
                'detection_summary': {},
                'anomaly_regions': [],
                'scratches': [],
                'digs': [],
                'blobs': [],
                'inpainted_image': None,
                'defect_mask': None
            }
            
            # Main anomaly detection with inpainting
            inpainted_image, defect_mask = self.detect_and_inpaint_anomalies(image)
            results['inpainted_image'] = inpainted_image
            results['defect_mask'] = defect_mask
            
            # Detect general defect regions
            anomaly_regions = self.detect_defect_regions(image)
            results['anomaly_regions'] = anomaly_regions
            
            # Detect specific defect types
            scratches = self.detect_scratches(image)
            results['scratches'] = scratches
            
            digs = self.detect_digs(image)
            results['digs'] = digs
            
            blobs = self.detect_blobs(image)
            results['blobs'] = blobs
            
            # Summary statistics
            results['detection_summary'] = {
                'total_anomaly_regions': len(anomaly_regions),
                'total_scratches': len(scratches),
                'total_digs': len(digs),
                'total_blobs': len(blobs),
                'total_defects': len(anomaly_regions) + len(scratches) + len(digs) + len(blobs),
                'defect_mask_area': int(np.sum(defect_mask > 0)) if defect_mask is not None else 0,
                'defect_density': (len(anomaly_regions) + len(scratches) + len(digs) + len(blobs)) / (image.shape[0] * image.shape[1])
            }
            
            self.logger.info(f"Comprehensive detection complete. Found {results['detection_summary']['total_defects']} total defects")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive_anomaly_detection: {e}")
            return {
                'error': str(e),
                'detection_summary': {'total_defects': 0}
            }


def visualize_anomaly_detection(image: np.ndarray, results: Dict, save_path: Optional[str] = None):
    """
    Visualize anomaly detection results.
    
    Args:
        image: Original image
        results: Detection results from comprehensive_anomaly_detection
        save_path: Optional path to save visualization
    """
    if 'error' in results:
        print(f"Cannot visualize failed detection: {results['error']}")
        return
    
    # Create visualization
    vis_img = image.copy()
    
    # Draw anomaly regions
    for region in results.get('anomaly_regions', []):
        contour = region.get('contour')
        if contour is not None:
            cv2.drawContours(vis_img, [contour], -1, (0, 0, 255), 2)
            
        # Draw centroid
        cx, cy = region['centroid']
        cv2.circle(vis_img, (cx, cy), 3, (0, 0, 255), -1)
    
    # Draw scratches
    for scratch in results.get('scratches', []):
        x1, y1, x2, y2 = scratch['line']
        cv2.line(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw digs
    for dig in results.get('digs', []):
        cx, cy = dig['center']
        radius = int(np.sqrt(dig['area'] / np.pi))
        cv2.circle(vis_img, (cx, cy), radius, (0, 255, 0), 2)
    
    # Draw blobs
    for blob in results.get('blobs', []):
        x, y, w, h = blob['bbox']
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    # Add summary text
    summary = results.get('detection_summary', {})
    text_lines = [
        f"Anomaly Regions: {summary.get('total_anomaly_regions', 0)}",
        f"Scratches: {summary.get('total_scratches', 0)}",
        f"Digs: {summary.get('total_digs', 0)}",
        f"Blobs: {summary.get('total_blobs', 0)}",
        f"Total Defects: {summary.get('total_defects', 0)}"
    ]
    
    for i, text in enumerate(text_lines):
        cv2.putText(vis_img, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"Visualization saved to: {save_path}")
    else:
        cv2.imshow('Anomaly Detection Results', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Test the AnomalyDetector functionality."""
    print("Testing AnomalyDetector...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create detector
    detector = AnomalyDetector(
        blackhat_threshold=25,
        morphology_kernel_size=15,
        min_defect_area=10,
        max_defect_area=2000
    )
    
    # Ask for image path
    image_path = input("Enter path to test image (or press Enter to create synthetic): ").strip()
    
    if not image_path:
        print("Creating synthetic test image with defects...")
        # Create synthetic image with various defects
        img = np.ones((400, 400, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Add some dark spots (digs)
        cv2.circle(img, (100, 100), 8, (50, 50, 50), -1)
        cv2.circle(img, (300, 150), 5, (30, 30, 30), -1)
        cv2.circle(img, (200, 300), 12, (40, 40, 40), -1)
        
        # Add some lines (scratches)
        cv2.line(img, (150, 50), (250, 120), (60, 60, 60), 2)
        cv2.line(img, (80, 200), (180, 250), (70, 70, 70), 1)
        
        # Add some contamination blobs
        cv2.ellipse(img, (320, 300), (15, 10), 0, 0, 360, (80, 80, 80), -1)
        cv2.ellipse(img, (50, 350), (20, 8), 45, 0, 360, (90, 90, 90), -1)
        
        # Add noise
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        test_path = "synthetic_defects.png"
        cv2.imwrite(test_path, img)
        image_path = test_path
        print(f"Created synthetic test image: {test_path}")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load and process image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")
    
    # Perform comprehensive detection
    results = detector.comprehensive_anomaly_detection(img)
    
    if 'error' in results:
        print(f"Detection failed: {results['error']}")
        return
    
    # Print results
    summary = results['detection_summary']
    print(f"\nDetection Results:")
    print(f"  Anomaly regions: {summary['total_anomaly_regions']}")
    print(f"  Scratches: {summary['total_scratches']}")
    print(f"  Digs: {summary['total_digs']}")
    print(f"  Blobs: {summary['total_blobs']}")
    print(f"  Total defects: {summary['total_defects']}")
    print(f"  Defect density: {summary['defect_density']:.6f} defects/pixel")
    
    # Visualize results
    visualize_anomaly_detection(img, results, "anomaly_detection_result.png")
    
    # Save inpainted image
    if results['inpainted_image'] is not None:
        cv2.imwrite("inpainted_image.png", results['inpainted_image'])
        print("Inpainted image saved as 'inpainted_image.png'")
    
    # Save defect mask
    if results['defect_mask'] is not None:
        cv2.imwrite("defect_mask.png", results['defect_mask'])
        print("Defect mask saved as 'defect_mask.png'")


if __name__ == "__main__":
    import os
    main()
