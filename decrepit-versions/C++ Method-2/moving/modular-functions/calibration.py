#!/usr/bin/env python3
"""
Calibration Module
=================
Standalone module for calculating pixel-to-micron conversion
using calibration targets and reference measurements.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationProcessor:
    """Handles calibration measurements and conversions."""
    
    def __init__(self):
        self.calibration_data = {}
        self.default_um_per_px = 0.5  # Default fallback value
    
    def detect_calibration_features(self, image: np.ndarray, 
                                   feature_type: str = "auto") -> List[Tuple[float, float]]:
        """
        Detect calibration features in the image.
        
        Args:
            image: Input calibration image
            feature_type: Type of features to detect ('dots', 'lines', 'auto')
            
        Returns:
            List of feature centroids
        """
        logger.info(f"Detecting calibration features of type: {feature_type}")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        centroids = []
        
        if feature_type in ["dots", "auto"]:
            centroids = self._detect_dots(gray)
            if len(centroids) > 0:
                logger.info(f"Found {len(centroids)} dot features")
                return centroids
        
        if feature_type in ["lines", "auto"]:
            centroids = self._detect_line_intersections(gray)
            if len(centroids) > 0:
                logger.info(f"Found {len(centroids)} line intersection features")
                return centroids
        
        if feature_type in ["circles", "auto"]:
            centroids = self._detect_circles(gray)
            if len(centroids) > 0:
                logger.info(f"Found {len(centroids)} circle features")
                return centroids
        
        logger.warning("No calibration features detected")
        return []
    
    def _detect_dots(self, gray: np.ndarray) -> List[Tuple[float, float]]:
        """Detect dots using SimpleBlobDetector."""
        try:
            # Setup SimpleBlobDetector parameters
            params = cv2.SimpleBlobDetector.Params()
            
            # Threshold parameters
            params.minThreshold = 10
            params.maxThreshold = 200
            params.thresholdStep = 10
            
            # Filter by Area
            params.filterByArea = True
            params.minArea = 20
            params.maxArea = 5000
            
            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.6
            
            # Filter by Convexity
            params.filterByConvexity = True
            params.minConvexity = 0.8
            
            # Filter by Inertia
            params.filterByInertia = True
            params.minInertiaRatio = 0.1
            
            # Create detector
            detector = cv2.SimpleBlobDetector.create(params)
            keypoints = detector.detect(gray)
            
            centroids = [kp.pt for kp in keypoints]
            
        except Exception as e:
            logger.warning(f"SimpleBlobDetector failed: {e}, using fallback")
            centroids = []
        
        # Fallback to HoughCircles if blob detector fails
        if not centroids:
            centroids = self._detect_circles(gray)
        
        return centroids
    
    def _detect_circles(self, gray: np.ndarray) -> List[Tuple[float, float]]:
        """Detect circles using HoughCircles."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=60, param2=30, minRadius=5, maxRadius=50
        )
        
        centroids = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                centroids.append((float(x), float(y)))
        
        return centroids
    
    def _detect_line_intersections(self, gray: np.ndarray) -> List[Tuple[float, float]]:
        """Detect line intersections in grid patterns."""
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None or len(lines) < 4:
            return []
        
        # Convert to line equations
        line_equations = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            line_equations.append((a, b, rho))
        
        # Find intersections
        intersections = []
        for i in range(len(line_equations)):
            for j in range(i+1, len(line_equations)):
                intersection = self._line_intersection(line_equations[i], line_equations[j])
                if intersection is not None:
                    x, y = intersection
                    # Check if intersection is within image bounds
                    if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                        intersections.append((x, y))
        
        # Remove duplicate intersections
        unique_intersections = []
        for point in intersections:
            is_unique = True
            for existing in unique_intersections:
                if np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2) < 10:
                    is_unique = False
                    break
            if is_unique:
                unique_intersections.append(point)
        
        return unique_intersections
    
    def _line_intersection(self, line1: Tuple[float, float, float], 
                          line2: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        """Calculate intersection of two lines."""
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:  # Lines are parallel
            return None
        
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        
        return (x, y)
    
    def calculate_um_per_px(self, centroids: List[Tuple[float, float]], 
                           known_spacing_um: float, 
                           method: str = "nearest_neighbor") -> Optional[float]:
        """
        Calculate um_per_px from detected features.
        
        Args:
            centroids: List of feature centroids
            known_spacing_um: Known physical spacing between features
            method: Calculation method ('nearest_neighbor', 'grid', 'average')
            
        Returns:
            Calculated um_per_px value
        """
        if len(centroids) < 2:
            logger.error("Need at least 2 features for calibration")
            return None
        
        logger.info(f"Calculating um_per_px using {method} method")
        
        if method == "nearest_neighbor":
            return self._calculate_nearest_neighbor(centroids, known_spacing_um)
        elif method == "grid":
            return self._calculate_grid_spacing(centroids, known_spacing_um)
        elif method == "average":
            return self._calculate_average_spacing(centroids, known_spacing_um)
        else:
            logger.error(f"Unknown calculation method: {method}")
            return None
    
    def _calculate_nearest_neighbor(self, centroids: List[Tuple[float, float]], 
                                  known_spacing_um: float) -> float:
        """Calculate spacing using nearest neighbor distances."""
        distances = []
        
        for i, point1 in enumerate(centroids):
            min_dist = float('inf')
            for j, point2 in enumerate(centroids):
                if i != j:
                    dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                    if dist < min_dist:
                        min_dist = dist
            if min_dist != float('inf'):
                distances.append(min_dist)
        
        if not distances:
            return self.default_um_per_px
        
        avg_distance_px = np.mean(distances)
        um_per_px = known_spacing_um / avg_distance_px
        
        logger.info(f"Average nearest neighbor distance: {avg_distance_px:.2f} px")
        logger.info(f"Calculated um_per_px: {um_per_px:.4f}")
        
        return um_per_px
    
    def _calculate_grid_spacing(self, centroids: List[Tuple[float, float]], 
                               known_spacing_um: float) -> float:
        """Calculate spacing assuming regular grid pattern."""
        if len(centroids) < 4:
            return self._calculate_nearest_neighbor(centroids, known_spacing_um)
        
        # Sort points to identify grid structure
        centroids = sorted(centroids, key=lambda p: (p[1], p[0]))  # Sort by y, then x
        
        # Calculate horizontal and vertical spacings
        h_spacings = []
        v_spacings = []
        
        # Group points by rows (similar y-coordinates)
        rows = []
        current_row = [centroids[0]]
        
        for point in centroids[1:]:
            if abs(point[1] - current_row[0][1]) < 20:  # Same row
                current_row.append(point)
            else:
                rows.append(sorted(current_row, key=lambda p: p[0]))  # Sort by x
                current_row = [point]
        rows.append(sorted(current_row, key=lambda p: p[0]))
        
        # Calculate horizontal spacings within rows
        for row in rows:
            for i in range(len(row) - 1):
                h_spacings.append(row[i+1][0] - row[i][0])
        
        # Calculate vertical spacings between rows
        for i in range(len(rows) - 1):
            if len(rows[i]) > 0 and len(rows[i+1]) > 0:
                v_spacings.append(rows[i+1][0][1] - rows[i][0][1])
        
        # Use the most consistent spacing
        all_spacings = h_spacings + v_spacings
        if all_spacings:
            avg_spacing_px = np.median(all_spacings)  # Use median for robustness
            um_per_px = known_spacing_um / avg_spacing_px
            
            logger.info(f"Grid spacing: {avg_spacing_px:.2f} px")
            logger.info(f"Calculated um_per_px: {um_per_px:.4f}")
            
            return um_per_px
        
        return self._calculate_nearest_neighbor(centroids, known_spacing_um)
    
    def _calculate_average_spacing(self, centroids: List[Tuple[float, float]], 
                                  known_spacing_um: float) -> float:
        """Calculate average of all pairwise distances."""
        distances = []
        
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                              (centroids[i][1] - centroids[j][1])**2)
                distances.append(dist)
        
        if not distances:
            return self.default_um_per_px
        
        # Use the most common distance (approximate mode)
        hist, bin_edges = np.histogram(distances, bins=20)
        mode_idx = np.argmax(hist)
        modal_distance = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        
        um_per_px = known_spacing_um / modal_distance
        
        logger.info(f"Modal distance: {modal_distance:.2f} px")
        logger.info(f"Calculated um_per_px: {um_per_px:.4f}")
        
        return um_per_px
    
    def calibrate_from_image(self, image_path: str, known_spacing_um: float, 
                            feature_type: str = "auto", 
                            method: str = "nearest_neighbor") -> Optional[float]:
        """
        Perform complete calibration from calibration image.
        
        Args:
            image_path: Path to calibration image
            known_spacing_um: Known spacing between features in microns
            feature_type: Type of features ('dots', 'lines', 'circles', 'auto')
            method: Calculation method
            
        Returns:
            Calculated um_per_px value
        """
        logger.info(f"Calibrating from image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Detect features
        centroids = self.detect_calibration_features(image, feature_type)
        
        if not centroids:
            logger.error("No calibration features detected")
            return None
        
        # Calculate um_per_px
        um_per_px = self.calculate_um_per_px(centroids, known_spacing_um, method)
        
        if um_per_px is not None:
            # Store calibration data
            self.calibration_data = {
                'um_per_px': um_per_px,
                'known_spacing_um': known_spacing_um,
                'feature_count': len(centroids),
                'feature_type': feature_type,
                'method': method,
                'image_path': image_path,
                'centroids': centroids
            }
            
            logger.info(f"Calibration successful: {um_per_px:.4f} um/px")
        
        return um_per_px
    
    def calibrate_from_fiber_dimensions(self, cladding_diameter_px: float, 
                                       cladding_diameter_um: float = 125.0) -> float:
        """
        Calibrate using known fiber cladding diameter.
        
        Args:
            cladding_diameter_px: Measured cladding diameter in pixels
            cladding_diameter_um: Known cladding diameter in microns
            
        Returns:
            Calculated um_per_px value
        """
        if cladding_diameter_px <= 0:
            logger.error("Invalid cladding diameter in pixels")
            return self.default_um_per_px
        
        um_per_px = cladding_diameter_um / cladding_diameter_px
        
        # Store calibration data
        self.calibration_data = {
            'um_per_px': um_per_px,
            'cladding_diameter_px': cladding_diameter_px,
            'cladding_diameter_um': cladding_diameter_um,
            'method': 'fiber_cladding',
        }
        
        logger.info(f"Fiber calibration: {cladding_diameter_px:.2f} px = {cladding_diameter_um} um")
        logger.info(f"Calculated um_per_px: {um_per_px:.4f}")
        
        return um_per_px
    
    def save_calibration(self, filename: str = "calibration.json") -> bool:
        """Save calibration data to file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            save_data = self.calibration_data.copy()
            if 'centroids' in save_data:
                save_data['centroids'] = [list(c) for c in save_data['centroids']]
            
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"Calibration data saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filename: str = "calibration.json") -> bool:
        """Load calibration data from file."""
        try:
            with open(filename, 'r') as f:
                self.calibration_data = json.load(f)
            
            logger.info(f"Calibration data loaded from {filename}")
            logger.info(f"um_per_px: {self.calibration_data.get('um_per_px', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def get_um_per_px(self) -> float:
        """Get the current um_per_px value."""
        return self.calibration_data.get('um_per_px', self.default_um_per_px)
    
    def convert_px_to_um(self, pixels: float) -> float:
        """Convert pixels to microns."""
        return pixels * self.get_um_per_px()
    
    def convert_um_to_px(self, microns: float) -> float:
        """Convert microns to pixels."""
        um_per_px = self.get_um_per_px()
        return microns / um_per_px if um_per_px > 0 else 0

def test_calibration():
    """Test the calibration module."""
    logger.info("Testing calibration module...")
    
    # Create synthetic calibration image
    cal_image = np.ones((400, 400), dtype=np.uint8) * 128
    
    # Add calibration dots in a 5x5 grid
    spacing_px = 60  # 60 pixels spacing
    known_spacing_um = 50.0  # 50 microns actual spacing
    
    for i in range(5):
        for j in range(5):
            x = 100 + j * spacing_px
            y = 100 + i * spacing_px
            cv2.circle(cal_image, (x, y), 8, 255, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, cal_image.shape)
    cal_image = np.clip(cal_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Test calibration processor
    processor = CalibrationProcessor()
    
    # Test feature detection
    centroids = processor.detect_calibration_features(cal_image, "dots")
    logger.info(f"Detected {len(centroids)} calibration features")
    
    # Test calibration calculation
    if centroids:
        um_per_px = processor.calculate_um_per_px(centroids, known_spacing_um, "grid")
        logger.info(f"Calculated um_per_px: {um_per_px:.4f}")
        
        # Expected value
        expected = known_spacing_um / spacing_px
        error = abs(um_per_px - expected) / expected * 100
        logger.info(f"Expected: {expected:.4f}, Error: {error:.2f}%")
        
        # Test conversions
        test_pixels = 120
        microns = processor.convert_px_to_um(test_pixels)
        back_to_pixels = processor.convert_um_to_px(microns)
        
        logger.info(f"Conversion test: {test_pixels} px -> {microns:.2f} um -> {back_to_pixels:.2f} px")
    
    # Test fiber calibration
    fiber_um_per_px = processor.calibrate_from_fiber_dimensions(250, 125.0)
    logger.info(f"Fiber calibration: {fiber_um_per_px:.4f} um/px")
    
    # Test save/load
    save_success = processor.save_calibration("test_calibration.json")
    logger.info(f"Save calibration: {save_success}")
    
    if save_success:
        new_processor = CalibrationProcessor()
        load_success = new_processor.load_calibration("test_calibration.json")
        logger.info(f"Load calibration: {load_success}")
        
        if load_success:
            loaded_um_per_px = new_processor.get_um_per_px()
            logger.info(f"Loaded um_per_px: {loaded_um_per_px:.4f}")
    
    logger.info("Calibration module testing completed!")
    
    return {
        'synthetic_image': cal_image,
        'detected_centroids': centroids,
        'calculated_um_per_px': um_per_px if centroids else None,
        'fiber_um_per_px': fiber_um_per_px
    }

if __name__ == "__main__":
    # Run tests
    test_results = test_calibration()
    logger.info("Calibration module is ready for use!")
