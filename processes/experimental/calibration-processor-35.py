#!/usr/bin/env python3
"""
Calibration Module for Pixel-to-Micron Conversion
===============================================
Standalone module for calculating pixel-to-micron conversion ratio using
calibration target images (e.g., stage micrometer). Supports multiple
feature detection methods and statistical analysis.

Features:
- SimpleBlobDetector for circular features
- HoughCircles fallback detection
- Statistical analysis of inter-feature distances
- Histogram-based characteristic distance calculation
- JSON export/import of calibration data
- Visualization of detected features
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import argparse


class CalibrationProcessor:
    """Class for processing calibration images and calculating pixel-to-micron ratios."""
    
    def __init__(self):
        """Initialize the calibration processor."""
        self.detected_features = []
        self.calculated_scale = None
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load calibration image and convert to grayscale.
        
        Args:
            image_path: Path to calibration image
            
        Returns:
            Grayscale image as numpy array
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path_obj = Path(image_path)
        
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Calibration image not found: {image_path}")
        
        image = cv2.imread(str(image_path_obj))
        if image is None:
            raise ValueError(f"Failed to load calibration image: {image_path}")
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.info(f"Calibration image '{image_path_obj.name}' loaded successfully")
        
        return gray_image
    
    def detect_features_blob(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect calibration features using SimpleBlobDetector.
        
        Args:
            image: Grayscale calibration image
            
        Returns:
            List of (x, y) coordinates of detected features
        """
        try:
            # Setup SimpleBlobDetector parameters
            params = cv2.SimpleBlobDetector_Params()
            
            # Threshold parameters
            params.minThreshold = 10
            params.maxThreshold = 200
            
            # Filter by Area
            params.filterByArea = True
            params.minArea = 20
            params.maxArea = 5000
            
            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.6
            
            # Filter by Convexity
            params.filterByConvexity = True
            params.minConvexity = 0.80
            
            # Filter by Inertia
            params.filterByInertia = True
            params.minInertiaRatio = 0.1
            
            # Create detector and detect blobs
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(image)
            
            centroids = []
            if keypoints:
                for kp in keypoints:
                    centroids.append(kp.pt)
                logging.info(f"SimpleBlobDetector found {len(centroids)} features")
            else:
                logging.warning("SimpleBlobDetector found no features")
                
        except AttributeError:
            logging.warning("SimpleBlobDetector not available in this OpenCV version")
            centroids = []
        
        return centroids
    
    def detect_features_hough(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect calibration features using HoughCircles (fallback method).
        
        Args:
            image: Grayscale calibration image
            
        Returns:
            List of (x, y) coordinates of detected features
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=20,
            param1=60, 
            param2=30, 
            minRadius=5, 
            maxRadius=50
        )
        
        centroids = []
        if circles is not None:
            circles_int = np.uint16(np.around(circles))
            for circle in circles_int[0, :]:  # type: ignore
                x, y = circle[0], circle[1]
                centroids.append((float(x), float(y)))
            logging.info(f"HoughCircles found {len(centroids)} features")
        else:
            logging.warning("HoughCircles found no features")
        
        return centroids
    
    def detect_features(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect calibration features using multiple methods.
        
        Args:
            image: Grayscale calibration image
            
        Returns:
            List of (x, y) coordinates of detected features
            
        Raises:
            ValueError: If no features could be detected
        """
        # Try SimpleBlobDetector first
        centroids = self.detect_features_blob(image)
        
        # Fallback to HoughCircles if blob detection failed
        if not centroids:
            logging.info("Trying HoughCircles as fallback")
            centroids = self.detect_features_hough(image)
        
        if not centroids:
            raise ValueError("No calibration features detected in the image")
        
        self.detected_features = centroids
        return centroids
    
    def calculate_inter_feature_distances(self, centroids: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate distances between all pairs of detected features.
        
        Args:
            centroids: List of (x, y) coordinates
            
        Returns:
            List of inter-feature distances in pixels
        """
        if len(centroids) < 2:
            raise ValueError("At least two features required for distance calculation")
        
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.sqrt(
                    (centroids[i][0] - centroids[j][0])**2 + 
                    (centroids[i][1] - centroids[j][1])**2
                )
                distances.append(dist)
        
        return distances
    
    def find_characteristic_distance(self, distances: List[float]) -> float:
        """
        Find the characteristic spacing distance from list of all distances.
        
        Uses histogram analysis for many distances, median for few distances.
        
        Args:
            distances: List of inter-feature distances
            
        Returns:
            Characteristic distance in pixels
        """
        if not distances:
            raise ValueError("No distances provided")
        
        # Sort distances
        distances_sorted = sorted(distances)
        
        if len(distances) > 10:
            # Use histogram to find most frequent distance
            hist, bin_edges = np.histogram(distances_sorted, bins='auto')
            
            if len(hist) > 0 and len(bin_edges) > 1:
                peak_bin_index = np.argmax(hist)
                characteristic_distance = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2.0
                logging.info(f"Characteristic distance from histogram: {characteristic_distance:.2f}px")
            else:
                # Fallback to median of smaller distances
                characteristic_distance = np.median(distances_sorted[:max(1, len(distances_sorted)//2)])
                logging.warning(f"Histogram failed, using median: {characteristic_distance:.2f}px")
        else:
            # For few distances, use the smallest (likely unit spacing)
            characteristic_distance = distances_sorted[0]
            logging.info(f"Characteristic distance (smallest): {characteristic_distance:.2f}px")
        
        return float(characteristic_distance)
    
    def calculate_scale(self, 
                       centroids: List[Tuple[float, float]], 
                       known_spacing_um: float) -> float:
        """
        Calculate the micrometers per pixel scale factor.
        
        Args:
            centroids: List of detected feature coordinates
            known_spacing_um: Known physical spacing between features in micrometers
            
        Returns:
            Scale factor in micrometers per pixel
            
        Raises:
            ValueError: If calculation fails
        """
        # Calculate inter-feature distances
        distances = self.calculate_inter_feature_distances(centroids)
        
        # Find characteristic distance
        characteristic_distance_px = self.find_characteristic_distance(distances)
        
        # Validate characteristic distance
        if characteristic_distance_px <= 1e-6:
            raise ValueError(f"Characteristic distance too small: {characteristic_distance_px:.6f}px")
        
        # Calculate scale
        um_per_px = known_spacing_um / characteristic_distance_px
        
        # Validate scale is reasonable (typical microscope scales: 0.1-10 µm/pixel)
        if um_per_px < 0.05 or um_per_px > 20.0:
            logging.warning(f"Calculated scale {um_per_px:.6f} µm/px seems unreasonable")
            logging.warning("Typical range is 0.1-10 µm/px. Please verify inputs.")
        
        self.calculated_scale = um_per_px
        
        logging.info(f"Calculated scale: {um_per_px:.6f} µm/px")
        logging.info(f"Based on known spacing: {known_spacing_um} µm")
        logging.info(f"Characteristic distance: {characteristic_distance_px:.2f}px")
        
        return um_per_px
    
    def save_calibration_data(self, 
                            data: Dict[str, Any], 
                            file_path: str) -> bool:
        """
        Save calibration data to JSON file.
        
        Args:
            data: Dictionary containing calibration data
            file_path: Path to save JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path_obj = Path(file_path)
            with open(file_path_obj, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            logging.info(f"Calibration data saved to: {file_path}")
            return True
        except IOError as e:
            logging.error(f"Failed to save calibration data: {e}")
            return False
    
    def load_calibration_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load calibration data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with calibration data or None if failed
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logging.warning(f"Calibration file not found: {file_path}")
                return None
            
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Calibration data loaded from: {file_path}")
            return data
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load calibration data: {e}")
            return None
    
    def visualize_features(self, 
                          image: np.ndarray, 
                          centroids: List[Tuple[float, float]],
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected calibration features.
        
        Args:
            image: Original grayscale image
            centroids: List of detected feature coordinates
            save_path: Optional path to save visualization
            
        Returns:
            Annotated image
        """
        # Create color version
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
        
        # Draw detected features
        for i, (x, y) in enumerate(centroids):
            # Draw circle around feature
            cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(result, (int(x), int(y)), 2, (0, 0, 255), -1)
            # Add feature number
            cv2.putText(result, str(i+1), (int(x+15), int(y+5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Add summary text
        text_y = 30
        cv2.putText(result, f"Features detected: {len(centroids)}", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.calculated_scale:
            text_y += 30
            cv2.putText(result, f"Scale: {self.calculated_scale:.4f} um/px", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, result)
            logging.info(f"Visualization saved to: {save_path}")
        
        return result
    
    def run_calibration_process(self,
                              image_path: str,
                              known_spacing_um: float,
                              output_file: str = "calibration.json",
                              visualize_path: Optional[str] = None) -> Optional[float]:
        """
        Run the complete calibration process.
        
        Args:
            image_path: Path to calibration image
            known_spacing_um: Known spacing between features in micrometers
            output_file: Path for output JSON file
            visualize_path: Optional path for visualization image
            
        Returns:
            Calculated scale in µm/px or None if failed
        """
        try:
            # Load image
            image = self.load_image(image_path)
            
            # Detect features
            centroids = self.detect_features(image)
            
            # Calculate scale
            um_per_px = self.calculate_scale(centroids, known_spacing_um)
            
            # Prepare calibration data
            calibration_data = {
                "um_per_px": um_per_px,
                "image_path": str(image_path),
                "known_spacing_um": known_spacing_um,
                "features_detected": len(centroids),
                "feature_coordinates": centroids,
                "timestamp": str(Path(image_path).stat().st_mtime),
                "method": "automatic_feature_detection"
            }
            
            # Save calibration data
            self.save_calibration_data(calibration_data, output_file)
            
            # Create visualization if requested
            if visualize_path:
                self.visualize_features(image, centroids, visualize_path)
            
            return um_per_px
            
        except Exception as e:
            logging.error(f"Calibration process failed: {e}")
            return None


def validate_scale(um_per_px: float) -> bool:
    """
    Validate that the calculated scale is reasonable.
    
    Args:
        um_per_px: Scale factor in micrometers per pixel
        
    Returns:
        True if scale seems reasonable, False otherwise
    """
    # Typical microscope scales range from 0.1 to 10 µm/pixel
    min_reasonable = 0.05
    max_reasonable = 20.0
    
    if min_reasonable <= um_per_px <= max_reasonable:
        return True
    else:
        logging.warning(f"Scale {um_per_px:.6f} µm/px outside typical range "
                       f"({min_reasonable}-{max_reasonable} µm/px)")
        return False


def generate_test_calibration_image(output_path: str, 
                                  grid_size: Tuple[int, int] = (5, 5),
                                  spacing_px: int = 100,
                                  dot_radius: int = 5) -> bool:
    """
    Generate a test calibration image with known spacing.
    
    Args:
        output_path: Path to save test image
        grid_size: Number of dots in (rows, cols)
        spacing_px: Spacing between dots in pixels
        dot_radius: Radius of each dot in pixels
        
    Returns:
        True if successful, False otherwise
    """
    try:
        rows, cols = grid_size
        
        # Calculate image size
        img_width = (cols - 1) * spacing_px + 2 * spacing_px
        img_height = (rows - 1) * spacing_px + 2 * spacing_px
        
        # Create white background
        image = np.ones((img_height, img_width), dtype=np.uint8) * 255
        
        # Draw grid of black dots
        for row in range(rows):
            for col in range(cols):
                center_x = spacing_px + col * spacing_px
                center_y = spacing_px + row * spacing_px
                cv2.circle(image, (center_x, center_y), dot_radius, (0,), -1)
        
        # Save image
        cv2.imwrite(output_path, image)
        logging.info(f"Test calibration image saved to: {output_path}")
        logging.info(f"Grid: {rows}x{cols}, Spacing: {spacing_px}px, Dot radius: {dot_radius}px")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to generate test image: {e}")
        return False


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Calibration Module")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Calibrate command
    calibrate_parser = subparsers.add_parser('calibrate', help='Run calibration process')
    calibrate_parser.add_argument('image', help='Path to calibration image')
    calibrate_parser.add_argument('spacing', type=float, help='Known spacing in micrometers')
    calibrate_parser.add_argument('--output', '-o', default='calibration.json',
                                help='Output JSON file (default: calibration.json)')
    calibrate_parser.add_argument('--visualize', '-v', help='Save visualization image')
    
    # Generate test image command
    generate_parser = subparsers.add_parser('generate', help='Generate test calibration image')
    generate_parser.add_argument('output', help='Output path for test image')
    generate_parser.add_argument('--grid-size', nargs=2, type=int, default=[5, 5],
                                help='Grid size as rows cols (default: 5 5)')
    generate_parser.add_argument('--spacing', type=int, default=100,
                                help='Spacing between dots in pixels (default: 100)')
    generate_parser.add_argument('--dot-radius', type=int, default=5,
                                help='Dot radius in pixels (default: 5)')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load and display calibration data')
    load_parser.add_argument('file', help='Path to calibration JSON file')
    
    # General arguments
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'calibrate':
            # Run calibration process
            processor = CalibrationProcessor()
            result = processor.run_calibration_process(
                args.image,
                args.spacing,
                args.output,
                args.visualize
            )
            
            if result:
                print(f"Calibration successful: {result:.6f} µm/px")
                if validate_scale(result):
                    print("Scale validation: PASSED")
                else:
                    print("Scale validation: WARNING - scale may be unreasonable")
                return 0
            else:
                print("Calibration failed")
                return 1
                
        elif args.command == 'generate':
            # Generate test calibration image
            success = generate_test_calibration_image(
                args.output,
                tuple(args.grid_size),
                args.spacing,
                args.dot_radius
            )
            
            if success:
                # Calculate what the known spacing would be for this test image
                test_spacing_um = args.spacing * 0.5  # Example: assume 0.5 µm/px
                print(f"Test image generated: {args.output}")
                print(f"For testing, use known spacing: {test_spacing_um} µm")
                print(f"Expected result: 0.5 µm/px")
                return 0
            else:
                return 1
                
        elif args.command == 'load':
            # Load and display calibration data
            processor = CalibrationProcessor()
            data = processor.load_calibration_data(args.file)
            
            if data:
                print("Calibration Data:")
                print(f"  Scale: {data.get('um_per_px', 'N/A')} µm/px")
                print(f"  Image: {data.get('image_path', 'N/A')}")
                print(f"  Known spacing: {data.get('known_spacing_um', 'N/A')} µm")
                print(f"  Features detected: {data.get('features_detected', 'N/A')}")
                print(f"  Method: {data.get('method', 'N/A')}")
                return 0
            else:
                print("Failed to load calibration data")
                return 1
                
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
