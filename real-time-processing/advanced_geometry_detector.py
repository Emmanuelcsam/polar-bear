#!/usr/bin/env python3
"""
Advanced Real-Time Geometric Shape Detection and Analysis System
================================================================

This cutting-edge system performs comprehensive geometric analysis on live video feeds,
detecting all shapes, lines, polygons, circles, and complex geometries while calculating
their properties including angles, areas, perimeters, and more.

INSTALLATION REQUIREMENTS:
pip install opencv-python opencv-contrib-python numpy scipy matplotlib

For GPU support (optional):
- Install CUDA Toolkit from NVIDIA
- Build OpenCV with CUDA support

USAGE:
python advanced_geometry_detector.py              # Use default webcam
python advanced_geometry_detector.py -s 1         # Use external camera
python advanced_geometry_detector.py -f video.mp4 # Use video file

Author: Advanced Computer Vision System
Date: 2025
"""

import sys
import cv2
import numpy as np
import math
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import argparse
from concurrent.futures import ThreadPoolExecutor
import warnings
import traceback
import os

warnings.filterwarnings('ignore')

# Check for required libraries
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("ERROR: OpenCV not installed. Run: pip install opencv-python opencv-contrib-python")
    sys.exit(1)

# Check for CUDA availability
try:
    CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if CUDA_AVAILABLE:
        print(f"CUDA is available! GPU count: {cv2.cuda.getCudaEnabledDeviceCount()}")
    else:
        print("CUDA not available. Running on CPU only.")
except:
    CUDA_AVAILABLE = False
    print("CUDA module not found. Running on CPU only.")

@dataclass
class GeometricShape:
    """Data class for storing shape properties"""
    shape_type: str
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
    perimeter: float
    vertices: List[Tuple[int, int]]
    angles: List[float]
    bounding_box: Tuple[int, int, int, int]
    orientation: float
    eccentricity: float
    solidity: float
    aspect_ratio: float
    extent: float
    equivalent_diameter: float
    compactness: float
    hu_moments: np.ndarray
    color: Tuple[int, int, int]
    confidence: float

class AdvancedGeometryDetector:
    """Advanced real-time geometric shape detection and analysis system"""
    
    def __init__(self, use_gpu: bool = True, num_threads: int = 4):
        """
        Initialize the detector
        
        Args:
            use_gpu: Enable GPU acceleration if available
            num_threads: Number of processing threads
        """
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.num_threads = min(num_threads, os.cpu_count() or 4)
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
        # Shape detection parameters
        self.min_area = 100
        self.max_area = 100000
        self.epsilon_factor = 0.02
        
        # Color ranges for shape highlighting (BGR format for OpenCV)
        self.colors = {
            'triangle': (255, 0, 0),      # Blue
            'rectangle': (0, 255, 0),      # Green
            'square': (0, 255, 255),       # Yellow
            'pentagon': (255, 0, 255),     # Magenta
            'hexagon': (0, 165, 255),      # Orange
            'circle': (255, 255, 0),       # Cyan
            'ellipse': (128, 0, 128),      # Purple
            'polygon': (255, 255, 255),    # White
            'line': (0, 0, 255),           # Red
            'unknown': (128, 128, 128)     # Gray
        }
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        
        # Processing queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        print(f"Detector initialized with {self.num_threads} threads")
        print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess frame for shape detection with error handling
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed grayscale and edge images
        """
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame")
            
            if self.use_gpu:
                try:
                    # GPU accelerated preprocessing
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    
                    # Convert to grayscale
                    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Apply bilateral filter for noise reduction
                    gpu_filtered = cv2.cuda.bilateralFilter(gpu_gray, -1, 50, 50)
                    
                    # Apply CLAHE for contrast enhancement
                    clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gpu_enhanced = clahe.apply(gpu_filtered)
                    
                    # Download results
                    gray = gpu_enhanced.download()
                except Exception as e:
                    print(f"GPU processing failed, falling back to CPU: {e}")
                    self.use_gpu = False
                    return self.preprocess_frame(frame)
            else:
                # CPU preprocessing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                
                # CLAHE for contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            # Multi-scale edge detection
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            edges3 = cv2.Canny(gray, 100, 200)
            
            # Combine edges
            edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
            
            return gray, edges
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Return empty images on error
            return np.zeros_like(frame[:,:,0]), np.zeros_like(frame[:,:,0])
    
    def detect_lines(self, edges: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect lines using Hough transform with error handling
        
        Args:
            edges: Edge image
            
        Returns:
            List of line endpoints
        """
        lines = []
        
        try:
            if edges is None or edges.size == 0:
                return lines
            
            # Standard Hough Line Transform
            hough_lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
            if hough_lines is not None:
                for i in range(min(len(hough_lines), 50)):  # Limit to 50 lines
                    rho, theta = hough_lines[i, 0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    lines.append(((x1, y1), (x2, y2)))
            
            # Probabilistic Hough Line Transform
            prob_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                         minLineLength=30, maxLineGap=10)
            if prob_lines is not None:
                for i in range(min(len(prob_lines), 50)):  # Limit to 50 lines
                    x1, y1, x2, y2 = prob_lines[i, 0]
                    lines.append(((x1, y1), (x2, y2)))
                    
        except Exception as e:
            print(f"Error detecting lines: {e}")
        
        return lines
    
    def detect_circles(self, gray: np.ndarray) -> List[Tuple[Tuple[int, int], int]]:
        """
        Detect circles using Hough Circle Transform with error handling
        
        Args:
            gray: Grayscale image
            
        Returns:
            List of circles (center, radius)
        """
        circles = []
        
        try:
            if gray is None or gray.size == 0:
                return circles
            
            # Multi-scale circle detection
            for dp in [1.0, 1.5, 2.0]:
                detected = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, 
                                           minDist=50, param1=100, param2=30, 
                                           minRadius=10, maxRadius=200)
                if detected is not None:
                    detected = np.uint16(np.around(detected))
                    for i in range(min(len(detected[0]), 20)):  # Limit to 20 circles per scale
                        circle = detected[0, i]
                        # Validate circle is within image bounds
                        if (0 <= circle[0] < gray.shape[1] and 
                            0 <= circle[1] < gray.shape[0]):
                            circles.append(((int(circle[0]), int(circle[1])), int(circle[2])))
                            
        except Exception as e:
            print(f"Error detecting circles: {e}")
        
        return circles
    
    def calculate_shape_properties(self, contour: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive geometric properties of a shape with error handling
        
        Args:
            contour: Shape contour
            
        Returns:
            Dictionary of geometric properties
        """
        try:
            # Basic properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Prevent division by zero
            if area == 0:
                area = 1
            if perimeter == 0:
                perimeter = 1
            
            # Moments
            moments = cv2.moments(contour)
            
            # Centroid
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = 0, 0
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Prevent zero dimensions
            if w == 0:
                w = 1
            if h == 0:
                h = 1
            
            # Oriented bounding box
            if len(contour) >= 5:
                try:
                    (ox, oy), (width, height), angle = cv2.minAreaRect(contour)
                    orientation = angle
                except:
                    orientation = 0
                    width, height = w, h
            else:
                orientation = 0
                width, height = w, h
            
            # Aspect ratio
            aspect_ratio = float(w) / h
            
            # Extent
            rect_area = w * h
            extent = float(area) / rect_area
            
            # Solidity
            try:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    hull_area = 1
                solidity = float(area) / hull_area
            except:
                solidity = 1.0
            
            # Equivalent diameter
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            
            # Compactness
            compactness = (perimeter * perimeter) / area
            
            # Eccentricity
            if moments['m00'] != 0:
                mu20 = moments['mu20'] / moments['m00']
                mu02 = moments['mu02'] / moments['m00']
                mu11 = moments['mu11'] / moments['m00']
                
                discriminant = (mu20 - mu02)**2 + 4*mu11**2
                if discriminant >= 0:
                    lambda1 = 0.5 * (mu20 + mu02 + np.sqrt(discriminant))
                    lambda2 = 0.5 * (mu20 + mu02 - np.sqrt(discriminant))
                    
                    if lambda1 != 0 and lambda1 >= lambda2:
                        eccentricity = np.sqrt(1 - (lambda2 / lambda1))
                    else:
                        eccentricity = 0
                else:
                    eccentricity = 0
            else:
                eccentricity = 0
            
            # Hu moments
            try:
                hu_moments = cv2.HuMoments(moments)
            except:
                hu_moments = np.zeros((7, 1))
            
            return {
                'area': area,
                'perimeter': perimeter,
                'center': (cx, cy),
                'bounding_box': (x, y, w, h),
                'orientation': orientation,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity,
                'equivalent_diameter': equivalent_diameter,
                'compactness': compactness,
                'eccentricity': eccentricity,
                'hu_moments': hu_moments.flatten()
            }
            
        except Exception as e:
            print(f"Error calculating shape properties: {e}")
            # Return default properties
            return {
                'area': 0,
                'perimeter': 0,
                'center': (0, 0),
                'bounding_box': (0, 0, 1, 1),
                'orientation': 0,
                'aspect_ratio': 1,
                'extent': 0,
                'solidity': 0,
                'equivalent_diameter': 0,
                'compactness': 0,
                'eccentricity': 0,
                'hu_moments': np.zeros(7)
            }
    
    def calculate_polygon_angles(self, vertices: np.ndarray) -> List[float]:
        """
        Calculate interior angles of a polygon with error handling
        
        Args:
            vertices: Polygon vertices
            
        Returns:
            List of angles in degrees
        """
        angles = []
        
        try:
            n = len(vertices)
            if n < 3:
                return angles
            
            for i in range(n):
                # Get three consecutive vertices
                p1 = vertices[(i - 1) % n][0] if len(vertices[(i - 1) % n]) > 0 else [0, 0]
                p2 = vertices[i][0] if len(vertices[i]) > 0 else [0, 0]
                p3 = vertices[(i + 1) % n][0] if len(vertices[(i + 1) % n]) > 0 else [0, 0]
                
                # Calculate vectors
                v1 = np.array(p1) - np.array(p2)
                v2 = np.array(p3) - np.array(p2)
                
                # Calculate angle using dot product
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 * norm2 != 0:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    angles.append(angle)
                else:
                    angles.append(0)
                    
        except Exception as e:
            print(f"Error calculating angles: {e}")
        
        return angles
    
    def classify_shape(self, contour: np.ndarray, properties: Dict[str, Any]) -> Tuple[str, float]:
        """
        Classify shape using advanced algorithms with error handling
        
        Args:
            contour: Shape contour
            properties: Shape properties
            
        Returns:
            Shape type and confidence score
        """
        try:
            # Approximate polygon
            epsilon = self.epsilon_factor * properties['perimeter']
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            # Calculate circularity
            if properties['perimeter'] > 0:
                circularity = 4 * np.pi * properties['area'] / (properties['perimeter'] ** 2)
            else:
                circularity = 0
            
            # Shape classification logic
            confidence = 1.0
            
            if vertices == 2:
                return 'line', 0.9
            
            elif vertices == 3:
                # Check for triangle
                angles = self.calculate_polygon_angles(approx)
                if len(angles) == 3:
                    angle_sum = sum(angles)
                    if 170 < angle_sum < 190:  # Should be ~180 degrees
                        return 'triangle', 0.95
                return 'triangle', 0.8
            
            elif vertices == 4:
                # Check for rectangle/square
                angles = self.calculate_polygon_angles(approx)
                if len(angles) == 4 and all(85 < angle < 95 for angle in angles):
                    # Check if square
                    if 0.9 < properties['aspect_ratio'] < 1.1:
                        return 'square', 0.95
                    else:
                        return 'rectangle', 0.95
                else:
                    return 'quadrilateral', 0.8
            
            elif vertices == 5:
                return 'pentagon', 0.9
            
            elif vertices == 6:
                return 'hexagon', 0.9
            
            elif vertices > 6:
                # Check for circle/ellipse
                if circularity > 0.8:
                    # Check if circle or ellipse
                    if properties['eccentricity'] < 0.3:
                        return 'circle', 0.95
                    else:
                        return 'ellipse', 0.9
                else:
                    return f'polygon_{vertices}', 0.85
            
            else:
                # Use machine learning features for complex shapes
                if circularity > 0.7:
                    if properties['eccentricity'] < 0.5:
                        return 'circle', 0.8
                    else:
                        return 'ellipse', 0.8
                else:
                    return 'unknown', 0.5
                    
        except Exception as e:
            print(f"Error classifying shape: {e}")
            return 'unknown', 0.0
    
    def detect_shapes(self, frame: np.ndarray) -> List[GeometricShape]:
        """
        Detect all shapes in frame with comprehensive error handling
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected geometric shapes
        """
        shapes = []
        
        try:
            if frame is None or frame.size == 0:
                return shapes
            
            # Preprocess frame
            gray, edges = self.preprocess_frame(frame)
            
            # Find contours
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours in parallel
            futures = []
            
            for i, contour in enumerate(contours[:100]):  # Limit to 100 contours
                try:
                    area = cv2.contourArea(contour)
                    if self.min_area < area < self.max_area:
                        future = self.executor.submit(self.process_contour, contour, frame)
                        futures.append(future)
                except Exception as e:
                    print(f"Error processing contour {i}: {e}")
            
            # Collect results with timeout
            for future in futures:
                try:
                    shape = future.result(timeout=0.1)  # 100ms timeout
                    if shape:
                        shapes.append(shape)
                except Exception as e:
                    print(f"Error getting future result: {e}")
            
            # Detect lines
            try:
                lines = self.detect_lines(edges)
                for i, line in enumerate(lines[:20]):  # Limit to 20 lines
                    # Create line shape
                    line_contour = np.array([line[0], line[1]], dtype=np.int32)
                    center = ((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2)
                    length = np.linalg.norm(np.array(line[1]) - np.array(line[0]))
                    
                    shape = GeometricShape(
                        shape_type='line',
                        contour=line_contour,
                        center=center,
                        area=0,
                        perimeter=length,
                        vertices=[line[0], line[1]],
                        angles=[],
                        bounding_box=(min(line[0][0], line[1][0]), min(line[0][1], line[1][1]),
                                     abs(line[1][0] - line[0][0]), abs(line[1][1] - line[0][1])),
                        orientation=np.arctan2(line[1][1] - line[0][1], line[1][0] - line[0][0]) * 180 / np.pi,
                        eccentricity=1.0,
                        solidity=1.0,
                        aspect_ratio=length,
                        extent=1.0,
                        equivalent_diameter=0,
                        compactness=0,
                        hu_moments=np.zeros(7),
                        color=self.colors['line'],
                        confidence=0.9
                    )
                    shapes.append(shape)
            except Exception as e:
                print(f"Error detecting lines: {e}")
            
            # Detect circles
            try:
                circles = self.detect_circles(gray)
                for i, (center, radius) in enumerate(circles[:20]):  # Limit to 20 circles
                    # Create circle shape
                    circle_points = []
                    for angle in range(0, 360, 10):
                        x = int(center[0] + radius * np.cos(angle * np.pi / 180))
                        y = int(center[1] + radius * np.sin(angle * np.pi / 180))
                        circle_points.append([x, y])
                    
                    circle_contour = np.array(circle_points, dtype=np.int32)
                    area = np.pi * radius * radius
                    perimeter = 2 * np.pi * radius
                    
                    shape = GeometricShape(
                        shape_type='circle',
                        contour=circle_contour,
                        center=center,
                        area=area,
                        perimeter=perimeter,
                        vertices=[],
                        angles=[],
                        bounding_box=(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius),
                        orientation=0,
                        eccentricity=0,
                        solidity=1.0,
                        aspect_ratio=1.0,
                        extent=np.pi / 4,
                        equivalent_diameter=2 * radius,
                        compactness=12.57,  # 4π
                        hu_moments=np.zeros(7),
                        color=self.colors['circle'],
                        confidence=0.95
                    )
                    shapes.append(shape)
            except Exception as e:
                print(f"Error detecting circles: {e}")
                
        except Exception as e:
            print(f"Error in detect_shapes: {e}")
            traceback.print_exc()
        
        return shapes
    
    def process_contour(self, contour: np.ndarray, frame: np.ndarray) -> Optional[GeometricShape]:
        """
        Process single contour to extract shape with error handling
        
        Args:
            contour: Contour to process
            frame: Original frame for color extraction
            
        Returns:
            GeometricShape object or None
        """
        try:
            # Calculate properties
            properties = self.calculate_shape_properties(contour)
            
            # Classify shape
            shape_type, confidence = self.classify_shape(contour, properties)
            
            # Extract vertices
            epsilon = self.epsilon_factor * properties['perimeter']
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = [tuple(v[0]) for v in approx if len(v) > 0]
            
            # Calculate angles
            angles = self.calculate_polygon_angles(approx) if len(approx) >= 3 else []
            
            # Get color
            color = self.colors.get(shape_type.split('_')[0], self.colors['unknown'])
            
            # Create shape object
            shape = GeometricShape(
                shape_type=shape_type,
                contour=contour,
                center=properties['center'],
                area=properties['area'],
                perimeter=properties['perimeter'],
                vertices=vertices,
                angles=angles,
                bounding_box=properties['bounding_box'],
                orientation=properties['orientation'],
                eccentricity=properties['eccentricity'],
                solidity=properties['solidity'],
                aspect_ratio=properties['aspect_ratio'],
                extent=properties['extent'],
                equivalent_diameter=properties['equivalent_diameter'],
                compactness=properties['compactness'],
                hu_moments=properties['hu_moments'],
                color=color,
                confidence=confidence
            )
            
            return shape
            
        except Exception as e:
            print(f"Error processing contour: {e}")
            return None
    
    def draw_shapes(self, frame: np.ndarray, shapes: List[GeometricShape]) -> np.ndarray:
        """
        Draw detected shapes with annotations and error handling
        
        Args:
            frame: Frame to draw on
            shapes: List of detected shapes
            
        Returns:
            Annotated frame
        """
        try:
            if frame is None or frame.size == 0:
                return frame
            
            output = frame.copy()
            
            for shape in shapes:
                try:
                    # Draw contour
                    if shape.contour is not None and len(shape.contour) > 0:
                        cv2.drawContours(output, [shape.contour], -1, shape.color, 2)
                    
                    # Draw vertices
                    for vertex in shape.vertices:
                        if 0 <= vertex[0] < output.shape[1] and 0 <= vertex[1] < output.shape[0]:
                            cv2.circle(output, vertex, 5, (0, 255, 0), -1)
                    
                    # Draw center
                    if 0 <= shape.center[0] < output.shape[1] and 0 <= shape.center[1] < output.shape[0]:
                        cv2.circle(output, shape.center, 7, (255, 0, 0), -1)
                    
                    # Draw bounding box
                    x, y, w, h = shape.bounding_box
                    x = max(0, min(x, output.shape[1] - 1))
                    y = max(0, min(y, output.shape[0] - 1))
                    w = min(w, output.shape[1] - x)
                    h = min(h, output.shape[0] - y)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 0), 1)
                    
                    # Add text annotations
                    text_y = max(20, y - 10)
                    text_x = max(10, x)
                    
                    # Shape type and confidence
                    text = f"{shape.shape_type} ({shape.confidence:.2f})"
                    cv2.putText(output, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, shape.color, 2)
                    text_y -= 20
                    
                    # Area and perimeter
                    if text_y > 20:
                        text = f"A: {shape.area:.1f} P: {shape.perimeter:.1f}"
                        cv2.putText(output, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.4, (255, 255, 255), 1)
                        text_y -= 15
                    
                    # Angles (for polygons)
                    if shape.angles and text_y > 20:
                        angles_text = "∠: " + ", ".join([f"{angle:.1f}°" for angle in shape.angles[:3]])
                        cv2.putText(output, angles_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.4, (255, 255, 255), 1)
                        text_y -= 15
                    
                    # Orientation
                    if shape.shape_type != 'circle' and text_y > 20:
                        text = f"θ: {shape.orientation:.1f}°"
                        cv2.putText(output, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.4, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Error drawing shape: {e}")
                    continue
            
            return output
            
        except Exception as e:
            print(f"Error in draw_shapes: {e}")
            return frame
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        try:
            current_time = time.time()
            fps = 1 / (current_time - self.last_time)
            self.last_time = current_time
            self.fps_buffer.append(fps)
            return np.mean(self.fps_buffer) if self.fps_buffer else 0
        except:
            return 0
    
    def draw_info_panel(self, frame: np.ndarray, shapes: List[GeometricShape], fps: float) -> np.ndarray:
        """
        Draw information panel on frame with error handling
        
        Args:
            frame: Frame to draw on
            shapes: Detected shapes
            fps: Current FPS
            
        Returns:
            Frame with info panel
        """
        try:
            if frame is None or frame.size == 0:
                return frame
            
            # Create info panel
            panel_height = 150
            panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
            panel[:] = (50, 50, 50)
            
            # Add title
            cv2.putText(panel, "Advanced Geometry Detector", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add FPS
            cv2.putText(panel, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add GPU status
            gpu_text = "GPU: Enabled" if self.use_gpu else "GPU: Disabled"
            color = (0, 255, 0) if self.use_gpu else (0, 0, 255)
            cv2.putText(panel, gpu_text, (150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Shape statistics
            shape_counts = {}
            for shape in shapes:
                shape_type = shape.shape_type.split('_')[0]
                shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
            
            # Draw shape counts
            x_offset = 10
            y_offset = 90
            for shape_type, count in shape_counts.items():
                color = self.colors.get(shape_type, (255, 255, 255))
                text = f"{shape_type}: {count}"
                cv2.putText(panel, text, (x_offset, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                x_offset += 150
                if x_offset > frame.shape[1] - 150:
                    x_offset = 10
                    y_offset += 25
            
            # Total shapes
            cv2.putText(panel, f"Total Shapes: {len(shapes)}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Combine frame and panel
            output = np.vstack([panel, frame])
            
            return output
            
        except Exception as e:
            print(f"Error drawing info panel: {e}")
            return frame
    
    def run(self, source=0):
        """
        Run the detector on video source with comprehensive error handling
        
        Args:
            source: Video source (0 for webcam, or video file path)
        """
        cap = None
        try:
            # Initialize video capture
            print(f"Opening video source: {source}")
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                print(f"Error: Could not open video source {source}")
                print("Tips:")
                print("- For webcam, try different indices: 0, 1, 2, etc.")
                print("- For video file, check if path is correct")
                print("- On Linux, try: sudo apt-get install v4l-utils")
                print("- Check camera permissions")
                return
            
            # Try to set camera properties
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
            except:
                print("Warning: Could not set camera properties")
            
            # Get actual camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {width}x{height} @ {fps}fps")
            
            print("\nAdvanced Geometry Detector Started")
            print(f"GPU Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
            print(f"Threads: {self.num_threads}")
            print("\nControls:")
            print("  'q' - Quit")
            print("  'g' - Toggle GPU acceleration")
            print("  'p' - Pause/Resume")
            print("  's' - Save screenshot")
            print("  '+/-' - Adjust detection sensitivity")
            print("  'r' - Reset detector")
            
            paused = False
            frame_count = 0
            
            # Create window
            window_name = 'Advanced Geometry Detector'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width, height + 150)
            
            while True:
                try:
                    if not paused:
                        ret, frame = cap.read()
                        if not ret:
                            print("Failed to read frame")
                            # Try to reconnect
                            cap.release()
                            time.sleep(1)
                            cap = cv2.VideoCapture(source)
                            continue
                        
                        # Process frame
                        shapes = self.detect_shapes(frame)
                        
                        # Draw shapes
                        annotated_frame = self.draw_shapes(frame, shapes)
                        
                        # Calculate FPS
                        fps = self.calculate_fps()
                        
                        # Draw info panel
                        output = self.draw_info_panel(annotated_frame, shapes, fps)
                        
                        # Display frame
                        cv2.imshow(window_name, output)
                        
                        frame_count += 1
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('g'):
                        self.use_gpu = not self.use_gpu and CUDA_AVAILABLE
                        print(f"GPU Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
                    elif key == ord('p'):
                        paused = not paused
                        print(f"{'Paused' if paused else 'Resumed'}")
                    elif key == ord('s'):
                        filename = f"geometry_capture_{frame_count}.png"
                        cv2.imwrite(filename, output)
                        print(f"Screenshot saved: {filename}")
                    elif key == ord('+'):
                        self.epsilon_factor = min(0.1, self.epsilon_factor + 0.005)
                        print(f"Epsilon factor: {self.epsilon_factor:.3f}")
                    elif key == ord('-'):
                        self.epsilon_factor = max(0.001, self.epsilon_factor - 0.005)
                        print(f"Epsilon factor: {self.epsilon_factor:.3f}")
                    elif key == ord('r'):
                        print("Resetting detector...")
                        self.fps_buffer.clear()
                        self.last_time = time.time()
                        
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"Fatal error: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            print("Cleaning up...")
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown()
            print("Cleanup complete")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Geometry Detector')
    parser.add_argument('-s', '--source', default=0, 
                       help='Video source (0 for default webcam, 1+ for other cameras, or video file path)')
    parser.add_argument('-g', '--gpu', action='store_true', default=True,
                       help='Enable GPU acceleration (default: True)')
    parser.add_argument('-t', '--threads', type=int, default=4,
                       help='Number of processing threads (default: 4)')
    parser.add_argument('-f', '--file', type=str, default=None,
                       help='Video file path (overrides source)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit
    if args.file:
        args.source = args.file
    elif str(args.source).isdigit():
        args.source = int(args.source)
    
    return args

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        print("="*60)
        print("Advanced Geometry Detector v2.0")
        print("="*60)
        
        # Initialize detector
        detector = AdvancedGeometryDetector(use_gpu=args.gpu, num_threads=args.threads)
        
        # Run detector
        detector.run(source=args.source)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        print("Program ended")

if __name__ == "__main__":
    main()
