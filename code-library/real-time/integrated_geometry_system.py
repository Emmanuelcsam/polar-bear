#!/usr/bin/env python3
"""
Integrated Real-Time Geometry Detection and Analysis System
==========================================================

A comprehensive system combining all geometry detection capabilities:
- Multi-backend camera support (OpenCV, Pylon/Basler, IP cameras)
- Advanced shape detection (polygons, circles, ellipses, lines)
- Specialized tube angle measurement
- GPU acceleration with CUDA
- Performance benchmarking and optimization
- Comprehensive logging and error handling
- Unit testing framework

Author: Advanced Computer Vision System
Date: 2025
License: BSD-3-Clause
"""

import cv2
import numpy as np
import logging
import time
import json
import threading
import queue
import unittest
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any, Union, Callable
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import warnings
from abc import ABC, abstractmethod
from enum import Enum
import os
import sys
import psutil
import traceback
import shared_config # Import the shared configuration module

# Optional imports with fallback
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    PYLON_AVAILABLE = False
    pylon = None

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = None

warnings.filterwarnings('ignore')

# ==================== Configuration ====================

class Config:
    """Central configuration for the system"""
    # Camera settings
    DEFAULT_CAMERA_INDEX = shared_config.CONFIG.get("camera_source", 0)
    DEFAULT_WIDTH = shared_config.CONFIG.get("camera_width", 1280)
    DEFAULT_HEIGHT = shared_config.CONFIG.get("camera_height", 720)
    DEFAULT_FPS = shared_config.CONFIG.get("camera_fps", 30)
    
    # Detection parameters
    MIN_SHAPE_AREA = shared_config.CONFIG.get("min_shape_area", 100)
    MAX_SHAPE_AREA = shared_config.CONFIG.get("max_shape_area", 100000)
    EPSILON_FACTOR = shared_config.CONFIG.get("epsilon_factor", 0.02)
    CANNY_LOW = shared_config.CONFIG.get("canny_low", 50)
    CANNY_HIGH = shared_config.CONFIG.get("canny_high", 150)
    
    # Performance settings
    MAX_THREADS = shared_config.CONFIG.get("max_threads", 4)
    FRAME_BUFFER_SIZE = shared_config.CONFIG.get("frame_buffer_size", 10)
    FPS_BUFFER_SIZE = shared_config.CONFIG.get("fps_buffer_size", 30)
    
    # Kalman filter settings
    KALMAN_PROCESS_NOISE = shared_config.CONFIG.get("kalman_process_noise", 0.01)
    KALMAN_MEASUREMENT_NOISE = shared_config.CONFIG.get("kalman_measurement_noise", 0.1)
    
    # Logging
    LOG_LEVEL = getattr(logging, shared_config.CONFIG.get("log_level", "INFO").upper())
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # GPU settings
    USE_GPU_DEFAULT = shared_config.CONFIG.get("use_gpu", True)
    
    # Pylon/Basler specific
    PYLON_GRAB_STRATEGY = shared_config.CONFIG.get("pylon_grab_strategy", 'LatestImageOnly') if PYLON_AVAILABLE else None
    PYLON_PIXEL_FORMAT = shared_config.CONFIG.get("pylon_pixel_format", 'Mono8') if PYLON_AVAILABLE else None

# ==================== Logging Setup ====================

def setup_logging(name: str = __name__, level: int = Config.LOG_LEVEL) -> logging.Logger:
    """Setup logging with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(Config.LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(f'geometry_detection_{time.strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ==================== Data Classes ====================

class ShapeType(Enum):
    """Enumeration of detected shape types"""
    UNKNOWN = "unknown"
    LINE = "line"
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    SQUARE = "square"
    PENTAGON = "pentagon"
    HEXAGON = "hexagon"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    TUBE = "tube"

@dataclass
class GeometricShape:
    """Comprehensive geometric shape data"""
    shape_type: ShapeType
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
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['shape_type'] = self.shape_type.value
        data['contour'] = self.contour.tolist()
        data['hu_moments'] = self.hu_moments.tolist()
        return data

@dataclass
class TubeAngle:
    """Tube angle measurement data"""
    bevel_angle: float
    tilt_angle: float
    distance: float
    center: Tuple[float, float]
    axes: Tuple[float, float]
    image_angle: float
    axis_ratio: float
    confidence: float
    wall_thickness: Optional[float] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class PerformanceMetrics:
    """Performance measurement data"""
    fps: float
    latency_ms: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory: float
    shapes_detected: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

# ==================== Camera Backends ====================

class CameraBackend(ABC):
    """Abstract base class for camera backends"""
    
    @abstractmethod
    def open(self) -> bool:
        """Open camera connection"""
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close camera connection"""
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """Get camera properties"""
        pass
    
    @abstractmethod
    def set_property(self, prop: str, value: Any) -> bool:
        """Set camera property"""
        pass

class OpenCVCamera(CameraBackend):
    """OpenCV camera backend"""
    
    def __init__(self, source: Union[int, str] = 0, backend: int = cv2.CAP_ANY):
        self.source = source
        self.backend = backend
        self.cap = None
        self.logger = logging.getLogger(f"{__name__}.OpenCVCamera")
    
    def open(self) -> bool:
        """Open OpenCV camera"""
        try:
            self.cap = cv2.VideoCapture(self.source, self.backend)
            if self.cap.isOpened():
                # Set default properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.DEFAULT_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.DEFAULT_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, Config.DEFAULT_FPS)
                self.logger.info(f"Opened camera at {self.source}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to open camera: {e}")
        return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from OpenCV camera"""
        if self.cap is not None:
            return self.cap.read()
        return False, None
    
    def close(self) -> None:
        """Close OpenCV camera"""
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Camera closed")
    
    def get_properties(self) -> Dict[str, Any]:
        """Get camera properties"""
        if self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE)
        }
    
    def set_property(self, prop: str, value: Any) -> bool:
        """Set camera property"""
        if self.cap is None:
            return False
        
        prop_map = {
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS,
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'exposure': cv2.CAP_PROP_EXPOSURE
        }
        
        if prop in prop_map:
            return self.cap.set(prop_map[prop], value)
        return False

class PylonCamera(CameraBackend):
    """Pylon/Basler camera backend"""
    
    def __init__(self, device_index: int = 0):
        if not PYLON_AVAILABLE:
            raise ImportError("pypylon not available. Install with: pip install pypylon")
        
        self.device_index = device_index
        self.camera = None
        self.converter = None
        self.logger = logging.getLogger(f"{__name__}.PylonCamera")
    
    def open(self) -> bool:
        """Open Pylon camera"""
        try:
            factory = pylon.TlFactory.GetInstance()
            devices = factory.EnumerateDevices()
            
            if not devices:
                self.logger.error("No Pylon devices found")
                return False
            
            if self.device_index >= len(devices):
                self.logger.error(f"Device index {self.device_index} out of range")
                return False
            
            self.camera = pylon.InstantCamera(factory.CreateDevice(devices[self.device_index]))
            self.camera.Open()
            
            # Configure camera
            self.camera.PixelFormat = Config.PYLON_PIXEL_FORMAT
            self.camera.ExposureAuto = "Off"
            self.camera.GainAuto = "Off"
            
            # Start grabbing
            strategy = getattr(pylon, f"GrabStrategy_{Config.PYLON_GRAB_STRATEGY}")
            self.camera.StartGrabbing(strategy)
            
            # Setup converter
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_Mono8
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            
            self.logger.info(f"Opened Pylon camera: {devices[self.device_index].GetFriendlyName()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open Pylon camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from Pylon camera"""
        if self.camera is None or not self.camera.IsGrabbing():
            return False, None
        
        try:
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result).GetArray()
                grab_result.Release()
                
                # Convert to BGR if mono
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
                return True, image
            
            grab_result.Release()
            return False, None
            
        except Exception as e:
            self.logger.error(f"Failed to read frame: {e}")
            return False, None
    
    def close(self) -> None:
        """Close Pylon camera"""
        if self.camera is not None:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            self.camera.Close()
            self.logger.info("Pylon camera closed")
    
    def get_properties(self) -> Dict[str, Any]:
        """Get camera properties"""
        if self.camera is None:
            return {}
        
        try:
            return {
                'width': self.camera.Width.Value,
                'height': self.camera.Height.Value,
                'fps': self.camera.ResultingFrameRate.Value,
                'exposure': self.camera.ExposureTime.Value,
                'gain': self.camera.Gain.Value,
                'model': self.camera.GetDeviceInfo().GetModelName()
            }
        except:
            return {}
    
    def set_property(self, prop: str, value: Any) -> bool:
        """Set camera property"""
        if self.camera is None:
            return False
        
        try:
            if prop == 'exposure':
                self.camera.ExposureTime.Value = value
            elif prop == 'gain':
                self.camera.Gain.Value = value
            elif prop == 'width':
                self.camera.Width.Value = int(value)
            elif prop == 'height':
                self.camera.Height.Value = int(value)
            else:
                return False
            return True
        except:
            return False

# ==================== Detection Algorithms ====================

class KalmanFilter:
    """Kalman filter for temporal smoothing"""
    
    def __init__(self, state_dim: int = 4, measure_dim: int = 2):
        self.kf = cv2.KalmanFilter(state_dim, measure_dim)
        self.kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(measure_dim, state_dim, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * Config.KALMAN_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(measure_dim, dtype=np.float32) * Config.KALMAN_MEASUREMENT_NOISE
        self.initialized = False
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update Kalman filter with measurement"""
        if not self.initialized:
            self.kf.statePost = np.zeros((self.kf.statePost.shape[0], 1), dtype=np.float32)
            self.kf.statePost[:len(measurement)] = measurement.reshape(-1, 1)
            self.initialized = True
        
        self.kf.predict()
        self.kf.correct(measurement.astype(np.float32))
        return self.kf.statePost[:len(measurement)].flatten()

class GeometryDetector:
    """Main geometry detection engine"""
    
    def __init__(self, use_gpu: bool = Config.USE_GPU_DEFAULT):
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.logger = logging.getLogger(f"{__name__}.GeometryDetector")
        self.kalman_filters = {}
        
        # Shape colors
        self.colors = {
            ShapeType.TRIANGLE: (255, 0, 0),      # Blue
            ShapeType.RECTANGLE: (0, 255, 0),      # Green
            ShapeType.SQUARE: (0, 255, 255),       # Yellow
            ShapeType.PENTAGON: (255, 0, 255),     # Magenta
            ShapeType.HEXAGON: (0, 165, 255),      # Orange
            ShapeType.CIRCLE: (255, 255, 0),       # Cyan
            ShapeType.ELLIPSE: (128, 0, 128),      # Purple
            ShapeType.POLYGON: (255, 255, 255),    # White
            ShapeType.LINE: (0, 0, 255),           # Red
            ShapeType.UNKNOWN: (128, 128, 128)     # Gray
        }
        
        self.logger.info(f"Detector initialized. GPU: {self.use_gpu}")
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess frame for detection"""
        try:
            if self.use_gpu:
                # GPU preprocessing
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                gpu_blurred = cv2.cuda.bilateralFilter(gpu_gray, -1, 50, 50)
                
                # CLAHE
                clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gpu_enhanced = clahe.apply(gpu_blurred)
                
                gray = gpu_enhanced.download()
            else:
                # CPU preprocessing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            # Multi-scale edge detection
            edges = self._multi_scale_edges(gray)
            
            return gray, edges
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            return np.zeros_like(frame[:,:,0]), np.zeros_like(frame[:,:,0])
    
    def _multi_scale_edges(self, gray: np.ndarray) -> np.ndarray:
        """Multi-scale edge detection"""
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 100, 200)
        
        edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def detect_shapes(self, frame: np.ndarray) -> List[GeometricShape]:
        """Detect all shapes in frame"""
        shapes = []
        
        try:
            gray, edges = self.preprocess_frame(frame)
            
            # Find contours
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                shape = self._process_contour(contour, frame)
                if shape is not None:
                    shapes.append(shape)
            
            # Detect lines
            lines = self._detect_lines(edges)
            shapes.extend(lines)
            
            # Detect circles
            circles = self._detect_circles(gray)
            shapes.extend(circles)
            
            # Apply temporal smoothing
            shapes = self._apply_temporal_smoothing(shapes)
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            traceback.print_exc()
        
        return shapes
    
    def _process_contour(self, contour: np.ndarray, frame: np.ndarray) -> Optional[GeometricShape]:
        """Process single contour"""
        try:
            area = cv2.contourArea(contour)
            if not (Config.MIN_SHAPE_AREA < area < Config.MAX_SHAPE_AREA):
                return None
            
            # Calculate properties
            properties = self._calculate_shape_properties(contour)
            
            # Classify shape
            shape_type, confidence = self._classify_shape(contour, properties)
            
            # Get color
            color = self.colors.get(shape_type, self.colors[ShapeType.UNKNOWN])
            
            # Extract vertices
            epsilon = Config.EPSILON_FACTOR * properties['perimeter']
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = [tuple(v[0]) for v in approx]
            
            # Calculate angles
            angles = self._calculate_polygon_angles(approx) if len(approx) >= 3 else []
            
            return GeometricShape(
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
            
        except Exception as e:
            self.logger.error(f"Contour processing error: {e}")
            return None
    
    def _calculate_shape_properties(self, contour: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive shape properties"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Prevent division by zero
        area = max(area, 1)
        perimeter = max(perimeter, 1)
        
        # Moments
        moments = cv2.moments(contour)
        
        # Centroid
        cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0
        cy = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        w = max(w, 1)
        h = max(h, 1)
        
        # Oriented bounding box
        if len(contour) >= 5:
            (ox, oy), (width, height), angle = cv2.minAreaRect(contour)
            orientation = angle
        else:
            orientation = 0
            width, height = w, h
        
        # Shape descriptors
        aspect_ratio = float(w) / h
        rect_area = w * h
        extent = float(area) / rect_area
        
        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        hull_area = max(hull_area, 1)
        solidity = float(area) / hull_area
        
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
        hu_moments = cv2.HuMoments(moments).flatten()
        
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
            'hu_moments': hu_moments
        }
    
    def _classify_shape(self, contour: np.ndarray, properties: Dict[str, Any]) -> Tuple[ShapeType, float]:
        """Classify shape using multiple criteria"""
        epsilon = Config.EPSILON_FACTOR * properties['perimeter']
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Circularity
        circularity = 4 * np.pi * properties['area'] / (properties['perimeter'] ** 2)
        
        confidence = 1.0
        
        if vertices == 2:
            return ShapeType.LINE, 0.9
        
        elif vertices == 3:
            angles = self._calculate_polygon_angles(approx)
            if len(angles) == 3 and 170 < sum(angles) < 190:
                return ShapeType.TRIANGLE, 0.95
            return ShapeType.TRIANGLE, 0.8
        
        elif vertices == 4:
            angles = self._calculate_polygon_angles(approx)
            if len(angles) == 4 and all(85 < angle < 95 for angle in angles):
                if 0.9 < properties['aspect_ratio'] < 1.1:
                    return ShapeType.SQUARE, 0.95
                else:
                    return ShapeType.RECTANGLE, 0.95
            return ShapeType.RECTANGLE, 0.8
        
        elif vertices == 5:
            return ShapeType.PENTAGON, 0.9
        
        elif vertices == 6:
            return ShapeType.HEXAGON, 0.9
        
        elif vertices > 6 or circularity > 0.7:
            if circularity > 0.8:
                if properties['eccentricity'] < 0.3:
                    return ShapeType.CIRCLE, 0.95
                else:
                    return ShapeType.ELLIPSE, 0.9
            else:
                return ShapeType.POLYGON, 0.85
        
        else:
            return ShapeType.UNKNOWN, 0.5
    
    def _calculate_polygon_angles(self, vertices: np.ndarray) -> List[float]:
        """Calculate interior angles of polygon"""
        angles = []
        n = len(vertices)
        
        if n < 3:
            return angles
        
        for i in range(n):
            p1 = vertices[(i - 1) % n][0]
            p2 = vertices[i][0]
            p3 = vertices[(i + 1) % n][0]
            
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 * norm2 != 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
            else:
                angles.append(0)
        
        return angles
    
    def _detect_lines(self, edges: np.ndarray) -> List[GeometricShape]:
        """Detect lines using Hough transform"""
        lines_shapes = []
        
        try:
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                   minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                for line in lines[:50]:  # Limit to 50 lines
                    x1, y1, x2, y2 = line[0]
                    
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    # Create line contour
                    contour = np.array([[x1, y1], [x2, y2]], dtype=np.int32)
                    
                    shape = GeometricShape(
                        shape_type=ShapeType.LINE,
                        contour=contour,
                        center=center,
                        area=0,
                        perimeter=length,
                        vertices=[(x1, y1), (x2, y2)],
                        angles=[],
                        bounding_box=(min(x1, x2), min(y1, y2), 
                                     abs(x2 - x1), abs(y2 - y1)),
                        orientation=angle,
                        eccentricity=1.0,
                        solidity=1.0,
                        aspect_ratio=length,
                        extent=1.0,
                        equivalent_diameter=0,
                        compactness=0,
                        hu_moments=np.zeros(7),
                        color=self.colors[ShapeType.LINE],
                        confidence=0.9
                    )
                    lines_shapes.append(shape)
                    
        except Exception as e:
            self.logger.error(f"Line detection error: {e}")
        
        return lines_shapes
    
    def _detect_circles(self, gray: np.ndarray) -> List[GeometricShape]:
        """Detect circles using Hough Circle Transform"""
        circle_shapes = []
        
        try:
            # Multi-scale circle detection
            for dp in [1.0, 1.5, 2.0]:
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp,
                                         minDist=50, param1=100, param2=30,
                                         minRadius=10, maxRadius=200)
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    
                    for circle in circles[0, :20]:  # Limit to 20 circles
                        center = (int(circle[0]), int(circle[1]))
                        radius = int(circle[2])
                        
                        # Validate bounds
                        if (0 <= center[0] < gray.shape[1] and 
                            0 <= center[1] < gray.shape[0]):
                            
                            # Create circle contour
                            angles = np.linspace(0, 2*np.pi, 36)
                            circle_points = []
                            for angle in angles:
                                x = int(center[0] + radius * np.cos(angle))
                                y = int(center[1] + radius * np.sin(angle))
                                circle_points.append([x, y])
                            
                            contour = np.array(circle_points, dtype=np.int32)
                            area = np.pi * radius * radius
                            perimeter = 2 * np.pi * radius
                            
                            shape = GeometricShape(
                                shape_type=ShapeType.CIRCLE,
                                contour=contour,
                                center=center,
                                area=area,
                                perimeter=perimeter,
                                vertices=[],
                                angles=[],
                                bounding_box=(center[0] - radius, center[1] - radius,
                                            2 * radius, 2 * radius),
                                orientation=0,
                                eccentricity=0,
                                solidity=1.0,
                                aspect_ratio=1.0,
                                extent=np.pi / 4,
                                equivalent_diameter=2 * radius,
                                compactness=12.57,
                                hu_moments=np.zeros(7),
                                color=self.colors[ShapeType.CIRCLE],
                                confidence=0.95
                            )
                            circle_shapes.append(shape)
                            
        except Exception as e:
            self.logger.error(f"Circle detection error: {e}")
        
        return circle_shapes
    
    def _apply_temporal_smoothing(self, shapes: List[GeometricShape]) -> List[GeometricShape]:
        """Apply Kalman filtering for temporal smoothing"""
        smoothed = []
        
        for shape in shapes:
            # Create unique key for shape tracking
            key = f"{shape.shape_type.value}_{shape.center[0]}_{shape.center[1]}"
            
            if key not in self.kalman_filters:
                self.kalman_filters[key] = KalmanFilter()
            
            # Update Kalman filter
            measurement = np.array([shape.center[0], shape.center[1]])
            smoothed_pos = self.kalman_filters[key].update(measurement)
            
            # Update shape center
            shape.center = (int(smoothed_pos[0]), int(smoothed_pos[1]))
            smoothed.append(shape)
        
        # Clean old filters
        if len(self.kalman_filters) > 100:
            self.kalman_filters.clear()
        
        return smoothed

class TubeAngleDetector:
    """Specialized tube angle detection"""
    
    def __init__(self, camera_matrix: Optional[np.ndarray] = None, 
                 dist_coeffs: Optional[np.ndarray] = None,
                 tube_diameter: float = 10.0):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tube_diameter = tube_diameter
        self.logger = logging.getLogger(f"{__name__}.TubeAngleDetector")
        self.kalman = self._setup_kalman()
    
    def _setup_kalman(self) -> cv2.KalmanFilter:
        """Setup Kalman filter for angle smoothing"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array([[1, 1, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 1],
                                           [0, 0, 0, 1]], dtype=np.float32)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 0, 1, 0]], dtype=np.float32)
        kalman.processNoiseCov = 0.01 * np.eye(4, dtype=np.float32)
        kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        return kalman
    
    def detect_tube_angle(self, frame: np.ndarray) -> Optional[TubeAngle]:
        """Detect tube angle from frame"""
        try:
            ellipse = self._detect_tube_ellipse(frame)
            if ellipse is None:
                return None
            
            # Check for concentric ellipses
            concentric = self._detect_concentric_ellipses(frame)
            
            if concentric:
                return self._refine_pose_with_concentric(concentric[0], concentric[1])
            else:
                return self._estimate_3d_pose(ellipse)
                
        except Exception as e:
            self.logger.error(f"Tube angle detection error: {e}")
            return None
    
    def _detect_tube_ellipse(self, frame: np.ndarray) -> Optional[Tuple]:
        """Detect tube end face ellipse"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ellipse = None
        best_score = 0
        
        for contour in contours:
            if len(contour) < 5:
                continue
            
            area = cv2.contourArea(contour)
            if not (500 < area < 50000):
                continue
            
            ellipse = cv2.fitEllipse(contour)
            score = self._calculate_ellipse_fit_score(contour, ellipse)
            
            if score > best_score:
                best_score = score
                best_ellipse = ellipse
        
        return best_ellipse if best_score > 0.7 else None
    
    def _calculate_ellipse_fit_score(self, contour: np.ndarray, ellipse: Tuple) -> float:
        """Calculate ellipse fit quality"""
        center, axes, angle = ellipse
        
        # Generate ellipse points
        ellipse_points = cv2.ellipse2Poly(
            (int(center[0]), int(center[1])),
            (int(axes[0]/2), int(axes[1]/2)),
            int(angle), 0, 360, 5
        )
        
        # Calculate distances
        distances = []
        for pt in contour[:, 0]:
            min_dist = np.min([np.linalg.norm(pt - ept) for ept in ellipse_points])
            distances.append(min_dist)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Normalize score
        score = 1.0 / (1.0 + mean_dist / 10.0) * np.exp(-std_dist / 10.0)
        
        return score
    
    def _detect_concentric_ellipses(self, frame: np.ndarray) -> List[Tuple]:
        """Detect concentric ellipses for tube walls"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        edges = cv2.Canny(enhanced, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        ellipses = []
        for contour in contours:
            if len(contour) >= 5:
                area = cv2.contourArea(contour)
                if 500 < area < 50000:
                    ellipse = cv2.fitEllipse(contour)
                    ellipses.append(ellipse)
        
        # Find concentric pairs
        concentric_pairs = []
        for i, e1 in enumerate(ellipses):
            for j, e2 in enumerate(ellipses[i+1:], i+1):
                if self._are_concentric(e1, e2):
                    concentric_pairs.append((e1, e2))
        
        return concentric_pairs
    
    def _are_concentric(self, e1: Tuple, e2: Tuple, tolerance: float = 20) -> bool:
        """Check if two ellipses are concentric"""
        center1, axes1, angle1 = e1
        center2, axes2, angle2 = e2
        
        # Check center distance
        center_dist = np.linalg.norm(np.array(center1) - np.array(center2))
        if center_dist > tolerance:
            return False
        
        # Check angle similarity
        angle_diff = abs(angle1 - angle2) % 180
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        if angle_diff > 15:
            return False
        
        # Check if one is inside the other
        if max(axes1) < min(axes2) or max(axes2) < min(axes1):
            return False
        
        return True
    
    def _estimate_3d_pose(self, ellipse: Tuple) -> TubeAngle:
        """Estimate 3D pose from ellipse"""
        center, axes, angle = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        
        # Calculate tilt angle from axis ratio
        axis_ratio = minor_axis / major_axis
        tilt_angle = np.arccos(axis_ratio) * 180 / np.pi
        
        # Estimate distance
        if self.camera_matrix is not None:
            focal_length = (self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / 2
            distance = (self.tube_diameter * focal_length) / major_axis
        else:
            distance = (self.tube_diameter * 500) / major_axis
        
        # Calculate bevel angle
        bevel_angle = np.arccos(axis_ratio) * 180 / np.pi
        
        # Apply Kalman filter
        measurement = np.array([[bevel_angle], [tilt_angle]], dtype=np.float32)
        self.kalman.predict()
        self.kalman.correct(measurement)
        
        smoothed_bevel = float(self.kalman.statePost[0])
        smoothed_tilt = float(self.kalman.statePost[2])
        
        return TubeAngle(
            bevel_angle=smoothed_bevel,
            tilt_angle=smoothed_tilt,
            distance=distance,
            center=center,
            axes=axes,
            image_angle=angle,
            axis_ratio=axis_ratio,
            confidence=axis_ratio
        )
    
    def _refine_pose_with_concentric(self, inner: Tuple, outer: Tuple) -> TubeAngle:
        """Refine pose using concentric ellipses"""
        # Average centers
        center = ((inner[0][0] + outer[0][0]) / 2,
                 (inner[0][1] + outer[0][1]) / 2)
        
        # Average orientations
        angle = (inner[2] + outer[2]) / 2
        
        # Use outer ellipse for measurements
        axes = outer[1]
        
        # Calculate more accurate tilt
        inner_ratio = min(inner[1]) / max(inner[1])
        outer_ratio = min(outer[1]) / max(outer[1])
        avg_ratio = (inner_ratio + outer_ratio) / 2
        
        tilt_angle = np.arccos(avg_ratio) * 180 / np.pi
        
        # Wall thickness
        wall_thickness = (max(outer[1]) - max(inner[1])) / 2
        
        # Create pose with refined measurements
        pose = self._estimate_3d_pose(outer)
        pose.center = center
        pose.tilt_angle = tilt_angle
        pose.wall_thickness = wall_thickness
        pose.confidence = 0.95
        
        return pose

# ==================== Performance Monitor ====================

class PerformanceMonitor:
    """Monitor and benchmark system performance"""
    
    def __init__(self):
        self.fps_buffer = deque(maxlen=Config.FPS_BUFFER_SIZE)
        self.last_time = time.time()
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
    
    def update(self) -> float:
        """Update FPS calculation"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer) if self.fps_buffer else 0
    
    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        resources = {
            'cpu': psutil.cpu_percent(interval=0.1),
            'memory': psutil.virtual_memory().percent,
            'gpu': 0,
            'gpu_memory': 0
        }
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    resources['gpu'] = gpu.load * 100
                    resources['gpu_memory'] = gpu.memoryUtil * 100
            except:
                pass
        
        return resources
    
    def get_metrics(self, shapes_detected: int) -> PerformanceMetrics:
        """Get current performance metrics"""
        fps = self.update()
        latency = 1000.0 / (fps + 1e-6)
        resources = self.get_system_resources()
        
        return PerformanceMetrics(
            fps=fps,
            latency_ms=latency,
            cpu_usage=resources['cpu'],
            memory_usage=resources['memory'],
            gpu_usage=resources['gpu'],
            gpu_memory=resources['gpu_memory'],
            shapes_detected=shapes_detected
        )

# ==================== Visualization ====================

class Visualizer:
    """Handle visualization and UI rendering"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Visualizer")
    
    def draw_shapes(self, frame: np.ndarray, shapes: List[GeometricShape]) -> np.ndarray:
        """Draw detected shapes on frame"""
        output = frame.copy()
        
        for shape in shapes:
            try:
                # Draw contour
                if shape.contour is not None and len(shape.contour) > 0:
                    cv2.drawContours(output, [shape.contour], -1, shape.color, 2)
                
                # Draw vertices
                for vertex in shape.vertices:
                    if self._is_valid_point(vertex, output.shape):
                        cv2.circle(output, vertex, 5, (0, 255, 0), -1)
                
                # Draw center
                if self._is_valid_point(shape.center, output.shape):
                    cv2.circle(output, shape.center, 7, (255, 0, 0), -1)
                
                # Draw bounding box
                x, y, w, h = shape.bounding_box
                x, y, w, h = self._clip_bbox(x, y, w, h, output.shape)
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 0), 1)
                
                # Add text annotations
                self._draw_shape_info(output, shape)
                
            except Exception as e:
                self.logger.error(f"Error drawing shape: {e}")
        
        return output
    
    def draw_tube_angle(self, frame: np.ndarray, tube_angle: TubeAngle) -> np.ndarray:
        """Draw tube angle visualization"""
        output = frame.copy()
        
        if tube_angle is None:
            return output
        
        # Draw ellipse
        center = (int(tube_angle.center[0]), int(tube_angle.center[1]))
        axes = (int(tube_angle.axes[0]/2), int(tube_angle.axes[1]/2))
        angle = int(tube_angle.image_angle)
        
        cv2.ellipse(output, center, axes, angle, 0, 360, (0, 255, 0), 2)
        cv2.circle(output, center, 5, (255, 0, 0), -1)
        
        # Draw angle information
        y_offset = 30
        info = [
            f"Bevel Angle: {tube_angle.bevel_angle:.1f}°",
            f"Tilt Angle: {tube_angle.tilt_angle:.1f}°",
            f"Distance: {tube_angle.distance:.1f}mm",
            f"Confidence: {tube_angle.confidence:.2f}"
        ]
        
        for text in info:
            cv2.putText(output, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        return output
    
    def draw_performance_overlay(self, frame: np.ndarray, metrics: PerformanceMetrics) -> np.ndarray:
        """Draw performance overlay"""
        output = frame.copy()
        
        # Create semi-transparent overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Draw metrics
        y_offset = 30
        texts = [
            f"FPS: {metrics.fps:.1f}",
            f"Latency: {metrics.latency_ms:.1f}ms",
            f"CPU: {metrics.cpu_usage:.1f}%",
            f"Memory: {metrics.memory_usage:.1f}%",
            f"Shapes: {metrics.shapes_detected}"
        ]
        
        if metrics.gpu_usage > 0:
            texts.append(f"GPU: {metrics.gpu_usage:.1f}%")
        
        for text in texts:
            cv2.putText(output, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return output
    
    def _draw_shape_info(self, frame: np.ndarray, shape: GeometricShape):
        """Draw shape information text"""
        x, y, _, _ = shape.bounding_box
        y = max(20, y - 10)
        x = max(10, x)
        
        # Shape type and confidence
        text = f"{shape.shape_type.value} ({shape.confidence:.2f})"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, shape.color, 2)
        
        # Area and perimeter
        if y > 40:
            text = f"A: {shape.area:.0f} P: {shape.perimeter:.0f}"
            cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 255), 1)
    
    def _is_valid_point(self, point: Tuple[int, int], shape: Tuple) -> bool:
        """Check if point is within frame bounds"""
        return 0 <= point[0] < shape[1] and 0 <= point[1] < shape[0]
    
    def _clip_bbox(self, x: int, y: int, w: int, h: int, shape: Tuple) -> Tuple[int, int, int, int]:
        """Clip bounding box to frame bounds"""
        x = max(0, min(x, shape[1] - 1))
        y = max(0, min(y, shape[0] - 1))
        w = min(w, shape[1] - x)
        h = min(h, shape[0] - y)
        return x, y, w, h

# ==================== Main Application ====================

class GeometryDetectionSystem:
    """Main integrated geometry detection system"""
    
    def __init__(self, camera_backend: str = None, 
                 camera_source: Union[int, str] = None,
                 use_gpu: bool = None,
                 enable_tube_detection: bool = None,
                 enable_benchmarking: bool = None):
        
        self.logger = logging.getLogger(f"{__name__}.GeometryDetectionSystem")
        
        # Load initial configuration from shared_config
        self.current_shared_config = shared_config.get_config()

        # Initialize parameters from shared_config, falling back to defaults or constructor args
        self.camera_backend = self.current_shared_config.get("camera_backend", camera_backend if camera_backend is not None else 'opencv')
        self.camera_source = self.current_shared_config.get("camera_source", camera_source if camera_source is not None else Config.DEFAULT_CAMERA_INDEX)
        self.use_gpu = self.current_shared_config.get("use_gpu", use_gpu if use_gpu is not None else Config.USE_GPU_DEFAULT)
        self.enable_tube_detection = self.current_shared_config.get("enable_tube_detection", enable_tube_detection if enable_tube_detection is not None else True)
        self.enable_benchmarking = self.current_shared_config.get("enable_benchmarking", enable_benchmarking if enable_benchmarking is not None else True)

        # Initialize camera
        if self.camera_backend == 'pylon' and PYLON_AVAILABLE:
            self.camera = PylonCamera(self.camera_source if isinstance(self.camera_source, int) else 0)
        else:
            self.camera = OpenCVCamera(self.camera_source)
        
        # Initialize components
        self.detector = GeometryDetector(self.use_gpu)
        self.tube_detector = TubeAngleDetector() if self.enable_tube_detection else None
        self.performance_monitor = PerformanceMonitor() if self.enable_benchmarking else None
        self.visualizer = Visualizer()
        
        # State
        self.running = False
        self.paused = False
        self.recording = False
        self.writer = None
        
        # Results storage
        self.results_history = deque(maxlen=1000)
        
        self.status = "initialized" # Add a status variable
        self.logger.info("Geometry Detection System initialized")

    def get_script_info(self):
        return {
            "name": "Integrated Geometry Detection System",
            "status": self.status,
            "parameters": {
                "camera_backend": self.camera_backend,
                "camera_source": self.camera_source,
                "use_gpu": self.use_gpu,
                "enable_tube_detection": self.enable_tube_detection,
                "enable_benchmarking": self.enable_benchmarking,
                "min_shape_area": Config.MIN_SHAPE_AREA,
                "max_shape_area": Config.MAX_SHAPE_AREA,
                "epsilon_factor": Config.EPSILON_FACTOR,
                "canny_low": Config.CANNY_LOW,
                "canny_high": Config.CANNY_HIGH,
                "log_level": self.current_shared_config.get("log_level"),
                "data_source": self.current_shared_config.get("data_source"),
                "processing_enabled": self.current_shared_config.get("processing_enabled"),
                "threshold_value": self.current_shared_config.get("threshold_value")
            },
            "camera_properties": self.camera.get_properties() if self.camera else {},
            "detector_info": self.detector.get_script_info() if hasattr(self.detector, 'get_script_info') else "N/A",
            "performance_metrics": self.performance_monitor.get_metrics(0) if self.performance_monitor else "N/A"
        }

    def set_script_parameter(self, key, value):
        if key in self.current_shared_config or hasattr(Config, key.upper()):
            # Update shared_config
            shared_config.set_config_value(key, value)
            self.current_shared_config[key] = value # Update local copy

            # Apply changes to relevant components
            if key == "camera_source":
                self.camera_source = value
                # Re-initialize camera if running, or mark for re-init on next start
                if self.running and self.camera:
                    self.camera.close()
                    if self.camera_backend == 'pylon' and PYLON_AVAILABLE:
                        self.camera = PylonCamera(self.camera_source if isinstance(self.camera_source, int) else 0)
                    else:
                        self.camera = OpenCVCamera(self.camera_source)
                    self.camera.open()
            elif key == "use_gpu":
                self.use_gpu = value
                self.detector.use_gpu = value # Update detector's GPU setting
            elif key == "enable_tube_detection":
                self.enable_tube_detection = value
                self.tube_detector = TubeAngleDetector() if value else None
            elif key == "enable_benchmarking":
                self.enable_benchmarking = value
                self.performance_monitor = PerformanceMonitor() if value else None
            elif key.upper() in dir(Config): # For parameters directly in Config class
                setattr(Config, key.upper(), value)
                # Re-initialize detector if detection parameters changed
                if key in ["min_shape_area", "max_shape_area", "epsilon_factor", "canny_low", "canny_high"]:
                    self.detector = GeometryDetector(self.use_gpu)
            
            self.status = f"parameter '{key}' updated"
            return True
        return False
    
    def run(self):
        """Main application loop"""
        if not self.camera.open():
            self.logger.error("Failed to open camera")
            self.status = "camera_error"
            return
        
        self.running = True
        self.status = "running"
        self.logger.info("Starting detection loop")
        
        # Get camera properties
        props = self.camera.get_properties()
        self.logger.info(f"Camera properties: {props}")
        
        # Create window
        window_name = "Integrated Geometry Detection System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                # Handle pause
                if self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('p'):
                        self.paused = False
                    elif key == ord('q'):
                        break
                    continue
                
                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    self.logger.warning("Failed to read frame")
                    continue
                
                # Process frame only if processing is enabled
                if self.current_shared_config.get("processing_enabled", True):
                    # Detect shapes
                    shapes = self.detector.detect_shapes(frame)
                    
                    # Detect tube angles if enabled
                    tube_angle = None
                    if self.tube_detector:
                        tube_angle = self.tube_detector.detect_tube_angle(frame)
                    
                    # Draw results
                    output = self.visualizer.draw_shapes(frame, shapes)
                    if tube_angle:
                        output = self.visualizer.draw_tube_angle(output, tube_angle)
                    
                    # Draw performance overlay
                    if self.performance_monitor:
                        metrics = self.performance_monitor.get_metrics(len(shapes))
                        output = self.visualizer.draw_performance_overlay(output, metrics)
                        
                        # Store results
                        self.results_history.append({
                            'timestamp': time.time(),
                            'shapes': [s.to_dict() for s in shapes],
                            'tube_angle': asdict(tube_angle) if tube_angle else None,
                            'metrics': asdict(metrics)
                        })
                else:
                    output = frame.copy() # Display original frame if processing is disabled
                
                # Record if enabled
                if self.recording and self.writer:
                    self.writer.write(output)
                
                # Display
                cv2.imshow(window_name, output)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard(key):
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def _handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == ord('q'):
            return False
        elif key == ord('p'):
            self.paused = not self.paused
            self.status = "paused" if self.paused else "running"
            self.logger.info(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == ord('s'):
            self._save_screenshot()
        elif key == ord('r'):
            self.recording = not self.recording
            if self.recording:
                self._start_recording()
            else:
                self._stop_recording()
        elif key == ord('b'):
            self._save_benchmark_results()
        elif key == ord('g'):
            self.use_gpu = not self.use_gpu
            self.detector.use_gpu = self.use_gpu # Update detector's GPU setting
            shared_config.set_config_value("use_gpu", self.use_gpu) # Update shared config
            self.logger.info(f"GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        elif key == ord('+'):
            Config.EPSILON_FACTOR = min(0.1, Config.EPSILON_FACTOR + 0.005)
            shared_config.set_config_value("epsilon_factor", Config.EPSILON_FACTOR) # Update shared config
            self.logger.info(f"Epsilon factor: {Config.EPSILON_FACTOR:.3f}")
        elif key == ord('-'):
            Config.EPSILON_FACTOR = max(0.001, Config.EPSILON_FACTOR - 0.005)
            shared_config.set_config_value("epsilon_factor", Config.EPSILON_FACTOR) # Update shared config
            self.logger.info(f"Epsilon factor: {Config.EPSILON_FACTOR:.3f}")
        
        return True
    
    def _save_screenshot(self):
        """Save current frame"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"geometry_screenshot_{timestamp}.png"
        
        ret, frame = self.camera.read()
        if ret and frame is not None:
            cv2.imwrite(filename, frame)
            self.logger.info(f"Screenshot saved: {filename}")
    
    def _start_recording(self):
        """Start video recording"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"geometry_recording_{timestamp}.avi"
        
        props = self.camera.get_properties()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(filename, fourcc, 20.0, 
                                     (props['width'], props['height']))
        self.logger.info(f"Recording started: {filename}")
    
    def _stop_recording(self):
        """Stop video recording"""
        if self.writer:
            self.writer.release()
            self.writer = None
            self.logger.info("Recording stopped")
    
    def _save_benchmark_results(self):
        """Save benchmark results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"geometry_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(list(self.results_history), f, indent=2)
        
        self.logger.info(f"Benchmark results saved: {filename}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")
        
        if self.writer:
            self.writer.release()
        
        self.camera.close()
        cv2.destroyAllWindows()
        
        self.logger.info("Cleanup complete")

# ==================== Unit Tests ====================

class TestGeometryDetection(unittest.TestCase):
    """Unit tests for geometry detection system"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.detector = GeometryDetector(use_gpu=False)
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Draw test shapes
        cv2.rectangle(self.test_image, (50, 50), (150, 150), (0, 0, 0), -1)
        cv2.circle(self.test_image, (300, 100), 50, (0, 0, 0), -1)
        pts = np.array([[400, 50], [450, 150], [350, 150]], np.int32)
        cv2.fillPoly(self.test_image, [pts], (0, 0, 0))
    
    def test_shape_detection(self):
        """Test basic shape detection"""
        shapes = self.detector.detect_shapes(self.test_image)
        self.assertGreater(len(shapes), 0, "No shapes detected")
        
        # Check shape types
        shape_types = [s.shape_type for s in shapes]
        self.assertIn(ShapeType.RECTANGLE, shape_types, "Rectangle not detected")
        self.assertIn(ShapeType.CIRCLE, shape_types, "Circle not detected")
        self.assertIn(ShapeType.TRIANGLE, shape_types, "Triangle not detected")
    
    def test_shape_properties(self):
        """Test shape property calculation"""
        # Create simple square
        square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
        props = self.detector._calculate_shape_properties(square)
        
        self.assertAlmostEqual(props['area'], 10000, delta=100)
        self.assertAlmostEqual(props['perimeter'], 400, delta=10)
        self.assertAlmostEqual(props['aspect_ratio'], 1.0, delta=0.1)
    
    def test_angle_calculation(self):
        """Test polygon angle calculation"""
        # Create right triangle
        triangle = np.array([[[0, 0]], [[100, 0]], [[0, 100]]], dtype=np.int32)
        angles = self.detector._calculate_polygon_angles(triangle)
        
        self.assertEqual(len(angles), 3)
        self.assertAlmostEqual(angles[0], 90, delta=5)
    
    def test_kalman_filter(self):
        """Test Kalman filter"""
        kf = KalmanFilter()
        
        # Test with constant position
        for _ in range(10):
            result = kf.update(np.array([100, 200]))
        
        self.assertAlmostEqual(result[0], 100, delta=1)
        self.assertAlmostEqual(result[1], 200, delta=1)
    
    def test_performance_monitor(self):
        """Test performance monitoring"""
        monitor = PerformanceMonitor()
        
        # Simulate frame processing
        for _ in range(10):
            time.sleep(0.033)  # ~30 FPS
            fps = monitor.update()
        
        self.assertGreater(fps, 20)
        self.assertLess(fps, 40)
        
        metrics = monitor.get_metrics(5)
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.cpu_usage, 0)
    
    def test_data_serialization(self):
        """Test data class serialization"""
        shape = GeometricShape(
            shape_type=ShapeType.CIRCLE,
            contour=np.array([[0, 0], [1, 1]]),
            center=(100, 100),
            area=314.0,
            perimeter=62.8,
            vertices=[],
            angles=[],
            bounding_box=(50, 50, 100, 100),
            orientation=0,
            eccentricity=0,
            solidity=1.0,
            aspect_ratio=1.0,
            extent=0.785,
            equivalent_diameter=20,
            compactness=12.57,
            hu_moments=np.zeros(7),
            color=(255, 255, 0),
            confidence=0.95
        )
        
        data = shape.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['shape_type'], 'circle')
        
        # Test JSON serialization
        json_str = json.dumps(data)
        self.assertIsInstance(json_str, str)

# ==================== Main Entry Point ====================

system_instance = None

def get_script_info():
    """Returns information about the script, its status, and exposed parameters."""
    if system_instance:
        return system_instance.get_script_info()
    return {"name": "Integrated Geometry Detection System", "status": "not_initialized", "parameters": {}}

def set_script_parameter(key, value):
    """Sets a specific parameter for the script and updates shared_config."""
    if system_instance:
        return system_instance.set_script_parameter(key, value)
    return False

def main():
    """Main entry point"""
    global system_instance

    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Geometry Detection System')
    parser.add_argument('-s', '--source', default=None,
                       help='Camera source (index or path), can be overridden by shared_config')
    parser.add_argument('-b', '--backend', default=None,
                       choices=['opencv', 'pylon'],
                       help='Camera backend, can be overridden by shared_config')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration, can be overridden by shared_config')
    parser.add_argument('--no-tube', action='store_true',
                       help='Disable tube angle detection, can be overridden by shared_config')
    parser.add_argument('--no-benchmark', action='store_true',
                       help='Disable performance benchmarking, can be overridden by shared_config')
    parser.add_argument('--test', action='store_true',
                       help='Run unit tests')
    parser.add_argument('--log-level', default=None,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level, can be overridden by shared_config')
    
    args = parser.parse_args()
    
    # Set logging level based on args or shared_config
    log_level_str = shared_config.CONFIG.get("log_level", args.log_level if args.log_level is not None else "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level_str.upper()))
    
    # Run tests if requested
    if args.test:
        unittest.main(argv=[''], exit=False)
        return
    
    # Convert source to int if numeric
    source = args.source
    if source is not None and str(source).isdigit():
        source = int(source)
    
    print("="*60)
    print("Integrated Geometry Detection System".center(60))
    print("="*60)
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"CUDA Available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    print(f"Pylon Available: {PYLON_AVAILABLE}")
    print("="*60)
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  's' - Save screenshot")
    print("  'r' - Start/Stop recording")
    print("  'b' - Save benchmark results")
    print("  'g' - Toggle GPU")
    print("  '+/-' - Adjust sensitivity")
    print("\n")
    
    # Create and run system
    try:
        system_instance = GeometryDetectionSystem(
            camera_backend=args.backend,
            camera_source=source,
            use_gpu=not args.no_gpu if args.no_gpu else None, # Pass None if not explicitly set by arg
            enable_tube_detection=not args.no_tube if args.no_tube else None,
            enable_benchmarking=not args.no_benchmark if args.no_benchmark else None
        )
        
        system_instance.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

