#!/usr/bin/env python3
"""
Real-time Fiber Optic Analysis System
Integrates camera capture, YOLO detection, fiber anomaly detection, and segmentation analysis.
"""

import cv2
import numpy as np
import time
import threading
import os
import json
from pathlib import Path
import logging
from datetime import datetime

# Import our custom modules
from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE
from detection import OmniFiberAnalyzer, OmniConfig
from separation import UnifiedSegmentationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

class RealTimeFiberAnalyzer:
    """
    Comprehensive real-time fiber optic analysis system.
    Integrates camera capture, object detection, anomaly detection, and segmentation.
    """
    
    def __init__(self, config_path=None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.camera_grabber = None
        self.yolo_detector = None
        self.fiber_analyzer = None
        self.segmentation_system = None
        
        # Analysis state
        self.is_running = False
        self.analysis_thread = None
        self.latest_frame = None
        self.latest_results = None
        self.frame_lock = threading.Lock()
        
        # Initialize all systems
        self._initialize_systems()
        
    def _load_config(self, config_path):
        """Load configuration from file or use defaults."""
        default_config = {
            'camera': {
                'use_pylon': True,
                'fallback_camera_index': 0,
                'exposure_time': 5000,
                'frame_width': 1920,
                'frame_height': 1080
            },
            'detection': {
                'yolo_confidence': 0.5,
                'yolo_nms_threshold': 0.4,
                'fiber_confidence_threshold': 0.3,
                'anomaly_threshold_multiplier': 2.5
            },
            'analysis': {
                'enable_segmentation': True,
                'enable_anomaly_detection': True,
                'analysis_interval': 1.0,  # seconds
                'save_results': True,
                'output_directory': 'realtime_output'
            },
            'display': {
                'show_live_feed': True,
                'show_detections': True,
                'show_analysis': True,
                'window_width': 1280,
                'window_height': 720
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for section in user_config:
                        if section in default_config:
                            default_config[section].update(user_config[section])
                        else:
                            default_config[section] = user_config[section]
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_systems(self):
        """Initialize all analysis systems."""
        self.logger.info("Initializing analysis systems...")
        
        # Initialize camera system
        self._initialize_camera()
        
        # Initialize YOLO detector
        self._initialize_yolo_detector()
        
        # Initialize fiber analyzer
        self._initialize_fiber_analyzer()
        
        # Initialize segmentation system
        if self.config['analysis']['enable_segmentation']:
            self._initialize_segmentation_system()
        
        self.logger.info("All systems initialized successfully")
    
    def _initialize_camera(self):
        """Initialize camera system (Pylon or fallback)."""
        try:
            if self.config['camera']['use_pylon'] and PYLON_AVAILABLE:
                self.camera_grabber = PylonFrameGrabber()
                self.logger.info("Using Pylon camera system")
            else:
                # Fallback to OpenCV camera
                self.camera_grabber = OpenCVCameraGrabber(
                    camera_index=self.config['camera']['fallback_camera_index'],
                    width=self.config['camera']['frame_width'],
                    height=self.config['camera']['frame_height']
                )
                self.logger.info("Using OpenCV camera system")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def _initialize_yolo_detector(self):
        """Initialize YOLO object detector."""
        try:
            # Check for required YOLO files
            required_files = ['yolov3.weights', 'yolov3.cfg', 'coco.names']
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                self.logger.warning(f"Missing YOLO files: {missing_files}")
                self.yolo_detector = None
                return
            
            # Initialize YOLO detector
            self.yolo_detector = YOLODetector(
                weights_path='yolov3.weights',
                config_path='yolov3.cfg',
                classes_path='coco.names',
                confidence_threshold=self.config['detection']['yolo_confidence'],
                nms_threshold=self.config['detection']['yolo_nms_threshold']
            )
            self.logger.info("YOLO detector initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO detector: {e}")
            self.yolo_detector = None
    
    def _initialize_fiber_analyzer(self):
        """Initialize fiber optic anomaly analyzer."""
        try:
            if not self.config['analysis']['enable_anomaly_detection']:
                self.fiber_analyzer = None
                return
            
            # Create configuration for fiber analyzer
            fiber_config = OmniConfig(
                confidence_threshold=self.config['detection']['fiber_confidence_threshold'],
                anomaly_threshold_multiplier=self.config['detection']['anomaly_threshold_multiplier'],
                enable_visualization=False  # Disable for real-time
            )
            
            self.fiber_analyzer = OmniFiberAnalyzer(fiber_config)
            self.logger.info("Fiber analyzer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize fiber analyzer: {e}")
            self.fiber_analyzer = None
    
    def _initialize_segmentation_system(self):
        """Initialize fiber segmentation system."""
        try:
            # Check if methods directory exists
            methods_dir = "zones_methods"
            if not os.path.exists(methods_dir):
                self.logger.warning(f"Methods directory not found: {methods_dir}")
                self.segmentation_system = None
                return
            
            self.segmentation_system = UnifiedSegmentationSystem(methods_dir)
            self.logger.info("Segmentation system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize segmentation system: {e}")
            self.segmentation_system = None
    
    def start(self):
        """Start the real-time analysis system."""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("Starting real-time analysis system...")
        
        # Start camera
        if hasattr(self.camera_grabber, 'start'):
            self.camera_grabber.start()
        else:
            self.camera_grabber.start_grabbing()
        
        # Start analysis thread
        self.is_running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        # Start display loop
        self._display_loop()
    
    def stop(self):
        """Stop the real-time analysis system."""
        self.logger.info("Stopping real-time analysis system...")
        self.is_running = False
        
        # Stop camera
        if hasattr(self.camera_grabber, 'stop'):
            self.camera_grabber.stop()
        else:
            self.camera_grabber.stop_grabbing()
        
        # Wait for analysis thread
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
        
        cv2.destroyAllWindows()
        self.logger.info("System stopped")
    
    def _analysis_loop(self):
        """Main analysis loop running in separate thread."""
        last_analysis_time = 0
        analysis_interval = self.config['analysis']['analysis_interval']
        
        while self.is_running:
            try:
                # Get current frame
                frame = self.camera_grabber.read()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Update latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Perform analysis at specified interval
                current_time = time.time()
                if current_time - last_analysis_time >= analysis_interval:
                    self._perform_analysis(frame)
                    last_analysis_time = current_time
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(0.1)
    
    def _perform_analysis(self, frame):
        """Perform comprehensive analysis on the frame."""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'frame_shape': frame.shape,
                'yolo_detections': [],
                'fiber_analysis': None,
                'segmentation': None
            }
            
            # YOLO object detection
            if self.yolo_detector:
                yolo_results = self.yolo_detector.detect(frame)
                results['yolo_detections'] = yolo_results
            
            # Fiber anomaly detection
            if self.fiber_analyzer:
                # Save frame temporarily for analysis
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                try:
                    fiber_results = self.fiber_analyzer.detect_anomalies_comprehensive(temp_path)
                    results['fiber_analysis'] = fiber_results
                except Exception as e:
                    self.logger.warning(f"Fiber analysis failed: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Segmentation analysis
            if self.segmentation_system:
                # Save frame temporarily for segmentation
                temp_path = "temp_frame.png"
                cv2.imwrite(temp_path, frame)
                
                try:
                    seg_results = self.segmentation_system.process_image(
                        Path(temp_path), 
                        self.config['analysis']['output_directory']
                    )
                    results['segmentation'] = seg_results
                except Exception as e:
                    self.logger.warning(f"Segmentation analysis failed: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Save results if enabled
            if self.config['analysis']['save_results']:
                self._save_results(results)
            
            # Update latest results
            with self.frame_lock:
                self.latest_results = results
                
        except Exception as e:
            self.logger.error(f"Error in analysis: {e}")
    
    def _save_results(self, results):
        """Save analysis results to file."""
        try:
            output_dir = Path(self.config['analysis']['output_directory'])
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"analysis_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _display_loop(self):
        """Main display loop."""
        cv2.namedWindow('Real-time Fiber Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Fiber Analysis', 
                        self.config['display']['window_width'],
                        self.config['display']['window_height'])
        
        while self.is_running:
            try:
                # Get current frame and results
                with self.frame_lock:
                    frame = self.latest_frame.copy() if self.latest_frame is not None else None
                    results = self.latest_results
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Create display frame
                display_frame = self._create_display_frame(frame, results)
                
                # Show frame
                cv2.imshow('Real-time Fiber Analysis', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_current_frame(frame, results)
                elif key == ord('h'):
                    self._show_help()
                
            except Exception as e:
                self.logger.error(f"Error in display loop: {e}")
                time.sleep(0.1)
        
        self.stop()
    
    def _create_display_frame(self, frame, results):
        """Create the display frame with analysis overlays."""
        display_frame = frame.copy()
        
        if results is None:
            return display_frame
        
        # Draw YOLO detections
        if self.config['display']['show_detections'] and results.get('yolo_detections'):
            for detection in results['yolo_detections']:
                label, confidence, (x, y, w, h) = detection
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{label}: {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw fiber analysis results
        if self.config['display']['show_analysis'] and results.get('fiber_analysis'):
            fiber_results = results['fiber_analysis']
            if fiber_results and 'verdict' in fiber_results:
                verdict = fiber_results['verdict']
                status = "ANOMALY" if verdict['is_anomalous'] else "NORMAL"
                color = (0, 0, 255) if verdict['is_anomalous'] else (0, 255, 0)
                cv2.putText(display_frame, f"Fiber Status: {status}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display_frame, f"Confidence: {verdict['confidence']:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw segmentation results
        if self.config['display']['show_analysis'] and results.get('segmentation'):
            seg_results = results['segmentation']
            if seg_results and 'center' in seg_results:
                center = seg_results['center']
                core_radius = seg_results.get('core_radius', 0)
                cladding_radius = seg_results.get('cladding_radius', 0)
                
                # Draw circles for core and cladding
                if center and core_radius:
                    cv2.circle(display_frame, (int(center[0]), int(center[1])), 
                              int(core_radius), (255, 0, 0), 2)
                if center and cladding_radius:
                    cv2.circle(display_frame, (int(center[0]), int(center[1])), 
                              int(cladding_radius), (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def _save_current_frame(self, frame, results):
        """Save current frame with analysis results."""
        try:
            output_dir = Path(self.config['analysis']['output_directory'])
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = output_dir / f"frame_{timestamp}.jpg"
            
            cv2.imwrite(str(frame_path), frame)
            self.logger.info(f"Saved frame to {frame_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save frame: {e}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
Real-time Fiber Analysis System
Controls:
- Q: Quit
- S: Save current frame
- H: Show this help
        """
        print(help_text)


class YOLODetector:
    """YOLO object detector wrapper."""
    
    def __init__(self, weights_path, config_path, classes_path, 
                 confidence_threshold=0.5, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Load YOLO model
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Check for GPU
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    def detect(self, image):
        """Detect objects in image."""
        height, width, _ = image.shape
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get output layers
        layer_names = self.net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except TypeError:
            output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Forward pass
        outs = self.net.forward(output_layers)
        
        # Process detections
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                results.append((self.classes[class_ids[i]], confidences[i], (x, y, w, h)))
        
        return results


class OpenCVCameraGrabber:
    """OpenCV camera grabber as fallback for Pylon."""
    
    def __init__(self, camera_index=0, width=1920, height=1080):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.is_running = False
        self.thread = None
    
    def start_grabbing(self):
        """Start camera grabbing."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()
    
    def _grab_loop(self):
        """Camera grabbing loop."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                time.sleep(0.1)
    
    def read(self):
        """Get latest frame."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def stop_grabbing(self):
        """Stop camera grabbing."""
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()


def main():
    """Main entry point."""
    print("Real-time Fiber Optic Analysis System")
    print("=" * 50)
    
    # Create and start analyzer
    analyzer = RealTimeFiberAnalyzer()
    
    try:
        analyzer.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        analyzer.stop()


if __name__ == "__main__":
    main() 