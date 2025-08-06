#!/usr/bin/env python3
"""
Simplified Real-time Fiber Optic Analysis System
Focuses on core fiber analysis without heavy dependencies.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

class SimpleFiberAnalyzer:
    """
    Simplified real-time fiber optic analysis system.
    Focuses on fiber anomaly detection and basic analysis.
    """
    
    def __init__(self, config_path=None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.camera_grabber = None
        self.fiber_analyzer = None
        
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
                'frame_width': 1280,
                'frame_height': 720
            },
            'analysis': {
                'enable_anomaly_detection': True,
                'analysis_interval': 2.0,  # seconds
                'save_results': True,
                'output_directory': 'realtime_output'
            },
            'display': {
                'show_live_feed': True,
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
        
        # Initialize fiber analyzer
        self._initialize_fiber_analyzer()
        
        self.logger.info("All systems initialized successfully")
    
    def _initialize_camera(self):
        """Initialize camera system (Pylon or fallback)."""
        try:
            if self.config['camera']['use_pylon'] and PYLON_AVAILABLE:
                self.camera_grabber = PylonFrameGrabber()
                self.logger.info("Using Pylon camera system")
            else:
                # Fallback to OpenCV camera
                self.camera_grabber = SimpleCameraGrabber(
                    camera_index=self.config['camera']['fallback_camera_index'],
                    width=self.config['camera']['frame_width'],
                    height=self.config['camera']['frame_height']
                )
                self.logger.info("Using OpenCV camera system")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def _initialize_fiber_analyzer(self):
        """Initialize fiber optic anomaly analyzer."""
        try:
            if not self.config['analysis']['enable_anomaly_detection']:
                self.fiber_analyzer = None
                return
            
            # Create configuration for fiber analyzer
            fiber_config = OmniConfig(
                confidence_threshold=0.3,
                anomaly_threshold_multiplier=2.5,
                enable_visualization=False  # Disable for real-time
            )
            
            self.fiber_analyzer = OmniFiberAnalyzer(fiber_config)
            
            # Build a simple reference model from the test image if available
            if os.path.exists("good.bmp"):
                self.logger.info("Building reference model from good.bmp...")
                self.fiber_analyzer._build_minimal_reference("good.bmp")
                self.logger.info("Reference model built successfully")
            
            self.logger.info("Fiber analyzer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize fiber analyzer: {e}")
            self.fiber_analyzer = None
    
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
                'fiber_analysis': None,
                'basic_analysis': self._basic_image_analysis(frame)
            }
            
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
            
            # Save results if enabled
            if self.config['analysis']['save_results']:
                self._save_results(results)
            
            # Update latest results
            with self.frame_lock:
                self.latest_results = results
                
        except Exception as e:
            self.logger.error(f"Error in analysis: {e}")
    
    def _basic_image_analysis(self, frame):
        """Perform basic image analysis."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Basic statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            else:
                area = 0
                circularity = 0
            
            return {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'edge_density': float(edge_density),
                'largest_area': float(area),
                'circularity': float(circularity)
            }
        except Exception as e:
            self.logger.error(f"Basic analysis error: {e}")
            return {}
    
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
        cv2.namedWindow('Simple Fiber Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Simple Fiber Analysis', 
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
                cv2.imshow('Simple Fiber Analysis', display_frame)
                
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
        
        # Draw basic analysis results
        if self.config['display']['show_analysis'] and results.get('basic_analysis'):
            basic = results['basic_analysis']
            if basic:
                cv2.putText(display_frame, f"Mean Intensity: {basic.get('mean_intensity', 0):.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Edge Density: {basic.get('edge_density', 0):.3f}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Circularity: {basic.get('circularity', 0):.3f}", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
Simple Fiber Analysis System
Controls:
- Q: Quit
- S: Save current frame
- H: Show this help
        """
        print(help_text)


class SimpleCameraGrabber:
    """Simple OpenCV camera grabber."""
    
    def __init__(self, camera_index=0, width=1280, height=720):
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
    print("Simple Real-time Fiber Optic Analysis System")
    print("=" * 50)
    
    # Create and start analyzer
    analyzer = SimpleFiberAnalyzer()
    
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