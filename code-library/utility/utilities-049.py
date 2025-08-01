#!/usr/bin/env python3
"""
Real-Time Calibration and Testing Tool
======================================

Interactive tool for calibrating and testing the geometry detection system
with live feedback and parameter adjustment.

Features:
- Real-time parameter tuning with visual feedback
- Camera calibration wizard
- Shape detection accuracy testing
- Interactive ROI selection
- Parameter optimization suggestions
- Configuration save/load

Usage: python realtime_calibration_tool.py
"""

import cv2
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Callable
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from queue import Queue
import os
import shared_config # Import the shared configuration module

# Import from integrated system
from src.core.integrated_geometry_system import (
    GeometryDetector,
    GeometryDetectionSystem,
    CameraBackend,
    OpenCVCamera,
    ShapeType,
    Config,
    setup_logging
)

@dataclass
class CalibrationConfig:
    """Calibration configuration parameters"""
    # Detection parameters
    min_shape_area: int = 100
    max_shape_area: int = 100000
    epsilon_factor: float = 0.02
    canny_low: int = 50
    canny_high: int = 150
    
    # Camera parameters
    brightness: float = 0.0
    contrast: float = 1.0
    exposure: float = -1.0
    
    # ROI (Region of Interest)
    roi_enabled: bool = False
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 0
    roi_height: int = 0
    
    # Filtering
    shape_filter: List[str] = None
    confidence_threshold: float = 0.7
    
    # Display options
    show_contours: bool = True
    show_centers: bool = True
    show_bounding_boxes: bool = True
    show_labels: bool = True
    show_fps: bool = True
    
    def __post_init__(self):
        if self.shape_filter is None:
            self.shape_filter = [s.value for s in ShapeType]

class InteractiveCalibrator:
    """Interactive calibration system with GUI"""
    
    def __init__(self):
        self.logger = setup_logging("InteractiveCalibrator")
        
        # Load initial configuration from shared_config
        self.current_shared_config = shared_config.get_config()
        
        # Configuration - initialize with shared_config values if available
        self.config = CalibrationConfig(
            min_shape_area=self.current_shared_config.get("min_shape_area", 100),
            max_shape_area=self.current_shared_config.get("max_shape_area", 100000),
            epsilon_factor=self.current_shared_config.get("epsilon_factor", 0.02),
            canny_low=self.current_shared_config.get("canny_low", 50),
            canny_high=self.current_shared_config.get("canny_high", 150),
            brightness=self.current_shared_config.get("brightness", 0.0),
            contrast=self.current_shared_config.get("contrast", 1.0),
            exposure=self.current_shared_config.get("exposure", -1.0),
            roi_enabled=self.current_shared_config.get("roi_enabled", False),
            roi_x=self.current_shared_config.get("roi_x", 0),
            roi_y=self.current_shared_config.get("roi_y", 0),
            roi_width=self.current_shared_config.get("roi_width", 0),
            roi_height=self.current_shared_config.get("roi_height", 0),
            shape_filter=self.current_shared_config.get("shape_filter", [s.value for s in ShapeType]),
            confidence_threshold=self.current_shared_config.get("confidence_threshold", 0.7),
            show_contours=self.current_shared_config.get("show_contours", True),
            show_centers=self.current_shared_config.get("show_centers", True),
            show_bounding_boxes=self.current_shared_config.get("show_bounding_boxes", True),
            show_labels=self.current_shared_config.get("show_labels", True),
            show_fps=self.current_shared_config.get("show_fps", True)
        )
        self.original_config = CalibrationConfig() # Keep original defaults
        
        # Detection system
        self.camera = None
        self.detector = None
        
        # State
        self.running = False
        self.paused = False
        self.roi_selection_mode = False
        self.roi_start_point = None
        self.current_frame = None
        self.detected_shapes = []
        
        # Performance tracking
        self.fps_history = []
        self.detection_history = []
        
        # GUI elements
        self.root = None
        self.sliders = {}
        self.checkboxes = {}
        self.labels = {}
        
        # Threading
        self.gui_queue = Queue()
        
        self.status = "initialized" # Add a status variable
        self.logger.info("Interactive Calibrator initialized")

    def get_script_info(self):
        """Returns information about the script, its status, and exposed parameters."""
        return {
            "name": "Real-time Calibration Tool",
            "status": self.status,
            "parameters": asdict(self.config), # Expose current config as parameters
            "performance": {
                "avg_fps": np.mean(self.fps_history) if self.fps_history else 0,
                "avg_shapes_per_frame": np.mean(self.detection_history) if self.detection_history else 0
            }
        }

    def set_script_parameter(self, key, value):
        """Sets a specific parameter for the script and updates shared_config."""
        if hasattr(self.config, key):
            # Special handling for shape_filter as it's a list
            if key == "shape_filter" and isinstance(value, list):
                setattr(self.config, key, value)
            else:
                setattr(self.config, key, value)

            shared_config.set_config_value(key, value) # Update shared config
            
            # Update GUI elements if they exist
            if key in self.sliders:
                self.sliders[key].set(value)
            if key in self.checkboxes:
                self.checkboxes[key].set(value)
            
            # Apply changes to detector/camera if running
            if self.running:
                if key in ["min_shape_area", "max_shape_area", "epsilon_factor", "canny_low", "canny_high", "confidence_threshold"]:
                    self.update_detection_params()
                elif key in ["brightness", "contrast", "exposure"]:
                    self.update_camera_params()
                elif key == "shape_filter":
                    self.update_shape_filter()
                elif key.startswith("show_"):
                    self.update_display_options()
            
            self.status = f"parameter '{key}' updated"
            return True
        return False
    
    def create_gui(self):
        """Create the calibration GUI"""
        self.root = tk.Tk()
        self.root.title("Geometry Detection Calibration Tool")
        self.root.geometry("400x800")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True)
        
        # Detection parameters tab
        detection_frame = ttk.Frame(notebook)
        notebook.add(detection_frame, text='Detection')
        self._create_detection_controls(detection_frame)
        
        # Camera settings tab
        camera_frame = ttk.Frame(notebook)
        notebook.add(camera_frame, text='Camera')
        self._create_camera_controls(camera_frame)
        
        # Filters tab
        filter_frame = ttk.Frame(notebook)
        notebook.add(filter_frame, text='Filters')
        self._create_filter_controls(filter_frame)
        
        # Display options tab
        display_frame = ttk.Frame(notebook)
        notebook.add(display_frame, text='Display')
        self._create_display_controls(display_frame)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(button_frame, text="Start/Stop", 
                  command=self.toggle_detection).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Pause/Resume", 
                  command=self.toggle_pause).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset", 
                  command=self.reset_parameters).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Config", 
                  command=self.save_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Load Config", 
                  command=self.load_config).pack(side='left', padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill='x', side='bottom')
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _create_detection_controls(self, parent):
        """Create detection parameter controls"""
        # Min area
        self._create_slider(parent, "Min Area", "min_shape_area", 
                          10, 5000, self.config.min_shape_area, 
                          self.update_detection_params)
        
        # Max area
        self._create_slider(parent, "Max Area", "max_shape_area", 
                          100, 200000, self.config.max_shape_area, 
                          self.update_detection_params)
        
        # Epsilon factor
        self._create_slider(parent, "Epsilon Factor", "epsilon_factor", 
                          0.001, 0.1, self.config.epsilon_factor, 
                          self.update_detection_params, resolution=0.001)
        
        # Canny thresholds
        self._create_slider(parent, "Canny Low", "canny_low", 
                          10, 200, self.config.canny_low, 
                          self.update_detection_params)
        
        self._create_slider(parent, "Canny High", "canny_high", 
                          50, 300, self.config.canny_high, 
                          self.update_detection_params)
        
        # Confidence threshold
        self._create_slider(parent, "Confidence Threshold", "confidence_threshold", 
                          0.0, 1.0, self.config.confidence_threshold, 
                          self.update_detection_params, resolution=0.01)
        
        # ROI button
        ttk.Button(parent, text="Select ROI", 
                  command=self.start_roi_selection).pack(pady=10)
        
        # ROI status
        self.roi_label = ttk.Label(parent, text="ROI: Disabled")
        self.roi_label.pack()
    
    def _create_camera_controls(self, parent):
        """Create camera controls"""
        # Brightness
        self._create_slider(parent, "Brightness", "brightness", 
                          -100, 100, self.config.brightness, 
                          self.update_camera_params)
        
        # Contrast
        self._create_slider(parent, "Contrast", "contrast", 
                          0.5, 3.0, self.config.contrast, 
                          self.update_camera_params, resolution=0.1)
        
        # Exposure
        self._create_slider(parent, "Exposure", "exposure", 
                          -10, 0, self.config.exposure, 
                          self.update_camera_params)
        
        # Camera info
        info_frame = ttk.LabelFrame(parent, text="Camera Info")
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.camera_info_label = ttk.Label(info_frame, text="No camera connected")
        self.camera_info_label.pack()
    
    def _create_filter_controls(self, parent):
        """Create filter controls"""
        # Shape type filters
        filter_frame = ttk.LabelFrame(parent, text="Shape Types")
        filter_frame.pack(fill='x', padx=5, pady=5)
        
        self.shape_vars = {}
        for shape_type in ShapeType:
            var = tk.BooleanVar(value=True)
            self.shape_vars[shape_type.value] = var
            ttk.Checkbutton(filter_frame, text=shape_type.value, 
                           variable=var,
                           command=self.update_shape_filter).pack(anchor='w')
    
    def _create_display_controls(self, parent):
        """Create display option controls"""
        options = [
            ("Show Contours", "show_contours"),
            ("Show Centers", "show_centers"),
            ("Show Bounding Boxes", "show_bounding_boxes"),
            ("Show Labels", "show_labels"),
            ("Show FPS", "show_fps")
        ]
        
        for text, attr in options:
            var = tk.BooleanVar(value=getattr(self.config, attr))
            self.checkboxes[attr] = var
            ttk.Checkbutton(parent, text=text, variable=var,
                           command=self.update_display_options).pack(anchor='w')
        
        # Statistics
        stats_frame = ttk.LabelFrame(parent, text="Statistics")
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        self.stats_label = ttk.Label(stats_frame, text="No data")
        self.stats_label.pack()
    
    def _create_slider(self, parent, label, attr, min_val, max_val, 
                      initial, callback, resolution=1):
        """Create a labeled slider"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=5, pady=2)
        
        # Label with value
        label_text = f"{label}: {initial}"
        label_var = tk.StringVar(value=label_text)
        self.labels[attr] = label_var
        ttk.Label(frame, textvariable=label_var).pack()
        
        # Slider
        var = tk.DoubleVar(value=initial)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, 
                          variable=var, orient='horizontal')
        slider.pack(fill='x')
        
        # Callback
        def on_change(value):
            val = float(value)
            if resolution < 1:
                label_var.set(f"{label}: {val:.3f}")
            else:
                label_var.set(f"{label}: {int(val)}")
            setattr(self.config, attr, val)
            callback()
        
        slider.config(command=on_change)
        self.sliders[attr] = slider
    
    def update_detection_params(self):
        """Update detection parameters"""
        if self.detector:
            # Update Config class parameters
            Config.MIN_SHAPE_AREA = self.config.min_shape_area
            Config.MAX_SHAPE_AREA = self.config.max_shape_area
            Config.EPSILON_FACTOR = self.config.epsilon_factor
            Config.CANNY_LOW = self.config.canny_low
            Config.CANNY_HIGH = self.config.canny_high
    
    def update_camera_params(self):
        """Update camera parameters"""
        if self.camera:
            self.camera.set_property('brightness', self.config.brightness)
            self.camera.set_property('contrast', self.config.contrast)
            self.camera.set_property('exposure', self.config.exposure)
    
    def update_shape_filter(self):
        """Update shape type filter"""
        self.config.shape_filter = [
            shape_type for shape_type, var in self.shape_vars.items()
            if var.get()
        ]
    
    def update_display_options(self):
        """Update display options"""
        for attr, var in self.checkboxes.items():
            setattr(self.config, attr, var.get())
    
    def toggle_detection(self):
        """Start/stop detection"""
        if not self.running:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        """Start detection system"""
        try:
            # Initialize camera
            self.camera = OpenCVCamera(0)
            if not self.camera.open():
                messagebox.showerror("Error", "Failed to open camera")
                return
            
            # Initialize detector
            self.detector = GeometryDetector(use_gpu=False)
            
            # Update parameters
            self.update_detection_params()
            self.update_camera_params()
            
            # Update camera info
            props = self.camera.get_properties()
            info_text = f"Resolution: {props['width']}x{props['height']}\n"
            info_text += f"FPS: {props['fps']}"
            self.camera_info_label.config(text=info_text)
            
            self.running = True
            self.status_var.set("Detection running")
            
            # Start detection thread
            threading.Thread(target=self.detection_loop, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Failed to start detection: {e}")
            messagebox.showerror("Error", f"Failed to start: {str(e)}")
    
    def stop_detection(self):
        """Stop detection system"""
        self.running = False
        
        if self.camera:
            self.camera.close()
            self.camera = None
        
        self.detector = None
        self.status_var.set("Detection stopped")
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        self.status_var.set("Paused" if self.paused else "Running")
    
    def detection_loop(self):
        """Main detection loop"""
        cv2.namedWindow('Calibration View', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Calibration View', self.mouse_callback)
        
        while self.running:
            if self.paused:
                cv2.waitKey(50)
                continue
            
            # Read frame
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            self.current_frame = frame.copy()
            
            # Apply ROI if enabled
            if self.config.roi_enabled:
                roi_frame = frame[self.config.roi_y:self.config.roi_y + self.config.roi_height,
                                 self.config.roi_x:self.config.roi_x + self.config.roi_width]
            else:
                roi_frame = frame
            
            # Detect shapes
            start_time = time.time()
            shapes = self.detector.detect_shapes(roi_frame)
            detection_time = time.time() - start_time
            
            # Filter shapes
            filtered_shapes = self.filter_shapes(shapes)
            self.detected_shapes = filtered_shapes
            
            # Update statistics
            fps = 1.0 / detection_time
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            self.detection_history.append(len(filtered_shapes))
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
            
            # Draw results
            output = self.draw_results(frame, filtered_shapes)
            
            # Draw ROI rectangle
            if self.config.roi_enabled:
                cv2.rectangle(output, 
                            (self.config.roi_x, self.config.roi_y),
                            (self.config.roi_x + self.config.roi_width, 
                             self.config.roi_y + self.config.roi_height),
                            (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Calibration View', output)
            
            # Update GUI stats
            self.update_statistics()
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_detection()
                break
            elif key == ord('s'):
                self.save_screenshot(output)
        
        cv2.destroyAllWindows()
    
    def filter_shapes(self, shapes):
        """Filter shapes based on configuration"""
        filtered = []
        
        for shape in shapes:
            # Type filter
            if shape.shape_type.value not in self.config.shape_filter:
                continue
            
            # Confidence filter
            if shape.confidence < self.config.confidence_threshold:
                continue
            
            # Area filter (already applied in detector, but double-check)
            if not (self.config.min_shape_area <= shape.area <= self.config.max_shape_area):
                continue
            
            filtered.append(shape)
        
        return filtered
    
    def draw_results(self, frame, shapes):
        """Draw detection results"""
        output = frame.copy()
        
        for shape in shapes:
            color = shape.color
            
            # Draw contour
            if self.config.show_contours:
                cv2.drawContours(output, [shape.contour], -1, color, 2)
            
            # Draw center
            if self.config.show_centers:
                cv2.circle(output, shape.center, 5, (255, 0, 0), -1)
            
            # Draw bounding box
            if self.config.show_bounding_boxes:
                x, y, w, h = shape.bounding_box
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 0), 1)
            
            # Draw label
            if self.config.show_labels:
                label = f"{shape.shape_type.value} ({shape.confidence:.2f})"
                cv2.putText(output, label, 
                           (shape.center[0] - 40, shape.center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw FPS
        if self.config.show_fps and self.fps_history:
            avg_fps = np.mean(self.fps_history)
            cv2.putText(output, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw detection count
        cv2.putText(output, f"Shapes: {len(shapes)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if self.roi_selection_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_start_point = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if self.roi_start_point:
                    # Calculate ROI
                    x1, y1 = self.roi_start_point
                    x2, y2 = x, y
                    
                    self.config.roi_x = min(x1, x2)
                    self.config.roi_y = min(y1, y2)
                    self.config.roi_width = abs(x2 - x1)
                    self.config.roi_height = abs(y2 - y1)
                    self.config.roi_enabled = True
                    
                    # Update GUI
                    roi_text = f"ROI: ({self.config.roi_x}, {self.config.roi_y}) "
                    roi_text += f"{self.config.roi_width}x{self.config.roi_height}"
                    self.roi_label.config(text=roi_text)
                    
                    self.roi_selection_mode = False
                    self.roi_start_point = None
                    self.status_var.set("ROI selected")
    
    def start_roi_selection(self):
        """Start ROI selection mode"""
        self.roi_selection_mode = True
        self.status_var.set("Click and drag to select ROI")
    
    def reset_parameters(self):
        """Reset all parameters to default"""
        self.config = CalibrationConfig()
        
        # Update sliders
        for attr, slider in self.sliders.items():
            if hasattr(self.config, attr):
                slider.set(getattr(self.config, attr))
        
        # Update checkboxes
        for attr, var in self.checkboxes.items():
            if hasattr(self.config, attr):
                var.set(getattr(self.config, attr))
        
        # Update shape filters
        for shape_type, var in self.shape_vars.items():
            var.set(True)
        
        # Update detection parameters
        self.update_detection_params()
        self.update_camera_params()
        
        self.status_var.set("Parameters reset")
    
    def save_config(self):
        """Save configuration to file and update shared_config"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        
        if filename:
            try:
                config_data = asdict(self.config)
                with open(filename, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                # Update shared_config as well
                shared_config.update_config(config_data)
                
                self.status_var.set(f"Configuration saved to {os.path.basename(filename)}")
                self.logger.info(f"Configuration saved to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def load_config(self):
        """Load configuration from file and update GUI/shared_config"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        
        loaded_data = None
        if filename:
            try:
                with open(filename, 'r') as f:
                    loaded_data = json.load(f)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load from file: {str(e)}")
                return
        
        # Prioritize loaded file, then shared_config, then defaults
        config_to_apply = self.current_shared_config.copy()
        if loaded_data:
            config_to_apply.update(loaded_data)

        # Update configuration object
        for key, value in config_to_apply.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Update GUI elements
        for attr, slider in self.sliders.items():
            if hasattr(self.config, attr):
                slider.set(getattr(self.config, attr))
        
        for attr, var in self.checkboxes.items():
            if hasattr(self.config, attr):
                var.set(getattr(self.config, attr))
        
        # Update shape filters
        for shape_type, var in self.shape_vars.items():
            var.set(shape_type in self.config.shape_filter)
        
        # Apply parameters
        self.update_detection_params()
        self.update_camera_params()
        
        # Update shared_config with the newly loaded configuration
        shared_config.update_config(asdict(self.config))

        self.status_var.set(f"Configuration loaded from {os.path.basename(filename) if filename else 'shared_config'}")
        self.logger.info(f"Configuration loaded from {filename if filename else 'shared_config'}")
    
    def save_screenshot(self, frame):
        """Save screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_{timestamp}.png"
        cv2.imwrite(filename, frame)
        self.status_var.set(f"Screenshot saved: {filename}")
    
    def update_statistics(self):
        """Update statistics display"""
        if not self.fps_history:
            return
        
        stats_text = f"Avg FPS: {np.mean(self.fps_history):.1f}\n"
        stats_text += f"Shapes detected: {len(self.detected_shapes)}\n"
        
        if self.detection_history:
            stats_text += f"Avg shapes/frame: {np.mean(self.detection_history):.1f}\n"
            stats_text += f"Max shapes: {max(self.detection_history)}"
        
        self.stats_label.config(text=stats_text)
    
    def on_closing(self):
        """Handle window close event"""
        if self.running:
            self.stop_detection()
        self.root.quit()
    
    def run(self):
        """Run the calibration tool"""
        self.create_gui()
        self.root.mainloop()

class BatchCalibrator:
    """Batch calibration for finding optimal parameters"""
    
    def __init__(self):
        self.logger = setup_logging("BatchCalibrator")
        self.test_results = []
    
    def find_optimal_parameters(self, test_images: List[str], 
                              ground_truth: Dict) -> CalibrationConfig:
        """Find optimal parameters using grid search"""
        self.logger.info("Starting batch calibration...")
        
        # Parameter ranges to test
        param_ranges = {
            'min_shape_area': [50, 100, 200, 500],
            'epsilon_factor': [0.01, 0.02, 0.03, 0.04],
            'canny_low': [30, 50, 70],
            'canny_high': [100, 150, 200],
            'confidence_threshold': [0.5, 0.7, 0.8, 0.9]
        }
        
        best_config = None
        best_score = 0
        
        # Grid search
        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)
        
        self.logger.info(f"Testing {total_combinations} parameter combinations...")
        
        # Test each combination
        # (Implementation would test each combination and find best)
        
        return best_config

def create_test_pattern():
    """Create a test pattern for calibration"""
    # Create test image with known shapes
    width, height = 1280, 720
    pattern = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Grid of shapes
    shapes_info = []
    
    # Circles
    for i in range(3):
        for j in range(3):
            center = (100 + i * 200, 100 + j * 200)
            radius = 30 + i * 10
            cv2.circle(pattern, center, radius, (0, 0, 255), -1)
            shapes_info.append(('circle', center, radius))
    
    # Rectangles
    for i in range(3):
        x = 700 + i * 150
        y = 100 + i * 150
        w, h = 80, 60
        cv2.rectangle(pattern, (x, y), (x + w, y + h), (0, 255, 0), -1)
        shapes_info.append(('rectangle', (x, y, w, h)))
    
    # Triangles
    for i in range(2):
        pts = np.array([
            [900 + i * 100, 500],
            [950 + i * 100, 600],
            [850 + i * 100, 600]
        ], np.int32)
        cv2.fillPoly(pattern, [pts], (255, 0, 0))
        shapes_info.append(('triangle', pts))
    
    # Add text
    cv2.putText(pattern, "Calibration Test Pattern", (width//2 - 200, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    return pattern, shapes_info

calibrator_instance = None

def get_script_info():
    """Returns information about the script, its status, and exposed parameters."""
    if calibrator_instance:
        return calibrator_instance.get_script_info()
    return {"name": "Real-time Calibration Tool", "status": "not_initialized", "parameters": {}}

def set_script_parameter(key, value):
    """Sets a specific parameter for the script and updates shared_config."""
    if calibrator_instance:
        return calibrator_instance.set_script_parameter(key, value)
    return False

def main():
    """Main entry point"""
    global calibrator_instance

    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Calibration Tool')
    parser.add_argument('--test-pattern', action='store_true',
                       help='Show test pattern')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch calibration')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GEOMETRY DETECTION CALIBRATION TOOL")
    print("="*60)
    
    if args.test_pattern:
        print("\nShowing test pattern...")
        pattern, info = create_test_pattern()
        
        cv2.imshow('Test Pattern', pattern)
        print(f"Test pattern contains {len(info)} shapes")
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif args.batch:
        print("\nBatch calibration not fully implemented in this demo")
        print("Would perform grid search over parameter space")
    
    else:
        print("\nStarting interactive calibration tool...")
        print("\nFeatures:")
        print("- Real-time parameter adjustment")
        print("- Visual feedback")
        print("- ROI selection")
        print("- Configuration save/load")
        print("\nLaunching GUI...\n")
        
        calibrator_instance = InteractiveCalibrator()
        calibrator_instance.run()

if __name__ == "__main__":
    main()

