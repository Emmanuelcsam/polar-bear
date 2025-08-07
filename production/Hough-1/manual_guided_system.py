#!/usr/bin/env python3
"""
Manual-Guided Feature Detection System

Complete system with textbox controls, real-time imaging, and overlays.
Combines line and circle detection with full manual parameter control.

Author: AI Assistant
Date: August 2025  
Version: 3.0.0 - Complete Textbox Implementation
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings("ignore")

# Import existing modules
try:
    from hough_lines import HoughLinesDetector
    from hough_circles import HoughCirclesDetector
    from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE
    from bmp_video_emulator import BMPVideoEmulator, EmulatedPylonGrabber
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}")
    PYLON_AVAILABLE = False
    
    # Fallback detector implementations
    class HoughLinesDetector:
        def __init__(self, **kwargs):
            self.rho = 1
            self.theta_degrees = 1.0
            self.threshold = 50
            self.min_line_length = 50
            self.max_line_gap = 10
            self.blur_kernel_size = 5
            self.blur_sigma = 1.0
            self.canny_low = 50
            self.canny_high = 150
            self.use_probabilistic = True
            
        def detect_lines(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Apply blur
            blur_size = self.blur_kernel_size if self.blur_kernel_size % 2 == 1 else self.blur_kernel_size + 1
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), self.blur_sigma)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            
            # Detect lines
            if self.use_probabilistic:
                lines = cv2.HoughLinesP(edges, self.rho, np.pi/180*self.theta_degrees, 
                                      self.threshold, minLineLength=self.min_line_length, 
                                      maxLineGap=self.max_line_gap)
            else:
                lines = cv2.HoughLines(edges, self.rho, np.pi/180*self.theta_degrees, self.threshold)
            
            # Draw lines on result image
            result_img = frame.copy()
            if lines is not None:
                if self.use_probabilistic:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    for line in lines:
                        rho, theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            return lines, result_img
            
        def update_parameters(self, **kwargs):
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    class HoughCirclesDetector:
        def __init__(self, **kwargs):
            self.dp = 1.0
            self.min_dist = 50
            self.param1 = 100
            self.param2 = 50
            self.min_radius = 5
            self.max_radius = 200
            self.blur_kernel_size = 5
            self.blur_sigma = 1.0
            
        def detect_circles(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Apply Gaussian blur
            blur_size = self.blur_kernel_size if self.blur_kernel_size % 2 == 1 else self.blur_kernel_size + 1
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), self.blur_sigma)
            
            # Detect circles
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.dp, self.min_dist,
                                     param1=self.param1, param2=self.param2, 
                                     minRadius=self.min_radius, maxRadius=self.max_radius)
            
            # Draw circles on result image
            result_img = frame.copy()
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw outer circle
                    cv2.circle(result_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw center
                    cv2.circle(result_img, (i[0], i[1]), 2, (0, 0, 255), 3)
            
            return circles, result_img
            
        def update_parameters(self, **kwargs):
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)


class ManualGuidedGUI:
    """Main GUI for manual-guided feature detection system with textbox controls."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Manual-Guided Feature Detection System v3.0")
        self.root.geometry("1600x1000")
        
        # Initialize detectors
        self.lines_detector = HoughLinesDetector()
        self.circles_detector = HoughCirclesDetector()
        
        # Current image and processing state
        self.current_image = None
        self.processed_image = None
        self.is_processing = False
        
        # Video source
        self.camera = None
        self.video_thread = None
        self.emulator = None
        self.grabber = None
        
        # Parameter tracking
        self.lines_vars = {}
        self.circles_vars = {}
        self.lines_entries = {}
        self.circles_entries = {}
        
        # Setup GUI
        self.setup_gui()
        
        # Load previous settings
        self.load_settings()
        
    def setup_gui(self):
        """Setup the main GUI components."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls (wider to accommodate textboxes)
        control_frame = ttk.Frame(main_frame, width=500)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_frame.pack_propagate(False)
        
        # Right panel for image display
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup control panels
        self.setup_file_controls(control_frame)
        self.setup_processing_controls(control_frame)
        self.setup_lines_controls(control_frame)
        self.setup_circles_controls(control_frame)
        
        # Setup image display
        self.setup_display(display_frame)
        
        # Setup status bar
        self.setup_status_bar()
        
    def setup_file_controls(self, parent):
        """Setup file loading and video source controls."""
        file_frame = ttk.LabelFrame(parent, text="Input Source", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        # File controls
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_buttons_frame, text="Load Image", 
                  command=self.load_image, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_buttons_frame, text="Save Result", 
                  command=self.save_result, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
        # Video controls
        video_buttons_frame = ttk.Frame(file_frame)
        video_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        if PYLON_AVAILABLE:
            ttk.Button(video_buttons_frame, text="Start Camera", 
                      command=self.start_camera, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(video_buttons_frame, text="Start Emulator", 
                  command=self.start_emulator, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(video_buttons_frame, text="Stop Video", 
                  command=self.stop_video, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
    def setup_processing_controls(self, parent):
        """Setup processing and display controls."""
        proc_frame = ttk.LabelFrame(parent, text="Processing Controls", padding=10)
        proc_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Enable/disable detection types
        detection_frame = ttk.Frame(proc_frame)
        detection_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.enable_lines_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(detection_frame, text="Enable Line Detection", 
                       variable=self.enable_lines_var,
                       command=self.update_processing).pack(side=tk.LEFT, padx=(0, 10))
        
        self.enable_circles_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(detection_frame, text="Enable Circle Detection", 
                       variable=self.enable_circles_var,
                       command=self.update_processing).pack(side=tk.LEFT)
        
        # Processing buttons
        button_frame = ttk.Frame(proc_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="Process Current Image", 
                  command=self.process_current_image, width=20).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Reset Parameters", 
                  command=self.reset_parameters, width=20).pack(side=tk.LEFT)
        
    def setup_lines_controls(self, parent):
        """Setup manual controls for line detection parameters using textboxes."""
        lines_frame = ttk.LabelFrame(parent, text="Lines Detection Parameters", padding=10)
        lines_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Create scrollable frame
        lines_canvas = tk.Canvas(lines_frame, height=300)
        lines_scrollbar = ttk.Scrollbar(lines_frame, orient="vertical", command=lines_canvas.yview)
        lines_scrollable_frame = ttk.Frame(lines_canvas)
        
        lines_scrollable_frame.bind(
            "<Configure>",
            lambda e: lines_canvas.configure(scrollregion=lines_canvas.bbox("all"))
        )
        
        lines_canvas.create_window((0, 0), window=lines_scrollable_frame, anchor="nw")
        lines_canvas.configure(yscrollcommand=lines_scrollbar.set)
        
        # Lines parameter controls with expanded ranges
        lines_params_config = [
            ('rho', 'Distance Resolution', 1, '(1-10)', '1'),
            ('theta_degrees', 'Angle Resolution (°)', 1.0, '(0.1-5.0)', '1.0'),
            ('threshold', 'Accumulator Threshold', 50, '(10-300)', '50'),
            ('min_line_length', 'Min Line Length', 50, '(5-500)', '50'),
            ('max_line_gap', 'Max Line Gap', 10, '(1-100)', '10'),
            ('blur_kernel_size', 'Blur Kernel Size', 5, '(1-31, odd)', '5'),
            ('blur_sigma', 'Blur Sigma', 1.0, '(0.1-10.0)', '1.0'),
            ('canny_low', 'Canny Low Threshold', 50, '(10-300)', '50'),
            ('canny_high', 'Canny High Threshold', 150, '(50-500)', '150'),
        ]
        
        for i, (param_name, display_name, default_val, range_hint, default_str) in enumerate(lines_params_config):
            # Parameter label
            ttk.Label(lines_scrollable_frame, text=f"{display_name}:").grid(
                row=i, column=0, sticky=tk.W, padx=(5, 5), pady=2)
            
            # Parameter entry
            var = tk.StringVar(value=default_str)
            entry = ttk.Entry(lines_scrollable_frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, padx=(5, 5), pady=2)
            entry.bind('<Return>', lambda e, p=param_name: self.update_lines_param_from_entry(p))
            entry.bind('<FocusOut>', lambda e, p=param_name: self.update_lines_param_from_entry(p))
            
            # Range hint
            ttk.Label(lines_scrollable_frame, text=range_hint).grid(
                row=i, column=2, sticky=tk.W, padx=(5, 5), pady=2)
            
            self.lines_vars[param_name] = var
            self.lines_entries[param_name] = entry
        
        # Probabilistic method checkbox
        self.use_probabilistic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lines_scrollable_frame, text="Use Probabilistic Method", 
                       variable=self.use_probabilistic_var,
                       command=self.update_probabilistic_method).grid(
            row=len(lines_params_config), column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Preset buttons
        preset_frame = ttk.Frame(lines_scrollable_frame)
        preset_frame.grid(row=len(lines_params_config)+1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="Fine Lines", 
                  command=lambda: self.load_lines_preset("fine"), width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Balanced", 
                  command=lambda: self.load_lines_preset("balanced"), width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Thick Lines", 
                  command=lambda: self.load_lines_preset("thick"), width=10).pack(side=tk.LEFT, padx=2)
        
        lines_canvas.pack(side="left", fill="both", expand=True)
        lines_scrollbar.pack(side="right", fill="y")
        
    def setup_circles_controls(self, parent):
        """Setup manual controls for circle detection parameters using textboxes."""
        circles_frame = ttk.LabelFrame(parent, text="Circles Detection Parameters", padding=10)
        circles_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Create scrollable frame
        circles_canvas = tk.Canvas(circles_frame, height=250)
        circles_scrollbar = ttk.Scrollbar(circles_frame, orient="vertical", command=circles_canvas.yview)
        circles_scrollable_frame = ttk.Frame(circles_canvas)
        
        circles_scrollable_frame.bind(
            "<Configure>",
            lambda e: circles_canvas.configure(scrollregion=circles_canvas.bbox("all"))
        )
        
        circles_canvas.create_window((0, 0), window=circles_scrollable_frame, anchor="nw")
        circles_canvas.configure(yscrollcommand=circles_scrollbar.set)
        
        # Circles parameter controls with expanded ranges
        circles_params_config = [
            ('dp', 'Accumulator Resolution', 1.0, '(0.5-3.0)', '1.0'),
            ('min_dist', 'Min Distance Between Circles', 50, '(10-1000)', '50'),
            ('param1', 'Edge Detection Threshold', 100, '(50-500)', '100'),
            ('param2', 'Center Detection Threshold', 50, '(10-300)', '50'),
            ('min_radius', 'Min Circle Radius', 5, '(1-500)', '5'),
            ('max_radius', 'Max Circle Radius', 200, '(10-2000)', '200'),
            ('blur_kernel_size', 'Blur Kernel Size', 5, '(1-31, odd)', '5'),
            ('blur_sigma', 'Blur Sigma', 1.0, '(0.1-10.0)', '1.0'),
        ]
        
        for i, (param_name, display_name, default_val, range_hint, default_str) in enumerate(circles_params_config):
            # Parameter label
            ttk.Label(circles_scrollable_frame, text=f"{display_name}:").grid(
                row=i, column=0, sticky=tk.W, padx=(5, 5), pady=2)
            
            # Parameter entry
            var = tk.StringVar(value=default_str)
            entry = ttk.Entry(circles_scrollable_frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, sticky=tk.W, padx=(5, 5), pady=2)
            entry.bind('<Return>', lambda e, p=param_name: self.update_circles_param_from_entry(p))
            entry.bind('<FocusOut>', lambda e, p=param_name: self.update_circles_param_from_entry(p))
            
            # Range hint
            ttk.Label(circles_scrollable_frame, text=range_hint).grid(
                row=i, column=2, sticky=tk.W, padx=(5, 5), pady=2)
            
            self.circles_vars[param_name] = var
            self.circles_entries[param_name] = entry
        
        # Preset buttons
        preset_frame = ttk.Frame(circles_scrollable_frame)
        preset_frame.grid(row=len(circles_params_config), column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="Sensitive", 
                  command=lambda: self.load_circles_preset("sensitive"), width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Balanced", 
                  command=lambda: self.load_circles_preset("balanced"), width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Conservative", 
                  command=lambda: self.load_circles_preset("conservative"), width=10).pack(side=tk.LEFT, padx=2)
        
        circles_canvas.pack(side="left", fill="both", expand=True)
        circles_scrollbar.pack(side="right", fill="y")
        
    def setup_display(self, parent):
        """Setup image display area."""
        display_notebook = ttk.Notebook(parent)
        display_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Original image tab
        original_frame = ttk.Frame(display_notebook)
        display_notebook.add(original_frame, text="Original Image")
        
        self.original_canvas = tk.Canvas(original_frame, bg='black')
        original_scrollbar_v = ttk.Scrollbar(original_frame, orient="vertical", command=self.original_canvas.yview)
        original_scrollbar_h = ttk.Scrollbar(original_frame, orient="horizontal", command=self.original_canvas.xview)
        self.original_canvas.configure(yscrollcommand=original_scrollbar_v.set, xscrollcommand=original_scrollbar_h.set)
        
        self.original_canvas.pack(side="left", fill="both", expand=True)
        original_scrollbar_v.pack(side="right", fill="y")
        original_scrollbar_h.pack(side="bottom", fill="x")
        
        # Lines result tab
        lines_frame = ttk.Frame(display_notebook)
        display_notebook.add(lines_frame, text="Lines Detection")
        
        self.lines_canvas = tk.Canvas(lines_frame, bg='black')
        lines_scrollbar_v = ttk.Scrollbar(lines_frame, orient="vertical", command=self.lines_canvas.yview)
        lines_scrollbar_h = ttk.Scrollbar(lines_frame, orient="horizontal", command=self.lines_canvas.xview)
        self.lines_canvas.configure(yscrollcommand=lines_scrollbar_v.set, xscrollcommand=lines_scrollbar_h.set)
        
        self.lines_canvas.pack(side="left", fill="both", expand=True)
        lines_scrollbar_v.pack(side="right", fill="y")
        lines_scrollbar_h.pack(side="bottom", fill="x")
        
        # Circles result tab
        circles_frame = ttk.Frame(display_notebook)
        display_notebook.add(circles_frame, text="Circles Detection")
        
        self.circles_canvas = tk.Canvas(circles_frame, bg='black')
        circles_scrollbar_v = ttk.Scrollbar(circles_frame, orient="vertical", command=self.circles_canvas.yview)
        circles_scrollbar_h = ttk.Scrollbar(circles_frame, orient="horizontal", command=self.circles_canvas.xview)
        self.circles_canvas.configure(yscrollcommand=circles_scrollbar_v.set, xscrollcommand=circles_scrollbar_h.set)
        
        self.circles_canvas.pack(side="left", fill="both", expand=True)
        circles_scrollbar_v.pack(side="right", fill="y")
        circles_scrollbar_h.pack(side="bottom", fill="x")
        
        # Combined result tab
        combined_frame = ttk.Frame(display_notebook)
        display_notebook.add(combined_frame, text="Combined Detection")
        
        self.combined_canvas = tk.Canvas(combined_frame, bg='black')
        combined_scrollbar_v = ttk.Scrollbar(combined_frame, orient="vertical", command=self.combined_canvas.yview)
        combined_scrollbar_h = ttk.Scrollbar(combined_frame, orient="horizontal", command=self.combined_canvas.xview)
        self.combined_canvas.configure(yscrollcommand=combined_scrollbar_v.set, xscrollcommand=combined_scrollbar_h.set)
        
        self.combined_canvas.pack(side="left", fill="both", expand=True)
        combined_scrollbar_v.pack(side="right", fill="y")
        combined_scrollbar_h.pack(side="bottom", fill="x")
        
    def setup_status_bar(self):
        """Setup status bar."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status labels
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.frame_count_var = tk.StringVar(value="Frames: 0")
        self.frame_count_label = ttk.Label(self.status_frame, textvariable=self.frame_count_var)
        self.frame_count_label.pack(side=tk.LEFT, padx=20)
        
        self.detection_stats_var = tk.StringVar(value="Lines: 0 | Circles: 0")
        self.detection_stats_label = ttk.Label(self.status_frame, textvariable=self.detection_stats_var)
        self.detection_stats_label.pack(side=tk.LEFT, padx=20)
        
        # Processing time
        self.processing_time_var = tk.StringVar(value="Processing: 0ms")
        self.processing_time_label = ttk.Label(self.status_frame, textvariable=self.processing_time_var)
        self.processing_time_label.pack(side=tk.RIGHT, padx=5)
        
    def update_lines_param_from_entry(self, param_name):
        """Update lines parameter from textbox entry."""
        try:
            value_str = self.lines_vars[param_name].get()
            
            if param_name in ['rho', 'theta_degrees', 'blur_sigma']:
                value = float(value_str)
            elif param_name == 'blur_kernel_size':
                value = int(value_str)
                # Ensure odd number for blur kernel
                if value % 2 == 0:
                    value += 1
                    self.lines_vars[param_name].set(str(value))
            else:
                value = int(value_str)
            
            # Update detector parameter
            setattr(self.lines_detector, param_name, value)
            
            # Process current image if available
            if self.current_image is not None:
                self.process_current_image()
                
        except ValueError as e:
            messagebox.showerror("Invalid Value", f"Invalid value for {param_name}: {value_str}")
            # Reset to current detector value
            current_value = getattr(self.lines_detector, param_name, 0)
            self.lines_vars[param_name].set(str(current_value))
    
    def update_circles_param_from_entry(self, param_name):
        """Update circles parameter from textbox entry."""
        try:
            value_str = self.circles_vars[param_name].get()
            
            if param_name in ['dp', 'blur_sigma']:
                value = float(value_str)
            elif param_name == 'blur_kernel_size':
                value = int(value_str)
                # Ensure odd number for blur kernel
                if value % 2 == 0:
                    value += 1
                    self.circles_vars[param_name].set(str(value))
            else:
                value = int(value_str)
            
            # Update detector parameter
            setattr(self.circles_detector, param_name, value)
            
            # Process current image if available
            if self.current_image is not None:
                self.process_current_image()
                
        except ValueError as e:
            messagebox.showerror("Invalid Value", f"Invalid value for {param_name}: {value_str}")
            # Reset to current detector value
            current_value = getattr(self.circles_detector, param_name, 0)
            self.circles_vars[param_name].set(str(current_value))
    
    def update_probabilistic_method(self):
        """Update probabilistic method setting."""
        self.lines_detector.use_probabilistic = self.use_probabilistic_var.get()
        if self.current_image is not None:
            self.process_current_image()
    
    def update_processing(self):
        """Update processing based on enable/disable checkboxes."""
        if self.current_image is not None:
            self.process_current_image()
    
    def load_lines_preset(self, preset_name):
        """Load predefined preset for line detection."""
        presets = {
            'fine': {
                'rho': 1, 'theta_degrees': 0.5, 'threshold': 30, 'min_line_length': 20,
                'max_line_gap': 5, 'blur_kernel_size': 3, 'blur_sigma': 0.5,
                'canny_low': 30, 'canny_high': 100
            },
            'balanced': {
                'rho': 1, 'theta_degrees': 1.0, 'threshold': 50, 'min_line_length': 50,
                'max_line_gap': 10, 'blur_kernel_size': 5, 'blur_sigma': 1.0,
                'canny_low': 50, 'canny_high': 150
            },
            'thick': {
                'rho': 2, 'theta_degrees': 2.0, 'threshold': 100, 'min_line_length': 100,
                'max_line_gap': 20, 'blur_kernel_size': 7, 'blur_sigma': 2.0,
                'canny_low': 100, 'canny_high': 200
            }
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            for param_name, value in preset.items():
                if param_name in self.lines_vars:
                    self.lines_vars[param_name].set(str(value))
                    self.update_lines_param_from_entry(param_name)
    
    def load_circles_preset(self, preset_name):
        """Load predefined preset for circle detection."""
        presets = {
            'sensitive': {
                'dp': 1.0, 'min_dist': 30, 'param1': 50, 'param2': 30,
                'min_radius': 1, 'max_radius': 100, 'blur_kernel_size': 3, 'blur_sigma': 0.5
            },
            'balanced': {
                'dp': 1.0, 'min_dist': 50, 'param1': 100, 'param2': 50,
                'min_radius': 5, 'max_radius': 200, 'blur_kernel_size': 5, 'blur_sigma': 1.0
            },
            'conservative': {
                'dp': 2.0, 'min_dist': 100, 'param1': 200, 'param2': 100,
                'min_radius': 10, 'max_radius': 500, 'blur_kernel_size': 9, 'blur_sigma': 2.0
            }
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            for param_name, value in preset.items():
                if param_name in self.circles_vars:
                    self.circles_vars[param_name].set(str(value))
                    self.update_circles_param_from_entry(param_name)
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.load_lines_preset('balanced')
        self.load_circles_preset('balanced')
        self.use_probabilistic_var.set(True)
        self.update_probabilistic_method()
    
    def load_image(self):
        """Load an image file."""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('BMP files', '*.bmp'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=file_types
        )
        
        if filename:
            try:
                self.current_image = cv2.imread(filename)
                if self.current_image is not None:
                    self.status_var.set(f"Loaded: {Path(filename).name}")
                    self.display_image(self.current_image, self.original_canvas)
                    self.process_current_image()
                else:
                    messagebox.showerror("Error", f"Failed to load image: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def save_result(self):
        """Save the current processed result."""
        if self.processed_image is not None:
            file_types = [
                ('PNG files', '*.png'),
                ('JPEG files', '*.jpg'),
                ('BMP files', '*.bmp'),
                ('All files', '*.*')
            ]
            
            filename = filedialog.asksaveasfilename(
                title="Save processed image",
                filetypes=file_types,
                defaultextension='.png'
            )
            
            if filename:
                try:
                    cv2.imwrite(filename, self.processed_image)
                    self.status_var.set(f"Saved: {Path(filename).name}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error saving image: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No processed image to save")
    
    def start_camera(self):
        """Start real camera feed."""
        if PYLON_AVAILABLE:
            try:
                self.grabber = PylonFrameGrabber()
                self.grabber.start()
                self.is_processing = True
                self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
                self.video_thread.start()
                self.status_var.set("Camera started")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Pylon camera support not available")
    
    def start_emulator(self):
        """Start BMP video emulator."""
        try:
            self.emulator = BMPVideoEmulator("good.bmp", frame_rate=30)
            self.grabber = EmulatedPylonGrabber(use_emulation=True, image_path="good.bmp")
            self.grabber.start()
            self.is_processing = True
            self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
            self.video_thread.start()
            self.status_var.set("Emulator started")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start emulator: {str(e)}")
    
    def stop_video(self):
        """Stop video processing."""
        self.is_processing = False
        if self.grabber:
            try:
                self.grabber.stop()
            except:
                pass
            self.grabber = None
        if self.emulator:
            try:
                self.emulator.stop()
            except:
                pass
            self.emulator = None
        self.status_var.set("Video stopped")
    
    def video_processing_loop(self):
        """Main video processing loop."""
        frame_count = 0
        
        while self.is_processing and self.grabber:
            try:
                frame = self.grabber.read()
                if frame is not None:
                    self.current_image = frame.copy()
                    frame_count += 1
                    
                    # Update displays in main thread
                    self.root.after(0, lambda: self.display_image(self.current_image, self.original_canvas))
                    self.root.after(0, self.process_current_image)
                    self.root.after(0, lambda: self.frame_count_var.set(f"Frames: {frame_count}"))
                    
                time.sleep(1/30)  # Limit to ~30 FPS
            except Exception as e:
                logging.error(f"Error in video processing: {e}")
                break
    
    def process_current_image(self):
        """Process current image with both line and circle detection."""
        if self.current_image is None:
            return
        
        start_time = time.time()
        
        try:
            combined_image = self.current_image.copy()
            lines_count = 0
            circles_count = 0
            
            # Process lines if enabled
            if self.enable_lines_var.get():
                lines, lines_image = self.lines_detector.detect_lines(self.current_image)
                self.display_image(lines_image, self.lines_canvas)
                
                if lines is not None:
                    lines_count = len(lines)
                    # Add lines to combined image
                    if self.lines_detector.use_probabilistic:
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        for line in lines:
                            rho, theta = line[0]
                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            x1 = int(x0 + 1000*(-b))
                            y1 = int(y0 + 1000*(a))
                            x2 = int(x0 - 1000*(-b))
                            y2 = int(y0 - 1000*(a))
                            cv2.line(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Process circles if enabled
            if self.enable_circles_var.get():
                circles, circles_image = self.circles_detector.detect_circles(self.current_image)
                self.display_image(circles_image, self.circles_canvas)
                
                if circles is not None:
                    circles_count = len(circles[0]) if len(circles.shape) == 3 else len(circles)
                    # Add circles to combined image
                    if len(circles.shape) == 3:
                        circles = np.uint16(np.around(circles))
                        for i in circles[0, :]:
                            cv2.circle(combined_image, (i[0], i[1]), i[2], (255, 0, 0), 2)
                            cv2.circle(combined_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            
            # Display combined result
            self.processed_image = combined_image
            self.display_image(combined_image, self.combined_canvas)
            
            # Update status
            processing_time = (time.time() - start_time) * 1000
            self.processing_time_var.set(f"Processing: {processing_time:.1f}ms")
            self.detection_stats_var.set(f"Lines: {lines_count} | Circles: {circles_count}")
            
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            self.status_var.set(f"Processing error: {str(e)}")
    
    def display_image(self, cv_image, canvas):
        """Display OpenCV image on tkinter canvas."""
        try:
            # Convert BGR to RGB
            if len(cv_image.shape) == 3:
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv_image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            canvas.delete("all")
            canvas.config(scrollregion=canvas.bbox("all"))
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo  # Keep a reference
            
            # Update scroll region
            canvas.config(scrollregion=canvas.bbox("all"))
            
        except Exception as e:
            logging.error(f"Error displaying image: {e}")
    
    def save_settings(self):
        """Save current settings to file."""
        try:
            settings = {
                'lines_params': {param: var.get() for param, var in self.lines_vars.items()},
                'circles_params': {param: var.get() for param, var in self.circles_vars.items()},
                'use_probabilistic': self.use_probabilistic_var.get(),
                'enable_lines': self.enable_lines_var.get(),
                'enable_circles': self.enable_circles_var.get()
            }
            
            with open('manual_guided_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load settings from file."""
        try:
            with open('manual_guided_settings.json', 'r') as f:
                settings = json.load(f)
            
            # Load lines parameters
            lines_params = settings.get('lines_params', {})
            for param, value in lines_params.items():
                if param in self.lines_vars:
                    self.lines_vars[param].set(value)
                    self.update_lines_param_from_entry(param)
            
            # Load circles parameters
            circles_params = settings.get('circles_params', {})
            for param, value in circles_params.items():
                if param in self.circles_vars:
                    self.circles_vars[param].set(value)
                    self.update_circles_param_from_entry(param)
            
            # Load other settings
            self.use_probabilistic_var.set(settings.get('use_probabilistic', True))
            self.enable_lines_var.set(settings.get('enable_lines', True))
            self.enable_circles_var.set(settings.get('enable_circles', True))
            
        except (FileNotFoundError, json.JSONDecodeError):
            # Use default values if file doesn't exist or is corrupted
            pass
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
    
    def on_closing(self):
        """Handle window closing."""
        self.stop_video()
        self.save_settings()
        self.root.destroy()
    
    def run(self):
        """Run the GUI application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main function to run the manual-guided detection system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Manual-Guided Feature Detection System v3.0...")
    print("=" * 60)
    print("Features:")
    print("• Complete textbox control for all parameters")
    print("• Real-time imaging with overlays")
    print("• Expanded parameter ranges")
    print("• Preset configurations")
    print("• Multi-tab result display")
    print("• Camera and emulator support")
    print("=" * 60)
    
    try:
        app = ManualGuidedGUI()
        app.run()
    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Error", f"Application error: {str(e)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
