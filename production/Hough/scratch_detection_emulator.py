#!/usr/bin/env python3
"""
BMP Video Emulator with Hough Lines Detection (Scratch Detection).
Emulates real-time video feed by looping a BMP image and integrates with Hough line detection
for manual parameter adjustment and real-time scratch detection.
"""

import cv2
import numpy as np
import time
import threading
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from PIL import Image, ImageTk

# Import the pylon grabber module
from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE

# Import the hough lines detector
from hough_lines import HoughLinesDetector, HoughLinesProcessor

# Import the BMP video emulator components
from bmp_video_emulator import BMPVideoEmulator, EmulatedPylonGrabber


class VideoDisplayLines:
    """
    Video display widget that shows frames with line detection in real-time.
    """

    def __init__(self, parent, width=640, height=480):
        self.parent = parent
        self.width = width
        self.height = height
        self.current_frame = None
        self.is_displaying = False

        # Hough lines processor
        self.hough_processor = HoughLinesProcessor()

        # Create canvas for video display
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='black')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add a label for video info
        self.info_label = ttk.Label(parent, text="No video feed", anchor=tk.CENTER)
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_frame(self, frame):
        """Update the display with a new frame."""
        if frame is None:
            self.info_label.config(text="No frame available")
            return

        try:
            # Apply Hough lines detection if enabled
            processed_frame = self.hough_processor.process_frame(frame)

            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Resize frame to fit display
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height))

            # Convert to PIL Image
            pil_image = Image.fromarray(frame_resized)
            self.photo = ImageTk.PhotoImage(pil_image)

            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(self.width//2, self.height//2, image=self.photo)

            # Update info
            height, width = processed_frame.shape[:2]
            hough_status = "ON" if self.hough_processor.is_processing_enabled() else "OFF"
            lines_count = self.hough_processor.detector.lines_detected
            method = "Prob" if self.hough_processor.detector.use_probabilistic else "Std"
            self.info_label.config(text=f"Frame: {width}x{height} | Display: {self.width}x{self.height} | Hough: {hough_status} | Lines: {lines_count} | Method: {method}")

        except Exception as e:
            self.info_label.config(text=f"Error displaying frame: {e}")

    def start_display(self):
        """Start the video display loop."""
        self.is_displaying = True

    def stop_display(self):
        """Stop the video display loop."""
        self.is_displaying = False
        self.canvas.delete("all")
        self.info_label.config(text="Display stopped")


class ScratchDetectionGUI:
    """
    GUI for controlling the BMP video emulator with live line detection for scratch detection.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("BMP Video Emulator - Scratch Detection (Hough Lines)")
        self.root.geometry("1200x800")

        # Initialize components
        self.emulator = None
        self.grabber = None
        self.is_running = False
        self.video_display = None

        self._create_widgets()
        self._setup_bindings()

    def _create_widgets(self):
        """Create and arrange GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Configuration section
        config_frame = ttk.LabelFrame(left_panel, text="Configuration", padding="5")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Image path
        ttk.Label(config_frame, text="Image Path:").grid(row=0, column=0, sticky=tk.W)
        self.image_path_var = tk.StringVar(value="good.bmp")
        self.image_path_entry = ttk.Entry(config_frame, textvariable=self.image_path_var, width=30)
        self.image_path_entry.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))

        # Frame rate
        ttk.Label(config_frame, text="Frame Rate:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.frame_rate_var = tk.IntVar(value=30)
        self.frame_rate_spinbox = ttk.Spinbox(config_frame, from_=1, to=120, textvariable=self.frame_rate_var, width=10)
        self.frame_rate_spinbox.grid(row=1, column=1, padx=(5, 0), pady=(5, 0), sticky=tk.W)

        # Use emulation checkbox
        self.use_emulation_var = tk.BooleanVar(value=True)
        self.use_emulation_check = ttk.Checkbutton(config_frame, text="Use Emulation", variable=self.use_emulation_var)
        self.use_emulation_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # Hough Lines section
        hough_frame = ttk.LabelFrame(left_panel, text="Hough Lines Detection (Scratch Detection)", padding="5")
        hough_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Enable Hough lines
        self.enable_hough_var = tk.BooleanVar(value=True)
        self.enable_hough_check = ttk.Checkbutton(hough_frame, text="Enable Line Detection",
                                                 variable=self.enable_hough_var,
                                                 command=self._toggle_hough_detection)
        self.enable_hough_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        # Use probabilistic method
        self.use_probabilistic_var = tk.BooleanVar(value=True)
        self.use_probabilistic_check = ttk.Checkbutton(hough_frame, text="Use Probabilistic Method",
                                                      variable=self.use_probabilistic_var,
                                                      command=self._update_hough_params_from_entry)
        self.use_probabilistic_check.grid(row=1, column=0, columnspan=2, sticky=tk.W)

        # Hough parameters frame
        hough_params_frame = ttk.Frame(hough_frame)
        hough_params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        # Rho parameter (distance resolution)
        ttk.Label(hough_params_frame, text="Rho (Distance):").grid(row=0, column=0, sticky=tk.W)
        self.rho_var = tk.StringVar(value="1")
        self.rho_entry = ttk.Entry(hough_params_frame, textvariable=self.rho_var, width=10)
        self.rho_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 5))
        self.rho_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.rho_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(1-10)").grid(row=0, column=2, sticky=tk.W)

        # Theta parameter (angle resolution in degrees)
        ttk.Label(hough_params_frame, text="Theta (Angle °):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.theta_var = tk.StringVar(value="1.0")
        self.theta_entry = ttk.Entry(hough_params_frame, textvariable=self.theta_var, width=10)
        self.theta_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.theta_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.theta_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(0.1-5.0)").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))

        # Threshold parameter
        ttk.Label(hough_params_frame, text="Threshold:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.threshold_var = tk.StringVar(value="50")
        self.threshold_entry = ttk.Entry(hough_params_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.threshold_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.threshold_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(10-300)").grid(row=2, column=2, sticky=tk.W, pady=(5, 0))

        # Min line length (for probabilistic)
        ttk.Label(hough_params_frame, text="Min Line Length:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.min_line_length_var = tk.StringVar(value="30")
        self.min_line_length_entry = ttk.Entry(hough_params_frame, textvariable=self.min_line_length_var, width=10)
        self.min_line_length_entry.grid(row=3, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.min_line_length_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.min_line_length_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(5-200)").grid(row=3, column=2, sticky=tk.W, pady=(5, 0))

        # Max line gap (for probabilistic)
        ttk.Label(hough_params_frame, text="Max Line Gap:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.max_line_gap_var = tk.StringVar(value="5")
        self.max_line_gap_entry = ttk.Entry(hough_params_frame, textvariable=self.max_line_gap_var, width=10)
        self.max_line_gap_entry.grid(row=4, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.max_line_gap_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.max_line_gap_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(1-50)").grid(row=4, column=2, sticky=tk.W, pady=(5, 0))

        # Gaussian Blur Kernel Size
        ttk.Label(hough_params_frame, text="Blur Kernel Size:").grid(row=5, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_kernel_var = tk.StringVar(value="5")
        self.blur_kernel_entry = ttk.Entry(hough_params_frame, textvariable=self.blur_kernel_var, width=10)
        self.blur_kernel_entry.grid(row=5, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.blur_kernel_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.blur_kernel_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(1-15, odd)").grid(row=5, column=2, sticky=tk.W, pady=(5, 0))

        # Gaussian Blur Sigma
        ttk.Label(hough_params_frame, text="Blur Sigma:").grid(row=6, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_sigma_var = tk.StringVar(value="1.0")
        self.blur_sigma_entry = ttk.Entry(hough_params_frame, textvariable=self.blur_sigma_var, width=10)
        self.blur_sigma_entry.grid(row=6, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.blur_sigma_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.blur_sigma_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(0.1-5.0)").grid(row=6, column=2, sticky=tk.W, pady=(5, 0))

        # Canny Low Threshold
        ttk.Label(hough_params_frame, text="Canny Low:").grid(row=7, column=0, sticky=tk.W, pady=(5, 0))
        self.canny_low_var = tk.StringVar(value="50")
        self.canny_low_entry = ttk.Entry(hough_params_frame, textvariable=self.canny_low_var, width=10)
        self.canny_low_entry.grid(row=7, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.canny_low_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.canny_low_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(10-200)").grid(row=7, column=2, sticky=tk.W, pady=(5, 0))

        # Canny High Threshold
        ttk.Label(hough_params_frame, text="Canny High:").grid(row=8, column=0, sticky=tk.W, pady=(5, 0))
        self.canny_high_var = tk.StringVar(value="150")
        self.canny_high_entry = ttk.Entry(hough_params_frame, textvariable=self.canny_high_var, width=10)
        self.canny_high_entry.grid(row=8, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.canny_high_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.canny_high_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(50-400)").grid(row=8, column=2, sticky=tk.W, pady=(5, 0))

        # Preset configurations
        preset_frame = ttk.Frame(hough_params_frame)
        preset_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(preset_frame, text="Presets:").grid(row=0, column=0, sticky=tk.W)

        preset_btn_frame = ttk.Frame(preset_frame)
        preset_btn_frame.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Button(preset_btn_frame, text="Fine Lines", command=lambda: self._load_preset("fine"), width=10).grid(row=0, column=0, padx=(0, 2))
        ttk.Button(preset_btn_frame, text="Balanced", command=lambda: self._load_preset("balanced"), width=10).grid(row=0, column=1, padx=(2, 2))
        ttk.Button(preset_btn_frame, text="Thick Lines", command=lambda: self._load_preset("thick"), width=10).grid(row=0, column=2, padx=(2, 0))

        # Control section
        control_frame = ttk.LabelFrame(left_panel, text="Controls", padding="5")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Start/Stop button
        self.start_stop_btn = ttk.Button(control_frame, text="Start Video", command=self._toggle_emulation)
        self.start_stop_btn.grid(row=0, column=0, padx=(0, 5))

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=1, padx=(5, 0))

        # Information section
        info_frame = ttk.LabelFrame(left_panel, text="Information", padding="5")
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Frame count
        ttk.Label(info_frame, text="Frame Count:").grid(row=0, column=0, sticky=tk.W)
        self.frame_count_var = tk.StringVar(value="0")
        self.frame_count_label = ttk.Label(info_frame, textvariable=self.frame_count_var)
        self.frame_count_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        # Pylon availability
        ttk.Label(info_frame, text="Pylon Available:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        pylon_status = "Yes" if PYLON_AVAILABLE else "No"
        self.pylon_status_var = tk.StringVar(value=pylon_status)
        self.pylon_status_label = ttk.Label(info_frame, textvariable=self.pylon_status_var)
        self.pylon_status_label.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))

        # Log section
        log_frame = ttk.LabelFrame(left_panel, text="Log", padding="5")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Log text area
        self.log_text = tk.Text(log_frame, height=8, width=40)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar for log
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        # Right panel for video display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video display section
        video_frame = ttk.LabelFrame(right_panel, text="Video Display - Scratch Detection", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create video display
        self.video_display = VideoDisplayLines(video_frame, width=640, height=480)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(4, weight=1)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def _setup_bindings(self):
        """Setup event bindings."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _toggle_hough_detection(self):
        """Toggle Hough lines detection on/off."""
        if self.video_display:
            enabled = self.video_display.hough_processor.toggle_processing()
            self._log_message(f"Hough lines detection {'enabled' if enabled else 'disabled'}")

    def _load_preset(self, preset_name):
        """Load a preset configuration for Hough parameters."""
        presets = {
            "fine": {
                "rho": "1", "theta": "0.5", "threshold": "30",
                "min_line_length": "20", "max_line_gap": "2",
                "blur_kernel": "3", "blur_sigma": "0.5",
                "canny_low": "30", "canny_high": "100", "probabilistic": True
            },
            "balanced": {
                "rho": "1", "theta": "1.0", "threshold": "50",
                "min_line_length": "30", "max_line_gap": "5",
                "blur_kernel": "5", "blur_sigma": "1.0",
                "canny_low": "50", "canny_high": "150", "probabilistic": True
            },
            "thick": {
                "rho": "2", "theta": "2.0", "threshold": "80",
                "min_line_length": "50", "max_line_gap": "10",
                "blur_kernel": "7", "blur_sigma": "2.0",
                "canny_low": "80", "canny_high": "200", "probabilistic": True
            }
        }

        if preset_name in presets:
            preset = presets[preset_name]
            self.rho_var.set(preset["rho"])
            self.theta_var.set(preset["theta"])
            self.threshold_var.set(preset["threshold"])
            self.min_line_length_var.set(preset["min_line_length"])
            self.max_line_gap_var.set(preset["max_line_gap"])
            self.blur_kernel_var.set(preset["blur_kernel"])
            self.blur_sigma_var.set(preset["blur_sigma"])
            self.canny_low_var.set(preset["canny_low"])
            self.canny_high_var.set(preset["canny_high"])
            self.use_probabilistic_var.set(preset["probabilistic"])

            # Apply the preset
            self._update_hough_params_from_entry()
            self._log_message(f"Loaded {preset_name.title()} preset")

    def _update_hough_params_from_entry(self, event=None):
        """Update Hough lines parameters from text entry fields."""
        if self.video_display:
            try:
                # Get values from entry fields and validate them
                rho = int(self.rho_var.get())
                theta_degrees = float(self.theta_var.get())
                threshold = int(self.threshold_var.get())
                min_line_length = int(self.min_line_length_var.get())
                max_line_gap = int(self.max_line_gap_var.get())
                blur_kernel = int(self.blur_kernel_var.get())
                blur_sigma = float(self.blur_sigma_var.get())
                canny_low = int(self.canny_low_var.get())
                canny_high = int(self.canny_high_var.get())
                use_probabilistic = self.use_probabilistic_var.get()

                # Validate ranges and clamp if necessary
                rho = max(1, min(10, rho))
                theta_degrees = max(0.1, min(5.0, theta_degrees))
                threshold = max(10, min(300, threshold))
                min_line_length = max(5, min(200, min_line_length))
                max_line_gap = max(1, min(50, max_line_gap))
                blur_kernel = max(1, min(15, blur_kernel))
                # Ensure kernel size is odd
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                blur_sigma = max(0.1, min(5.0, blur_sigma))
                canny_low = max(10, min(200, canny_low))
                canny_high = max(50, min(400, canny_high))

                # Ensure canny_high > canny_low
                if canny_high <= canny_low:
                    canny_high = canny_low + 50

                # Update the entry fields with validated values
                self.rho_var.set(str(rho))
                self.theta_var.set(f"{theta_degrees:.1f}")
                self.threshold_var.set(str(threshold))
                self.min_line_length_var.set(str(min_line_length))
                self.max_line_gap_var.set(str(max_line_gap))
                self.blur_kernel_var.set(str(blur_kernel))
                self.blur_sigma_var.set(f"{blur_sigma:.1f}")
                self.canny_low_var.set(str(canny_low))
                self.canny_high_var.set(str(canny_high))

                # Update detector parameters
                self.video_display.hough_processor.detector.update_parameters(
                    rho=rho,
                    theta_degrees=theta_degrees,
                    threshold=threshold,
                    min_line_length=min_line_length,
                    max_line_gap=max_line_gap,
                    blur_kernel_size=blur_kernel,
                    blur_sigma=blur_sigma,
                    canny_low=canny_low,
                    canny_high=canny_high,
                    use_probabilistic=use_probabilistic
                )

                method = "Probabilistic" if use_probabilistic else "Standard"
                self._log_message(f"Updated Hough parameters ({method}): rho={rho}, theta={theta_degrees:.1f}°, threshold={threshold}, min_length={min_line_length}, max_gap={max_line_gap}")

            except ValueError as e:
                self._log_message(f"Invalid parameter value: {e}")
                # Reset to current detector values
                detector = self.video_display.hough_processor.detector
                self.rho_var.set(str(detector.rho))
                self.theta_var.set(f"{detector.theta_degrees:.1f}")
                self.threshold_var.set(str(detector.threshold))
                self.min_line_length_var.set(str(detector.min_line_length))
                self.max_line_gap_var.set(str(detector.max_line_gap))
                self.blur_kernel_var.set(str(detector.blur_kernel_size))
                self.blur_sigma_var.set(f"{detector.blur_sigma:.1f}")
                self.canny_low_var.set(str(detector.canny_low))
                self.canny_high_var.set(str(detector.canny_high))
                self.use_probabilistic_var.set(detector.use_probabilistic)

    def _toggle_emulation(self):
        """Toggle emulation start/stop."""
        if not self.is_running:
            self._start_emulation()
        else:
            self._stop_emulation()

    def _start_emulation(self):
        """Start the video emulation."""
        try:
            image_path = self.image_path_var.get()
            frame_rate = self.frame_rate_var.get()
            use_emulation = self.use_emulation_var.get()

            # Create and start the grabber
            self.grabber = EmulatedPylonGrabber(
                use_emulation=use_emulation,
                image_path=image_path,
                frame_rate=frame_rate
            )
            self.grabber.start()

            # Set initial Hough detection state
            if self.video_display:
                enable_hough = self.enable_hough_var.get()
                if not enable_hough:
                    self.video_display.hough_processor.toggle_processing()

                # Update Hough parameters from entry fields
                self._update_hough_params_from_entry()

            self.is_running = True
            self.start_stop_btn.config(text="Stop Video")
            self.status_var.set("Running")
            self._log_message("Scratch detection emulation started successfully")

            # Start video display and frame count update
            self.video_display.start_display()
            self._update_display()
            self._update_frame_count()

        except Exception as e:
            self._log_message(f"Error starting emulation: {e}")
            messagebox.showerror("Error", f"Failed to start emulation: {e}")

    def _stop_emulation(self):
        """Stop the video emulation."""
        try:
            if self.grabber:
                self.grabber.stop()
                self.grabber.join(timeout=2.0)

            self.is_running = False
            self.start_stop_btn.config(text="Start Video")
            self.status_var.set("Stopped")
            self._log_message("Scratch detection emulation stopped")

            # Stop video display
            if self.video_display:
                self.video_display.stop_display()

        except Exception as e:
            self._log_message(f"Error stopping emulation: {e}")

    def _update_display(self):
        """Update the video display with current frame."""
        if self.is_running and self.grabber:
            frame = self.grabber.read()
            if self.video_display:
                self.video_display.update_frame(frame)

        if self.is_running:
            # Update display at 30 FPS for smooth video
            self.root.after(33, self._update_display)  # ~30 FPS

    def _update_frame_count(self):
        """Update the frame count display."""
        if self.is_running and self.grabber and hasattr(self.grabber, 'emulator'):
            frame_count = self.grabber.emulator.get_frame_count()
            self.frame_count_var.set(str(frame_count))

        if self.is_running:
            self.root.after(100, self._update_frame_count)

    def _log_message(self, message):
        """Add a message to the log display."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)

    def _on_closing(self):
        """Handle window closing."""
        if self.is_running:
            self._stop_emulation()
        self.root.destroy()


def main():
    """Main function to run the scratch detection GUI application."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run GUI
    root = tk.Tk()
    app = ScratchDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
