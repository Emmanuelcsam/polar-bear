"""
BMP Video Emulator Module.
Emulates real-time video feed by looping a BMP image and integrating with pylon_grabber.
"""

import cv2
import numpy as np
import time
import threading
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import unittest
from unittest.mock import Mock, patch
import sys
import os
from PIL import Image, ImageTk
import tkinter as tk
import threading
import time
import logging
import cv2

# Import the pylon grabber module
from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE

# Import the hough circles detector
from hough_circles import HoughCirclesDetector, HoughCirclesProcessor


class VideoDisplay:
    """
    Video display widget that shows frames in real-time.
    """

    def __init__(self, parent, width=640, height=480):
        self.parent = parent
        self.width = width
        self.height = height
        self.current_frame = None
        self.is_displaying = False

        # Hough circles processor
        self.hough_processor = HoughCirclesProcessor()

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
            # Apply Hough circles detection if enabled
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
            circles_count = self.hough_processor.detector.circles_detected
            self.info_label.config(text=f"Frame: {width}x{height} | Display: {self.width}x{self.height} | Hough: {hough_status} | Circles: {circles_count}")

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


class BMPVideoEmulator:
    """
    Emulates real-time video by looping a BMP image at specified frame rate.
    Integrates with pylon_grabber for seamless camera simulation.
    """

    def __init__(self, image_path="good.bmp", frame_rate=30):
        """
        Initialize the BMP video emulator.

        Args:
            image_path (str): Path to the BMP image to loop
            frame_rate (int): Target frame rate for the emulated video
        """
        self.image_path = Path(image_path)
        self.frame_rate = frame_rate
        self.frame_interval = 1.0 / frame_rate

        # Threading components
        self.is_running = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_count = 0

        # Load and validate the image
        self._load_image()

        # Initialize frame timing
        self.last_frame_time = time.time()

    def _load_image(self):
        """Load and validate the BMP image."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.image_path}")

        # Load image using OpenCV
        self.original_frame = cv2.imread(str(self.image_path))
        if self.original_frame is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        logging.info(f"Loaded image: {self.image_path}")
        logging.info(f"Image dimensions: {self.original_frame.shape}")

    def start(self):
        """Start the video emulation thread."""
        if self.is_running.is_set():
            logging.warning("Emulator is already running")
            return

        self.is_running.set()
        self.emulation_thread = threading.Thread(target=self._emulation_loop, daemon=True)
        self.emulation_thread.start()
        logging.info("BMP video emulator started")

    def stop(self):
        """Stop the video emulation thread."""
        self.is_running.clear()
        if hasattr(self, 'emulation_thread'):
            self.emulation_thread.join(timeout=1.0)
        logging.info("BMP video emulator stopped")

    def _emulation_loop(self):
        """Main emulation loop that continuously provides frames."""
        while self.is_running.is_set():
            current_time = time.time()

            # Maintain frame rate timing
            if current_time - self.last_frame_time >= self.frame_interval:
                with self.lock:
                    self.latest_frame = self.original_frame.copy()
                    self.frame_count += 1

                self.last_frame_time = current_time

            # Small sleep to prevent CPU overload
            time.sleep(0.001)

    def read(self):
        """
        Returns the most recent frame, similar to pylon_grabber interface.

        Returns:
            numpy.ndarray or None: The current frame or None if not available
        """
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_frame_count(self):
        """Get the total number of frames processed."""
        with self.lock:
            return self.frame_count


class EmulatedPylonGrabber(PylonFrameGrabber):
    """
    Extended PylonFrameGrabber that uses BMP emulation when real camera is unavailable.
    """

    def __init__(self, use_emulation=True, image_path="good.bmp", frame_rate=30):
        """
        Initialize the emulated pylon grabber.

        Args:
            use_emulation (bool): Whether to use emulation instead of real camera
            image_path (str): Path to BMP image for emulation
            frame_rate (int): Frame rate for emulation
        """
        super().__init__()
        self.use_emulation = use_emulation
        self.emulator = None

        if use_emulation or not PYLON_AVAILABLE:
            self.emulator = BMPVideoEmulator(image_path, frame_rate)

    def run(self):
        """Override the run method to use emulation when needed."""
        if self.use_emulation or not PYLON_AVAILABLE:
            logging.info("Using BMP video emulation")
            self._emulation_run()
        else:
            logging.info("Using real Pylon camera")
            super().run()

    def _emulation_run(self):
        """Run the emulation instead of real camera."""
        logging.info("Emulated PylonFrameGrabber thread started.")

        try:
            self.emulator.start()
            self.is_running.set()

            while self.is_running.is_set():
                frame = self.emulator.read()
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame.copy()
                time.sleep(0.001)  # Small delay to prevent CPU overload

        except Exception as e:
            logging.critical(f"Error in emulated grabber: {e}", exc_info=True)
        finally:
            if self.emulator:
                self.emulator.stop()
            self.is_running.clear()
            logging.info("Emulated PylonFrameGrabber thread finished.")


class VideoEmulatorGUI:
    """
    GUI for controlling the BMP video emulator with live video display.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("BMP Video Emulator - Pylon Viewer Style")
        self.root.geometry("1000x700")

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

        # Hough Circles section
        hough_frame = ttk.LabelFrame(left_panel, text="Hough Circles Detection", padding="5")
        hough_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Enable Hough circles
        self.enable_hough_var = tk.BooleanVar(value=True)
        self.enable_hough_check = ttk.Checkbutton(hough_frame, text="Enable Circle Detection",
                                                 variable=self.enable_hough_var,
                                                 command=self._toggle_hough_detection)
        self.enable_hough_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        # Hough parameters frame
        hough_params_frame = ttk.Frame(hough_frame)
        hough_params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        # DP parameter (accumulator resolution)
        ttk.Label(hough_params_frame, text="DP (Resolution):").grid(row=0, column=0, sticky=tk.W)
        self.dp_var = tk.StringVar(value="1.0")
        self.dp_entry = ttk.Entry(hough_params_frame, textvariable=self.dp_var, width=10)
        self.dp_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 5))
        self.dp_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.dp_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(0.1-5.0)").grid(row=0, column=2, sticky=tk.W)

        # Min distance between circles (heavily increased range)
        ttk.Label(hough_params_frame, text="Min Distance:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.min_dist_var = tk.StringVar(value="50")
        self.min_dist_entry = ttk.Entry(hough_params_frame, textvariable=self.min_dist_var, width=10)
        self.min_dist_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.min_dist_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.min_dist_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(1-1000)").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))

        # Param1 (edge detection threshold) - heavily increased range
        ttk.Label(hough_params_frame, text="Edge Threshold:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.param1_var = tk.StringVar(value="100")
        self.param1_entry = ttk.Entry(hough_params_frame, textvariable=self.param1_var, width=10)
        self.param1_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.param1_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.param1_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(1-500)").grid(row=2, column=2, sticky=tk.W, pady=(5, 0))

        # Param2 (center detection threshold) - heavily increased range
        ttk.Label(hough_params_frame, text="Center Threshold:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.param2_var = tk.StringVar(value="50")
        self.param2_entry = ttk.Entry(hough_params_frame, textvariable=self.param2_var, width=10)
        self.param2_entry.grid(row=3, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.param2_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.param2_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(1-300)").grid(row=3, column=2, sticky=tk.W, pady=(5, 0))

        # Min radius - heavily increased range
        ttk.Label(hough_params_frame, text="Min Radius:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.min_radius_var = tk.StringVar(value="5")
        self.min_radius_entry = ttk.Entry(hough_params_frame, textvariable=self.min_radius_var, width=10)
        self.min_radius_entry.grid(row=4, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.min_radius_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.min_radius_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(0-500)").grid(row=4, column=2, sticky=tk.W, pady=(5, 0))

        # Max radius - heavily increased range
        ttk.Label(hough_params_frame, text="Max Radius:").grid(row=5, column=0, sticky=tk.W, pady=(5, 0))
        self.max_radius_var = tk.StringVar(value="200")
        self.max_radius_entry = ttk.Entry(hough_params_frame, textvariable=self.max_radius_var, width=10)
        self.max_radius_entry.grid(row=5, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.max_radius_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.max_radius_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(1-2000)").grid(row=5, column=2, sticky=tk.W, pady=(5, 0))

        # Gaussian Blur Kernel Size
        ttk.Label(hough_params_frame, text="Blur Kernel Size:").grid(row=6, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_kernel_var = tk.StringVar(value="9")
        self.blur_kernel_entry = ttk.Entry(hough_params_frame, textvariable=self.blur_kernel_var, width=10)
        self.blur_kernel_entry.grid(row=6, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.blur_kernel_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.blur_kernel_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(1-51, odd)").grid(row=6, column=2, sticky=tk.W, pady=(5, 0))

        # Gaussian Blur Sigma
        ttk.Label(hough_params_frame, text="Blur Sigma:").grid(row=7, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_sigma_var = tk.StringVar(value="2.0")
        self.blur_sigma_entry = ttk.Entry(hough_params_frame, textvariable=self.blur_sigma_var, width=10)
        self.blur_sigma_entry.grid(row=7, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.blur_sigma_entry.bind('<Return>', self._update_hough_params_from_entry)
        self.blur_sigma_entry.bind('<FocusOut>', self._update_hough_params_from_entry)
        ttk.Label(hough_params_frame, text="(0.1-10.0)").grid(row=7, column=2, sticky=tk.W, pady=(5, 0))

        # Preset configurations
        preset_frame = ttk.Frame(hough_params_frame)
        preset_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(preset_frame, text="Presets:").grid(row=0, column=0, sticky=tk.W)

        preset_btn_frame = ttk.Frame(preset_frame)
        preset_btn_frame.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Button(preset_btn_frame, text="Sensitive", command=lambda: self._load_preset("sensitive"), width=10).grid(row=0, column=0, padx=(0, 2))
        ttk.Button(preset_btn_frame, text="Balanced", command=lambda: self._load_preset("balanced"), width=10).grid(row=0, column=1, padx=(2, 2))
        ttk.Button(preset_btn_frame, text="Conservative", command=lambda: self._load_preset("conservative"), width=10).grid(row=0, column=2, padx=(2, 0))

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
        self.log_text = tk.Text(log_frame, height=10, width=40)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar for log
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        # Right panel for video display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video display section
        video_frame = ttk.LabelFrame(right_panel, text="Video Display", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create video display
        self.video_display = VideoDisplay(video_frame, width=640, height=480)

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
        """Toggle Hough circles detection on/off."""
        if self.video_display:
            enabled = self.video_display.hough_processor.toggle_processing()
            self._log_message(f"Hough circles detection {'enabled' if enabled else 'disabled'}")

    def _load_preset(self, preset_name):
        """Load a preset configuration for Hough parameters."""
        presets = {
            "sensitive": {
                "dp": "0.5", "min_dist": "20", "param1": "50", "param2": "15",
                "min_radius": "5", "max_radius": "500", "blur_kernel": "5", "blur_sigma": "1.0"
            },
            "balanced": {
                "dp": "1.0", "min_dist": "50", "param1": "100", "param2": "50",
                "min_radius": "5", "max_radius": "200", "blur_kernel": "9", "blur_sigma": "2.0"
            },
            "conservative": {
                "dp": "1.5", "min_dist": "150", "param1": "200", "param2": "100",
                "min_radius": "20", "max_radius": "300", "blur_kernel": "15", "blur_sigma": "4.0"
            }
        }

        if preset_name in presets:
            preset = presets[preset_name]
            self.dp_var.set(preset["dp"])
            self.min_dist_var.set(preset["min_dist"])
            self.param1_var.set(preset["param1"])
            self.param2_var.set(preset["param2"])
            self.min_radius_var.set(preset["min_radius"])
            self.max_radius_var.set(preset["max_radius"])
            self.blur_kernel_var.set(preset["blur_kernel"])
            self.blur_sigma_var.set(preset["blur_sigma"])

            # Apply the preset
            self._update_hough_params_from_entry()
            self._log_message(f"Loaded {preset_name.title()} preset")

    def _update_hough_params_from_entry(self, event=None):
        """Update Hough circles parameters from text entry fields."""
        if self.video_display:
            try:
                # Get values from entry fields and validate them
                dp = float(self.dp_var.get())
                min_dist = int(self.min_dist_var.get())
                param1 = int(self.param1_var.get())
                param2 = int(self.param2_var.get())
                min_radius = int(self.min_radius_var.get())
                max_radius = int(self.max_radius_var.get())
                blur_kernel = int(self.blur_kernel_var.get())
                blur_sigma = float(self.blur_sigma_var.get())

                # Validate ranges and clamp if necessary
                dp = max(0.1, min(5.0, dp))
                min_dist = max(1, min(1000, min_dist))
                param1 = max(1, min(500, param1))
                param2 = max(1, min(300, param2))
                min_radius = max(0, min(500, min_radius))
                max_radius = max(1, min(2000, max_radius))
                blur_kernel = max(1, min(51, blur_kernel))
                # Ensure kernel size is odd
                if blur_kernel % 2 == 0:
                    blur_kernel += 1
                blur_sigma = max(0.1, min(10.0, blur_sigma))

                # Update the entry fields with validated values
                self.dp_var.set(f"{dp:.1f}")
                self.min_dist_var.set(str(min_dist))
                self.param1_var.set(str(param1))
                self.param2_var.set(str(param2))
                self.min_radius_var.set(str(min_radius))
                self.max_radius_var.set(str(max_radius))
                self.blur_kernel_var.set(str(blur_kernel))
                self.blur_sigma_var.set(f"{blur_sigma:.1f}")

                # Update detector parameters
                self.video_display.hough_processor.detector.update_parameters(
                    dp=dp,
                    min_dist=min_dist,
                    param1=param1,
                    param2=param2,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    blur_kernel_size=blur_kernel,
                    blur_sigma=blur_sigma
                )

                self._log_message(f"Updated Hough parameters: dp={dp:.1f}, min_dist={min_dist}, param1={param1}, param2={param2}, min_radius={min_radius}, max_radius={max_radius}, blur_kernel={blur_kernel}, blur_sigma={blur_sigma:.1f}")

            except ValueError as e:
                self._log_message(f"Invalid parameter value: {e}")
                # Reset to current detector values
                detector = self.video_display.hough_processor.detector
                self.dp_var.set(f"{detector.dp:.1f}")
                self.min_dist_var.set(str(detector.min_dist))
                self.param1_var.set(str(detector.param1))
                self.param2_var.set(str(detector.param2))
                self.min_radius_var.set(str(detector.min_radius))
                self.max_radius_var.set(str(detector.max_radius))
                self.blur_kernel_var.set(str(detector.blur_kernel_size))
                self.blur_sigma_var.set(f"{detector.blur_sigma:.1f}")

    def _update_hough_params(self, value=None):
        """Update Hough circles parameters from GUI controls (legacy method for compatibility)."""
        # This method is now called by _update_hough_params_from_entry
        self._update_hough_params_from_entry()

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
            self._log_message("Video emulation started successfully")

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
            self._log_message("Video emulation stopped")

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
    """Main function to run the GUI application."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run GUI
    root = tk.Tk()
    app = VideoEmulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
