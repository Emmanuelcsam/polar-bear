#!/usr/bin/env python3
"""
BMP Video Emulator with SSIM Detection.
Emulates real-time video feed by looping a BMP image and integrates with SSIM detection
for manual parameter adjustment and real-time difference detection.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import logging
from pathlib import Path
from PIL import Image, ImageTk

# Import the pylon grabber module
from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE

# Import the SSIM detector
from ssim_detector_module import SSIMDetector, SSIMDetectorProcessor

# Import the BMP video emulator components
from bmp_video_emulator import BMPVideoEmulator, EmulatedPylonGrabber


class VideoDisplaySSIM:
    """
    Video display widget that shows frames with SSIM detection in real-time.
    """

    def __init__(self, parent, width=640, height=480):
        self.parent = parent
        self.width = width
        self.height = height
        self.current_frame = None
        self.is_displaying = False

        # SSIM detector processor
        self.ssim_processor = SSIMDetectorProcessor()

        # Create canvas for video display
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='black')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add a label for video info
        self.info_label = ttk.Label(parent, text="No video feed", anchor=tk.CENTER)
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_frame(self, frame):
        """Update the display with a new frame."""
        if frame is None or not self.is_displaying:
            self.info_label.config(text="No frame available")
            return

        try:
            # Apply SSIM detection if enabled
            processed_frame = self.ssim_processor.process_frame(frame)

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
            ssim_status = "ON" if self.ssim_processor.is_processing_enabled() else "OFF"
            stats = self.ssim_processor.detector.get_statistics()
            ssim_score = stats.get('current_ssim_score', 0.0)
            defects_count = stats.get('defects_detected', 0)

            self.info_label.config(
                text=f"Frame: {width}x{height} | Display: {self.width}x{self.height} | "
                     f"SSIM: {ssim_status} | Score: {ssim_score:.3f} | Defects: {defects_count}"
            )

        except Exception as e:
            self.info_label.config(text=f"Error displaying frame: {e}")
            logging.error(f"Error in VideoDisplaySSIM.update_frame: {e}")

    def start_display(self):
        """Start the video display loop."""
        self.is_displaying = True

    def stop_display(self):
        """Stop the video display loop."""
        self.is_displaying = False
        self.canvas.delete("all")
        self.info_label.config(text="Display stopped")


class SSIMDetectionGUI:
    """
    GUI for controlling the BMP video emulator with live SSIM detection.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("BMP Video Emulator - SSIM Detection")
        self.root.geometry("1200x900")

        # Initialize components
        self.emulator = None
        self.grabber = None
        self.is_running = False
        self.video_display = None
        self.update_thread = None
        self.stop_update = threading.Event()

        self._create_widgets()
        self._setup_bindings()

    def _create_widgets(self):
        """Create and arrange GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Configuration section
        config_frame = ttk.LabelFrame(left_panel, text="Configuration", padding="5")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Image path
        ttk.Label(config_frame, text="Live Image Path:").grid(row=0, column=0, sticky=tk.W)
        self.image_path_var = tk.StringVar(value="good.bmp")
        self.image_path_entry = ttk.Entry(config_frame, textvariable=self.image_path_var, width=25)
        self.image_path_entry.grid(row=0, column=1, padx=(5, 5), sticky=(tk.W, tk.E))

        ttk.Button(config_frame, text="Browse", command=self._browse_image).grid(row=0, column=2, padx=(0, 0))

        # Reference image path
        ttk.Label(config_frame, text="Reference Image:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.ref_image_path_var = tk.StringVar(value="good.bmp")
        self.ref_image_path_entry = ttk.Entry(config_frame, textvariable=self.ref_image_path_var, width=25)
        self.ref_image_path_entry.grid(row=1, column=1, padx=(5, 5), pady=(5, 0), sticky=(tk.W, tk.E))

        ttk.Button(config_frame, text="Browse", command=self._browse_ref_image).grid(row=1, column=2, pady=(5, 0))

        # Frame rate
        ttk.Label(config_frame, text="Frame Rate:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.frame_rate_var = tk.IntVar(value=30)
        self.frame_rate_spinbox = ttk.Spinbox(config_frame, from_=1, to=120, textvariable=self.frame_rate_var, width=10)
        self.frame_rate_spinbox.grid(row=2, column=1, padx=(5, 0), pady=(5, 0), sticky=tk.W)

        # Use emulation checkbox
        self.use_emulation_var = tk.BooleanVar(value=True)
        self.use_emulation_check = ttk.Checkbutton(config_frame, text="Use Emulation", variable=self.use_emulation_var)
        self.use_emulation_check.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.start_button = ttk.Button(control_frame, text="Start Emulation", command=self._start_emulation)
        self.start_button.grid(row=0, column=0, padx=(0, 5))

        self.stop_button = ttk.Button(control_frame, text="Stop Emulation", command=self._stop_emulation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1)

        # Set reference button
        self.set_ref_button = ttk.Button(control_frame, text="Set Reference", command=self._set_reference_image)
        self.set_ref_button.grid(row=1, column=0, pady=(5, 0))

        # Create test image button
        self.create_test_button = ttk.Button(control_frame, text="Create Test Image", command=self._create_test_image)
        self.create_test_button.grid(row=1, column=1, pady=(5, 0))

        # SSIM Detection section
        ssim_frame = ttk.LabelFrame(left_panel, text="SSIM Detection Parameters", padding="5")
        ssim_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Enable SSIM detection
        self.enable_ssim_var = tk.BooleanVar(value=True)
        self.enable_ssim_check = ttk.Checkbutton(ssim_frame, text="Enable SSIM Detection",
                                                variable=self.enable_ssim_var,
                                                command=self._toggle_ssim_detection)
        self.enable_ssim_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        # Presets
        ttk.Label(ssim_frame, text="Presets:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        preset_frame = ttk.Frame(ssim_frame)
        preset_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Button(preset_frame, text="Sensitive", width=10,
                  command=lambda: self._load_preset("sensitive")).grid(row=0, column=0, padx=(0, 2))
        ttk.Button(preset_frame, text="Balanced", width=10,
                  command=lambda: self._load_preset("balanced")).grid(row=0, column=1, padx=(2, 2))
        ttk.Button(preset_frame, text="Robust", width=10,
                  command=lambda: self._load_preset("robust")).grid(row=0, column=2, padx=(2, 0))

        # SSIM threshold
        row = 2
        ttk.Label(ssim_frame, text="SSIM Threshold:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        self.ssim_threshold_var = tk.DoubleVar(value=0.95)
        self.ssim_threshold_scale = ttk.Scale(ssim_frame, from_=0.1, to=1.0, variable=self.ssim_threshold_var,
                                             orient=tk.HORIZONTAL, command=self._update_ssim_params)
        self.ssim_threshold_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        self.ssim_threshold_entry = ttk.Entry(ssim_frame, textvariable=self.ssim_threshold_var, width=8)
        self.ssim_threshold_entry.grid(row=row, column=2, padx=(5, 0), pady=(5, 0))

        # Min defect area
        row += 1
        ttk.Label(ssim_frame, text="Min Defect Area:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        self.min_defect_area_var = tk.IntVar(value=50)
        self.min_defect_area_scale = ttk.Scale(ssim_frame, from_=10, to=1000, variable=self.min_defect_area_var,
                                              orient=tk.HORIZONTAL, command=self._update_ssim_params)
        self.min_defect_area_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        self.min_defect_area_entry = ttk.Entry(ssim_frame, textvariable=self.min_defect_area_var, width=8)
        self.min_defect_area_entry.grid(row=row, column=2, padx=(5, 0), pady=(5, 0))

        # Max defect area
        row += 1
        ttk.Label(ssim_frame, text="Max Defect Area:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        self.max_defect_area_var = tk.IntVar(value=5000)
        self.max_defect_area_scale = ttk.Scale(ssim_frame, from_=100, to=10000, variable=self.max_defect_area_var,
                                              orient=tk.HORIZONTAL, command=self._update_ssim_params)
        self.max_defect_area_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        self.max_defect_area_entry = ttk.Entry(ssim_frame, textvariable=self.max_defect_area_var, width=8)
        self.max_defect_area_entry.grid(row=row, column=2, padx=(5, 0), pady=(5, 0))

        # Blur kernel size
        row += 1
        ttk.Label(ssim_frame, text="Blur Kernel:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_kernel_var = tk.IntVar(value=5)
        self.blur_kernel_scale = ttk.Scale(ssim_frame, from_=1, to=31, variable=self.blur_kernel_var,
                                          orient=tk.HORIZONTAL, command=self._update_ssim_params)
        self.blur_kernel_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        self.blur_kernel_entry = ttk.Entry(ssim_frame, textvariable=self.blur_kernel_var, width=8)
        self.blur_kernel_entry.grid(row=row, column=2, padx=(5, 0), pady=(5, 0))

        # Use manual SSIM
        self.use_manual_ssim_var = tk.BooleanVar(value=False)
        self.use_manual_ssim_check = ttk.Checkbutton(ssim_frame, text="Use Manual SSIM",
                                                    variable=self.use_manual_ssim_var,
                                                    command=self._update_ssim_params)
        self.use_manual_ssim_check.grid(row=row+1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # Statistics section
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="5")
        stats_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.stats_text = tk.Text(stats_frame, height=6, width=40)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Log section
        log_frame = ttk.LabelFrame(left_panel, text="Log", padding="5")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        left_panel.rowconfigure(4, weight=1)

        self.log_text = tk.Text(log_frame, height=8, width=40)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        # Right panel for video display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video display
        self.video_display = VideoDisplaySSIM(right_panel, width=640, height=480)

        # Frame count display
        self.frame_count_label = ttk.Label(right_panel, text="Frame: 0", anchor=tk.CENTER)
        self.frame_count_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

    def _setup_bindings(self):
        """Set up event bindings."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Bind entry field updates
        self.ssim_threshold_entry.bind('<Return>', self._update_ssim_params_from_entry)
        self.min_defect_area_entry.bind('<Return>', self._update_ssim_params_from_entry)
        self.max_defect_area_entry.bind('<Return>', self._update_ssim_params_from_entry)
        self.blur_kernel_entry.bind('<Return>', self._update_ssim_params_from_entry)

    def _browse_image(self):
        """Browse for image file."""
        filetypes = [
            ("Image files", "*.bmp *.jpg *.jpeg *.png *.tiff"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(title="Select Live Image", filetypes=filetypes)
        if filename:
            self.image_path_var.set(filename)

    def _browse_ref_image(self):
        """Browse for reference image file."""
        filetypes = [
            ("Image files", "*.bmp *.jpg *.jpeg *.png *.tiff"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(title="Select Reference Image", filetypes=filetypes)
        if filename:
            self.ref_image_path_var.set(filename)

    def _set_reference_image(self):
        """Set the reference image for SSIM detection."""
        ref_path = self.ref_image_path_var.get()
        if not Path(ref_path).exists():
            messagebox.showerror("Error", f"Reference image not found: {ref_path}")
            return

        try:
            ref_image = cv2.imread(ref_path)
            if ref_image is None:
                messagebox.showerror("Error", f"Failed to load reference image: {ref_path}")
                return

            self.video_display.ssim_processor.set_reference_image(ref_image)
            self._log_message(f"Reference image set: {ref_path}")
            messagebox.showinfo("Success", "Reference image set successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set reference image: {e}")

    def _create_test_image(self):
        """Create a test image with defects based on good.bmp."""
        try:
            # Load the base image
            base_path = "good.bmp"
            if not Path(base_path).exists():
                messagebox.showerror("Error", f"Base image not found: {base_path}")
                return

            base_image = cv2.imread(base_path)
            if base_image is None:
                messagebox.showerror("Error", f"Failed to load base image: {base_path}")
                return

            # Create test image with various defects
            test_image = base_image.copy()
            h, w = test_image.shape[:2]

            # Make defects proportional to image size for visibility
            defect_size = min(h, w) // 20  # About 5% of smallest dimension

            self._log_message(f"Creating test image {w}x{h} with defect size ~{defect_size}")

            # Add large black rectangle
            rect_w, rect_h = w//8, h//8
            cv2.rectangle(test_image, (w//4, h//4), (w//4 + rect_w, h//4 + rect_h), (0, 0, 0), -1)
            self._log_message(f"Added black rectangle: {rect_w}x{rect_h} pixels")

            # Add large white circle
            circle_radius = defect_size
            cv2.circle(test_image, (3*w//4, h//4), circle_radius, (255, 255, 255), -1)
            self._log_message(f"Added white circle: radius {circle_radius} pixels")

            # Add thick diagonal line
            line_thickness = max(5, defect_size // 4)
            cv2.line(test_image, (w//6, h//6), (w//3, h//3), (128, 128, 128), line_thickness)
            self._log_message(f"Added diagonal line: thickness {line_thickness} pixels")

            # Add multiple smaller defects
            small_radius = max(10, defect_size//3)
            for i in range(5):
                x = (i + 1) * w // 6
                y = 2 * h // 3
                cv2.circle(test_image, (x, y), small_radius, (0, 255, 0), -1)
            self._log_message(f"Added 5 green circles: radius {small_radius} pixels each")

            # Save test image
            test_path = "ssim_test_defects.bmp"
            cv2.imwrite(test_path, test_image)

            # Update the image path to use the test image
            self.image_path_var.set(test_path)

            # Also set more sensitive detection parameters automatically
            self.ssim_threshold_var.set(0.999)  # Very sensitive
            self.min_defect_area_var.set(50)    # Lower minimum
            self.max_defect_area_var.set(100000)  # Higher maximum
            self._update_ssim_params()

            self._log_message(f"Test image created: {test_path}")
            self._log_message("Updated SSIM parameters for better detection")
            messagebox.showinfo("Success", f"Test image created: {test_path}\nImage path updated to use test image.\nSSIM parameters set to sensitive values.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create test image: {e}")
            self._log_message(f"Error creating test image: {e}")

    def _toggle_ssim_detection(self):
        """Toggle SSIM detection on/off."""
        if self.video_display:
            enabled = self.video_display.ssim_processor.toggle_processing()
            self._log_message(f"SSIM detection {'enabled' if enabled else 'disabled'}")

    def _load_preset(self, preset_name):
        """Load preset SSIM parameters."""
        presets = {
            "sensitive": {
                "ssim_threshold": 0.90,
                "min_defect_area": 20,
                "max_defect_area": 10000,
                "blur_kernel": 3,
                "use_manual_ssim": False
            },
            "balanced": {
                "ssim_threshold": 0.95,
                "min_defect_area": 50,
                "max_defect_area": 5000,
                "blur_kernel": 5,
                "use_manual_ssim": False
            },
            "robust": {
                "ssim_threshold": 0.98,
                "min_defect_area": 100,
                "max_defect_area": 2000,
                "blur_kernel": 7,
                "use_manual_ssim": True
            }
        }

        if preset_name in presets:
            preset = presets[preset_name]
            self.ssim_threshold_var.set(preset["ssim_threshold"])
            self.min_defect_area_var.set(preset["min_defect_area"])
            self.max_defect_area_var.set(preset["max_defect_area"])
            self.blur_kernel_var.set(preset["blur_kernel"])
            self.use_manual_ssim_var.set(preset["use_manual_ssim"])

            self._update_ssim_params()
            self._log_message(f"Loaded {preset_name} preset")

    def _update_ssim_params_from_entry(self, event=None):
        """Update SSIM parameters from entry fields."""
        self._update_ssim_params()

    def _update_ssim_params(self, value=None):
        """Update SSIM detection parameters."""
        if self.video_display:
            self.video_display.ssim_processor.detector.update_parameters(
                ssim_threshold=self.ssim_threshold_var.get(),
                min_defect_area=self.min_defect_area_var.get(),
                max_defect_area=self.max_defect_area_var.get(),
                blur_kernel_size=self.blur_kernel_var.get(),
                use_manual_ssim=self.use_manual_ssim_var.get()
            )

    def _start_emulation(self):
        """Start the video emulation."""
        if self.is_running:
            return

        try:
            image_path = self.image_path_var.get()
            if not Path(image_path).exists():
                messagebox.showerror("Error", f"Image file not found: {image_path}")
                return

            frame_rate = self.frame_rate_var.get()

            if self.use_emulation_var.get():
                # Use BMP emulation
                self.emulator = BMPVideoEmulator(image_path, frame_rate)
                self.emulator.start()
                self.grabber = self.emulator
            else:
                # Use real camera (if available)
                if PYLON_AVAILABLE:
                    self.grabber = EmulatedPylonGrabber(use_emulation=False)
                    self.grabber.start()
                else:
                    messagebox.showerror("Error", "Pylon SDK not available. Please use emulation mode.")
                    return

            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            # Start video display
            self.video_display.start_display()

            # Start update thread
            self.stop_update.clear()
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

            self._log_message("Emulation started successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start emulation: {e}")
            self._log_message(f"Error starting emulation: {e}")

    def _stop_emulation(self):
        """Stop the video emulation."""
        if not self.is_running:
            return

        try:
            self.is_running = False
            self.stop_update.set()

            # Stop components
            if self.emulator:
                self.emulator.stop()
                self.emulator = None

            if self.grabber and hasattr(self.grabber, 'stop'):
                self.grabber.stop()
                self.grabber = None

            # Stop video display
            if self.video_display:
                self.video_display.stop_display()

            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            self._log_message("Emulation stopped")

        except Exception as e:
            self._log_message(f"Error stopping emulation: {e}")

    def _update_loop(self):
        """Main update loop running in a separate thread."""
        frame_count = 0
        last_stats_update = time.time()

        while not self.stop_update.is_set() and self.is_running:
            try:
                if self.grabber:
                    frame = self.grabber.read()
                    if frame is not None:
                        frame_count += 1

                        # Update display safely
                        self.root.after_idle(self._update_frame_safe, frame)

                        # Update frame count
                        self.root.after_idle(lambda: self.frame_count_label.config(text=f"Frame: {frame_count}"))

                        # Update statistics periodically
                        if time.time() - last_stats_update > 1.0:
                            self.root.after_idle(self._update_statistics)
                            last_stats_update = time.time()

                time.sleep(0.03)  # ~30 FPS

            except Exception as e:
                self.root.after_idle(lambda: self._log_message(f"Error in update loop: {e}"))
                break

    def _update_frame_safe(self, frame):
        """Thread-safe frame update."""
        try:
            if self.video_display and self.is_running:
                self.video_display.update_frame(frame)
        except Exception as e:
            self._log_message(f"Error updating frame: {e}")

    def _update_statistics(self):
        """Update statistics display."""
        try:
            if self.video_display:
                stats = self.video_display.ssim_processor.detector.get_statistics()

                stats_text = f"""SSIM Detection Statistics:
Frames Processed: {stats['frames_processed']}
Defects Detected: {stats['defects_detected']}
Detection Rate: {stats['detection_rate']:.3f}
Current SSIM Score: {stats['current_ssim_score']:.3f}

Parameters:
SSIM Threshold: {self.ssim_threshold_var.get():.3f}
Min Defect Area: {self.min_defect_area_var.get()}
Max Defect Area: {self.max_defect_area_var.get()}
Blur Kernel: {self.blur_kernel_var.get()}
Manual SSIM: {self.use_manual_ssim_var.get()}"""

                self.stats_text.delete('1.0', tk.END)
                self.stats_text.insert('1.0', stats_text)

        except Exception as e:
            self._log_message(f"Error updating statistics: {e}")

    def _log_message(self, message):
        """Add a message to the log display."""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"

            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)

            # Keep only last 100 lines
            lines = self.log_text.get('1.0', tk.END).split('\n')
            if len(lines) > 100:
                self.log_text.delete('1.0', f'{len(lines) - 100}.0')

        except Exception as e:
            print(f"Error logging message: {e}")

    def _on_closing(self):
        """Handle window closing."""
        if self.is_running:
            self._stop_emulation()

        self.root.quit()
        self.root.destroy()


def main():
    """Main function to run the SSIM detection GUI application."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run GUI
    root = tk.Tk()
    app = SSIMDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
