#!/usr/bin/env python3
"""
BMP Video Emulator with Blob Detection.
Emulates real-time video feed by looping a BMP image and integrates with blob detection
for manual parameter adjustment and real-time blob detection.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk

# Import the pylon grabber module
from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE

# Import the blob detector
from blob_detector_module import BlobDetector, BlobDetectorProcessor

# Import the BMP video emulator components
from bmp_video_emulator import BMPVideoEmulator, EmulatedPylonGrabber


class VideoDisplayBlobs:
    """
    Video display widget that shows frames with blob detection in real-time.
    """

    def __init__(self, parent, width=640, height=480):
        self.parent = parent
        self.width = width
        self.height = height
        self.current_frame = None
        self.is_displaying = False

        # Blob detector processor
        self.blob_processor = BlobDetectorProcessor()

        # Create canvas for video display
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='black')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add a label for video info
        self.info_label = ttk.Label(parent, text="No video feed", anchor=tk.CENTER)
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

    def update_frame(self, frame):
        """Update the display with a new frame."""
        if frame is None or not self.is_displaying:
            return

        try:
            # Process frame with blob detection if enabled
            if self.blob_processor.is_processing_enabled():
                processed_frame = self.blob_processor.process_frame(frame)
            else:
                processed_frame = frame

            # Check if processed frame is valid
            if processed_frame is None:
                return

            # Resize frame to fit canvas
            h, w = processed_frame.shape[:2]
            if h == 0 or w == 0:
                return

            aspect_ratio = w / h

            if aspect_ratio > self.width / self.height:
                new_width = self.width
                new_height = max(1, int(self.width / aspect_ratio))
            else:
                new_height = self.height
                new_width = max(1, int(self.height * aspect_ratio))

            # Ensure dimensions are valid
            if new_width <= 0 or new_height <= 0:
                return

            resized_frame = cv2.resize(processed_frame, (new_width, new_height))

            # Convert to RGB for PIL
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            tk_image = ImageTk.PhotoImage(pil_image)

            # Update canvas
            self.canvas.delete("all")
            x_offset = (self.width - new_width) // 2
            y_offset = (self.height - new_height) // 2
            self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=tk_image)
            self.canvas.image = tk_image  # Keep a reference

            # Update info
            stats = self.blob_processor.detector.get_statistics()
            info_text = f"Frame: {new_width}x{new_height} | Blobs: {stats['blobs_detected']} | Processed: {stats['frames_processed']}"
            self.info_label.config(text=info_text)

        except Exception as e:
            self.info_label.config(text=f"Display error: {str(e)}")

    def start_display(self):
        """Start the video display loop."""
        self.is_displaying = True

    def stop_display(self):
        """Stop the video display loop."""
        self.is_displaying = False
        self.canvas.delete("all")
        self.info_label.config(text="Display stopped")


class BlobDetectionGUI:
    """
    GUI for controlling the BMP video emulator with live blob detection.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("BMP Video Emulator - Blob Detection")
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
        ttk.Label(config_frame, text="Image Path:").grid(row=0, column=0, sticky=tk.W)
        self.image_path_var = tk.StringVar(value="blob_test.bmp")
        self.image_path_entry = ttk.Entry(config_frame, textvariable=self.image_path_var, width=25)
        self.image_path_entry.grid(row=0, column=1, padx=(5, 5), sticky=(tk.W, tk.E))

        ttk.Button(config_frame, text="Browse", command=self._browse_image).grid(row=0, column=2, padx=(0, 0))

        # Frame rate
        ttk.Label(config_frame, text="Frame Rate:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.frame_rate_var = tk.IntVar(value=30)
        self.frame_rate_spinbox = ttk.Spinbox(config_frame, from_=1, to=120, textvariable=self.frame_rate_var, width=10)
        self.frame_rate_spinbox.grid(row=1, column=1, padx=(5, 0), pady=(5, 0), sticky=tk.W)

        # Use emulation checkbox
        self.use_emulation_var = tk.BooleanVar(value=True)
        self.use_emulation_check = ttk.Checkbutton(config_frame, text="Use Emulation", variable=self.use_emulation_var)
        self.use_emulation_check.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.start_button = ttk.Button(control_frame, text="Start Emulation", command=self._start_emulation)
        self.start_button.grid(row=0, column=0, padx=(0, 5))

        self.stop_button = ttk.Button(control_frame, text="Stop Emulation", command=self._stop_emulation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1)

        # Blob Detection section
        blob_frame = ttk.LabelFrame(left_panel, text="Blob Detection", padding="5")
        blob_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Enable blob detection
        self.enable_blob_var = tk.BooleanVar(value=True)
        self.enable_blob_check = ttk.Checkbutton(blob_frame, text="Enable Blob Detection",
                                                variable=self.enable_blob_var,
                                                command=self._toggle_blob_detection)
        self.enable_blob_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        # Blob parameters frame
        blob_params_frame = ttk.Frame(blob_frame)
        blob_params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        # Min blob area
        ttk.Label(blob_params_frame, text="Min Area:").grid(row=0, column=0, sticky=tk.W)
        self.min_area_var = tk.StringVar(value="50")
        self.min_area_entry = ttk.Entry(blob_params_frame, textvariable=self.min_area_var, width=10)
        self.min_area_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 5))
        self.min_area_entry.bind('<Return>', self._update_blob_params_from_entry)
        self.min_area_entry.bind('<FocusOut>', self._update_blob_params_from_entry)
        ttk.Label(blob_params_frame, text="(10-1000)").grid(row=0, column=2, sticky=tk.W)

        # Max blob area
        ttk.Label(blob_params_frame, text="Max Area:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.max_area_var = tk.StringVar(value="5000")
        self.max_area_entry = ttk.Entry(blob_params_frame, textvariable=self.max_area_var, width=10)
        self.max_area_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.max_area_entry.bind('<Return>', self._update_blob_params_from_entry)
        self.max_area_entry.bind('<FocusOut>', self._update_blob_params_from_entry)
        ttk.Label(blob_params_frame, text="(100-50000)").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))

        # Min circularity
        ttk.Label(blob_params_frame, text="Min Circularity:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.min_circularity_var = tk.StringVar(value="0.3")
        self.min_circularity_entry = ttk.Entry(blob_params_frame, textvariable=self.min_circularity_var, width=10)
        self.min_circularity_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.min_circularity_entry.bind('<Return>', self._update_blob_params_from_entry)
        self.min_circularity_entry.bind('<FocusOut>', self._update_blob_params_from_entry)
        ttk.Label(blob_params_frame, text="(0.1-1.0)").grid(row=2, column=2, sticky=tk.W, pady=(5, 0))

        # Blur kernel size
        ttk.Label(blob_params_frame, text="Blur Kernel:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_kernel_var = tk.StringVar(value="5")
        self.blur_kernel_entry = ttk.Entry(blob_params_frame, textvariable=self.blur_kernel_var, width=10)
        self.blur_kernel_entry.grid(row=3, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.blur_kernel_entry.bind('<Return>', self._update_blob_params_from_entry)
        self.blur_kernel_entry.bind('<FocusOut>', self._update_blob_params_from_entry)
        ttk.Label(blob_params_frame, text="(1-51, odd)").grid(row=3, column=2, sticky=tk.W, pady=(5, 0))

        # Blur sigma
        ttk.Label(blob_params_frame, text="Blur Sigma:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_sigma_var = tk.StringVar(value="1.0")
        self.blur_sigma_entry = ttk.Entry(blob_params_frame, textvariable=self.blur_sigma_var, width=10)
        self.blur_sigma_entry.grid(row=4, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.blur_sigma_entry.bind('<Return>', self._update_blob_params_from_entry)
        self.blur_sigma_entry.bind('<FocusOut>', self._update_blob_params_from_entry)
        ttk.Label(blob_params_frame, text="(0.1-10.0)").grid(row=4, column=2, sticky=tk.W, pady=(5, 0))

        # Threshold value
        ttk.Label(blob_params_frame, text="Threshold:").grid(row=5, column=0, sticky=tk.W, pady=(5, 0))
        self.threshold_var = tk.StringVar(value="127")
        self.threshold_entry = ttk.Entry(blob_params_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.grid(row=5, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.threshold_entry.bind('<Return>', self._update_blob_params_from_entry)
        self.threshold_entry.bind('<FocusOut>', self._update_blob_params_from_entry)
        ttk.Label(blob_params_frame, text="(1-255)").grid(row=5, column=2, sticky=tk.W, pady=(5, 0))

        # Presets section
        presets_frame = ttk.LabelFrame(left_panel, text="Detection Presets", padding="5")
        presets_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(presets_frame, text="Small Blobs", command=lambda: self._load_preset("small")).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(presets_frame, text="Medium Blobs", command=lambda: self._load_preset("medium")).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(presets_frame, text="Large Blobs", command=lambda: self._load_preset("large")).grid(row=0, column=2)

        # Statistics section
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="5")
        stats_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.stats_text = tk.Text(stats_frame, height=6, width=40)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Log section
        log_frame = ttk.LabelFrame(left_panel, text="Log", padding="5")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_panel.rowconfigure(5, weight=1)

        self.log_text = tk.Text(log_frame, height=8, width=40)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.rowconfigure(0, weight=1)

        # Right panel for video display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video display
        video_frame = ttk.LabelFrame(right_panel, text="Video Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.rowconfigure(0, weight=1)

        self.video_display = VideoDisplayBlobs(video_frame, 800, 600)

    def _setup_bindings(self):
        """Setup event bindings."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _browse_image(self):
        """Browse for image file."""
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")]
        )
        if filename:
            self.image_path_var.set(filename)

    def _toggle_blob_detection(self):
        """Toggle blob detection on/off."""
        if self.video_display:
            enabled = self.video_display.blob_processor.toggle_processing()
            self._log_message(f"Blob detection {'enabled' if enabled else 'disabled'}")

    def _load_preset(self, preset_name):
        """Load detection presets."""
        presets = {
            "small": {
                "min_area": "20",
                "max_area": "500",
                "min_circularity": "0.5",
                "blur_kernel": "3",
                "blur_sigma": "0.8",
                "threshold": "120"
            },
            "medium": {
                "min_area": "100",
                "max_area": "3000",
                "min_circularity": "0.3",
                "blur_kernel": "5",
                "blur_sigma": "1.0",
                "threshold": "127"
            },
            "large": {
                "min_area": "500",
                "max_area": "10000",
                "min_circularity": "0.2",
                "blur_kernel": "7",
                "blur_sigma": "1.5",
                "threshold": "100"
            }
        }

        if preset_name in presets:
            preset = presets[preset_name]
            self.min_area_var.set(preset["min_area"])
            self.max_area_var.set(preset["max_area"])
            self.min_circularity_var.set(preset["min_circularity"])
            self.blur_kernel_var.set(preset["blur_kernel"])
            self.blur_sigma_var.set(preset["blur_sigma"])
            self.threshold_var.set(preset["threshold"])
            self._update_blob_params_from_entry()
            self._log_message(f"Loaded {preset_name} blob preset")

    def _update_blob_params_from_entry(self, event=None):
        """Update blob detection parameters from entry fields."""
        if not self.video_display:
            return

        try:
            detector = self.video_display.blob_processor.detector

            min_area = max(10, int(self.min_area_var.get()))
            max_area = max(100, int(self.max_area_var.get()))
            min_circularity = max(0.1, min(1.0, float(self.min_circularity_var.get())))
            blur_kernel = max(1, int(self.blur_kernel_var.get()))
            blur_sigma = max(0.1, min(10.0, float(self.blur_sigma_var.get())))
            threshold = max(1, min(255, int(self.threshold_var.get())))

            # Ensure max_area is larger than min_area
            if max_area <= min_area:
                max_area = min_area + 100
                self.max_area_var.set(str(max_area))

            detector.update_parameters(
                min_blob_area=min_area,
                max_blob_area=max_area,
                min_blob_circularity=min_circularity,
                blur_kernel_size=blur_kernel,
                blur_sigma=blur_sigma,
                threshold_value=threshold
            )

        except ValueError as e:
            self._log_message(f"Parameter error: {e}")
        except Exception as e:
            self._log_message(f"Unexpected parameter update error: {e}")

    def _start_emulation(self):
        """Start the emulation."""
        if self.is_running:
            return

        try:
            image_path = self.image_path_var.get()
            frame_rate = self.frame_rate_var.get()
            use_emulation = self.use_emulation_var.get()

            if not Path(image_path).exists():
                messagebox.showerror("Error", f"Image file not found: {image_path}")
                return

            # Create emulated grabber
            self.grabber = EmulatedPylonGrabber(
                use_emulation=use_emulation,
                image_path=image_path,
                frame_rate=frame_rate
            )

            # Start the grabber thread (non-blocking)
            self.grabber.start()

            # Start video display
            if self.video_display:
                self.video_display.start_display()

            # Start update thread
            self.stop_update.clear()
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            self._log_message(f"Started emulation with {image_path} at {frame_rate} FPS")

        except Exception as e:
            self._log_message(f"Error starting emulation: {e}")
            messagebox.showerror("Error", f"Failed to start emulation: {e}")

    def _stop_emulation(self):
        """Stop the emulation."""
        if not self.is_running:
            return

        self._log_message("Stopping emulation...")

        try:
            # Set running flag to false first
            self.is_running = False

            # Stop update thread
            self.stop_update.set()
            if self.update_thread and self.update_thread.is_alive():
                self._log_message("Waiting for update thread to stop...")
                self.update_thread.join(timeout=2.0)
                if self.update_thread.is_alive():
                    self._log_message("Warning: Update thread did not stop gracefully")

            # Stop grabber
            if self.grabber:
                self._log_message("Stopping grabber...")
                self.grabber.stop()
                # Wait a bit for the grabber to stop
                time.sleep(0.1)
                self.grabber = None

            # Stop video display
            if self.video_display:
                self.video_display.stop_display()

            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            self._log_message("Stopped emulation")

        except Exception as e:
            self._log_message(f"Error stopping emulation: {e}")

    def _update_loop(self):
        """Main update loop for video display."""
        self._log_message("Update loop started")

        while not self.stop_update.is_set():
            try:
                if self.grabber and self.video_display and self.is_running:
                    frame = self.grabber.read()
                    if frame is not None:
                        # Schedule GUI update in main thread
                        self.root.after(0, self._update_frame_safe, frame)

                time.sleep(1/30)  # ~30 FPS update rate

            except Exception as e:
                self._log_message(f"Update loop error: {e}")
                time.sleep(0.1)

        self._log_message("Update loop stopped")

    def _update_frame_safe(self, frame):
        """Safely update frame in main GUI thread."""
        try:
            if self.video_display and self.is_running:
                self.video_display.update_frame(frame)
                self._update_statistics()
        except Exception as e:
            self._log_message(f"Frame update error: {e}")

    def _update_statistics(self):
        """Update statistics display."""
        if not self.video_display:
            return

        try:
            stats = self.video_display.blob_processor.detector.get_statistics()

            stats_text = f"""Blob Detection Statistics:
Blobs Detected: {stats['blobs_detected']}
Frames Processed: {stats['frames_processed']}
Detection Rate: {stats['detection_rate']:.3f}

Current Parameters:
Min Area: {self.min_area_var.get()}
Max Area: {self.max_area_var.get()}
Min Circularity: {self.min_circularity_var.get()}
Blur Kernel: {self.blur_kernel_var.get()}
Blur Sigma: {self.blur_sigma_var.get()}
Threshold: {self.threshold_var.get()}"""

            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)

        except Exception as e:
            pass  # Ignore statistics update errors

    def _log_message(self, message):
        """Add a message to the log display."""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"

            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)

            # Keep log size manageable
            lines = self.log_text.get(1.0, tk.END).split('\n')
            if len(lines) > 100:
                self.log_text.delete(1.0, f"{len(lines) - 100}.0")

        except Exception:
            pass  # Ignore log errors

    def _on_closing(self):
        """Handle window closing event."""
        try:
            if self.is_running:
                self._log_message("Closing application, stopping emulation...")
                self._stop_emulation()
                # Give it a moment to stop
                time.sleep(0.5)

            self.root.destroy()
        except Exception as e:
            print(f"Error during closing: {e}")
            self.root.destroy()


def main():
    """Main function to run the blob detection GUI application."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run GUI
    root = tk.Tk()
    app = BlobDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
