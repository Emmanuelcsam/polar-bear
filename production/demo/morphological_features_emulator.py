#!/usr/bin/env python3
"""
BMP Video Emulator with Morphological Features Analysis.
Emulates real-time video feed by looping a BMP image and integrates with morphological
features detection for manual parameter adjustment and real-time analysis.
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

# Import the morphological detector
from morphological_features_module import MorphologicalDetector, MorphologicalProcessor

# Import the BMP video emulator components
from bmp_video_emulator import BMPVideoEmulator, EmulatedPylonGrabber


class VideoDisplayMorphological:
    """
    Video display widget that shows frames with morphological analysis in real-time.
    """

    def __init__(self, parent, width=640, height=480):
        self.parent = parent
        self.width = width
        self.height = height
        self.current_frame = None
        self.is_displaying = False

        # Morphological processor
        self.morph_processor = MorphologicalProcessor()

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
            # Process frame with morphological analysis
            if self.morph_processor.is_processing_enabled():
                processed_frame = self.morph_processor.process_frame(frame)
            else:
                processed_frame = frame

            # Convert from BGR to RGB for tkinter
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Resize frame to fit canvas while maintaining aspect ratio
            h, w = rgb_frame.shape[:2]
            scale = min(self.width/w, self.height/h)
            new_w, new_h = int(w*scale), int(h*scale)

            resized_frame = cv2.resize(rgb_frame, (new_w, new_h))

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(resized_frame)
            photo = ImageTk.PhotoImage(image=pil_image)

            # Update canvas
            self.canvas.delete("all")
            x_offset = (self.width - new_w) // 2
            y_offset = (self.height - new_h) // 2
            self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)

            # Keep a reference to prevent garbage collection
            self.canvas.image = photo

            # Update info
            stats = self.morph_processor.get_detector().get_statistics()
            info_text = f"Frame: {stats['frames_processed']} | Features: {stats['features_extracted']} | Components: {stats['components_found']} | Defects: {stats['defects_detected']}"
            self.info_label.config(text=info_text)

        except Exception as e:
            logging.error(f"Error updating morphological display: {e}")

    def start_display(self):
        """Start the video display loop."""
        self.is_displaying = True

    def stop_display(self):
        """Stop the video display loop."""
        self.is_displaying = False
        self.canvas.delete("all")
        self.info_label.config(text="Display stopped")


class MorphologicalFeaturesGUI:
    """
    GUI for controlling the BMP video emulator with live morphological features analysis.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("BMP Video Emulator - Morphological Features Analysis")
        self.root.geometry("1300x900")

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
        self.image_path_var = tk.StringVar(value="pictures/good.bmp")
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

        # Morphological Analysis section
        morph_frame = ttk.LabelFrame(left_panel, text="Morphological Analysis", padding="5")
        morph_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Enable morphological analysis
        self.enable_morph_var = tk.BooleanVar(value=True)
        self.enable_morph_check = ttk.Checkbutton(morph_frame, text="Enable Morphological Analysis",
                                                variable=self.enable_morph_var,
                                                command=self._toggle_morph_analysis)
        self.enable_morph_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        # Analysis types frame
        analysis_frame = ttk.LabelFrame(morph_frame, text="Analysis Types", padding="5")
        analysis_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        # Analysis type checkboxes
        self.analysis_vars = {}
        analysis_types = [
            ('features', 'Morphological Features'),
            ('complexity', 'Shape Complexity'),
            ('skeleton', 'Skeleton Features'),
            ('defects', 'Defect Detection'),
            ('components', 'Connected Components')
        ]

        for i, (key, label) in enumerate(analysis_types):
            var = tk.BooleanVar(value=True)
            self.analysis_vars[key] = var
            check = ttk.Checkbutton(analysis_frame, text=label, variable=var,
                                   command=self._update_analysis_types)
            check.grid(row=i//2, column=i%2, sticky=tk.W, padx=(0, 10))

        # Parameters frame
        params_frame = ttk.LabelFrame(morph_frame, text="Parameters", padding="5")
        params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        # Kernel sizes
        ttk.Label(params_frame, text="Kernel Sizes:").grid(row=0, column=0, sticky=tk.W)
        self.kernel_sizes_var = tk.StringVar(value="3,5,7")
        self.kernel_sizes_entry = ttk.Entry(params_frame, textvariable=self.kernel_sizes_var, width=15)
        self.kernel_sizes_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 5))
        self.kernel_sizes_entry.bind('<Return>', self._update_morph_params_from_entry)
        self.kernel_sizes_entry.bind('<FocusOut>', self._update_morph_params_from_entry)
        ttk.Label(params_frame, text="(3,5,7,11)").grid(row=0, column=2, sticky=tk.W)

        # Min component area
        ttk.Label(params_frame, text="Min Component Area:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.min_area_var = tk.StringVar(value="50")
        self.min_area_entry = ttk.Entry(params_frame, textvariable=self.min_area_var, width=15)
        self.min_area_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.min_area_entry.bind('<Return>', self._update_morph_params_from_entry)
        self.min_area_entry.bind('<FocusOut>', self._update_morph_params_from_entry)
        ttk.Label(params_frame, text="(10-1000)").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))

        # Defect threshold
        ttk.Label(params_frame, text="Defect Threshold:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.defect_threshold_var = tk.StringVar(value="30")
        self.defect_threshold_entry = ttk.Entry(params_frame, textvariable=self.defect_threshold_var, width=15)
        self.defect_threshold_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.defect_threshold_entry.bind('<Return>', self._update_morph_params_from_entry)
        self.defect_threshold_entry.bind('<FocusOut>', self._update_morph_params_from_entry)
        ttk.Label(params_frame, text="(1-255)").grid(row=2, column=2, sticky=tk.W, pady=(5, 0))

        # Morphological filter operation
        ttk.Label(params_frame, text="Filter Operation:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.filter_operation_var = tk.StringVar(value="gradient")
        filter_combo = ttk.Combobox(params_frame, textvariable=self.filter_operation_var,
                                   values=['opening', 'closing', 'gradient', 'tophat', 'blackhat'],
                                   state='readonly', width=12)
        filter_combo.grid(row=3, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        filter_combo.bind('<<ComboboxSelected>>', self._update_morph_params_from_entry)

        # Filter kernel size
        ttk.Label(params_frame, text="Filter Kernel Size:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.filter_kernel_size_var = tk.StringVar(value="5")
        self.filter_kernel_size_entry = ttk.Entry(params_frame, textvariable=self.filter_kernel_size_var, width=15)
        self.filter_kernel_size_entry.grid(row=4, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.filter_kernel_size_entry.bind('<Return>', self._update_morph_params_from_entry)
        self.filter_kernel_size_entry.bind('<FocusOut>', self._update_morph_params_from_entry)
        ttk.Label(params_frame, text="(1-21)").grid(row=4, column=2, sticky=tk.W, pady=(5, 0))

        # Blur parameters
        ttk.Label(params_frame, text="Blur Kernel Size:").grid(row=5, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_kernel_var = tk.StringVar(value="5")
        self.blur_kernel_entry = ttk.Entry(params_frame, textvariable=self.blur_kernel_var, width=15)
        self.blur_kernel_entry.grid(row=5, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.blur_kernel_entry.bind('<Return>', self._update_morph_params_from_entry)
        self.blur_kernel_entry.bind('<FocusOut>', self._update_morph_params_from_entry)
        ttk.Label(params_frame, text="(1-31, odd)").grid(row=5, column=2, sticky=tk.W, pady=(5, 0))

        ttk.Label(params_frame, text="Blur Sigma:").grid(row=6, column=0, sticky=tk.W, pady=(5, 0))
        self.blur_sigma_var = tk.StringVar(value="1.0")
        self.blur_sigma_entry = ttk.Entry(params_frame, textvariable=self.blur_sigma_var, width=15)
        self.blur_sigma_entry.grid(row=6, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.blur_sigma_entry.bind('<Return>', self._update_morph_params_from_entry)
        self.blur_sigma_entry.bind('<FocusOut>', self._update_morph_params_from_entry)
        ttk.Label(params_frame, text="(0.1-10.0)").grid(row=6, column=2, sticky=tk.W, pady=(5, 0))

        # Preset buttons
        preset_frame = ttk.LabelFrame(left_panel, text="Presets", padding="5")
        preset_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        presets = [
            ("Default", "default"),
            ("Fine Detail", "fine"),
            ("Coarse Features", "coarse"),
            ("Defect Focus", "defect")
        ]

        for i, (name, preset) in enumerate(presets):
            btn = ttk.Button(preset_frame, text=name, width=12,
                           command=lambda p=preset: self._load_preset(p))
            btn.grid(row=i//2, column=i%2, padx=2, pady=2)

        # Statistics section
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="5")
        stats_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.stats_text = tk.Text(stats_frame, height=8, width=35, font=('Courier', 9))
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Log section
        log_frame = ttk.LabelFrame(left_panel, text="Log", padding="5")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        left_panel.rowconfigure(5, weight=1)

        self.log_text = tk.Text(log_frame, height=6, width=35, font=('Courier', 8))
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        log_frame.rowconfigure(0, weight=1)

        # Right panel for video display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

        # Video display frame
        video_frame = ttk.LabelFrame(right_panel, text="Video Display - Morphological Analysis", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        # Create video display
        self.video_display = VideoDisplayMorphological(video_frame, width=640, height=480)

    def _setup_bindings(self):
        """Set up event bindings."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _browse_image(self):
        """Browse for image file."""
        filename = filedialog.askopenfilename(
            title="Select BMP Image",
            filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")]
        )
        if filename:
            self.image_path_var.set(filename)

    def _toggle_morph_analysis(self):
        """Toggle morphological analysis on/off."""
        if self.video_display:
            enabled = self.video_display.morph_processor.toggle_processing()
            self._log_message(f"Morphological analysis {'enabled' if enabled else 'disabled'}")

    def _update_analysis_types(self):
        """Update the analysis types based on checkboxes."""
        if not self.video_display:
            return

        enabled_types = [key for key, var in self.analysis_vars.items() if var.get()]
        self.video_display.morph_processor.get_detector().update_parameters(analysis_types=enabled_types)
        self._log_message(f"Analysis types updated: {enabled_types}")

    def _load_preset(self, preset_name):
        """Load a parameter preset."""
        presets = {
            "default": {
                "kernel_sizes": "3,5,7",
                "min_area": "50",
                "defect_threshold": "30",
                "filter_operation": "gradient",
                "filter_kernel_size": "5",
                "blur_kernel": "5",
                "blur_sigma": "1.0"
            },
            "fine": {
                "kernel_sizes": "3,5,7,9",
                "min_area": "25",
                "defect_threshold": "15",
                "filter_operation": "tophat",
                "filter_kernel_size": "3",
                "blur_kernel": "3",
                "blur_sigma": "0.5"
            },
            "coarse": {
                "kernel_sizes": "7,11,15",
                "min_area": "100",
                "defect_threshold": "50",
                "filter_operation": "opening",
                "filter_kernel_size": "9",
                "blur_kernel": "9",
                "blur_sigma": "2.0"
            },
            "defect": {
                "kernel_sizes": "5,7,9",
                "min_area": "30",
                "defect_threshold": "10",
                "filter_operation": "blackhat",
                "filter_kernel_size": "7",
                "blur_kernel": "5",
                "blur_sigma": "1.5"
            }
        }

        if preset_name in presets:
            preset = presets[preset_name]
            self.kernel_sizes_var.set(preset["kernel_sizes"])
            self.min_area_var.set(preset["min_area"])
            self.defect_threshold_var.set(preset["defect_threshold"])
            self.filter_operation_var.set(preset["filter_operation"])
            self.filter_kernel_size_var.set(preset["filter_kernel_size"])
            self.blur_kernel_var.set(preset["blur_kernel"])
            self.blur_sigma_var.set(preset["blur_sigma"])

            self._update_morph_params_from_entry()
            self._log_message(f"Loaded preset: {preset_name}")

    def _update_morph_params_from_entry(self, event=None):
        """Update morphological parameters from entry widgets."""
        if not self.video_display:
            return

        try:
            # Parse kernel sizes
            kernel_sizes_str = self.kernel_sizes_var.get().strip()
            kernel_sizes = [int(k.strip()) for k in kernel_sizes_str.split(',') if k.strip()]

            min_area = int(self.min_area_var.get())
            defect_threshold = int(self.defect_threshold_var.get())
            filter_operation = self.filter_operation_var.get()
            filter_kernel_size = int(self.filter_kernel_size_var.get())
            blur_kernel_size = int(self.blur_kernel_var.get())
            blur_sigma = float(self.blur_sigma_var.get())

            # Update detector parameters
            self.video_display.morph_processor.get_detector().update_parameters(
                kernel_sizes=kernel_sizes,
                min_component_area=min_area,
                defect_threshold=defect_threshold,
                filter_operation=filter_operation,
                filter_kernel_size=filter_kernel_size,
                blur_kernel_size=blur_kernel_size,
                blur_sigma=blur_sigma
            )

        except ValueError as e:
            self._log_message(f"Invalid parameter values: {e}")
        except Exception as e:
            self._log_message(f"Error updating parameters: {e}")

    def _start_emulation(self):
        """Start the video emulation."""
        if self.is_running:
            return

        try:
            image_path = self.image_path_var.get()
            frame_rate = self.frame_rate_var.get()
            use_emulation = self.use_emulation_var.get()

            # Check if image exists
            if not Path(image_path).exists():
                messagebox.showerror("Error", f"Image file not found: {image_path}")
                return

            # Create emulator or grabber
            if use_emulation or not PYLON_AVAILABLE:
                self.grabber = EmulatedPylonGrabber(
                    use_emulation=True,
                    image_path=image_path,
                    frame_rate=frame_rate
                )
                self._log_message(f"Started emulation with {image_path} at {frame_rate} FPS")
            else:
                self.grabber = PylonFrameGrabber()
                self._log_message("Started real camera capture")

            # Start the grabber
            self.grabber.start()

            # Start video display
            if self.video_display:
                self.video_display.start_display()

            # Start update thread
            self.is_running = True
            self.stop_update.clear()
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

            # Update button states
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

        except Exception as e:
            self._log_message(f"Error starting emulation: {e}")
            messagebox.showerror("Error", f"Failed to start emulation: {e}")

    def _stop_emulation(self):
        """Stop the video emulation."""
        if not self.is_running:
            return

        try:
            # Stop update thread
            self.is_running = False
            self.stop_update.set()

            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=1.0)

            # Stop grabber
            if self.grabber:
                self.grabber.stop()
                if hasattr(self.grabber, 'join'):
                    self.grabber.join(timeout=1.0)

            # Stop video display
            if self.video_display:
                self.video_display.stop_display()

            # Update button states
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            self._log_message("Emulation stopped")

        except Exception as e:
            self._log_message(f"Error stopping emulation: {e}")

    def _update_loop(self):
        """Main update loop for processing frames."""
        while self.is_running and not self.stop_update.is_set():
            try:
                if self.grabber and self.video_display:
                    frame = self.grabber.read()
                    if frame is not None:
                        # Update frame in main thread
                        self.root.after_idle(self._update_frame_safe, frame)

                        # Update statistics periodically
                        if self.grabber.frame_count % 30 == 0:  # Every 30 frames
                            self.root.after_idle(self._update_statistics)

                time.sleep(0.001)  # Small delay to prevent excessive CPU usage

            except Exception as e:
                self._log_message(f"Error in update loop: {e}")
                break

    def _update_frame_safe(self, frame):
        """Safely update frame in main thread."""
        try:
            if self.video_display and self.is_running:
                self.video_display.update_frame(frame)
        except Exception as e:
            self._log_message(f"Error updating frame: {e}")

    def _update_statistics(self):
        """Update the statistics display."""
        try:
            if self.video_display:
                stats = self.video_display.morph_processor.get_detector().get_statistics()

                stats_text = f"""Morphological Analysis Statistics:
─────────────────────────────────
Frames Processed: {stats['frames_processed']}
Features Extracted: {stats['features_extracted']}
Components Found: {stats['components_found']}
Defects Detected: {stats['defects_detected']}

Current Parameters:
─────────────────────────────────
Analysis Types: {len(self.video_display.morph_processor.get_detector().analysis_types)}
Kernel Sizes: {self.kernel_sizes_var.get()}
Min Component Area: {self.min_area_var.get()}
Defect Threshold: {self.defect_threshold_var.get()}
Filter Operation: {self.filter_operation_var.get()}
Filter Kernel: {self.filter_kernel_size_var.get()}
Blur Kernel: {self.blur_kernel_var.get()}
Blur Sigma: {self.blur_sigma_var.get()}
"""

                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, stats_text)

        except Exception as e:
            self._log_message(f"Error updating statistics: {e}")

    def _log_message(self, message):
        """Add a message to the log display."""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_line = f"[{timestamp}] {message}\n"

            self.log_text.insert(tk.END, log_line)
            self.log_text.see(tk.END)

            # Keep log size manageable
            if int(self.log_text.index('end-1c').split('.')[0]) > 100:
                self.log_text.delete(1.0, "50.0")

            self.root.update_idletasks()

        except Exception as e:
            print(f"Error logging message: {e}")

    def _on_closing(self):
        """Handle application closing."""
        try:
            self._stop_emulation()
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Error during closing: {e}")


def main():
    """Main function to run the morphological features GUI application."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run GUI
    root = tk.Tk()
    app = MorphologicalFeaturesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
