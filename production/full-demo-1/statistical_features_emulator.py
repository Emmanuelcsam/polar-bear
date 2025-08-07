#!/usr/bin/env python3
"""
Statistical Features Video Emulator.
Emulates real-time video feed by looping a BMP image and integrates with statistical features
for manual parameter adjustment and real-time feature extraction.
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

# Import the statistical features processor
from statistical_features_module import StatisticalFeaturesDetector, StatisticalFeaturesProcessor

# Import the BMP video emulator components
from bmp_video_emulator import BMPVideoEmulator, EmulatedPylonGrabber


class VideoDisplayStatisticalFeatures:
    """
    Video display widget that shows frames with statistical features in real-time.
    """

    def __init__(self, parent, width=640, height=480):
        self.parent = parent
        self.width = width
        self.height = height
        self.current_frame = None
        self.is_displaying = False

        # Statistical features processor
        self.stats_processor = StatisticalFeaturesProcessor()

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
            # Process frame with statistical features if enabled
            if self.stats_processor.is_processing_enabled():
                processed_frame = self.stats_processor.process_frame(frame)
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

            # Update info with performance stats
            stats = self.stats_processor.get_performance_stats()
            parallel_status = "PARALLEL" if stats.get('parallel_processing', False) else "SEQUENTIAL"
            frames_dropped = stats.get('frames_dropped', 0)
            avg_time = stats.get('avg_processing_time', 0.0)

            info_text = f"Frame: {new_width}x{new_height} | Features: {stats['current_feature_count']} | Processed: {stats['frames_processed']} | Mode: {parallel_status}"
            if frames_dropped > 0:
                info_text += f" | Dropped: {frames_dropped}"
            if avg_time > 0:
                info_text += f" | Avg: {avg_time:.3f}s"

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


class StatisticalFeaturesGUI:
    """
    GUI for controlling the BMP video emulator with live statistical features extraction.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Statistical Features Video Emulator")
        self.root.geometry("1400x900")

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
        self.image_path_var = tk.StringVar(value="good.bmp")
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

        # Statistical Features section
        stats_frame = ttk.LabelFrame(left_panel, text="Statistical Features", padding="5")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Enable statistical features
        self.enable_stats_var = tk.BooleanVar(value=True)
        self.enable_stats_check = ttk.Checkbutton(stats_frame, text="Enable Statistical Features",
                                                 variable=self.enable_stats_var,
                                                 command=self._toggle_stats_features)
        self.enable_stats_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        # Feature type checkboxes
        feature_types_frame = ttk.Frame(stats_frame)
        feature_types_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        self.enable_basic_stats_var = tk.BooleanVar(value=True)
        self.enable_basic_stats_check = ttk.Checkbutton(feature_types_frame, text="Basic Statistics",
                                                       variable=self.enable_basic_stats_var,
                                                       command=self._update_stats_params_from_checkboxes)
        self.enable_basic_stats_check.grid(row=0, column=0, sticky=tk.W)

        self.enable_histogram_var = tk.BooleanVar(value=True)
        self.enable_histogram_check = ttk.Checkbutton(feature_types_frame, text="Histogram Features",
                                                     variable=self.enable_histogram_var,
                                                     command=self._update_stats_params_from_checkboxes)
        self.enable_histogram_check.grid(row=1, column=0, sticky=tk.W)

        self.enable_texture_var = tk.BooleanVar(value=True)
        self.enable_texture_check = ttk.Checkbutton(feature_types_frame, text="Texture Statistics",
                                                   variable=self.enable_texture_var,
                                                   command=self._update_stats_params_from_checkboxes)
        self.enable_texture_check.grid(row=2, column=0, sticky=tk.W)

        self.enable_moment_var = tk.BooleanVar(value=True)
        self.enable_moment_check = ttk.Checkbutton(feature_types_frame, text="Moment Features",
                                                  variable=self.enable_moment_var,
                                                  command=self._update_stats_params_from_checkboxes)
        self.enable_moment_check.grid(row=3, column=0, sticky=tk.W)

        # Parameters frame
        params_frame = ttk.Frame(stats_frame)
        params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        # Histogram bins
        ttk.Label(params_frame, text="Histogram Bins:").grid(row=0, column=0, sticky=tk.W)
        self.histogram_bins_var = tk.StringVar(value="32")
        self.histogram_bins_entry = ttk.Entry(params_frame, textvariable=self.histogram_bins_var, width=10)
        self.histogram_bins_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 5))
        self.histogram_bins_entry.bind('<Return>', self._update_stats_params_from_entry)
        self.histogram_bins_entry.bind('<FocusOut>', self._update_stats_params_from_entry)
        ttk.Label(params_frame, text="(8-256)").grid(row=0, column=2, sticky=tk.W)

        # Texture window size
        ttk.Label(params_frame, text="Texture Window:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.texture_window_var = tk.StringVar(value="5")
        self.texture_window_entry = ttk.Entry(params_frame, textvariable=self.texture_window_var, width=10)
        self.texture_window_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.texture_window_entry.bind('<Return>', self._update_stats_params_from_entry)
        self.texture_window_entry.bind('<FocusOut>', self._update_stats_params_from_entry)
        ttk.Label(params_frame, text="(3-15)").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))

        # Feature update interval
        ttk.Label(params_frame, text="Update Interval:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.update_interval_var = tk.StringVar(value="1.0")
        self.update_interval_entry = ttk.Entry(params_frame, textvariable=self.update_interval_var, width=10)
        self.update_interval_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        self.update_interval_entry.bind('<Return>', self._update_stats_params_from_entry)
        self.update_interval_entry.bind('<FocusOut>', self._update_stats_params_from_entry)
        ttk.Label(params_frame, text="(0.1-10.0s)").grid(row=2, column=2, sticky=tk.W, pady=(5, 0))

        # Performance section
        performance_frame = ttk.LabelFrame(left_panel, text="Performance", padding="5")
        performance_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Parallel processing controls
        self.parallel_processing_var = tk.BooleanVar(value=True)
        self.parallel_processing_check = ttk.Checkbutton(performance_frame, text="Enable Parallel Processing",
                                                        variable=self.parallel_processing_var,
                                                        command=self._toggle_parallel_processing)
        self.parallel_processing_check.grid(row=0, column=0, columnspan=3, sticky=tk.W)

        # Max workers
        ttk.Label(performance_frame, text="Max Workers:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.max_workers_var = tk.StringVar(value="4")
        self.max_workers_entry = ttk.Entry(performance_frame, textvariable=self.max_workers_var, width=10)
        self.max_workers_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 5), pady=(5, 0))
        ttk.Label(performance_frame, text="(1-8)").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))

        # Performance info
        self.performance_info_label = ttk.Label(performance_frame, text="Performance: Ready")
        self.performance_info_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        # Presets section
        presets_frame = ttk.LabelFrame(left_panel, text="Feature Presets", padding="5")
        presets_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(presets_frame, text="Basic Only", command=lambda: self._load_preset("basic")).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(presets_frame, text="Full Analysis", command=lambda: self._load_preset("full")).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(presets_frame, text="Fast Mode", command=lambda: self._load_preset("fast")).grid(row=0, column=2)

        # Statistics section
        stats_display_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="5")
        stats_display_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.stats_text = tk.Text(stats_display_frame, height=8, width=45)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Log section
        log_frame = ttk.LabelFrame(left_panel, text="Log", padding="5")
        log_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_panel.rowconfigure(6, weight=1)

        self.log_text = tk.Text(log_frame, height=10, width=45)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.rowconfigure(0, weight=1)

        # Right panel for video display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Video display
        video_frame = ttk.LabelFrame(right_panel, text="Video Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.rowconfigure(0, weight=1)

        self.video_display = VideoDisplayStatisticalFeatures(video_frame, 800, 600)

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

    def _toggle_stats_features(self):
        """Toggle statistical features on/off."""
        if self.video_display:
            enabled = self.video_display.stats_processor.toggle_processing()
            self._log_message(f"Statistical features {'enabled' if enabled else 'disabled'}")

    def _load_preset(self, preset_name):
        """Load feature extraction presets."""
        presets = {
            "basic": {
                "enable_basic_stats": True,
                "enable_histogram_features": False,
                "enable_texture_stats": False,
                "enable_moment_features": False,
                "histogram_bins": "32",
                "texture_window_size": "5",
                "update_interval": "0.5"
            },
            "full": {
                "enable_basic_stats": True,
                "enable_histogram_features": True,
                "enable_texture_stats": True,
                "enable_moment_features": True,
                "histogram_bins": "64",
                "texture_window_size": "7",
                "update_interval": "1.0"
            },
            "fast": {
                "enable_basic_stats": True,
                "enable_histogram_features": True,
                "enable_texture_stats": False,
                "enable_moment_features": False,
                "histogram_bins": "16",
                "texture_window_size": "3",
                "update_interval": "0.2"
            }
        }

        if preset_name in presets:
            preset = presets[preset_name]
            self.enable_basic_stats_var.set(preset["enable_basic_stats"])
            self.enable_histogram_var.set(preset["enable_histogram_features"])
            self.enable_texture_var.set(preset["enable_texture_stats"])
            self.enable_moment_var.set(preset["enable_moment_features"])
            self.histogram_bins_var.set(preset["histogram_bins"])
            self.texture_window_var.set(preset["texture_window_size"])
            self.update_interval_var.set(preset["update_interval"])
            self._update_stats_params_from_checkboxes()
            self._log_message(f"Loaded {preset_name} feature preset")

    def _update_stats_params_from_checkboxes(self):
        """Update statistical features parameters from checkbox states."""
        if not self.video_display:
            return

        try:
            detector = self.video_display.stats_processor.detector

            detector.update_parameters(
                enable_basic_stats=self.enable_basic_stats_var.get(),
                enable_histogram_features=self.enable_histogram_var.get(),
                enable_texture_stats=self.enable_texture_var.get(),
                enable_moment_features=self.enable_moment_var.get()
            )

            self._log_message(f"Updated feature types: basic={self.enable_basic_stats_var.get()}, "
                            f"histogram={self.enable_histogram_var.get()}, "
                            f"texture={self.enable_texture_var.get()}, "
                            f"moment={self.enable_moment_var.get()}")

        except Exception as e:
            self._log_message(f"Parameter update error: {e}")

    def _update_stats_params_from_entry(self, event=None):
        """Update statistical features parameters from entry fields."""
        if not self.video_display:
            return

        try:
            detector = self.video_display.stats_processor.detector

            histogram_bins = max(8, min(256, int(self.histogram_bins_var.get())))
            texture_window_size = max(3, min(15, int(self.texture_window_var.get())))
            update_interval = max(0.1, min(10.0, float(self.update_interval_var.get())))

            detector.update_parameters(
                histogram_bins=histogram_bins,
                texture_window_size=texture_window_size,
                feature_update_interval=update_interval
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

            # Start simple update loop using tkinter's after method
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            # Use tkinter's after method instead of threading for better stability
            self._schedule_next_update()

            self._log_message(f"Started emulation with {image_path} at {frame_rate} FPS")

        except Exception as e:
            self._log_message(f"Error starting emulation: {e}")
            messagebox.showerror("Error", f"Failed to start emulation: {e}")

    def _schedule_next_update(self):
        """Schedule the next frame update using tkinter's after method."""
        if self.is_running:
            try:
                if self.grabber and self.video_display:
                    frame = self.grabber.read()
                    if frame is not None:
                        self.video_display.update_frame(frame)
                        self._update_statistics()

                # Schedule next update (30 FPS = ~33ms between frames)
                self.root.after(33, self._schedule_next_update)
            except Exception as e:
                self._log_message(f"Update error: {e}")
                if self.is_running:  # Only reschedule if still running
                    self.root.after(100, self._schedule_next_update)

    def _stop_emulation(self):
        """Stop the emulation."""
        if not self.is_running:
            return

        self._log_message("Stopping emulation...")

        try:
            # Set running flag to false first to stop the update cycle
            self.is_running = False

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

    def _update_statistics(self):
        """Update statistics display."""
        if not self.video_display:
            return

        try:
            stats = self.video_display.stats_processor.detector.get_statistics()
            current_features = self.video_display.stats_processor.detector.current_features

            stats_text = f"""Statistical Features Statistics:
Features Extracted: {stats['features_extracted']}
Current Features: {stats['current_feature_count']}
Frames Processed: {stats['frames_processed']}
Processing Rate: {stats['processing_rate']:.2f} fps
Avg Processing Time: {stats.get('avg_processing_time', 0.0):.3f}s

Current Parameters:
Basic Stats: {self.enable_basic_stats_var.get()}
Histogram Features: {self.enable_histogram_var.get()}
Texture Stats: {self.enable_texture_var.get()}
Moment Features: {self.enable_moment_var.get()}
Histogram Bins: {self.histogram_bins_var.get()}
Texture Window: {self.texture_window_var.get()}
Update Interval: {self.update_interval_var.get()}s

Key Features:"""

            # Add some key feature values if available
            if current_features:
                key_features = ['mean', 'std', 'entropy', 'hist_mode', 'texture_contrast']
                for feature in key_features:
                    if feature in current_features:
                        stats_text += f"\n{feature}: {current_features[feature]:.3f}"

            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)

            # Update performance info
            self._update_performance_info()

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

    def _toggle_parallel_processing(self):
        """Toggle parallel processing on/off."""
        try:
            if hasattr(self.video_display, 'stats_processor'):
                enabled = self.parallel_processing_var.get()
                # This would require creating a new processor with different settings
                # For now, just log the change
                self._log_message(f"Parallel processing {'enabled' if enabled else 'disabled'}")

                # Update performance info
                if enabled:
                    self.performance_info_label.config(text=f"Performance: Parallel ({self.max_workers_var.get()} workers)")
                else:
                    self.performance_info_label.config(text="Performance: Sequential")
        except Exception as e:
            self._log_message(f"Error toggling parallel processing: {e}")

    def _update_performance_info(self):
        """Update performance information display."""
        try:
            if hasattr(self.video_display, 'stats_processor'):
                stats = self.video_display.stats_processor.get_performance_stats()

                processing_enabled = stats.get('parallel_processing', False)
                mode = "SIMPLIFIED" if not processing_enabled else "PARALLEL"  # We're using simplified mode now
                dropped = stats.get('frames_dropped', 0)
                avg_time = stats.get('avg_processing_time', 0.0)

                info_text = f"Performance: {mode}"
                if dropped > 0:
                    info_text += f" | Dropped: {dropped}"
                if avg_time > 0:
                    info_text += f" | {avg_time:.3f}s"

                self.performance_info_label.config(text=info_text)

        except Exception as e:
            pass  # Ignore performance update errors

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
    """Main function to run the statistical features GUI application."""
    # Configure logging with verbose output
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('statistical_features_emulator.log')
        ]
    )

    # Create and run GUI
    root = tk.Tk()
    app = StatisticalFeaturesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
