#!/usr/bin/env python3
"""
Frequency Features Emulator - GUI for testing frequency domain analysis.
Provides real-time parameter adjustment for frequency-based defect detection.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import sys
import os
import threading
import time
from typing import Dict, Tuple, List

# Import the pylon grabber module
from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE

# Import the BMP video emulator components
from bmp_video_emulator import BMPVideoEmulator, EmulatedPylonGrabber


class VideoDisplayFrequencyFeatures:
    """
    Video display widget that shows frames with frequency features in real-time.
    """

    def __init__(self, parent, width=640, height=480):
        self.parent = parent
        self.width = width
        self.height = height
        self.current_frame = None
        self.is_displaying = False

        # Processing enabled flag
        self.processing_enabled = True

        # Filter parameters
        self.filter_type = 'lowpass'
        self.cutoff_freq = 0.3
        self.apply_filter = False
        self.filter_strength = 1.0
        self.inner_cutoff = 0.1
        self.noise_reduction = 0.0
        self.enhancement = 1.0
        self.defect_threshold = 0.5
        self.sensitivity = 1.0

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
            # Store current frame for processing
            self.current_frame = frame.copy()

            # Process frame if enabled
            if self.processing_enabled:
                processed_frame = self._process_frequency_features(frame)
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
            info_text = f"Processing: {self.processing_enabled} | Filter: {self.filter_type} | Cutoff: {self.cutoff_freq:.3f} | Strength: {self.filter_strength:.1f}"
            if self.noise_reduction > 0:
                info_text += f" | Noise: {self.noise_reduction:.2f}"
            if self.enhancement != 1.0:
                info_text += f" | Enhance: {self.enhancement:.1f}"
            self.info_label.config(text=info_text)

        except Exception as e:
            self.info_label.config(text=f"Error: {str(e)}")
            print(f"Error in update_frame: {e}")

    def update_parameters(self, filter_type=None, cutoff_freq=None, apply_filter=None, 
                        filter_strength=None, inner_cutoff=None, noise_reduction=None, enhancement=None,
                        defect_threshold=None, sensitivity=None):
        """Update processing parameters."""
        if filter_type is not None:
            self.filter_type = filter_type
        if cutoff_freq is not None:
            self.cutoff_freq = cutoff_freq
        if apply_filter is not None:
            self.apply_filter = apply_filter
        if filter_strength is not None:
            self.filter_strength = filter_strength
        if inner_cutoff is not None:
            self.inner_cutoff = inner_cutoff
        if noise_reduction is not None:
            self.noise_reduction = noise_reduction
        if enhancement is not None:
            self.enhancement = enhancement
        if defect_threshold is not None:
            self.defect_threshold = defect_threshold
        if sensitivity is not None:
            self.sensitivity = sensitivity

    def _process_frequency_features(self, frame):
        """Process frame with frequency features."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply frequency filtering if enabled
            if self.apply_filter:
                filtered_gray = self._apply_frequency_filter(gray)
            else:
                filtered_gray = gray

            # Convert back to BGR for display
            if len(frame.shape) == 3:
                output_frame = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2BGR)
            else:
                output_frame = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2BGR)

            h, w = output_frame.shape[:2]

            # Add frequency analysis text overlay
            cv2.putText(output_frame, "Frequency Features Analysis", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Compute FFT for analysis
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)

            # Display frequency statistics with sensitivity
            mean_freq = float(np.mean(magnitude))
            max_freq = float(np.max(magnitude))
            total_power = float(np.sum(magnitude ** 2))
            
            # Apply sensitivity to frequency analysis
            sensitivity_factor = self.sensitivity
            enhanced_mean = mean_freq * sensitivity_factor
            enhanced_max = max_freq * sensitivity_factor

            cv2.putText(output_frame, f"Mean Freq: {enhanced_mean:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(output_frame, f"Max Freq: {enhanced_max:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(output_frame, f"Total Power: {total_power:.1e}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(output_frame, f"Sensitivity: {sensitivity_factor:.1f}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Show filter status
            if self.apply_filter:
                cv2.putText(output_frame, f"Filter: {self.filter_type.upper()} @ {self.cutoff_freq:.2f}", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(output_frame, "Filter: OFF", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            return output_frame

        except Exception as e:
            print(f"Error in frequency processing: {e}")
            return frame

    def _apply_frequency_filter(self, gray):
        """Apply frequency domain filter to grayscale image."""
        try:
            h, w = gray.shape
            
            # Apply noise reduction if enabled
            if self.noise_reduction > 0:
                # Apply Gaussian blur for noise reduction
                kernel_size = int(self.noise_reduction * 10) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            
            # Compute FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)

            # Create frequency grid
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # Normalize distance to [0, 1]
            max_dist = np.sqrt(center_x**2 + center_y**2)
            dist_normalized = dist_from_center / max_dist

            # Create filter mask with enhanced control
            if self.filter_type == 'lowpass':
                # Apply filter strength
                mask = np.exp(-(dist_normalized / self.cutoff_freq) ** self.filter_strength)
            elif self.filter_type == 'highpass':
                # Apply filter strength
                mask = 1 - np.exp(-(dist_normalized / self.cutoff_freq) ** self.filter_strength)
            elif self.filter_type == 'bandpass':
                # Use separate inner and outer cutoffs
                inner_mask = np.exp(-(dist_normalized / self.inner_cutoff) ** self.filter_strength)
                outer_mask = 1 - np.exp(-(dist_normalized / self.cutoff_freq) ** self.filter_strength)
                mask = inner_mask * outer_mask
            elif self.filter_type == 'bandstop':
                # Bandstop (notch filter)
                inner_mask = 1 - np.exp(-(dist_normalized / self.inner_cutoff) ** self.filter_strength)
                outer_mask = np.exp(-(dist_normalized / self.cutoff_freq) ** self.filter_strength)
                mask = inner_mask + outer_mask
            else:
                mask = np.ones((h, w), dtype=float)

            # Apply filter
            f_shift_filtered = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_filtered = np.fft.ifft2(f_ishift)
            img_filtered = np.real(img_filtered)

            # Apply enhancement
            if self.enhancement != 1.0:
                img_filtered = img_filtered * self.enhancement

            # Normalize and clip
            img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
            
            return img_filtered

        except Exception as e:
            print(f"Error applying frequency filter: {e}")
            return gray

    def start_display(self):
        """Start the video display loop."""
        self.is_displaying = True

    def stop_display(self):
        """Stop the video display loop."""
        self.is_displaying = False
        self.canvas.delete("all")
        self.info_label.config(text="Display stopped")

    def toggle_processing(self):
        """Toggle frequency processing on/off."""
        self.processing_enabled = not self.processing_enabled
        return self.processing_enabled


class FrequencyFeaturesGUI:
    """
    Main GUI class for Frequency Features analysis and visualization.
    Provides controls for FFT analysis, filtering, and defect detection.
    """

    def __init__(self, root):
        """Initialize the FrequencyFeaturesGUI."""
        self.root = root
        self.root.title("Frequency Features Video Emulator")
        self.root.geometry("1400x900")

        # Initialize emulation components
        self.emulator = None
        self.grabber = None
        self.is_running = False
        self.video_display = None
        self.update_thread = None
        self.stop_update = threading.Event()

        # Image display parameters
        self.display_width = 640
        self.display_height = 480

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface with video emulation controls."""
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Video Emulation Controls
        emulation_frame = ttk.LabelFrame(control_frame, text="Video Emulation Controls")
        emulation_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

        # Image path configuration
        path_frame = ttk.Frame(emulation_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(path_frame, text="Image Path:").pack(side=tk.LEFT)
        self.image_path_var = tk.StringVar(value="pictures/good.bmp")
        ttk.Entry(path_frame, textvariable=self.image_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_image_path).pack(side=tk.LEFT)

        # Frame rate control
        rate_frame = ttk.Frame(emulation_frame)
        rate_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(rate_frame, text="Frame Rate (FPS):").pack(side=tk.LEFT)
        self.frame_rate_var = tk.StringVar(value="10.0")
        ttk.Entry(rate_frame, textvariable=self.frame_rate_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(rate_frame, text="(0.1 - 120.0)").pack(side=tk.LEFT)

        # Control buttons
        button_frame = ttk.Frame(emulation_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(button_frame, text="Start Emulation", command=self.start_emulation)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Emulation", command=self.stop_emulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.process_button = ttk.Button(button_frame, text="Toggle Processing", command=self.toggle_processing, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)

        self.load_button = ttk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Processing Parameters
        params_frame = ttk.LabelFrame(control_frame, text="Processing Parameters")
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

        # Filter controls
        filter_frame = ttk.Frame(params_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(filter_frame, text="Filter Type:").pack(side=tk.LEFT)
        self.filter_type_var = tk.StringVar(value="lowpass")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_type_var, values=["lowpass", "highpass", "bandpass", "bandstop"])
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind('<<ComboboxSelected>>', self._update_filter_params)

        # Cutoff frequency
        cutoff_frame = ttk.Frame(params_frame)
        cutoff_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(cutoff_frame, text="Cutoff Freq:").pack(side=tk.LEFT)
        self.cutoff_freq_var = tk.StringVar(value="0.3")
        ttk.Entry(cutoff_frame, textvariable=self.cutoff_freq_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(cutoff_frame, text="(0.001 - 2.0)").pack(side=tk.LEFT)
        cutoff_frame.bind('<Return>', self._update_filter_params)

        # Apply filter checkbox
        apply_filter_frame = ttk.Frame(params_frame)
        apply_filter_frame.pack(fill=tk.X, padx=5, pady=2)

        self.apply_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(apply_filter_frame, text="Apply Filter", variable=self.apply_filter_var, command=self._update_filter_params).pack(side=tk.LEFT)

        # Additional frequency parameters
        # Filter strength
        strength_frame = ttk.Frame(params_frame)
        strength_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(strength_frame, text="Filter Strength:").pack(side=tk.LEFT)
        self.filter_strength_var = tk.StringVar(value="1.0")
        ttk.Entry(strength_frame, textvariable=self.filter_strength_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(strength_frame, text="(0.1 - 10.0)").pack(side=tk.LEFT)
        strength_frame.bind('<Return>', self._update_filter_params)

        # Bandpass inner cutoff
        inner_cutoff_frame = ttk.Frame(params_frame)
        inner_cutoff_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(inner_cutoff_frame, text="Inner Cutoff:").pack(side=tk.LEFT)
        self.inner_cutoff_var = tk.StringVar(value="0.1")
        ttk.Entry(inner_cutoff_frame, textvariable=self.inner_cutoff_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(inner_cutoff_frame, text="(0.001 - 1.0)").pack(side=tk.LEFT)
        inner_cutoff_frame.bind('<Return>', self._update_filter_params)

        # Noise reduction
        noise_frame = ttk.Frame(params_frame)
        noise_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(noise_frame, text="Noise Reduction:").pack(side=tk.LEFT)
        self.noise_reduction_var = tk.StringVar(value="0.0")
        ttk.Entry(noise_frame, textvariable=self.noise_reduction_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(noise_frame, text="(0.0 - 1.0)").pack(side=tk.LEFT)
        noise_frame.bind('<Return>', self._update_filter_params)

        # Enhancement factor
        enhancement_frame = ttk.Frame(params_frame)
        enhancement_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(enhancement_frame, text="Enhancement:").pack(side=tk.LEFT)
        self.enhancement_var = tk.StringVar(value="1.0")
        ttk.Entry(enhancement_frame, textvariable=self.enhancement_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(enhancement_frame, text="(0.1 - 5.0)").pack(side=tk.LEFT)
        enhancement_frame.bind('<Return>', self._update_filter_params)

        # Threshold for defect detection
        threshold_frame = ttk.Frame(params_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(threshold_frame, text="Defect Threshold:").pack(side=tk.LEFT)
        self.defect_threshold_var = tk.StringVar(value="0.5")
        ttk.Entry(threshold_frame, textvariable=self.defect_threshold_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(threshold_frame, text="(0.0 - 1.0)").pack(side=tk.LEFT)
        threshold_frame.bind('<Return>', self._update_filter_params)

        # Frequency analysis sensitivity
        sensitivity_frame = ttk.Frame(params_frame)
        sensitivity_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(sensitivity_frame, text="Sensitivity:").pack(side=tk.LEFT)
        self.sensitivity_var = tk.StringVar(value="1.0")
        ttk.Entry(sensitivity_frame, textvariable=self.sensitivity_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(sensitivity_frame, text="(0.1 - 10.0)").pack(side=tk.LEFT)
        sensitivity_frame.bind('<Return>', self._update_filter_params)

        # Display area
        display_frame = ttk.Frame(self.root)
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Video display
        video_frame = ttk.LabelFrame(display_frame, text="Video Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.video_display = VideoDisplayFrequencyFeatures(video_frame, self.display_width, self.display_height)

        # Status area
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)

    def browse_image_path(self):
        """Browse for image file."""
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png"), ("BMP files", "*.bmp"), ("All files", "*.*")]
        )
        if filename:
            self.image_path_var.set(filename)

    def start_emulation(self):
        """Start video emulation."""
        try:
            # Check if path exists
            image_path = self.image_path_var.get()
            if not os.path.exists(image_path):
                self.status_label.config(text=f"Path not found: {image_path}")
                return

            # Initialize emulated grabber
            try:
                frame_rate = float(self.frame_rate_var.get())
                # Clamp frame rate to reasonable range
                frame_rate = max(0.1, min(120.0, frame_rate))
            except ValueError:
                frame_rate = 10.0
                
            self.grabber = EmulatedPylonGrabber(
                use_emulation=True,
                image_path=image_path,
                frame_rate=frame_rate
            )

            # Start the grabber thread (non-blocking)
            self.grabber.start()

            # Start video display
            self.video_display.start_display()

            # Update video display parameters
            self._update_filter_params()

            # Start update thread
            self.is_running = True
            self.stop_update.clear()
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.process_button.config(state=tk.NORMAL)
            self.status_label.config(text="Emulation running")

        except Exception as e:
            self.status_label.config(text=f"Error starting emulation: {e}")

    def stop_emulation(self):
        """Stop video emulation."""
        try:
            # Stop update loop
            self.is_running = False
            self.stop_update.set()

            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=1.0)

            # Stop grabber
            if self.grabber:
                self.grabber.stop()
                self.grabber = None

            # Stop video display
            if self.video_display:
                self.video_display.stop_display()

            # Update UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.process_button.config(state=tk.DISABLED)
            self.status_label.config(text="Emulation stopped")

        except Exception as e:
            self.status_label.config(text=f"Error stopping emulation: {e}")

    def toggle_processing(self):
        """Toggle frequency processing on/off."""
        if self.video_display:
            enabled = self.video_display.toggle_processing()
            self.status_label.config(text=f"Processing: {'ON' if enabled else 'OFF'}")

    def _update_filter_params(self, event=None):
        """Update filter parameters in the video display."""
        if self.video_display:
            try:
                # Convert string values to appropriate types
                cutoff_freq = float(self.cutoff_freq_var.get())
                filter_strength = float(self.filter_strength_var.get())
                inner_cutoff = float(self.inner_cutoff_var.get())
                noise_reduction = float(self.noise_reduction_var.get())
                enhancement = float(self.enhancement_var.get())
                defect_threshold = float(self.defect_threshold_var.get())
                sensitivity = float(self.sensitivity_var.get())
                
                self.video_display.update_parameters(
                    filter_type=self.filter_type_var.get(),
                    cutoff_freq=cutoff_freq,
                    apply_filter=self.apply_filter_var.get(),
                    filter_strength=filter_strength,
                    inner_cutoff=inner_cutoff,
                    noise_reduction=noise_reduction,
                    enhancement=enhancement,
                    defect_threshold=defect_threshold,
                    sensitivity=sensitivity
                )
            except ValueError as e:
                print(f"Invalid parameter value: {e}")

    def _update_loop(self):
        """Main update loop for video emulation."""
        while self.is_running and not self.stop_update.is_set():
            try:
                if self.grabber and self.video_display:
                    # Get frame from grabber
                    frame = self.grabber.read()
                    if frame is not None:
                        # Update video display
                        self.video_display.update_frame(frame)

                # Sleep based on frame rate
                try:
                    frame_rate = float(self.frame_rate_var.get())
                    if frame_rate > 0:
                        time.sleep(1.0 / frame_rate)
                    else:
                        time.sleep(0.1)
                except ValueError:
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error in update loop: {e}")
                break

    # Load and process image methods
    def load_image(self):
        """Load and display an image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png"), ("BMP files", "*.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path_var.set(file_path)
            self.status_label.config(text=f"Image path set: {file_path}")




def main():
    """Main function to run the Frequency Features Emulator."""
    root = tk.Tk()
    app = FrequencyFeaturesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
