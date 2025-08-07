#!/usr/bin/env python3
"""
VideoDisplayFrequency Widget - Display component for frequency domain analysis.
Shows original image and frequency spectrum side-by-side with filtering options.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Dict, Tuple, List
import sys
import os
import json

# Add modular scripts to path
sys.path.append('dev/modular_scripts')

# Define frequency features functions locally to avoid import issues
def compute_fft_features(gray):
    """Compute FFT features from grayscale image."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    power = magnitude**2
    phase = np.angle(fshift)

    center = np.array(power.shape) // 2

    return {
        'fft_mean_magnitude': float(np.mean(magnitude)),
        'fft_std_magnitude': float(np.std(magnitude)),
        'fft_max_magnitude': float(np.max(magnitude)),
        'fft_total_power': float(np.sum(power)),
        'fft_dc_component': float(magnitude[center[0], center[1]]),
        'fft_mean_phase': float(np.mean(phase)),
        'fft_std_phase': float(np.std(phase)),
        'fft_spectral_centroid': 0.0,  # Simplified for now
        'fft_spectral_spread': 0.0,    # Simplified for now
        'fft_high_freq_ratio': 0.0     # Simplified for now
    }

def apply_frequency_filter(image, filter_type, cutoff_freq):
    """Apply frequency domain filter."""
    h, w = image.shape
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    # Create frequency grid
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Normalize distance to [0, 1]
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist_normalized = dist_from_center / max_dist

    # Create filter mask
    if filter_type == 'lowpass':
        mask = dist_normalized <= cutoff_freq
    elif filter_type == 'highpass':
        mask = dist_normalized >= cutoff_freq
    else:
        mask = np.ones((h, w), dtype=bool)

    # Apply filter
    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.real(img_filtered)

    return np.clip(img_filtered, 0, 255).astype(np.uint8)

def detect_periodic_patterns(gray, threshold=0.5):
    """Detect periodic patterns in image."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Simple peak detection
    peaks = magnitude > threshold * np.max(magnitude)
    y_coords, x_coords = np.where(peaks)

    return list(zip(x_coords.tolist(), y_coords.tolist()))

def visualize_frequency_spectrum(gray):
    """Create visualization of frequency spectrum."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Log scale for better visualization
    log_magnitude = np.log(magnitude + 1)

    # Normalize to 0-255
    normalized = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return normalized

def create_frequency_mask(shape, filter_type, cutoff_freq):
    """Create frequency domain mask."""
    h, w = shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist_normalized = dist_from_center / max_dist

    if filter_type == 'lowpass':
        return dist_normalized <= cutoff_freq
    elif filter_type == 'highpass':
        return dist_normalized >= cutoff_freq
    else:
        return np.ones((h, w), dtype=bool)


class VideoDisplayFrequency(ttk.Frame):
    """
    Widget for displaying original image and frequency spectrum side-by-side.
    Includes frequency filtering and feature extraction capabilities.
    """

    def __init__(self, parent, width: int = 640, height: int = 480, **kwargs):
        """
        Initialize the VideoDisplayFrequency widget.

        Args:
            parent: Parent widget
            width: Display width for each panel
            height: Display height for each panel
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)

        # Display dimensions
        self.display_width = width
        self.display_height = height

        # Image data
        self.original_image = None
        self.processed_image = None
        self.frequency_spectrum = None
        self.fft_data = None
        self.fft_shift = None

        # Processing parameters with validation
        self.filter_params = {
            'filter_type': 'lowpass',
            'cutoff_freq': 0.3,
            'apply_filter': False
        }

        # Default parameters for reset
        self.default_params = self.filter_params.copy()

        # Preset configurations
        self.presets = {
            'Noise Removal': {'filter_type': 'lowpass', 'cutoff_freq': 0.2, 'apply_filter': True},
            'Edge Enhancement': {'filter_type': 'highpass', 'cutoff_freq': 0.1, 'apply_filter': True},
            'Pattern Isolation': {'filter_type': 'bandpass', 'cutoff_freq': 0.3, 'apply_filter': True}
        }

        # Custom presets storage
        self.custom_presets = {}

        # Parameter limits for validation
        self.param_limits = {
            'cutoff_freq': (0.01, 0.99),  # Valid range for cutoff frequency
            'filter_types': ['lowpass', 'highpass', 'bandpass']  # Valid filter types
        }

        # Extracted features
        self.frequency_features = {}
        self.periodic_patterns = []

        # Real-time update control
        self.update_pending = False
        self.update_delay = 50  # milliseconds
        self.last_update_params = None

        # Create widget components
        self._create_widgets()

    def _create_widgets(self):
        """Create and arrange widget components."""

        # Main container with two panels
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Original image
        left_panel = ttk.LabelFrame(main_frame, text="Original Image", padding="5")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        self.original_canvas = tk.Canvas(
            left_panel,
            width=self.display_width // 2,
            height=self.display_height,
            bg='black',
            highlightthickness=1,
            highlightbackground='gray'
        )
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # Right panel - Frequency spectrum
        right_panel = ttk.LabelFrame(main_frame, text="Frequency Spectrum", padding="5")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

        self.spectrum_canvas = tk.Canvas(
            right_panel,
            width=self.display_width // 2,
            height=self.display_height,
            bg='black',
            highlightthickness=1,
            highlightbackground='gray'
        )
        self.spectrum_canvas.pack(fill=tk.BOTH, expand=True)

        # Info panel below displays
        info_frame = ttk.LabelFrame(self, text="Frequency Features", padding="5")
        info_frame.pack(fill=tk.X, pady=(10, 0))

        # Create info labels in grid
        self.info_labels = {}
        info_items = [
            ('DC Component:', 'dc_component'),
            ('Mean Magnitude:', 'mean_magnitude'),
            ('Spectral Centroid:', 'spectral_centroid'),
            ('High Freq Ratio:', 'high_freq_ratio'),
            ('Periodic Patterns:', 'periodic_patterns'),
            ('Filter Applied:', 'filter_status')
        ]

        for i, (label_text, key) in enumerate(info_items):
            row = i // 3
            col = (i % 3) * 2

            ttk.Label(info_frame, text=label_text).grid(
                row=row, column=col, sticky=tk.W, padx=(5, 0), pady=2
            )

            self.info_labels[key] = ttk.Label(
                info_frame, text="--",
                font=('Courier', 10),
                foreground='blue'
            )
            self.info_labels[key].grid(
                row=row, column=col+1, sticky=tk.W, padx=(5, 20), pady=2
            )

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def update_frame(self, image: np.ndarray,
                    apply_filter: bool = False,
                    filter_type: str = 'lowpass',
                    cutoff_freq: float = 0.3) -> np.ndarray:
        """
        Update display with new frame and apply frequency processing.
        Includes parameter validation and real-time update optimization.

        Args:
            image: Input image (can be grayscale or color)
            apply_filter: Whether to apply frequency filter
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
            cutoff_freq: Cutoff frequency (0-1)

        Returns:
            Processed image
        """
        if image is None:
            return None

        # Validate parameters
        filter_type = self._validate_filter_type(filter_type)
        cutoff_freq = self._validate_cutoff_freq(cutoff_freq)

        # Check if parameters have changed
        new_params = (apply_filter, filter_type, cutoff_freq)
        if self.last_update_params == new_params and self.original_image is not None:
            # If parameters haven't changed and we have an image, use cached result
            if not self.update_pending:
                return self.processed_image

        self.last_update_params = new_params

        # Store original
        self.original_image = image.copy()

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Update filter parameters
        self.filter_params['apply_filter'] = apply_filter
        self.filter_params['filter_type'] = filter_type
        self.filter_params['cutoff_freq'] = cutoff_freq

        # Schedule real-time update
        self._schedule_update(gray)

        return self.processed_image

    def _validate_filter_type(self, filter_type: str) -> str:
        """
        Validate filter type parameter.

        Args:
            filter_type: Filter type to validate

        Returns:
            Valid filter type
        """
        if filter_type not in self.param_limits['filter_types']:
            print(f"Warning: Invalid filter type '{filter_type}'. Using 'lowpass'.")
            return 'lowpass'
        return filter_type

    def _validate_cutoff_freq(self, cutoff_freq: float) -> float:
        """
        Validate cutoff frequency parameter.

        Args:
            cutoff_freq: Cutoff frequency to validate

        Returns:
            Valid cutoff frequency
        """
        min_cutoff, max_cutoff = self.param_limits['cutoff_freq']

        if cutoff_freq < min_cutoff:
            print(f"Warning: Cutoff frequency {cutoff_freq} too low. Using {min_cutoff}.")
            return min_cutoff
        elif cutoff_freq > max_cutoff:
            print(f"Warning: Cutoff frequency {cutoff_freq} too high. Using {max_cutoff}.")
            return max_cutoff

        return cutoff_freq

    def _schedule_update(self, gray_image: np.ndarray):
        """
        Schedule a real-time update with debouncing.

        Args:
            gray_image: Grayscale image to process
        """
        # Cancel any pending update
        if self.update_pending:
            self.after_cancel(self.update_pending)

        # Schedule new update
        self.update_pending = self.after(
            self.update_delay,
            lambda: self._perform_update(gray_image)
        )

    def _perform_update(self, gray: np.ndarray):
        """
        Perform the actual update processing.

        Args:
            gray: Grayscale image
        """
        try:
            # Clear pending flag
            self.update_pending = False

            # Compute FFT
            self._compute_fft(gray)

            # Generate frequency spectrum visualization
            self.frequency_spectrum = visualize_frequency_spectrum(gray)

            # Apply frequency filter if requested
            if self.filter_params['apply_filter']:
                self.processed_image = apply_frequency_filter(
                    gray,
                    self.filter_params['filter_type'],
                    self.filter_params['cutoff_freq']
                )
            else:
                self.processed_image = gray.copy()

            # Extract frequency features
            self._extract_features(gray)

            # Update displays
            self._update_displays()

            # Update info labels
            self._update_info_labels()

        except Exception as e:
            print(f"Error in real-time update: {e}")

    def update_parameters_realtime(self, **kwargs):
        """
        Update processing parameters in real-time.
        Designed to be called from GUI controls for smooth updates.

        Args:
            **kwargs: Parameters to update (apply_filter, filter_type, cutoff_freq)
        """
        if self.original_image is None:
            return

        # Update parameters if provided
        if 'apply_filter' in kwargs:
            self.filter_params['apply_filter'] = kwargs['apply_filter']

        if 'filter_type' in kwargs:
            filter_type = self._validate_filter_type(kwargs['filter_type'])
            self.filter_params['filter_type'] = filter_type

        if 'cutoff_freq' in kwargs:
            cutoff_freq = self._validate_cutoff_freq(kwargs['cutoff_freq'])
            self.filter_params['cutoff_freq'] = cutoff_freq

        # Convert to grayscale if needed
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image.copy()

        # Schedule update
        self._schedule_update(gray)

    def _compute_fft(self, gray: np.ndarray):
        """
        Compute FFT of the image.

        Args:
            gray: Grayscale image
        """
        # Compute 2D FFT
        self.fft_data = np.fft.fft2(gray)
        self.fft_shift = np.fft.fftshift(self.fft_data)

    def _extract_features(self, gray: np.ndarray):
        """
        Extract frequency domain features.

        Args:
            gray: Grayscale image
        """
        # Extract FFT features
        self.frequency_features = compute_fft_features(gray)

        # Detect periodic patterns
        self.periodic_patterns = detect_periodic_patterns(gray, threshold=0.5)

    def _update_displays(self):
        """Update both canvas displays."""
        # Display original image
        if self.original_image is not None:
            self._display_image(self.original_image, self.original_canvas)

        # Display frequency spectrum
        if self.frequency_spectrum is not None:
            # Apply colormap for better visualization
            spectrum_colored = cv2.applyColorMap(self.frequency_spectrum, cv2.COLORMAP_JET)

            # Add filter overlay if filter is applied
            if self.filter_params['apply_filter']:
                spectrum_with_filter = self._add_filter_overlay(
                    spectrum_colored,
                    self.filter_params['filter_type'],
                    self.filter_params['cutoff_freq']
                )
                self._display_image(spectrum_with_filter, self.spectrum_canvas)
            else:
                self._display_image(spectrum_colored, self.spectrum_canvas)

    def _add_filter_overlay(self, spectrum_img: np.ndarray,
                           filter_type: str,
                           cutoff: float) -> np.ndarray:
        """
        Add filter visualization overlay to spectrum image.

        Args:
            spectrum_img: Spectrum visualization image
            filter_type: Type of filter
            cutoff: Cutoff frequency

        Returns:
            Image with filter overlay
        """
        overlay = spectrum_img.copy()
        h, w = overlay.shape[:2]
        center = (w // 2, h // 2)

        # Calculate radius based on cutoff
        max_radius = min(center[0], center[1])
        cutoff_radius = int(max_radius * cutoff)

        # Create semi-transparent overlay
        mask_overlay = np.zeros_like(overlay)

        if filter_type == 'lowpass':
            # Draw circle for lowpass
            cv2.circle(mask_overlay, center, cutoff_radius, (0, 255, 0), 2)
            cv2.putText(mask_overlay, "LP", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif filter_type == 'highpass':
            # Draw circle for highpass
            cv2.circle(mask_overlay, center, cutoff_radius, (0, 0, 255), 2)
            cv2.putText(mask_overlay, "HP", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif filter_type == 'bandpass':
            # Draw two circles for bandpass
            inner_radius = cutoff_radius // 2
            cv2.circle(mask_overlay, center, inner_radius, (255, 255, 0), 2)
            cv2.circle(mask_overlay, center, cutoff_radius, (255, 255, 0), 2)
            cv2.putText(mask_overlay, "BP", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Blend overlay with spectrum
        result = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)

        return result

    def _update_info_labels(self):
        """Update information labels with extracted features."""
        # Update DC component
        if 'fft_dc_component' in self.frequency_features:
            self.info_labels['dc_component'].config(
                text=f"{self.frequency_features['fft_dc_component']:.2f}"
            )

        # Update mean magnitude
        if 'fft_mean_magnitude' in self.frequency_features:
            self.info_labels['mean_magnitude'].config(
                text=f"{self.frequency_features['fft_mean_magnitude']:.2f}"
            )

        # Update spectral centroid
        if 'fft_spectral_centroid' in self.frequency_features:
            self.info_labels['spectral_centroid'].config(
                text=f"{self.frequency_features['fft_spectral_centroid']:.2f}"
            )

        # Update high frequency ratio
        if 'fft_high_freq_ratio' in self.frequency_features:
            self.info_labels['high_freq_ratio'].config(
                text=f"{self.frequency_features['fft_high_freq_ratio']:.3f}"
            )

        # Update periodic patterns count
        self.info_labels['periodic_patterns'].config(
            text=f"{len(self.periodic_patterns)} detected"
        )

        # Update filter status
        if self.filter_params['apply_filter']:
            filter_text = f"{self.filter_params['filter_type'].upper()} @ {self.filter_params['cutoff_freq']:.2f}"
            self.info_labels['filter_status'].config(text=filter_text, foreground='green')
        else:
            self.info_labels['filter_status'].config(text="None", foreground='gray')

    def _display_image(self, image: np.ndarray, canvas: tk.Canvas):
        """
        Display an image on a canvas.

        Args:
            image: Image to display
            canvas: Canvas widget
        """
        if image is None:
            return

        try:
            # Ensure image is 8-bit
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # Get canvas dimensions
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            # Use default dimensions if canvas not yet rendered
            if canvas_width <= 1:
                canvas_width = self.display_width // 2
            if canvas_height <= 1:
                canvas_height = self.display_height

            # Convert to RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Resize to fit canvas
            h, w = image_rgb.shape[:2]
            scale = min(canvas_width/w, canvas_height/h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            if new_w > 0 and new_h > 0:
                image_resized = cv2.resize(image_rgb, (new_w, new_h))

                # Convert to PIL Image
                pil_image = Image.fromarray(image_resized)
                photo = ImageTk.PhotoImage(pil_image)

                # Clear canvas and display image
                canvas.delete("all")
                x = canvas_width // 2
                y = canvas_height // 2
                canvas.create_image(x, y, image=photo, anchor=tk.CENTER)

                # Keep reference to prevent garbage collection
                canvas.image = photo

        except Exception as e:
            print(f"Error displaying image: {e}")

    def get_frequency_features(self) -> Dict[str, float]:
        """
        Get extracted frequency features.

        Returns:
            Dictionary of frequency features
        """
        return self.frequency_features.copy()

    def get_periodic_patterns(self) -> List[Tuple[int, int]]:
        """
        Get detected periodic patterns.

        Returns:
            List of frequency peaks (x, y)
        """
        return self.periodic_patterns.copy()

    def get_processed_image(self) -> Optional[np.ndarray]:
        """
        Get the processed image.

        Returns:
            Processed image or None
        """
        return self.processed_image.copy() if self.processed_image is not None else None

    def clear_display(self):
        """Clear both display canvases."""
        self.original_canvas.delete("all")
        self.spectrum_canvas.delete("all")

        # Reset labels
        for label in self.info_labels.values():
            label.config(text="--", foreground='blue')

        # Clear data
        self.original_image = None
        self.processed_image = None
        self.frequency_spectrum = None
        self.fft_data = None
        self.fft_shift = None
        self.frequency_features = {}
        self.periodic_patterns = []
        self.last_update_params = None
        self.update_pending = False

    def apply_preset(self, preset_name: str):
        """
        Apply a preset configuration.

        Args:
            preset_name: Name of the preset to apply
        """
        # Check built-in presets first
        if preset_name in self.presets:
            preset = self.presets[preset_name]
        elif preset_name in self.custom_presets:
            preset = self.custom_presets[preset_name]
        else:
            print(f"Preset '{preset_name}' not found")
            return

        # Apply preset parameters
        self.filter_params.update(preset)

        # Update display if we have an image
        if self.original_image is not None:
            # Convert to grayscale if needed
            if len(self.original_image.shape) == 3:
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.original_image.copy()

            # Schedule update
            self._schedule_update(gray)

    def save_custom_preset(self, preset_name: str):
        """
        Save current settings as a custom preset.

        Args:
            preset_name: Name for the custom preset
        """
        if not preset_name:
            return

        # Save current filter parameters
        self.custom_presets[preset_name] = self.filter_params.copy()

        # Save to file for persistence
        self._save_presets_to_file()

        print(f"Custom preset '{preset_name}' saved")

    def load_custom_presets(self):
        """
        Load custom presets from file.
        """
        presets_file = 'frequency_filter_presets.json'

        if os.path.exists(presets_file):
            try:
                with open(presets_file, 'r') as f:
                    self.custom_presets = json.load(f)
                print(f"Loaded {len(self.custom_presets)} custom presets")
            except Exception as e:
                print(f"Error loading presets: {e}")
                self.custom_presets = {}

    def _save_presets_to_file(self):
        """
        Save custom presets to file.
        """
        presets_file = 'frequency_filter_presets.json'

        try:
            with open(presets_file, 'w') as f:
                json.dump(self.custom_presets, f, indent=2)
        except Exception as e:
            print(f"Error saving presets: {e}")

    def delete_custom_preset(self, preset_name: str):
        """
        Delete a custom preset.

        Args:
            preset_name: Name of the preset to delete
        """
        if preset_name in self.custom_presets:
            del self.custom_presets[preset_name]
            self._save_presets_to_file()
            print(f"Custom preset '{preset_name}' deleted")

    def reset_to_defaults(self):
        """
        Reset all parameters to default values.
        """
        self.filter_params = self.default_params.copy()

        # Update display if we have an image
        if self.original_image is not None:
            # Convert to grayscale if needed
            if len(self.original_image.shape) == 3:
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.original_image.copy()

            # Schedule update
            self._schedule_update(gray)

    def get_all_preset_names(self) -> List[str]:
        """
        Get names of all available presets (built-in and custom).

        Returns:
            List of preset names
        """
        built_in = list(self.presets.keys())
        custom = list(self.custom_presets.keys())
        return built_in + custom


def demo_widget():
    """Demonstration of VideoDisplayFrequency widget."""

    root = tk.Tk()
    root.title("VideoDisplayFrequency Widget Demo")
    root.geometry("1200x700")

    # Create main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Create the frequency display widget
    freq_display = VideoDisplayFrequency(main_frame, width=1000, height=400)
    freq_display.pack(fill=tk.BOTH, expand=True)

    # Load custom presets if available
    freq_display.load_custom_presets()

    # Control panel with two sections
    control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
    control_frame.pack(fill=tk.X, pady=(10, 0))

    # Create notebook for organized controls
    control_notebook = ttk.Notebook(control_frame)
    control_notebook.pack(fill=tk.BOTH, expand=True)

    # Manual controls tab
    manual_tab = ttk.Frame(control_notebook)
    control_notebook.add(manual_tab, text="Manual Controls")

    # Presets tab
    presets_tab = ttk.Frame(control_notebook)
    control_notebook.add(presets_tab, text="Presets")

    # Filter controls
    filter_var = tk.BooleanVar(value=False)
    filter_type_var = tk.StringVar(value="lowpass")
    cutoff_var = tk.DoubleVar(value=0.3)

    # Real-time update callback for filter checkbox
    def on_filter_toggle():
        freq_display.update_parameters_realtime(apply_filter=filter_var.get())

    # Manual controls in manual tab
    manual_controls_frame = ttk.Frame(manual_tab, padding="10")
    manual_controls_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Checkbutton(manual_controls_frame, text="Apply Filter",
                   variable=filter_var,
                   command=on_filter_toggle).grid(row=0, column=0, padx=5, pady=5)

    ttk.Label(manual_controls_frame, text="Filter Type:").grid(row=0, column=1, padx=5, pady=5)

    # Real-time update callback for filter type
    def on_filter_type_change(event=None):
        freq_display.update_parameters_realtime(filter_type=filter_type_var.get())

    filter_combo = ttk.Combobox(manual_controls_frame, textvariable=filter_type_var,
                               values=["lowpass", "highpass", "bandpass"],
                               width=15, state="readonly")
    filter_combo.grid(row=0, column=2, padx=5, pady=5)
    filter_combo.bind('<<ComboboxSelected>>', on_filter_type_change)

    ttk.Label(manual_controls_frame, text="Cutoff:").grid(row=0, column=3, padx=5, pady=5)

    # Real-time update callback for cutoff frequency
    def on_cutoff_change(value):
        cutoff_label.config(text=f"{float(value):.2f}")
        freq_display.update_parameters_realtime(cutoff_freq=float(value))

    cutoff_scale = ttk.Scale(manual_controls_frame, from_=0.01, to=0.99,
                            variable=cutoff_var, orient=tk.HORIZONTAL,
                            length=200, command=on_cutoff_change)
    cutoff_scale.grid(row=0, column=4, padx=5, pady=5)

    cutoff_label = ttk.Label(manual_controls_frame, text="0.30")
    cutoff_label.grid(row=0, column=5, padx=5, pady=5)

    # Reset button in manual tab
    def reset_to_defaults():
        freq_display.reset_to_defaults()
        # Update GUI controls
        filter_var.set(freq_display.filter_params['apply_filter'])
        filter_type_var.set(freq_display.filter_params['filter_type'])
        cutoff_var.set(freq_display.filter_params['cutoff_freq'])
        cutoff_label.config(text=f"{freq_display.filter_params['cutoff_freq']:.2f}")

    ttk.Button(manual_controls_frame, text="Reset to Defaults",
              command=reset_to_defaults).grid(row=1, column=0, columnspan=2, pady=10, padx=5)

    # Presets controls in presets tab
    presets_frame = ttk.Frame(presets_tab, padding="10")
    presets_frame.pack(fill=tk.BOTH, expand=True)

    # Built-in presets section
    ttk.Label(presets_frame, text="Built-in Presets:", font=('TkDefaultFont', 10, 'bold')).grid(
        row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

    # Helper function to apply preset and update GUI
    def apply_preset_and_update(preset_name):
        freq_display.apply_preset(preset_name)
        # Update GUI controls to reflect new values
        filter_var.set(freq_display.filter_params['apply_filter'])
        filter_type_var.set(freq_display.filter_params['filter_type'])
        cutoff_var.set(freq_display.filter_params['cutoff_freq'])
        cutoff_label.config(text=f"{freq_display.filter_params['cutoff_freq']:.2f}")

    # Create buttons for built-in presets
    preset_buttons = [
        ("Noise Removal", "Remove high-frequency noise"),
        ("Edge Enhancement", "Enhance edges and details"),
        ("Pattern Isolation", "Isolate periodic patterns")
    ]

    for i, (preset_name, tooltip) in enumerate(preset_buttons):
        btn = ttk.Button(presets_frame, text=preset_name,
                        command=lambda name=preset_name: apply_preset_and_update(name),
                        width=20)
        btn.grid(row=1, column=i, padx=5, pady=5)
        # Could add tooltip here if needed

    # Custom presets section
    ttk.Separator(presets_frame, orient='horizontal').grid(
        row=2, column=0, columnspan=3, sticky='ew', pady=15)

    ttk.Label(presets_frame, text="Custom Presets:", font=('TkDefaultFont', 10, 'bold')).grid(
        row=3, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

    # Custom preset listbox and controls
    custom_preset_frame = ttk.Frame(presets_frame)
    custom_preset_frame.grid(row=4, column=0, columnspan=3, sticky='ew', pady=5)

    # Listbox for custom presets
    listbox_frame = ttk.Frame(custom_preset_frame)
    listbox_frame.grid(row=0, column=0, rowspan=4, padx=(0, 10))

    custom_listbox = tk.Listbox(listbox_frame, width=25, height=6)
    custom_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical")
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    custom_listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=custom_listbox.yview)

    # Update listbox with custom presets
    def refresh_custom_presets_list():
        custom_listbox.delete(0, tk.END)
        for preset_name in freq_display.custom_presets.keys():
            custom_listbox.insert(tk.END, preset_name)

    # Custom preset buttons
    def save_current_as_preset():
        preset_name = simpledialog.askstring("Save Preset",
                                            "Enter name for the preset:",
                                            parent=root)
        if preset_name:
            freq_display.save_custom_preset(preset_name)
            refresh_custom_presets_list()
            messagebox.showinfo("Success", f"Preset '{preset_name}' saved successfully!")

    def load_selected_preset():
        selection = custom_listbox.curselection()
        if selection:
            preset_name = custom_listbox.get(selection[0])
            apply_preset_and_update(preset_name)

    def delete_selected_preset():
        selection = custom_listbox.curselection()
        if selection:
            preset_name = custom_listbox.get(selection[0])
            if messagebox.askyesno("Confirm Delete",
                                  f"Delete preset '{preset_name}'?"):
                freq_display.delete_custom_preset(preset_name)
                refresh_custom_presets_list()

    ttk.Button(custom_preset_frame, text="Save Current",
              command=save_current_as_preset, width=15).grid(row=0, column=1, padx=5, pady=2)

    ttk.Button(custom_preset_frame, text="Load Selected",
              command=load_selected_preset, width=15).grid(row=1, column=1, padx=5, pady=2)

    ttk.Button(custom_preset_frame, text="Delete Selected",
              command=delete_selected_preset, width=15).grid(row=2, column=1, padx=5, pady=2)

    # Export/Import buttons
    def export_presets():
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Presets"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    export_data = {
                        'custom_presets': freq_display.custom_presets,
                        'version': '1.0'
                    }
                    json.dump(export_data, f, indent=2)
                messagebox.showinfo("Success", "Presets exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export presets: {e}")

    def import_presets():
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Import Presets"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    import_data = json.load(f)
                    if 'custom_presets' in import_data:
                        freq_display.custom_presets.update(import_data['custom_presets'])
                        freq_display._save_presets_to_file()
                        refresh_custom_presets_list()
                        messagebox.showinfo("Success", "Presets imported successfully!")
                    else:
                        messagebox.showerror("Error", "Invalid preset file format")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import presets: {e}")

    ttk.Button(custom_preset_frame, text="Export All",
              command=export_presets, width=15).grid(row=3, column=1, padx=5, pady=(10, 2))

    ttk.Button(custom_preset_frame, text="Import",
              command=import_presets, width=15).grid(row=4, column=1, padx=5, pady=2)

    # Initialize custom presets list
    refresh_custom_presets_list()

    # Generate test image with patterns
    def generate_test_image():
        """Generate test image with various frequency components."""
        img = np.zeros((400, 400), dtype=np.uint8)

        # Add different frequency patterns
        x = np.arange(400)
        y = np.arange(400)
        X, Y = np.meshgrid(x, y)

        # Low frequency component
        low_freq = np.sin(2 * np.pi * X / 100) * 50

        # High frequency component
        high_freq = np.sin(2 * np.pi * Y / 10) * 30

        # Diagonal pattern
        diagonal = np.sin(2 * np.pi * (X + Y) / 50) * 40

        # Combine patterns
        img = low_freq + high_freq + diagonal + 128
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Add some noise
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img

    def update_display():
        """Update the frequency display."""
        test_img = generate_test_image()
        freq_display.update_frame(
            test_img,
            apply_filter=filter_var.get(),
            filter_type=filter_type_var.get(),
            cutoff_freq=cutoff_var.get()
        )

    def load_image():
        """Load image from file."""
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"),
                      ("All files", "*.*")]
        )

        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                freq_display.update_frame(
                    img,
                    apply_filter=filter_var.get(),
                    filter_type=filter_type_var.get(),
                    cutoff_freq=cutoff_var.get()
                )

    # Action buttons at the bottom of manual tab
    action_frame = ttk.Frame(manual_controls_frame)
    action_frame.grid(row=2, column=0, columnspan=6, pady=10)

    ttk.Button(action_frame, text="Generate Test",
              command=update_display).pack(side=tk.LEFT, padx=5)

    ttk.Button(action_frame, text="Load Image",
              command=load_image).pack(side=tk.LEFT, padx=5)

    ttk.Button(action_frame, text="Clear Display",
              command=freq_display.clear_display).pack(side=tk.LEFT, padx=5)

    # Generate initial test image
    update_display()

    root.mainloop()


def main():
    """Main function that runs the demo widget."""
    demo_widget()


if __name__ == "__main__":
    main()
