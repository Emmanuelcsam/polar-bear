#!/usr/bin/env python3
"""
Test script for VideoDisplayFrequency widget.
Demonstrates all features and integration capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Import the VideoDisplayFrequency widget
from video_display_frequency import VideoDisplayFrequency


class FrequencyDisplayTest(tk.Tk):
    """Test application for VideoDisplayFrequency widget."""
    
    def __init__(self):
        super().__init__()
        
        self.title("VideoDisplayFrequency Widget Test")
        self.geometry("1400x800")
        
        # Create main container
        main_container = ttk.Frame(self, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create the frequency display widget
        self.freq_display = VideoDisplayFrequency(
            main_container, 
            width=1200, 
            height=500
        )
        self.freq_display.pack(fill=tk.BOTH, expand=True)
        
        # Create control panel
        self._create_controls(main_container)
        
        # Create status bar
        self._create_status_bar(main_container)
        
        # Initialize with test image
        self._generate_and_display_test()
        
    def _create_controls(self, parent):
        """Create control panel."""
        control_frame = ttk.LabelFrame(parent, text="Test Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Test pattern selection
        pattern_frame = ttk.Frame(control_frame)
        pattern_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(pattern_frame, text="Test Pattern:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.pattern_var = tk.StringVar(value="sinusoidal")
        pattern_combo = ttk.Combobox(
            pattern_frame, 
            textvariable=self.pattern_var,
            values=["sinusoidal", "checkerboard", "gradient", "noise", "circles"],
            width=15,
            state="readonly"
        )
        pattern_combo.pack(side=tk.LEFT, padx=5)
        pattern_combo.bind("<<ComboboxSelected>>", lambda e: self._generate_and_display_test())
        
        # Filter controls
        filter_frame = ttk.Frame(control_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.filter_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            filter_frame,
            text="Enable Filter",
            variable=self.filter_enabled_var,
            command=lambda: self.freq_display.update_parameters_realtime(
                apply_filter=self.filter_enabled_var.get()
            )
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(filter_frame, text="Type:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.filter_type_var = tk.StringVar(value="lowpass")
        filter_type_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_type_var,
            values=["lowpass", "highpass", "bandpass"],
            width=12,
            state="readonly"
        )
        filter_type_combo.pack(side=tk.LEFT, padx=(0, 10))
        filter_type_combo.bind("<<ComboboxSelected>>", 
                              lambda e: self.freq_display.update_parameters_realtime(
                                  filter_type=self.filter_type_var.get()
                              ))
        
        ttk.Label(filter_frame, text="Cutoff:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.cutoff_var = tk.DoubleVar(value=0.3)
        cutoff_scale = ttk.Scale(
            filter_frame,
            from_=0.05,
            to=0.95,
            variable=self.cutoff_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda v: self._update_cutoff(v)
        )
        cutoff_scale.pack(side=tk.LEFT, padx=(0, 5))
        
        self.cutoff_label = ttk.Label(filter_frame, text="0.30", width=5)
        self.cutoff_label.pack(side=tk.LEFT)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(
            button_frame,
            text="Load Image",
            command=self._load_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Save Processed",
            command=self._save_processed
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Print Features",
            command=self._print_features
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Clear",
            command=self._clear_display
        ).pack(side=tk.LEFT, padx=5)
        
    def _create_status_bar(self, parent):
        """Create status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)
        
    def _generate_test_pattern(self, pattern_type: str, size: tuple = (400, 400)) -> np.ndarray:
        """
        Generate various test patterns.
        
        Args:
            pattern_type: Type of pattern to generate
            size: Size of the image (height, width)
            
        Returns:
            Generated test image
        """
        h, w = size
        img = np.zeros((h, w), dtype=np.uint8)
        
        if pattern_type == "sinusoidal":
            # Create sinusoidal pattern with multiple frequencies
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)
            
            # Multiple frequency components
            pattern = (
                np.sin(2 * np.pi * X / 50) * 30 +  # Horizontal low freq
                np.sin(2 * np.pi * Y / 20) * 40 +  # Vertical high freq
                np.sin(2 * np.pi * (X + Y) / 100) * 20  # Diagonal
            )
            img = np.clip(pattern + 128, 0, 255).astype(np.uint8)
            
        elif pattern_type == "checkerboard":
            # Create checkerboard pattern
            block_size = 20
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    if ((i // block_size) + (j // block_size)) % 2 == 0:
                        img[i:i+block_size, j:j+block_size] = 255
                        
        elif pattern_type == "gradient":
            # Create radial gradient
            center = (w // 2, h // 2)
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            img = (255 * (1 - dist / max_dist)).astype(np.uint8)
            
        elif pattern_type == "noise":
            # Create noise pattern
            img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
            
        elif pattern_type == "circles":
            # Create concentric circles
            center = (w // 2, h // 2)
            for r in range(20, min(center), 30):
                cv2.circle(img, center, r, 255, 2)
        
        # Add some gaussian noise to all patterns
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def _generate_and_display_test(self):
        """Generate and display test pattern."""
        pattern_type = self.pattern_var.get()
        test_img = self._generate_test_pattern(pattern_type)
        
        # Update display with the test image
        self.freq_display.update_frame(
            test_img,
            apply_filter=self.filter_enabled_var.get(),
            filter_type=self.filter_type_var.get(),
            cutoff_freq=self.cutoff_var.get()
        )
        
        self.status_label.config(text=f"Generated {pattern_type} pattern")
        
    def _update_display(self):
        """Update display with current settings."""
        if self.freq_display.original_image is not None:
            self.freq_display.update_frame(
                self.freq_display.original_image,
                apply_filter=self.filter_enabled_var.get(),
                filter_type=self.filter_type_var.get(),
                cutoff_freq=self.cutoff_var.get()
            )
            
            filter_status = "enabled" if self.filter_enabled_var.get() else "disabled"
            self.status_label.config(text=f"Filter {filter_status}")
    
    def _update_cutoff(self, value):
        """Update cutoff frequency display and reprocess in real-time."""
        self.cutoff_label.config(text=f"{float(value):.2f}")
        self.freq_display.update_parameters_realtime(
            cutoff_freq=float(value)
        )
    
    def _load_image(self):
        """Load an image from file."""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.freq_display.update_frame(
                    img,
                    apply_filter=self.filter_enabled_var.get(),
                    filter_type=self.filter_type_var.get(),
                    cutoff_freq=self.cutoff_var.get()
                )
                self.status_label.config(text=f"Loaded: {Path(file_path).name}")
            else:
                messagebox.showerror("Error", "Failed to load image")
    
    def _save_processed(self):
        """Save the processed image."""
        from tkinter import filedialog
        
        processed_img = self.freq_display.get_processed_image()
        if processed_img is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            success = cv2.imwrite(file_path, processed_img)
            if success:
                self.status_label.config(text=f"Saved: {Path(file_path).name}")
                messagebox.showinfo("Success", "Image saved successfully")
            else:
                messagebox.showerror("Error", "Failed to save image")
    
    def _print_features(self):
        """Print extracted frequency features."""
        features = self.freq_display.get_frequency_features()
        patterns = self.freq_display.get_periodic_patterns()
        
        if not features:
            messagebox.showinfo("Info", "No features extracted yet")
            return
        
        # Create feature report window
        report_window = tk.Toplevel(self)
        report_window.title("Frequency Features Report")
        report_window.geometry("500x600")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(report_window, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, width=60, height=35)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget['yscrollcommand'] = scrollbar.set
        
        # Add feature report
        report = "=" * 50 + "\n"
        report += "FREQUENCY DOMAIN FEATURES REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += "FFT Features:\n"
        report += "-" * 30 + "\n"
        for key, value in features.items():
            report += f"  {key}: {value:.6f}\n"
        
        report += "\n"
        report += "Periodic Patterns Detected:\n"
        report += "-" * 30 + "\n"
        if patterns:
            report += f"  Total peaks found: {len(patterns)}\n"
            for i, (fx, fy) in enumerate(patterns[:10], 1):
                report += f"  Peak {i}: frequency = ({fx}, {fy})\n"
        else:
            report += "  No significant periodic patterns detected\n"
        
        report += "\n"
        report += "Filter Settings:\n"
        report += "-" * 30 + "\n"
        if self.filter_enabled_var.get():
            report += f"  Filter Type: {self.filter_type_var.get()}\n"
            report += f"  Cutoff Frequency: {self.cutoff_var.get():.3f}\n"
        else:
            report += "  No filter applied\n"
        
        # Insert report into text widget
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
        
        # Add close button
        ttk.Button(
            report_window,
            text="Close",
            command=report_window.destroy
        ).pack(pady=10)
        
        self.status_label.config(text="Feature report generated")
    
    def _clear_display(self):
        """Clear the display."""
        self.freq_display.clear_display()
        self.status_label.config(text="Display cleared")


def main():
    """Run the test application."""
    app = FrequencyDisplayTest()
    
    # Center window on screen
    app.update_idletasks()
    width = app.winfo_width()
    height = app.winfo_height()
    x = (app.winfo_screenwidth() // 2) - (width // 2)
    y = (app.winfo_screenheight() // 2) - (height // 2)
    app.geometry(f'{width}x{height}+{x}+{y}')
    
    app.mainloop()


if __name__ == "__main__":
    main()
