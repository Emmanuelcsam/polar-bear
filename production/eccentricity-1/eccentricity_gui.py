"""
Eccentricity Tester GUI Application
Real-time video processing with eccentricity analysis using Hough circles
combined with intensity profile and gradient analysis.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import logging
import time
import threading
from pathlib import Path
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

# Import required modules
from bmp_video_emulator import BMPVideoEmulator, EmulatedPylonGrabber, PYLON_AVAILABLE
from hough_circles import HoughCirclesDetector
from eccentricity_tester import EccentricityTester, EccentricityProcessor


class EccentricityVideoDisplay:
    """
    Enhanced video display widget that shows frames with eccentricity analysis.
    """
    
    def __init__(self, parent, width=640, height=480):
        self.parent = parent
        self.width = width
        self.height = height
        self.current_frame = None
        self.is_displaying = False
        
        # Eccentricity processor
        self.eccentricity_processor = EccentricityProcessor()
        
        # Create main frame
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for video display
        self.canvas = tk.Canvas(self.main_frame, width=width, height=height, bg='black')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add a label for video info
        self.info_label = ttk.Label(self.main_frame, text="No video feed", anchor=tk.CENTER)
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Analysis results
        self.last_results = {}
        
    def update_frame(self, frame):
        """Update the display with a new frame and perform eccentricity analysis."""
        if frame is None:
            self.info_label.config(text="No frame available")
            return
        
        try:
            # Process frame with eccentricity analysis
            processed_frame, results = self.eccentricity_processor.process_frame(frame)
            self.last_results = results
            
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
            if results and 'eccentricity_score' in results:
                score = results['eccentricity_score']
                status = f"Frame: {width}x{height} | Eccentricity: {score:.1f}%"
                if 'message' in results:
                    status += f" | {results['message']}"
            else:
                status = f"Frame: {width}x{height} | No analysis available"
            
            self.info_label.config(text=status)
            
        except Exception as e:
            self.info_label.config(text=f"Error displaying frame: {e}")
            logging.error(f"Error in update_frame: {e}", exc_info=True)
    
    def get_last_results(self):
        """Get the last analysis results."""
        return self.last_results
    
    def start_display(self):
        """Start the video display loop."""
        self.is_displaying = True
    
    def stop_display(self):
        """Stop the video display loop."""
        self.is_displaying = False
        self.canvas.delete("all")
        self.info_label.config(text="Display stopped")


class EccentricityAnalysisPanel:
    """
    Panel showing detailed eccentricity analysis metrics.
    """
    
    def __init__(self, parent):
        self.parent = parent
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Eccentricity Analysis", padding="5")
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Create metrics display
        self._create_metrics_display()
        
    def _create_metrics_display(self):
        """Create the metrics display widgets."""
        # Overall score (large display)
        score_frame = ttk.Frame(self.frame)
        score_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky='ew')
        
        ttk.Label(score_frame, text="Overall Score:", font=('TkDefaultFont', 12, 'bold')).pack(side=tk.LEFT)
        self.score_label = ttk.Label(score_frame, text="0.0%", font=('TkDefaultFont', 16, 'bold'))
        self.score_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Individual metrics
        metrics = [
            ("Radial Uniformity:", "radial_uniformity"),
            ("Intensity Uniformity:", "intensity_uniformity"),
            ("Intensity Symmetry:", "intensity_symmetry"),
            ("Gradient Consistency:", "gradient_consistency"),
            ("Gradient Circularity:", "gradient_circularity"),
            ("Shape Roundness:", "shape_roundness"),
            ("Eccentricity:", "shape_eccentricity"),
            ("Radial Deviation:", "radial_deviation")
        ]
        
        self.metric_labels = {}
        for i, (label_text, key) in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) * 2
            
            ttk.Label(self.frame, text=label_text).grid(row=row, column=col, sticky='w', padx=(0, 5))
            self.metric_labels[key] = ttk.Label(self.frame, text="0.000")
            self.metric_labels[key].grid(row=row, column=col+1, sticky='w', padx=(0, 20))
        
        # Status label
        self.status_label = ttk.Label(self.frame, text="Waiting for analysis...", 
                                     foreground='gray')
        self.status_label.grid(row=len(metrics)//2 + 2, column=0, columnspan=4, pady=(10, 0))
    
    def update_metrics(self, results):
        """Update the displayed metrics with new results."""
        if not results:
            return
        
        # Update overall score with color coding
        score = results.get('eccentricity_score', 0)
        self.score_label.config(text=f"{score:.1f}%")
        
        # Color code based on score
        if score > 90:
            color = 'green'
        elif score > 75:
            color = 'dark orange'
        elif score > 60:
            color = 'orange'
        else:
            color = 'red'
        
        self.score_label.config(foreground=color)
        
        # Update individual metrics
        for key, label in self.metric_labels.items():
            value = results.get(key, 0)
            if isinstance(value, (int, float)):
                label.config(text=f"{value:.3f}")
        
        # Update status
        if 'message' in results:
            self.status_label.config(text=results['message'])
        else:
            self.status_label.config(text="Analysis complete")


class EccentricityPlotPanel:
    """
    Panel for displaying live plots of eccentricity analysis.
    """
    
    def __init__(self, parent):
        self.parent = parent
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Analysis Plots", padding="5")
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.figure, self.axes = plt.subplots(2, 2, figsize=(8, 6))
        self.figure.tight_layout()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self._initialize_plots()
        
    def _initialize_plots(self):
        """Initialize the plot layouts."""
        # Plot 1: Radial edge profile
        self.ax1 = self.axes[0, 0]
        self.ax1.set_xlabel('Angle (degrees)')
        self.ax1.set_ylabel('Distance (pixels)')
        self.ax1.set_title('Radial Edge Profile')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score history
        self.ax2 = self.axes[0, 1]
        self.ax2.set_xlabel('Time (frames)')
        self.ax2.set_ylabel('Score (%)')
        self.ax2.set_title('Eccentricity Score History')
        self.ax2.set_ylim(0, 105)
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Polar plot
        self.axes[1, 0].remove()
        self.ax3 = self.figure.add_subplot(2, 2, 3, projection='polar')
        self.ax3.set_title('Edge Distance Polar Plot')
        
        # Plot 4: Metrics bar chart
        self.ax4 = self.axes[1, 1]
        self.ax4.set_ylabel('Score')
        self.ax4.set_title('Component Scores')
        self.ax4.set_ylim(0, 1.1)
        self.ax4.grid(True, alpha=0.3, axis='y')
        
        # History data
        self.score_history = []
        self.max_history = 100
        
    def update_plots(self, results):
        """Update plots with new analysis results."""
        if not results or 'detailed_metrics' not in results:
            return
        
        try:
            # Clear previous plots
            self.ax1.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Plot 1: Radial edge profile
            if 'radial_profile' in results['detailed_metrics']:
                radial_data = results['detailed_metrics']['radial_profile']
                edge_distances = radial_data.get('edge_distances', [])
                if len(edge_distances) > 0:
                    angles = np.degrees(np.linspace(0, 2 * np.pi, len(edge_distances), endpoint=False))
                    self.ax1.plot(angles, edge_distances, 'b-', linewidth=1)
                    self.ax1.axhline(y=radial_data['mean_radius'], color='r', linestyle='--', alpha=0.7)
                    self.ax1.fill_between(angles,
                                        radial_data['mean_radius'] - radial_data['deviation'],
                                        radial_data['mean_radius'] + radial_data['deviation'],
                                        alpha=0.2, color='red')
                    self.ax1.set_xlabel('Angle (degrees)')
                    self.ax1.set_ylabel('Distance (pixels)')
                    self.ax1.set_title('Radial Edge Profile')
                    self.ax1.grid(True, alpha=0.3)
            
            # Plot 2: Update score history
            score = results.get('eccentricity_score', 0)
            self.score_history.append(score)
            if len(self.score_history) > self.max_history:
                self.score_history.pop(0)
            
            self.ax2.clear()
            if len(self.score_history) > 1:
                x_vals = range(len(self.score_history))
                self.ax2.plot(x_vals, self.score_history, 'g-', linewidth=2)
                self.ax2.fill_between(x_vals, self.score_history, alpha=0.3, color='green')
            self.ax2.set_xlabel('Time (frames)')
            self.ax2.set_ylabel('Score (%)')
            self.ax2.set_title('Eccentricity Score History')
            self.ax2.set_ylim(0, 105)
            self.ax2.grid(True, alpha=0.3)
            
            # Plot 3: Polar plot
            if 'radial_profile' in results['detailed_metrics']:
                radial_data = results['detailed_metrics']['radial_profile']
                edge_distances = radial_data.get('edge_distances', [])
                if len(edge_distances) > 0:
                    angles = np.linspace(0, 2 * np.pi, len(edge_distances), endpoint=False)
                    self.ax3.plot(angles, edge_distances, 'b-')
                    self.ax3.fill(angles, edge_distances, alpha=0.3)
                    self.ax3.set_title('Edge Distance Polar Plot')
            
            # Plot 4: Component scores
            categories = ['Radial\nUnif.', 'Intensity\nUnif.', 'Intensity\nSym.', 
                         'Gradient\nCons.', 'Shape\nRound.']
            scores = [
                results.get('radial_uniformity', 0),
                results.get('intensity_uniformity', 0),
                results.get('intensity_symmetry', 0),
                results.get('gradient_consistency', 0),
                results.get('shape_roundness', 0)
            ]
            
            colors = ['blue', 'green', 'orange', 'red', 'purple']
            bars = self.ax4.bar(categories, scores, color=colors)
            self.ax4.set_ylim(0, 1.1)
            self.ax4.set_ylabel('Score')
            self.ax4.set_title('Component Scores')
            self.ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                self.ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{score:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            logging.error(f"Error updating plots: {e}", exc_info=True)


class EccentricityGUI:
    """
    Main GUI application for the eccentricity tester.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Eccentricity Tester - Real-time Analysis")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.emulator = None
        self.grabber = None
        self.is_running = False
        self.video_display = None
        self.analysis_panel = None
        self.plot_panel = None
        
        self._create_widgets()
        self._setup_bindings()
        
    def _create_widgets(self):
        """Create and arrange GUI widgets."""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Left panel (controls and analysis)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Configuration section
        config_frame = ttk.LabelFrame(left_frame, text="Configuration", padding="5")
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Image path
        path_frame = ttk.Frame(config_frame)
        path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame, text="Image:").pack(side=tk.LEFT)
        self.image_path_var = tk.StringVar(value="good.bmp")
        self.image_path_entry = ttk.Entry(path_frame, textvariable=self.image_path_var, width=25)
        self.image_path_entry.pack(side=tk.LEFT, padx=(5, 2))
        ttk.Button(path_frame, text="Browse", command=self._browse_image).pack(side=tk.LEFT)
        
        # Frame rate
        rate_frame = ttk.Frame(config_frame)
        rate_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rate_frame, text="Frame Rate:").pack(side=tk.LEFT)
        self.frame_rate_var = tk.IntVar(value=30)
        self.frame_rate_spinbox = ttk.Spinbox(rate_frame, from_=1, to=120, 
                                             textvariable=self.frame_rate_var, width=10)
        self.frame_rate_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Control buttons
        control_frame = ttk.Frame(config_frame)
        control_frame.pack(fill=tk.X, pady=(10, 5))
        self.start_stop_btn = ttk.Button(control_frame, text="Start Analysis", 
                                        command=self._toggle_analysis, style='Accent.TButton')
        self.start_stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(control_frame, text="Save Results", command=self._save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Plots", command=self._export_plots).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(config_frame, textvariable=self.status_var, 
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
        # Analysis panel
        self.analysis_panel = EccentricityAnalysisPanel(left_frame)
        
        # Hough parameters section
        hough_frame = ttk.LabelFrame(left_frame, text="Hough Circle Parameters", padding="5")
        hough_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create parameter controls
        params = [
            ("DP:", "dp", 1.0, 0.1, 5.0),
            ("Min Distance:", "min_dist", 50, 1, 1000),
            ("Edge Threshold:", "param1", 100, 1, 500),
            ("Center Threshold:", "param2", 50, 1, 300),
            ("Min Radius:", "min_radius", 5, 0, 500),
            ("Max Radius:", "max_radius", 200, 1, 2000)
        ]
        
        self.hough_vars = {}
        for i, (label, key, default, min_val, max_val) in enumerate(params):
            row = i // 2
            col = (i % 2) * 3
            
            ttk.Label(hough_frame, text=label).grid(row=row, column=col, sticky='w', padx=(0, 5))
            var = tk.DoubleVar(value=default) if isinstance(default, float) else tk.IntVar(value=default)
            self.hough_vars[key] = var
            
            if isinstance(default, float):
                entry = ttk.Entry(hough_frame, textvariable=var, width=8)
            else:
                entry = ttk.Spinbox(hough_frame, from_=min_val, to=max_val, 
                                   textvariable=var, width=8)
            entry.grid(row=row, column=col+1, padx=(0, 10))
            entry.bind('<Return>', self._update_hough_params)
            entry.bind('<FocusOut>', self._update_hough_params)
        
        # Middle panel (video display)
        middle_frame = ttk.Frame(main_paned)
        main_paned.add(middle_frame, weight=2)
        
        video_frame = ttk.LabelFrame(middle_frame, text="Video Display", padding="5")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create video display
        self.video_display = EccentricityVideoDisplay(video_frame, width=640, height=480)
        
        # Right panel (plots)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        self.plot_panel = EccentricityPlotPanel(right_frame)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def _setup_bindings(self):
        """Setup event bindings."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Update plots periodically when running
        self._schedule_plot_update()
        
    def _browse_image(self):
        """Browse for an image file."""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if filename:
            self.image_path_var.set(filename)
    
    def _toggle_analysis(self):
        """Toggle the analysis on/off."""
        if not self.is_running:
            self._start_analysis()
        else:
            self._stop_analysis()
    
    def _start_analysis(self):
        """Start the video analysis."""
        try:
            image_path = self.image_path_var.get()
            frame_rate = self.frame_rate_var.get()
            
            # Create and start the grabber
            self.grabber = EmulatedPylonGrabber(
                use_emulation=True,
                image_path=image_path,
                frame_rate=frame_rate
            )
            self.grabber.start()
            
            # Update Hough parameters
            self._update_hough_params()
            
            self.is_running = True
            self.start_stop_btn.config(text="Stop Analysis")
            self.status_var.set("Running analysis...")
            
            # Start video display
            self.video_display.start_display()
            self._update_display()
            
        except Exception as e:
            logging.error(f"Error starting analysis: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to start analysis: {e}")
    
    def _stop_analysis(self):
        """Stop the video analysis."""
        try:
            if self.grabber:
                self.grabber.stop()
                self.grabber.join(timeout=2.0)
            
            self.is_running = False
            self.start_stop_btn.config(text="Start Analysis")
            self.status_var.set("Stopped")
            
            if self.video_display:
                self.video_display.stop_display()
                
        except Exception as e:
            logging.error(f"Error stopping analysis: {e}", exc_info=True)
    
    def _update_display(self):
        """Update the video display with current frame."""
        if self.is_running and self.grabber:
            frame = self.grabber.read()
            if self.video_display:
                self.video_display.update_frame(frame)
                
                # Update analysis panel with results
                results = self.video_display.get_last_results()
                if self.analysis_panel and results:
                    self.analysis_panel.update_metrics(results)
        
        if self.is_running:
            # Update display at specified frame rate
            self.root.after(33, self._update_display)  # ~30 FPS
    
    def _schedule_plot_update(self):
        """Schedule periodic plot updates."""
        if self.is_running and self.plot_panel:
            results = self.video_display.get_last_results() if self.video_display else None
            if results:
                self.plot_panel.update_plots(results)
        
        # Schedule next update
        self.root.after(500, self._schedule_plot_update)  # Update plots every 500ms
    
    def _update_hough_params(self, event=None):
        """Update Hough circle detection parameters."""
        if self.video_display:
            try:
                detector = self.video_display.eccentricity_processor.hough_detector
                detector.update_parameters(
                    dp=self.hough_vars['dp'].get(),
                    min_dist=self.hough_vars['min_dist'].get(),
                    param1=self.hough_vars['param1'].get(),
                    param2=self.hough_vars['param2'].get(),
                    min_radius=self.hough_vars['min_radius'].get(),
                    max_radius=self.hough_vars['max_radius'].get()
                )
                logging.info("Updated Hough parameters")
            except Exception as e:
                logging.error(f"Error updating Hough parameters: {e}")
    
    def _save_results(self):
        """Save analysis results to file."""
        if not self.video_display:
            return
        
        results = self.video_display.get_last_results()
        if not results:
            messagebox.showwarning("No Results", "No analysis results to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("Eccentricity Analysis Results\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Image: {self.image_path_var.get()}\n\n")
                    
                    f.write("Overall Metrics:\n")
                    f.write(f"  Eccentricity Score: {results.get('eccentricity_score', 0):.1f}%\n")
                    f.write(f"  Center: {results.get('center', (0, 0))}\n")
                    f.write(f"  Radius: {results.get('radius', 0)}\n\n")
                    
                    f.write("Detailed Metrics:\n")
                    metrics = [
                        ("Radial Uniformity", 'radial_uniformity'),
                        ("Radial Deviation", 'radial_deviation'),
                        ("Intensity Uniformity", 'intensity_uniformity'),
                        ("Intensity Symmetry", 'intensity_symmetry'),
                        ("Gradient Consistency", 'gradient_consistency'),
                        ("Gradient Circularity", 'gradient_circularity'),
                        ("Shape Roundness", 'shape_roundness'),
                        ("Shape Eccentricity", 'shape_eccentricity')
                    ]
                    
                    for label, key in metrics:
                        value = results.get(key, 0)
                        f.write(f"  {label}: {value:.4f}\n")
                
                messagebox.showinfo("Success", f"Results saved to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")
    
    def _export_plots(self):
        """Export current plots to file."""
        if not self.plot_panel:
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.plot_panel.figure.savefig(filename, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plots exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plots: {e}")
    
    def _on_closing(self):
        """Handle window closing."""
        if self.is_running:
            self._stop_analysis()
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
    
    # Set style
    style = ttk.Style()
    style.configure('Accent.TButton', foreground='blue')
    
    app = EccentricityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
