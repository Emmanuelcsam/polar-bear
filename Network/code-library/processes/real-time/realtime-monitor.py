#!/usr/bin/env python3
"""
Real-time Monitoring and Advanced Tools for Fiber Optic Inspection
==================================================================

Provides real-time visualization, monitoring, calibration, and comparison tools
for the ultimate defect detection system.
"""

import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue
import time
from datetime import datetime
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
import sqlite3
import pandas as pd
from scipy import stats
import warnings
import shared_config # Import the shared configuration module

# Import main system components
from ultimate_defect_detector import UltimateDefectDetector, DefectDetectionConfig, DefectType
from integration_workflow import CompleteFiberInspectionSystem
from batch_processing import BatchProcessor, BatchConfig, DatabaseManager

warnings.filterwarnings('ignore')


class RealTimeMonitor:
    """
    Real-time monitoring GUI for fiber optic inspection
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fiber Optic Inspection Monitor - Real-time Analysis")
        self.root.geometry("1600x900")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Load initial configuration from shared_config
        self.current_shared_config = shared_config.get_config()

        # Initialize components
        self.inspector = None
        self.current_image = None
        self.current_results = None
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_processing = False
        self.database = DatabaseManager("realtime_monitor.db")
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'passed': 0,
            'failed': 0,
            'total_defects': 0,
            'processing_times': []
        }
        
        # Setup GUI
        self._setup_gui()
        self._setup_inspector()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()
        
        # Start GUI update loop
        self.root.after(100, self._update_gui)
        self.status = "initialized" # Add a status variable

    def get_script_info(self):
        """Returns information about the script, its status, and exposed parameters."""
        return {
            "name": "Real-time Monitor GUI",
            "status": self.status,
            "parameters": {
                "auto_process": self.auto_process_var.get(),
                "profile": self.profile_var.get(),
                "log_level": self.current_shared_config.get("log_level"),
                "data_source": self.current_shared_config.get("data_source"),
                "processing_enabled": self.current_shared_config.get("processing_enabled"),
                "threshold_value": self.current_shared_config.get("threshold_value")
            },
            "statistics": self.stats # Expose current statistics
        }

    def set_script_parameter(self, key, value):
        """Sets a specific parameter for the script and updates shared_config."""
        if key in self.current_shared_config:
            self.current_shared_config[key] = value
            shared_config.set_config_value(key, value) # Update shared config
            
            # Apply changes if they affect the running GUI or inspector
            if key == "auto_process":
                self.auto_process_var.set(value)
            elif key == "profile":
                self.profile_var.set(value)
                self._setup_inspector() # Re-initialize inspector with new profile
            # Add more conditions here for other parameters that need immediate effect
            
            self.status = f"parameter '{key}' updated"
            return True
        return False
    
    def _setup_gui(self):
        """Setup the GUI components"""
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self._open_image)
        file_menu.add_command(label="Open Folder", command=self._open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configuration", command=self._show_config_dialog)
        settings_menu.add_command(label="Calibration", command=self._show_calibration_dialog)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Method Comparison", command=self._show_comparison_tool)
        tools_menu.add_command(label="Batch Processing", command=self._show_batch_dialog)
        tools_menu.add_command(label="Statistics", command=self._show_statistics)
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Image display
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image display area
        self.image_frame = ttk.LabelFrame(left_panel, text="Image Display")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure for image display
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_original = self.fig.add_subplot(221)
        self.ax_processed = self.fig.add_subplot(222)
        self.ax_defects = self.fig.add_subplot(223)
        self.ax_confidence = self.fig.add_subplot(224)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_panel = ttk.Frame(left_panel)
        control_panel.pack(fill=tk.X, padx=5, pady=5)
        
        self.process_btn = ttk.Button(control_panel, text="Process Image", 
                                     command=self._process_current_image)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.auto_process_var = tk.BooleanVar(value=self.current_shared_config.get("auto_process", False))
        self.auto_process_cb = ttk.Checkbutton(control_panel, text="Auto Process", 
                                              variable=self.auto_process_var)
        self.auto_process_cb.pack(side=tk.LEFT, padx=5)
        
        # Processing profile
        ttk.Label(control_panel, text="Profile:").pack(side=tk.LEFT, padx=5)
        self.profile_var = tk.StringVar(value=self.current_shared_config.get("processing_profile", "balanced"))
        profile_combo = ttk.Combobox(control_panel, textvariable=self.profile_var,
                                    values=["fast", "balanced", "comprehensive"],
                                    width=15, state="readonly")
        profile_combo.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results and statistics
        right_panel = ttk.Frame(main_container, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        right_panel.pack_propagate(False)
        
        # Results display
        results_frame = ttk.LabelFrame(right_panel, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results text
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=20)
        results_scroll = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=results_scroll.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Real-time statistics
        stats_frame = ttk.LabelFrame(right_panel, text="Real-time Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Statistics labels
        self.stats_labels = {}
        stats_to_show = [
            ('total_processed', 'Total Processed:'),
            ('passed', 'Passed:'),
            ('failed', 'Failed:'),
            ('pass_rate', 'Pass Rate:'),
            ('avg_quality', 'Avg Quality:'),
            ('total_defects', 'Total Defects:'),
            ('avg_time', 'Avg Time:')
        ]
        
        for i, (key, label) in enumerate(stats_to_show):
            row = i // 2
            col = i % 2
            
            ttk.Label(stats_frame, text=label).grid(row=row, column=col*2, 
                                                    sticky=tk.W, padx=5, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="0")
            self.stats_labels[key].grid(row=row, column=col*2+1, 
                                       sticky=tk.W, padx=5, pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(right_panel, variable=self.progress_var,
                                          maximum=100, length=380)
        self.progress_bar.pack(padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_inspector(self):
        """Setup the inspection system"""
        config = self._get_current_config()
        self.inspector = CompleteFiberInspectionSystem(config)
    
    def _get_current_config(self) -> dict:
        """Get current configuration based on GUI settings and shared_config"""
        profile = self.profile_var.get()
        
        config = {
            "defect_detection": {
                "use_all_methods": profile == "comprehensive"
            },
            "output": {
                "save_all_intermediate": False,
                "generate_report": False
            }
        }
        
        if profile == "fast":
            config["defect_detection"]["method_groups"] = {
                "statistical": {"enabled": True, "methods": ["zscore"]},
                "morphological": {"enabled": True, "methods": ["tophat"]}
            }
        elif profile == "balanced":
            config["defect_detection"]["method_groups"] = {
                "statistical": {"enabled": True, "methods": ["zscore", "mad"]},
                "morphological": {"enabled": True, "methods": ["tophat", "blackhat"]},
                "ml": {"enabled": True, "methods": ["isolation_forest"]}
            }
        
        # Merge with shared_config, giving shared_config precedence for common parameters
        merged_config = self.current_shared_config.copy()
        merged_config.update(config) # GUI settings might override some shared settings
        
        return merged_config
    
    def _processing_worker(self):
        """Worker thread for image processing"""
        while True:
            try:
                # Get image from queue
                image_path = self.processing_queue.get(timeout=0.1)
                
                if image_path is None:
                    break
                
                # Process image
                start_time = time.time()
                results = self.inspector.inspect_fiber(image_path)
                processing_time = time.time() - start_time
                
                # Add processing time to results
                results['processing_time'] = processing_time
                
                # Put results in queue
                self.results_queue.put(results)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                self.results_queue.put({'error': str(e)})
    
    def _update_gui(self):
        """Update GUI with latest results"""
        try:
            # Check for new results
            while not self.results_queue.empty():
                results = self.results_queue.get_nowait()
                self._display_results(results)
                self._update_statistics(results)
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self._update_gui)
    
    def _open_image(self):
        """Open and display an image"""
        file_path = filedialog.askopenfilename(
            title="Select Fiber Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            self._display_image()
            
            if self.auto_process_var.get():
                self._process_current_image()
    
    def _open_folder(self):
        """Open folder for batch processing"""
        folder_path = filedialog.askdirectory(title="Select Folder")
        
        if folder_path:
            # Show batch dialog with selected folder
            self._show_batch_dialog(folder_path)
    
    def _display_image(self):
        """Display current image"""
        if self.current_image is None:
            return
        
        # Clear all axes
        for ax in [self.ax_original, self.ax_processed, self.ax_defects, self.ax_confidence]:
            ax.clear()
        
        # Display original
        if len(self.current_image.shape) == 3:
            self.ax_original.imshow(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
        else:
            self.ax_original.imshow(self.current_image, cmap='gray')
        self.ax_original.set_title('Original')
        self.ax_original.axis('off')
        
        # Clear other plots
        self.ax_processed.text(0.5, 0.5, 'Processed image\nwill appear here',
                              ha='center', va='center', transform=self.ax_processed.transAxes)
        self.ax_processed.set_title('Preprocessed')
        self.ax_processed.axis('off')
        
        self.ax_defects.text(0.5, 0.5, 'Defect mask\nwill appear here',
                            ha='center', va='center', transform=self.ax_defects.transAxes)
        self.ax_defects.set_title('Defects')
        self.ax_defects.axis('off')
        
        self.ax_confidence.text(0.5, 0.5, 'Confidence map\nwill appear here',
                               ha='center', va='center', transform=self.ax_confidence.transAxes)
        self.ax_confidence.set_title('Confidence')
        self.ax_confidence.axis('off')
        
        self.canvas.draw()
    
    def _process_current_image(self):
        """Process the current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please open an image first")
            return
        
        if self.is_processing:
            messagebox.showinfo("Processing", "Already processing an image")
            return
        
        # Save temporary image
        temp_path = "temp_realtime_image.png"
        cv2.imwrite(temp_path, self.current_image)
        
        # Add to processing queue
        self.processing_queue.put(temp_path)
        self.is_processing = True
        self.status_var.set("Processing...")
        self.progress_var.set(50)
    
    def _display_results(self, results: Dict[str, Any]):
        """Display processing results"""
        self.current_results = results
        self.is_processing = False
        self.progress_var.set(100)
        
        if 'error' in results:
            self.status_var.set(f"Error: {results['error']}")
            messagebox.showerror("Processing Error", results['error'])
            return
        
        # Update status
        status = results.get('pass_fail', {}).get('overall', 'UNKNOWN')
        quality = results.get('quality_metrics', {}).get('surface_quality_index', 0)
        self.status_var.set(f"Status: {status} | Quality: {quality:.1f}")
        
        # Display results text
        self.results_text.delete(1.0, tk.END)
        
        # Overall results
        self.results_text.insert(tk.END, "INSPECTION RESULTS\n", 'heading')
        self.results_text.insert(tk.END, "="*40 + "\n\n")
        
        self.results_text.insert(tk.END, f"Status: {status}\n", 
                                'pass' if status == 'PASS' else 'fail')
        self.results_text.insert(tk.END, f"Quality Index: {quality:.1f}/100\n")
        self.results_text.insert(tk.END, f"Total Defects: {len(results.get('defects', []))}\n")
        self.results_text.insert(tk.END, f"Processing Time: {results.get('processing_time', 0):.2f}s\n\n")
        
        # Regional results
        self.results_text.insert(tk.END, "REGIONAL ANALYSIS\n", 'heading')
        self.results_text.insert(tk.END, "-"*40 + "\n")
        
        pass_fail = results.get('pass_fail', {})
        for region in ['core', 'cladding', 'ferrule']:
            region_data = pass_fail.get('by_region', {}).get(region, {})
            if region_data:
                region_status = region_data.get('status', 'N/A')
                defect_count = region_data.get('defect_count', 0)
                
                self.results_text.insert(tk.END, f"\n{region.title()}:\n")
                self.results_text.insert(tk.END, f"  Status: {region_status}\n",
                                       'pass' if region_status == 'PASS' else 'fail')
                self.results_text.insert(tk.END, f"  Defects: {defect_count}\n")
                
                if region_data.get('failures'):
                    self.results_text.insert(tk.END, "  Failures:\n")
                    for failure in region_data['failures']:
                        self.results_text.insert(tk.END, f"    - {failure}\n", 'fail')
        
        # Defect details
        if results.get('defects'):
            self.results_text.insert(tk.END, "\n\nDEFECT DETAILS\n", 'heading')
            self.results_text.insert(tk.END, "-"*40 + "\n")
            
            for i, defect in enumerate(results['defects'][:10]):  # Show first 10
                self.results_text.insert(tk.END, 
                    f"\n{i+1}. {defect.get('type', 'Unknown')} in {defect.get('region', 'Unknown')}\n")
                self.results_text.insert(tk.END, 
                    f"   Confidence: {defect.get('confidence', 0):.2f}\n")
                self.results_text.insert(tk.END, 
                    f"   Size: {defect.get('area_um2', 0):.1f} \u00b5m\u00b2\n")
        
        # Configure text tags
        self.results_text.tag_config('heading', font=('Arial', 12, 'bold'))
        self.results_text.tag_config('pass', foreground='green', font=('Arial', 10, 'bold'))
        self.results_text.tag_config('fail', foreground='red', font=('Arial', 10, 'bold'))
        
        # Update image displays
        self._update_result_images(results)
        
        # Save to database
        self.database.save_inspection_result(results)
        
        # Reset progress
        self.progress_var.set(0)
    
    def _update_result_images(self, results: Dict[str, Any]):
        """Update result image displays"""
        # Clear axes
        self.ax_processed.clear()
        self.ax_defects.clear()
        self.ax_confidence.clear()
        
        # Display preprocessed image (if available)
        # This would need to be extracted from the processing
        self.ax_processed.text(0.5, 0.5, 'Preprocessed\n(Not available)',
                              ha='center', va='center', transform=self.ax_processed.transAxes)
        self.ax_processed.set_title('Preprocessed')
        self.ax_processed.axis('off')
        
        # Create defect overlay
        if self.current_image is not None:
            overlay = self.current_image.copy()
            
            # Draw defects
            for region_name, region_results in results.get('regions', {}).items():
                for defect in region_results.get('defects', []):
                    # Get defect bounding box
                    if 'bbox_px' in defect:
                        x, y, w, h = defect['bbox_px']
                        color = {
                            'core': (255, 0, 0),      # Red
                            'cladding': (0, 255, 0),  # Green
                            'ferrule': (0, 0, 255)    # Blue
                        }.get(region_name.lower(), (255, 255, 0))
                        
                        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
            
            if len(overlay.shape) == 3:
                self.ax_defects.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            else:
                self.ax_defects.imshow(overlay, cmap='gray')
        
        self.ax_defects.set_title(f'Defects ({len(results.get("defects", []))})')
        self.ax_defects.axis('off')
        
        # Confidence map placeholder
        self.ax_confidence.text(0.5, 0.5, 'Confidence map\n(Not available)',
                               ha='center', va='center', transform=self.ax_confidence.transAxes)
        self.ax_confidence.set_title('Confidence Map')
        self.ax_confidence.axis('off')
        
        self.canvas.draw()
    
    def _update_statistics(self, results: Dict[str, Any]):
        """Update real-time statistics"""
        if 'error' not in results:
            # Update counters
            self.stats['total_processed'] += 1
            
            status = results.get('pass_fail', {}).get('overall', 'UNKNOWN')
            if status == 'PASS':
                self.stats['passed'] += 1
            elif status == 'FAIL':
                self.stats['failed'] += 1
            
            self.stats['total_defects'] += len(results.get('defects', []))
            
            # Processing time
            proc_time = results.get('processing_time', 0)
            self.stats['processing_times'].append(proc_time)
            
            # Keep only last 100 processing times
            if len(self.stats['processing_times']) > 100:
                self.stats['processing_times'] = self.stats['processing_times'][-100:]
        
        # Update labels
        total = self.stats['total_processed']
        if total > 0:
            pass_rate = (self.stats['passed'] / total) * 100
            avg_time = np.mean(self.stats['processing_times'])
        else:
            pass_rate = 0
            avg_time = 0
        
        self.stats_labels['total_processed'].config(text=str(total))
        self.stats_labels['passed'].config(text=str(self.stats['passed']))
        self.stats_labels['failed'].config(text=str(self.stats['failed']))
        self.stats_labels['pass_rate'].config(text=f"{pass_rate:.1f}%")
        self.stats_labels['total_defects'].config(text=str(self.stats['total_defects']))
        self.stats_labels['avg_time'].config(text=f"{avg_time:.2f}s")
        
        # Average quality (would need to track this)
        self.stats_labels['avg_quality'].config(text="N/A")
    
    def _show_config_dialog(self):
        """Show configuration dialog"""
        ConfigDialog(self.root, self.inspector)
    
    def _show_calibration_dialog(self):
        """Show calibration dialog"""
        CalibrationTool(self.root)
    
    def _show_comparison_tool(self):
        """Show method comparison tool"""
        MethodComparisonTool(self.root)
    
    def _show_batch_dialog(self, folder=None):
        """Show batch processing dialog"""
        BatchDialog(self.root, folder)
    
    def _show_statistics(self):
        """Show detailed statistics"""
        StatisticsViewer(self.root, self.database)
    
    def _export_results(self):
        """Export current results"""
        if self.current_results is None:
            messagebox.showwarning("No Results", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv")
            ]
        )
        
        if file_path:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.json':
                with open(file_path, 'w') as f:
                    json.dump(self.current_results, f, indent=2, default=str)
            
            elif ext == '.xlsx':
                # Convert to DataFrame and export
                df = pd.DataFrame([self.current_results])
                df.to_excel(file_path, index=False)
            
            elif ext == '.csv':
                # Flatten results and export
                flat_results = self._flatten_results(self.current_results)
                df = pd.DataFrame([flat_results])
                df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Results exported to {file_path}")
    
    def _flatten_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested results for CSV export"""
        flat = {
            'status': results.get('pass_fail', {}).get('overall', 'UNKNOWN'),
            'quality_index': results.get('quality_metrics', {}).get('surface_quality_index', 0),
            'total_defects': len(results.get('defects', [])),
            'processing_time': results.get('processing_time', 0)
        }
        
        # Add regional counts
        for region in ['core', 'cladding', 'ferrule']:
            region_defects = len([d for d in results.get('defects', []) 
                                if d.get('region', '').lower() == region])
            flat[f'{region}_defects'] = region_defects
        
        return flat
    
    def run(self):
        """Run the monitor"""
        self.root.mainloop()
    
    def _setup_gui(self):
        """Setup the GUI components"""
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self._open_image)
        file_menu.add_command(label="Open Folder", command=self._open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configuration", command=self._show_config_dialog)
        settings_menu.add_command(label="Calibration", command=self._show_calibration_dialog)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Method Comparison", command=self._show_comparison_tool)
        tools_menu.add_command(label="Batch Processing", command=self._show_batch_dialog)
        tools_menu.add_command(label="Statistics", command=self._show_statistics)
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Image display
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image display area
        self.image_frame = ttk.LabelFrame(left_panel, text="Image Display")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure for image display
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_original = self.fig.add_subplot(221)
        self.ax_processed = self.fig.add_subplot(222)
        self.ax_defects = self.fig.add_subplot(223)
        self.ax_confidence = self.fig.add_subplot(224)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_panel = ttk.Frame(left_panel)
        control_panel.pack(fill=tk.X, padx=5, pady=5)
        
        self.process_btn = ttk.Button(control_panel, text="Process Image", 
                                     command=self._process_current_image)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.auto_process_var = tk.BooleanVar()
        self.auto_process_cb = ttk.Checkbutton(control_panel, text="Auto Process", 
                                              variable=self.auto_process_var)
        self.auto_process_cb.pack(side=tk.LEFT, padx=5)
        
        # Processing profile
        ttk.Label(control_panel, text="Profile:").pack(side=tk.LEFT, padx=5)
        self.profile_var = tk.StringVar(value="balanced")
        profile_combo = ttk.Combobox(control_panel, textvariable=self.profile_var,
                                    values=["fast", "balanced", "comprehensive"],
                                    width=15, state="readonly")
        profile_combo.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Results and statistics
        right_panel = ttk.Frame(main_container, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        right_panel.pack_propagate(False)
        
        # Results display
        results_frame = ttk.LabelFrame(right_panel, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results text
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=20)
        results_scroll = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=results_scroll.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Real-time statistics
        stats_frame = ttk.LabelFrame(right_panel, text="Real-time Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Statistics labels
        self.stats_labels = {}
        stats_to_show = [
            ('total_processed', 'Total Processed:'),
            ('passed', 'Passed:'),
            ('failed', 'Failed:'),
            ('pass_rate', 'Pass Rate:'),
            ('avg_quality', 'Avg Quality:'),
            ('total_defects', 'Total Defects:'),
            ('avg_time', 'Avg Time:')
        ]
        
        for i, (key, label) in enumerate(stats_to_show):
            row = i // 2
            col = i % 2
            
            ttk.Label(stats_frame, text=label).grid(row=row, column=col*2, 
                                                    sticky=tk.W, padx=5, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="0")
            self.stats_labels[key].grid(row=row, column=col*2+1, 
                                       sticky=tk.W, padx=5, pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(right_panel, variable=self.progress_var,
                                          maximum=100, length=380)
        self.progress_bar.pack(padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_inspector(self):
        """Setup the inspection system"""
        config = self._get_current_config()
        self.inspector = CompleteFiberInspectionSystem(config)
    
    def _get_current_config(self) -> dict:
        """Get current configuration based on GUI settings"""
        profile = self.profile_var.get()
        
        config = {
            "defect_detection": {
                "use_all_methods": profile == "comprehensive"
            },
            "output": {
                "save_all_intermediate": False,
                "generate_report": False
            }
        }
        
        if profile == "fast":
            config["defect_detection"]["method_groups"] = {
                "statistical": {"enabled": True, "methods": ["zscore"]},
                "morphological": {"enabled": True, "methods": ["tophat"]}
            }
        elif profile == "balanced":
            config["defect_detection"]["method_groups"] = {
                "statistical": {"enabled": True, "methods": ["zscore", "mad"]},
                "morphological": {"enabled": True, "methods": ["tophat", "blackhat"]},
                "ml": {"enabled": True, "methods": ["isolation_forest"]}
            }
        
        return config
    
    def _processing_worker(self):
        """Worker thread for image processing"""
        while True:
            try:
                # Get image from queue
                image_path = self.processing_queue.get(timeout=0.1)
                
                if image_path is None:
                    break
                
                # Process image
                start_time = time.time()
                results = self.inspector.inspect_fiber(image_path)
                processing_time = time.time() - start_time
                
                # Add processing time to results
                results['processing_time'] = processing_time
                
                # Put results in queue
                self.results_queue.put(results)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                self.results_queue.put({'error': str(e)})
    
    def _update_gui(self):
        """Update GUI with latest results"""
        try:
            # Check for new results
            while not self.results_queue.empty():
                results = self.results_queue.get_nowait()
                self._display_results(results)
                self._update_statistics(results)
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self._update_gui)
    
    def _open_image(self):
        """Open and display an image"""
        file_path = filedialog.askopenfilename(
            title="Select Fiber Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            self._display_image()
            
            if self.auto_process_var.get():
                self._process_current_image()
    
    def _open_folder(self):
        """Open folder for batch processing"""
        folder_path = filedialog.askdirectory(title="Select Folder")
        
        if folder_path:
            # Show batch dialog with selected folder
            self._show_batch_dialog(folder_path)
    
    def _display_image(self):
        """Display current image"""
        if self.current_image is None:
            return
        
        # Clear all axes
        for ax in [self.ax_original, self.ax_processed, self.ax_defects, self.ax_confidence]:
            ax.clear()
        
        # Display original
        if len(self.current_image.shape) == 3:
            self.ax_original.imshow(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
        else:
            self.ax_original.imshow(self.current_image, cmap='gray')
        self.ax_original.set_title('Original')
        self.ax_original.axis('off')
        
        # Clear other plots
        self.ax_processed.text(0.5, 0.5, 'Processed image\nwill appear here',
                              ha='center', va='center', transform=self.ax_processed.transAxes)
        self.ax_processed.set_title('Preprocessed')
        self.ax_processed.axis('off')
        
        self.ax_defects.text(0.5, 0.5, 'Defect mask\nwill appear here',
                            ha='center', va='center', transform=self.ax_defects.transAxes)
        self.ax_defects.set_title('Defects')
        self.ax_defects.axis('off')
        
        self.ax_confidence.text(0.5, 0.5, 'Confidence map\nwill appear here',
                               ha='center', va='center', transform=self.ax_confidence.transAxes)
        self.ax_confidence.set_title('Confidence')
        self.ax_confidence.axis('off')
        
        self.canvas.draw()
    
    def _process_current_image(self):
        """Process the current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please open an image first")
            return
        
        if self.is_processing:
            messagebox.showinfo("Processing", "Already processing an image")
            return
        
        # Save temporary image
        temp_path = "temp_realtime_image.png"
        cv2.imwrite(temp_path, self.current_image)
        
        # Add to processing queue
        self.processing_queue.put(temp_path)
        self.is_processing = True
        self.status_var.set("Processing...")
        self.progress_var.set(50)
    
    def _display_results(self, results: Dict[str, Any]):
        """Display processing results"""
        self.current_results = results
        self.is_processing = False
        self.progress_var.set(100)
        
        if 'error' in results:
            self.status_var.set(f"Error: {results['error']}")
            messagebox.showerror("Processing Error", results['error'])
            return
        
        # Update status
        status = results.get('pass_fail', {}).get('overall', 'UNKNOWN')
        quality = results.get('quality_metrics', {}).get('surface_quality_index', 0)
        self.status_var.set(f"Status: {status} | Quality: {quality:.1f}")
        
        # Display results text
        self.results_text.delete(1.0, tk.END)
        
        # Overall results
        self.results_text.insert(tk.END, "INSPECTION RESULTS\n", 'heading')
        self.results_text.insert(tk.END, "="*40 + "\n\n")
        
        self.results_text.insert(tk.END, f"Status: {status}\n", 
                                'pass' if status == 'PASS' else 'fail')
        self.results_text.insert(tk.END, f"Quality Index: {quality:.1f}/100\n")
        self.results_text.insert(tk.END, f"Total Defects: {len(results.get('defects', []))}\n")
        self.results_text.insert(tk.END, f"Processing Time: {results.get('processing_time', 0):.2f}s\n\n")
        
        # Regional results
        self.results_text.insert(tk.END, "REGIONAL ANALYSIS\n", 'heading')
        self.results_text.insert(tk.END, "-"*40 + "\n")
        
        pass_fail = results.get('pass_fail', {})
        for region in ['core', 'cladding', 'ferrule']:
            region_data = pass_fail.get('by_region', {}).get(region, {})
            if region_data:
                region_status = region_data.get('status', 'N/A')
                defect_count = region_data.get('defect_count', 0)
                
                self.results_text.insert(tk.END, f"\n{region.title()}:\n")
                self.results_text.insert(tk.END, f"  Status: {region_status}\n",
                                       'pass' if region_status == 'PASS' else 'fail')
                self.results_text.insert(tk.END, f"  Defects: {defect_count}\n")
                
                if region_data.get('failures'):
                    self.results_text.insert(tk.END, "  Failures:\n")
                    for failure in region_data['failures']:
                        self.results_text.insert(tk.END, f"    - {failure}\n", 'fail')
        
        # Defect details
        if results.get('defects'):
            self.results_text.insert(tk.END, "\n\nDEFECT DETAILS\n", 'heading')
            self.results_text.insert(tk.END, "-"*40 + "\n")
            
            for i, defect in enumerate(results['defects'][:10]):  # Show first 10
                self.results_text.insert(tk.END, 
                    f"\n{i+1}. {defect.get('type', 'Unknown')} in {defect.get('region', 'Unknown')}\n")
                self.results_text.insert(tk.END, 
                    f"   Confidence: {defect.get('confidence', 0):.2f}\n")
                self.results_text.insert(tk.END, 
                    f"   Size: {defect.get('area_um2', 0):.1f} μm²\n")
        
        # Configure text tags
        self.results_text.tag_config('heading', font=('Arial', 12, 'bold'))
        self.results_text.tag_config('pass', foreground='green', font=('Arial', 10, 'bold'))
        self.results_text.tag_config('fail', foreground='red', font=('Arial', 10, 'bold'))
        
        # Update image displays
        self._update_result_images(results)
        
        # Save to database
        self.database.save_inspection_result(results)
        
        # Reset progress
        self.progress_var.set(0)
    
    def _update_result_images(self, results: Dict[str, Any]):
        """Update result image displays"""
        # Clear axes
        self.ax_processed.clear()
        self.ax_defects.clear()
        self.ax_confidence.clear()
        
        # Display preprocessed image (if available)
        # This would need to be extracted from the processing
        self.ax_processed.text(0.5, 0.5, 'Preprocessed\n(Not available)',
                              ha='center', va='center', transform=self.ax_processed.transAxes)
        self.ax_processed.set_title('Preprocessed')
        self.ax_processed.axis('off')
        
        # Create defect overlay
        if self.current_image is not None:
            overlay = self.current_image.copy()
            
            # Draw defects
            for region_name, region_results in results.get('regions', {}).items():
                for defect in region_results.get('defects', []):
                    # Get defect bounding box
                    if 'bbox_px' in defect:
                        x, y, w, h = defect['bbox_px']
                        color = {
                            'core': (255, 0, 0),      # Red
                            'cladding': (0, 255, 0),  # Green
                            'ferrule': (0, 0, 255)    # Blue
                        }.get(region_name.lower(), (255, 255, 0))
                        
                        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
            
            if len(overlay.shape) == 3:
                self.ax_defects.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            else:
                self.ax_defects.imshow(overlay, cmap='gray')
        
        self.ax_defects.set_title(f'Defects ({len(results.get("defects", []))})')
        self.ax_defects.axis('off')
        
        # Confidence map placeholder
        self.ax_confidence.text(0.5, 0.5, 'Confidence map\n(Not available)',
                               ha='center', va='center', transform=self.ax_confidence.transAxes)
        self.ax_confidence.set_title('Confidence Map')
        self.ax_confidence.axis('off')
        
        self.canvas.draw()
    
    def _update_statistics(self, results: Dict[str, Any]):
        """Update real-time statistics"""
        if 'error' not in results:
            # Update counters
            self.stats['total_processed'] += 1
            
            status = results.get('pass_fail', {}).get('overall', 'UNKNOWN')
            if status == 'PASS':
                self.stats['passed'] += 1
            elif status == 'FAIL':
                self.stats['failed'] += 1
            
            self.stats['total_defects'] += len(results.get('defects', []))
            
            # Processing time
            proc_time = results.get('processing_time', 0)
            self.stats['processing_times'].append(proc_time)
            
            # Keep only last 100 processing times
            if len(self.stats['processing_times']) > 100:
                self.stats['processing_times'] = self.stats['processing_times'][-100:]
        
        # Update labels
        total = self.stats['total_processed']
        if total > 0:
            pass_rate = (self.stats['passed'] / total) * 100
            avg_time = np.mean(self.stats['processing_times'])
        else:
            pass_rate = 0
            avg_time = 0
        
        self.stats_labels['total_processed'].config(text=str(total))
        self.stats_labels['passed'].config(text=str(self.stats['passed']))
        self.stats_labels['failed'].config(text=str(self.stats['failed']))
        self.stats_labels['pass_rate'].config(text=f"{pass_rate:.1f}%")
        self.stats_labels['total_defects'].config(text=str(self.stats['total_defects']))
        self.stats_labels['avg_time'].config(text=f"{avg_time:.2f}s")
        
        # Average quality (would need to track this)
        self.stats_labels['avg_quality'].config(text="N/A")
    
    def _show_config_dialog(self):
        """Show configuration dialog"""
        ConfigDialog(self.root, self.inspector)
    
    def _show_calibration_dialog(self):
        """Show calibration dialog"""
        CalibrationTool(self.root)
    
    def _show_comparison_tool(self):
        """Show method comparison tool"""
        MethodComparisonTool(self.root)
    
    def _show_batch_dialog(self, folder=None):
        """Show batch processing dialog"""
        BatchDialog(self.root, folder)
    
    def _show_statistics(self):
        """Show detailed statistics"""
        StatisticsViewer(self.root, self.database)
    
    def _export_results(self):
        """Export current results"""
        if self.current_results is None:
            messagebox.showwarning("No Results", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv")
            ]
        )
        
        if file_path:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.json':
                with open(file_path, 'w') as f:
                    json.dump(self.current_results, f, indent=2, default=str)
            
            elif ext == '.xlsx':
                # Convert to DataFrame and export
                df = pd.DataFrame([self.current_results])
                df.to_excel(file_path, index=False)
            
            elif ext == '.csv':
                # Flatten results and export
                flat_results = self._flatten_results(self.current_results)
                df = pd.DataFrame([flat_results])
                df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Results exported to {file_path}")
    
    def _flatten_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested results for CSV export"""
        flat = {
            'status': results.get('pass_fail', {}).get('overall', 'UNKNOWN'),
            'quality_index': results.get('quality_metrics', {}).get('surface_quality_index', 0),
            'total_defects': len(results.get('defects', [])),
            'processing_time': results.get('processing_time', 0)
        }
        
        # Add regional counts
        for region in ['core', 'cladding', 'ferrule']:
            region_defects = len([d for d in results.get('defects', []) 
                                if d.get('region', '').lower() == region])
            flat[f'{region}_defects'] = region_defects
        
        return flat
    
    def run(self):
        """Run the monitor"""
        self.root.mainloop()


class ConfigDialog(tk.Toplevel):
    """Configuration dialog for inspection parameters"""
    
    def __init__(self, parent, inspector):
        super().__init__(parent)
        self.inspector = inspector
        self.title("Configuration Settings")
        self.geometry("800x600")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Preprocessing tab
        prep_frame = ttk.Frame(notebook)
        notebook.add(prep_frame, text="Preprocessing")
        self._create_preprocessing_tab(prep_frame)
        
        # Detection methods tab
        methods_frame = ttk.Frame(notebook)
        notebook.add(methods_frame, text="Detection Methods")
        self._create_methods_tab(methods_frame)
        
        # Thresholds tab
        thresh_frame = ttk.Frame(notebook)
        notebook.add(thresh_frame, text="Thresholds")
        self._create_thresholds_tab(thresh_frame)
        
        # Pass/Fail criteria tab
        criteria_frame = ttk.Frame(notebook)
        notebook.add(criteria_frame, text="Pass/Fail Criteria")
        self._create_criteria_tab(criteria_frame)
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save", command=self._save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Load", command=self._load_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Apply", command=self._apply_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _create_preprocessing_tab(self, parent):
        """Create preprocessing configuration tab"""
        # Illumination correction
        ttk.Label(parent, text="Illumination Correction", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, padx=10, pady=10)
        
        self.illum_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Enable illumination correction", 
                       variable=self.illum_var).grid(row=1, column=0, padx=20, pady=5, sticky=tk.W)
        
        # Noise reduction
        ttk.Label(parent, text="Noise Reduction", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=10, pady=10)
        
        self.noise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Enable noise reduction", 
                       variable=self.noise_var).grid(row=3, column=0, padx=20, pady=5, sticky=tk.W)
        
        # CLAHE parameters
        ttk.Label(parent, text="CLAHE Parameters", font=('Arial', 10, 'bold')).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, padx=10, pady=10)
        
        ttk.Label(parent, text="Clip Limit:").grid(row=5, column=0, padx=20, pady=5, sticky=tk.W)
        self.clahe_clip_var = tk.DoubleVar(value=3.0)
        ttk.Scale(parent, from_=0.5, to=10.0, variable=self.clahe_clip_var,
                 orient=tk.HORIZONTAL, length=200).grid(row=5, column=1, padx=10, pady=5)
        ttk.Label(parent, textvariable=self.clahe_clip_var).grid(row=5, column=2, padx=5, pady=5)
    
    def _create_methods_tab(self, parent):
        """Create detection methods configuration tab"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Method groups
        self.method_vars = {}
        method_groups = {
            "Statistical": ["zscore", "mad", "iqr", "grubbs", "dixon", "chauvenet"],
            "Spatial": ["lbp", "glcm", "fractal"],
            "Frequency": ["fft", "wavelet", "gabor"],
            "Morphological": ["tophat", "blackhat", "gradient"],
            "Machine Learning": ["isolation_forest", "one_class_svm", "dbscan"],
            "Physics-based": ["diffraction", "scattering", "interference"]
        }
        
        row = 0
        for group, methods in method_groups.items():
            ttk.Label(scrollable_frame, text=group, font=('Arial', 10, 'bold')).grid(
                row=row, column=0, columnspan=3, sticky=tk.W, padx=10, pady=10)
            row += 1
            
            for i, method in enumerate(methods):
                var = tk.BooleanVar(value=True)
                self.method_vars[f"{group.lower()}_{method}"] = var
                
                col = i % 3
                method_row = row + i // 3
                
                ttk.Checkbutton(scrollable_frame, text=method, variable=var).grid(
                    row=method_row, column=col, padx=20, pady=2, sticky=tk.W)
            
            row = method_row + 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_thresholds_tab(self, parent):
        """Create thresholds configuration tab"""
        # Statistical thresholds
        ttk.Label(parent, text="Statistical Thresholds", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, padx=10, pady=10)
        
        self.threshold_vars = {}
        
        thresholds = [
            ("Z-score threshold:", "zscore_threshold", 3.0, 1.0, 5.0),
            ("MAD threshold:", "mad_threshold", 3.5, 1.0, 5.0),
            ("IQR multiplier:", "iqr_multiplier", 1.5, 0.5, 3.0),
            ("Min contrast:", "min_contrast", 10.0, 5.0, 50.0),
            ("Min defect area (px):", "min_defect_area", 3, 1, 20)
        ]
        
        for i, (label, key, default, min_val, max_val) in enumerate(thresholds):
            ttk.Label(parent, text=label).grid(row=i+1, column=0, padx=20, pady=5, sticky=tk.W)
            
            if isinstance(default, int):
                var = tk.IntVar(value=default)
            else:
                var = tk.DoubleVar(value=default)
            
            self.threshold_vars[key] = var
            
            scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var,
                            orient=tk.HORIZONTAL, length=200)
            scale.grid(row=i+1, column=1, padx=10, pady=5)
            
            ttk.Label(parent, textvariable=var).grid(row=i+1, column=2, padx=5, pady=5)
    
    def _create_criteria_tab(self, parent):
        """Create pass/fail criteria configuration tab"""
        # Region-specific criteria
        self.criteria_vars = {}
        
        regions = ["Core", "Cladding", "Ferrule"]
        criteria = [
            ("Max defects:", "max_defects", [0, 5, 10]),
            ("Max scratch length (μm):", "max_scratch_length", [0, 50, 100]),
            ("Max pit diameter (μm):", "max_pit_diameter", [0, 10, 20])
        ]
        
        # Create headers
        ttk.Label(parent, text="Region", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, padx=10, pady=10)
        
        for i, region in enumerate(regions):
            ttk.Label(parent, text=region, font=('Arial', 10, 'bold')).grid(
                row=0, column=i+1, padx=10, pady=10)
        
        # Create criteria inputs
        for i, (label, key, defaults) in enumerate(criteria):
            ttk.Label(parent, text=label).grid(row=i+1, column=0, padx=10, pady=5, sticky=tk.W)
            
            for j, region in enumerate(regions):
                var = tk.IntVar(value=defaults[j])
                self.criteria_vars[f"{region.lower()}_{key}"] = var
                
                entry = ttk.Entry(parent, textvariable=var, width=10)
                entry.grid(row=i+1, column=j+1, padx=10, pady=5)
    
    def _save_config(self):
        """Save configuration to file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json")]
        )
        
        if file_path:
            config = self._gather_config()
            
            ext = Path(file_path).suffix.lower()
            if ext == '.yaml':
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            else:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            messagebox.showinfo("Save Complete", f"Configuration saved to {file_path}")
    
    def _load_config(self):
        """Load configuration from file"""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("Config files", "*.yaml *.json"), ("All files", "*.*")]
        )
        
        if file_path:
            ext = Path(file_path).suffix.lower()
            
            try:
                if ext == '.yaml':
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    with open(file_path, 'r') as f:
                        config = json.load(f)
                
                self._apply_loaded_config(config)
                messagebox.showinfo("Load Complete", "Configuration loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load configuration: {str(e)}")
    
    def _gather_config(self) -> dict:
        """Gather current configuration from GUI"""
        config = {
            "preprocessing": {
                "illumination_correction": self.illum_var.get(),
                "noise_reduction": self.noise_var.get(),
                "clahe_clip_limit": self.clahe_clip_var.get()
            },
            "methods": {},
            "thresholds": {},
            "criteria": {}
        }
        
        # Gather method selections
        for key, var in self.method_vars.items():
            config["methods"][key] = var.get()
        
        # Gather thresholds
        for key, var in self.threshold_vars.items():
            config["thresholds"][key] = var.get()
        
        # Gather criteria
        for key, var in self.criteria_vars.items():
            config["criteria"][key] = var.get()
        
        return config
    
    def _apply_loaded_config(self, config: dict):
        """Apply loaded configuration to GUI"""
        # Apply preprocessing settings
        if "preprocessing" in config:
            prep = config["preprocessing"]
            self.illum_var.set(prep.get("illumination_correction", True))
            self.noise_var.set(prep.get("noise_reduction", True))
            self.clahe_clip_var.set(prep.get("clahe_clip_limit", 3.0))
        
        # Apply method selections
        if "methods" in config:
            for key, value in config["methods"].items():
                if key in self.method_vars:
                    self.method_vars[key].set(value)
        
        # Apply thresholds
        if "thresholds" in config:
            for key, value in config["thresholds"].items():
                if key in self.threshold_vars:
                    self.threshold_vars[key].set(value)
        
        # Apply criteria
        if "criteria" in config:
            for key, value in config["criteria"].items():
                if key in self.criteria_vars:
                    self.criteria_vars[key].set(value)
    
    def _apply_config(self):
        """Apply configuration to inspector"""
        config = self._gather_config()
        # This would update the inspector configuration
        messagebox.showinfo("Configuration Applied", "Settings have been applied")
        self.destroy()


class CalibrationTool(tk.Toplevel):
    """Calibration tool for scale and system validation"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Calibration Tool")
        self.geometry("900x700")
        
        # Variables
        self.calibration_image = None
        self.scale_factor = None
        self.points = []
        self.current_line = None
        
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for image
        self.canvas = tk.Canvas(left_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        
        # Right panel - Controls
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)
        
        # Load image button
        ttk.Button(right_frame, text="Load Calibration Image",
                  command=self._load_image).pack(pady=10)
        
        # Calibration mode
        ttk.Label(right_frame, text="Calibration Mode:", 
                 font=('Arial', 10, 'bold')).pack(pady=(20, 5))
        
        self.mode_var = tk.StringVar(value="scale")
        ttk.Radiobutton(right_frame, text="Scale Calibration", 
                       variable=self.mode_var, value="scale").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(right_frame, text="Defect Standards", 
                       variable=self.mode_var, value="standards").pack(anchor=tk.W, padx=20)
        
        # Scale calibration
        scale_frame = ttk.LabelFrame(right_frame, text="Scale Calibration")
        scale_frame.pack(fill=tk.X, pady=20, padx=10)
        
        ttk.Label(scale_frame, text="Known Distance:").grid(row=0, column=0, padx=5, pady=5)
        self.known_distance_var = tk.DoubleVar(value=125.0)
        ttk.Entry(scale_frame, textvariable=self.known_distance_var, width=10).grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Label(scale_frame, text="μm").grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(scale_frame, text="Measured Pixels:").grid(row=1, column=0, padx=5, pady=5)
        self.measured_pixels_var = tk.StringVar(value="0")
        ttk.Label(scale_frame, textvariable=self.measured_pixels_var).grid(
            row=1, column=1, padx=5, pady=5)
        
        ttk.Label(scale_frame, text="Scale Factor:").grid(row=2, column=0, padx=5, pady=5)
        self.scale_factor_var = tk.StringVar(value="0.00")
        ttk.Label(scale_frame, textvariable=self.scale_factor_var).grid(
            row=2, column=1, padx=5, pady=5)
        ttk.Label(scale_frame, text="μm/px").grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Button(scale_frame, text="Calculate Scale",
                  command=self._calculate_scale).grid(row=3, column=0, columnspan=3, pady=10)
        
        # Validation
        validation_frame = ttk.LabelFrame(right_frame, text="Validation")
        validation_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Button(validation_frame, text="Validate with Standards",
                  command=self._validate_standards).pack(pady=10)
        
        self.validation_text = tk.Text(validation_frame, height=8, width=35)
        self.validation_text.pack(padx=5, pady=5)
        
        # Save calibration
        ttk.Button(right_frame, text="Save Calibration",
                  command=self._save_calibration).pack(pady=20)
    
    def _load_image(self):
        """Load calibration image"""
        file_path = filedialog.askopenfilename(
            title="Select Calibration Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            # Load and display image
            self.calibration_image = cv2.imread(file_path)
            self._display_image()
    
    def _display_image(self):
        """Display calibration image on canvas"""
        if self.calibration_image is None:
            return
        
        # Convert to RGB
        if len(self.calibration_image.shape) == 3:
            image_rgb = cv2.cvtColor(self.calibration_image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(self.calibration_image, cv2.COLOR_GRAY2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            h, w = image_rgb.shape[:2]
            scale = min(canvas_width/w, canvas_height/h, 1.0)
            
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h))
            
            self.display_scale = scale
        else:
            self.display_scale = 1.0
        
        # Convert to PhotoImage
        from PIL import Image, ImageTk
        image_pil = Image.fromarray(image_rgb)
        self.photo = ImageTk.PhotoImage(image_pil)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def _on_click(self, event):
        """Handle mouse click on canvas"""
        if self.calibration_image is None:
            return
        
        if self.mode_var.get() == "scale":
            # Scale calibration mode
            if len(self.points) < 2:
                # Add point
                self.points.append((event.x, event.y))
                
                # Draw point
                self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3,
                                      fill='red', tags='calibration')
                
                if len(self.points) == 2:
                    # Draw line
                    self.canvas.create_line(self.points[0][0], self.points[0][1],
                                          self.points[1][0], self.points[1][1],
                                          fill='red', width=2, tags='calibration')
                    
                    # Calculate distance
                    dx = self.points[1][0] - self.points[0][0]
                    dy = self.points[1][1] - self.points[0][1]
                    distance = np.sqrt(dx**2 + dy**2) / self.display_scale
                    
                    self.measured_pixels_var.set(f"{distance:.1f}")
            else:
                # Reset
                self.points = []
                self.canvas.delete('calibration')
                self.measured_pixels_var.set("0")
    
    def _on_motion(self, event):
        """Handle mouse motion on canvas"""
        if self.calibration_image is None:
            return
        
        if self.mode_var.get() == "scale" and len(self.points) == 1:
            # Draw temporary line
            if self.current_line:
                self.canvas.delete(self.current_line)
            
            self.current_line = self.canvas.create_line(
                self.points[0][0], self.points[0][1],
                event.x, event.y,
                fill='yellow', width=1, dash=(2, 2)
            )
    
    def _calculate_scale(self):
        """Calculate scale factor"""
        try:
            measured_pixels = float(self.measured_pixels_var.get())
            known_distance = self.known_distance_var.get()
            
            if measured_pixels > 0:
                self.scale_factor = known_distance / measured_pixels
                self.scale_factor_var.set(f"{self.scale_factor:.3f}")
            else:
                messagebox.showwarning("Invalid Measurement", 
                                     "Please measure a distance on the image first")
        except ValueError:
            messagebox.showerror("Error", "Invalid values entered")
    
    def _validate_standards(self):
        """Validate calibration with standard samples"""
        if self.scale_factor is None:
            messagebox.showwarning("No Calibration", "Please calibrate scale first")
            return
        
        # Simulate validation with known standards
        self.validation_text.delete(1.0, tk.END)
        self.validation_text.insert(tk.END, "CALIBRATION VALIDATION\n")
        self.validation_text.insert(tk.END, "="*30 + "\n\n")
        
        # Test measurements
        test_standards = [
            ("Cladding diameter", 125.0, 125.3),
            ("Core diameter", 9.0, 8.8),
            ("Test scratch", 25.0, 24.7),
            ("Test pit", 5.0, 5.2)
        ]
        
        self.validation_text.insert(tk.END, "Standard\tExpected\tMeasured\tError\n")
        self.validation_text.insert(tk.END, "-"*50 + "\n")
        
        total_error = 0
        for name, expected, measured in test_standards:
            error = abs(measured - expected) / expected * 100
            total_error += error
            
            self.validation_text.insert(tk.END, 
                f"{name}\t{expected:.1f}μm\t{measured:.1f}μm\t{error:.1f}%\n")
        
        avg_error = total_error / len(test_standards)
        self.validation_text.insert(tk.END, "\n" + "-"*50 + "\n")
        self.validation_text.insert(tk.END, f"Average Error: {avg_error:.2f}%\n")
        
        if avg_error < 2.0:
            self.validation_text.insert(tk.END, "\nCalibration PASSED ✓\n", 'pass')
        else:
            self.validation_text.insert(tk.END, "\nCalibration FAILED ✗\n", 'fail')
        
        # Configure tags
        self.validation_text.tag_config('pass', foreground='green', font=('Arial', 10, 'bold'))
        self.validation_text.tag_config('fail', foreground='red', font=('Arial', 10, 'bold'))
    
    def _save_calibration(self):
        """Save calibration data"""
        if self.scale_factor is None:
            messagebox.showwarning("No Calibration", "Please calibrate scale first")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Calibration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            calibration_data = {
                "scale_factor_um_per_px": self.scale_factor,
                "calibration_date": datetime.now().isoformat(),
                "known_distance_um": self.known_distance_var.get(),
                "measured_pixels": float(self.measured_pixels_var.get())
            }
            
            with open(file_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            messagebox.showinfo("Save Complete", f"Calibration saved to {file_path}")


class MethodComparisonTool(tk.Toplevel):
    """Tool for comparing different detection methods"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Detection Method Comparison")
        self.geometry("1200x800")
        
        # Test image
        self.test_image = None
        self.comparison_results = {}
        
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Load Test Image",
                  command=self._load_test_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Run Comparison",
                  command=self._run_comparison).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Export Results",
                  command=self._export_results).pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                          maximum=100, length=300)
        self.progress_bar.pack(side=tk.LEFT, padx=20)
        
        # Results area
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Comparison grid tab
        grid_frame = ttk.Frame(self.notebook)
        self.notebook.add(grid_frame, text="Visual Comparison")
        self._create_comparison_grid(grid_frame)
        
        # Metrics tab
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="Performance Metrics")
        self._create_metrics_tab(metrics_frame)
        
        # Analysis tab
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Statistical Analysis")
        self._create_analysis_tab(analysis_frame)
    
    def _create_comparison_grid(self, parent):
        """Create visual comparison grid"""
        # Create scrollable canvas
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.grid_frame = ttk.Frame(canvas)
        
        self.grid_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_metrics_tab(self, parent):
        """Create performance metrics tab"""
        # Create figure for metrics
        self.metrics_fig = Figure(figsize=(10, 6), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=parent)
        self.metrics_canvas.draw()
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_analysis_tab(self, parent):
        """Create statistical analysis tab"""
        # Text widget for analysis results
        self.analysis_text = tk.Text(parent, wrap=tk.WORD)
        scroll = ttk.Scrollbar(parent, command=self.analysis_text.yview)
        self.analysis_text.config(yscrollcommand=scroll.set)
        
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _load_test_image(self):
        """Load test image for comparison"""
        file_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.test_image = cv2.imread(file_path)
            messagebox.showinfo("Image Loaded", "Test image loaded successfully")
    
    def _run_comparison(self):
        """Run method comparison"""
        if self.test_image is None:
            messagebox.showwarning("No Image", "Please load a test image first")
            return
        
        # Define methods to compare
        methods = [
            ("DO2MR", self._run_do2mr),
            ("LEI", self._run_lei),
            ("Statistical", self._run_statistical),
            ("Morphological", self._run_morphological),
            ("Machine Learning", self._run_ml),
            ("Hybrid", self._run_hybrid)
        ]
        
        # Clear previous results
        self.comparison_results = {}
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        
        # Run each method
        total_methods = len(methods)
        for i, (method_name, method_func) in enumerate(methods):
            self.progress_var.set((i / total_methods) * 100)
            self.update()
            
            start_time = time.time()
            result = method_func()
            end_time = time.time()
            
            self.comparison_results[method_name] = {
                'mask': result,
                'time': end_time - start_time,
                'defect_count': self._count_defects(result)
            }
        
        self.progress_var.set(100)
        
        # Display results
        self._display_comparison_results()
        self._display_metrics()
        self._display_analysis()
    
    def _run_do2mr(self) -> np.ndarray:
        """Run DO2MR detection"""
        # Simplified DO2MR implementation
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
        
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        max_filtered = cv2.dilate(gray, kernel)
        min_filtered = cv2.erode(gray, kernel)
        residual = cv2.subtract(max_filtered, min_filtered)
        
        _, mask = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
    
    def _run_lei(self) -> np.ndarray:
        """Run LEI detection"""
        # Simplified LEI implementation
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
        
        # Apply directional filters
        scratch_map = np.zeros_like(gray)
        
        for angle in range(0, 180, 15):
            kernel = self._create_line_kernel(15, angle)
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            scratch_map = np.maximum(scratch_map, response)
        
        # Normalize and threshold
        scratch_map = cv2.normalize(scratch_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, mask = cv2.threshold(scratch_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
    
    def _run_statistical(self) -> np.ndarray:
        """Run statistical detection"""
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
        
        # Z-score based detection
        mean = np.mean(gray)
        std = np.std(gray)
        z_scores = np.abs((gray - mean) / std)
        
        mask = (z_scores > 3).astype(np.uint8) * 255
        
        return mask
    
    def _run_morphological(self) -> np.ndarray:
        """Run morphological detection"""
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
        
        # Top-hat and black-hat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine
        combined = cv2.add(tophat, blackhat)
        _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
    
    def _run_ml(self) -> np.ndarray:
        """Run machine learning detection"""
        # Simplified ML detection (would use actual ML in practice)
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
        
        # Extract simple features
        edges = cv2.Canny(gray, 50, 150)
        
        # Simple anomaly detection
        kernel = np.ones((5, 5), np.float32) / 25
        filtered = cv2.filter2D(gray, -1, kernel)
        diff = cv2.absdiff(gray, filtered)
        
        _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _run_hybrid(self) -> np.ndarray:
        """Run hybrid detection (combination of methods)"""
        # Combine multiple methods
        do2mr = self._run_do2mr()
        morph = self._run_morphological()
        
        # Voting
        combined = cv2.add(do2mr // 2, morph // 2)
        _, mask = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _create_line_kernel(self, length: int, angle: float) -> np.ndarray:
        """Create line kernel for LEI"""
        kernel = np.zeros((length, length), dtype=np.float32)
        center = length // 2
        
        angle_rad = np.radians(angle)
        for i in range(length):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1.0
        
        return kernel / np.sum(kernel)
    
    def _count_defects(self, mask: np.ndarray) -> int:
        """Count defects in mask"""
        if mask is None:
            return 0
        
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return num_labels - 1  # Exclude background
    
    def _display_comparison_results(self):
        """Display visual comparison results"""
        # Display original
        fig = Figure(figsize=(3, 3), dpi=80)
        ax = fig.add_subplot(111)
        
        if len(self.test_image.shape) == 3:
            ax.imshow(cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(self.test_image, cmap='gray')
        
        ax.set_title('Original')
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, master=self.grid_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        
        # Display each method result
        col = 1
        for method_name, result in self.comparison_results.items():
            fig = Figure(figsize=(3, 3), dpi=80)
            ax = fig.add_subplot(111)
            
            ax.imshow(result['mask'], cmap='hot')
            ax.set_title(f'{method_name}\n({result["defect_count"]} defects)')
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=self.grid_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=col, padx=5, pady=5)
            
            col += 1
    
    def _display_metrics(self):
        """Display performance metrics"""
        self.metrics_fig.clear()
        
        # Processing time comparison
        ax1 = self.metrics_fig.add_subplot(221)
        methods = list(self.comparison_results.keys())
        times = [r['time'] for r in self.comparison_results.values()]
        
        ax1.bar(methods, times, color='skyblue')
        ax1.set_ylabel('Processing Time (s)')
        ax1.set_title('Processing Time Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Defect count comparison
        ax2 = self.metrics_fig.add_subplot(222)
        counts = [r['defect_count'] for r in self.comparison_results.values()]
        
        ax2.bar(methods, counts, color='lightcoral')
        ax2.set_ylabel('Defect Count')
        ax2.set_title('Defect Count Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Efficiency (defects per second)
        ax3 = self.metrics_fig.add_subplot(223)
        efficiency = [c/t if t > 0 else 0 for c, t in zip(counts, times)]
        
        ax3.bar(methods, efficiency, color='lightgreen')
        ax3.set_ylabel('Defects/Second')
        ax3.set_title('Detection Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        
        # Overlap matrix
        ax4 = self.metrics_fig.add_subplot(224)
        overlap_matrix = self._calculate_overlap_matrix()
        
        im = ax4.imshow(overlap_matrix, cmap='YlOrRd', aspect='auto')
        ax4.set_xticks(range(len(methods)))
        ax4.set_yticks(range(len(methods)))
        ax4.set_xticklabels(methods, rotation=45)
        ax4.set_yticklabels(methods)
        ax4.set_title('Method Overlap (IoU)')
        
        # Add colorbar
        self.metrics_fig.colorbar(im, ax=ax4)
        
        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw()
    
    def _calculate_overlap_matrix(self) -> np.ndarray:
        """Calculate IoU overlap between methods"""
        methods = list(self.comparison_results.keys())
        n = len(methods)
        overlap_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                mask1 = self.comparison_results[methods[i]]['mask']
                mask2 = self.comparison_results[methods[j]]['mask']
                
                # Calculate IoU
                intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
                union = np.logical_or(mask1 > 0, mask2 > 0).sum()
                
                if union > 0:
                    overlap_matrix[i, j] = intersection / union
                else:
                    overlap_matrix[i, j] = 1.0 if i == j else 0.0
        
        return overlap_matrix
    
    def _display_analysis(self):
        """Display statistical analysis"""
        self.analysis_text.delete(1.0, tk.END)
        
        # Header
        self.analysis_text.insert(tk.END, "STATISTICAL ANALYSIS OF DETECTION METHODS\n", 'heading')
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        
        # Summary statistics
        self.analysis_text.insert(tk.END, "Summary Statistics\n", 'subheading')
        self.analysis_text.insert(tk.END, "-"*40 + "\n")
        
        times = [r['time'] for r in self.comparison_results.values()]
        counts = [r['defect_count'] for r in self.comparison_results.values()]
        
        self.analysis_text.insert(tk.END, f"Average processing time: {np.mean(times):.3f}s ± {np.std(times):.3f}s\n")
        self.analysis_text.insert(tk.END, f"Average defect count: {np.mean(counts):.1f} ± {np.std(counts):.1f}\n\n")
        
        # Method rankings
        self.analysis_text.insert(tk.END, "Method Rankings\n", 'subheading')
        self.analysis_text.insert(tk.END, "-"*40 + "\n")
        
        # Speed ranking
        self.analysis_text.insert(tk.END, "\nFastest Methods:\n")
        speed_ranking = sorted(self.comparison_results.items(), key=lambda x: x[1]['time'])
        for i, (method, result) in enumerate(speed_ranking[:3]):
            self.analysis_text.insert(tk.END, f"{i+1}. {method}: {result['time']:.3f}s\n")
        
        # Detection ranking
        self.analysis_text.insert(tk.END, "\nMost Sensitive Methods:\n")
        detection_ranking = sorted(self.comparison_results.items(), 
                                 key=lambda x: x[1]['defect_count'], reverse=True)
        for i, (method, result) in enumerate(detection_ranking[:3]):
            self.analysis_text.insert(tk.END, f"{i+1}. {method}: {result['defect_count']} defects\n")
        
        # Agreement analysis
        self.analysis_text.insert(tk.END, "\n\nMethod Agreement Analysis\n", 'subheading')
        self.analysis_text.insert(tk.END, "-"*40 + "\n")
        
        overlap_matrix = self._calculate_overlap_matrix()
        methods = list(self.comparison_results.keys())
        
        # Find most similar methods
        similarities = []
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                similarities.append((methods[i], methods[j], overlap_matrix[i, j]))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        self.analysis_text.insert(tk.END, "\nMost Similar Methods:\n")
        for m1, m2, iou in similarities[:3]:
            self.analysis_text.insert(tk.END, f"{m1} ↔ {m2}: {iou:.3f} IoU\n")
        
        # Configure tags
        self.analysis_text.tag_config('heading', font=('Arial', 14, 'bold'))
        self.analysis_text.tag_config('subheading', font=('Arial', 12, 'bold'))
    
    def _export_results(self):
        """Export comparison results"""
        if not self.comparison_results:
            messagebox.showwarning("No Results", "Please run comparison first")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            # Create DataFrame
            data = []
            for method, result in self.comparison_results.items():
                data.append({
                    'Method': method,
                    'Processing Time (s)': result['time'],
                    'Defect Count': result['defect_count'],
                    'Efficiency (defects/s)': result['defect_count'] / result['time'] if result['time'] > 0 else 0
                })
            
            df = pd.DataFrame(data)
            
            if file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Results exported to {file_path}")


class BatchDialog(tk.Toplevel):
    """Dialog for batch processing configuration"""
    
    def __init__(self, parent, initial_folder=None):
        super().__init__(parent)
        self.title("Batch Processing")
        self.geometry("600x500")
        
        # Variables
        self.input_folder_var = tk.StringVar(value=initial_folder or "")
        self.output_folder_var = tk.StringVar()
        self.profile_var = tk.StringVar(value="balanced")
        self.workers_var = tk.IntVar(value=mp.cpu_count())
        
        # Input folder
        input_frame = ttk.Frame(self)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(input_frame, text="Input Folder:").pack(side=tk.LEFT)
        ttk.Entry(input_frame, textvariable=self.input_folder_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Browse", command=self._browse_input).pack(side=tk.LEFT)
        
        # Output folder
        output_frame = ttk.Frame(self)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(output_frame, text="Output Folder:").pack(side=tk.LEFT)
        ttk.Entry(output_frame, textvariable=self.output_folder_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="Browse", command=self._browse_output).pack(side=tk.LEFT)
        
        # Settings
        settings_frame = ttk.LabelFrame(self, text="Processing Settings")
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Profile
        ttk.Label(settings_frame, text="Processing Profile:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        profile_combo = ttk.Combobox(settings_frame, textvariable=self.profile_var,
                                    values=["fast", "balanced", "comprehensive"],
                                    state="readonly", width=20)
        profile_combo.grid(row=0, column=1, padx=10, pady=5)
        
        # Workers
        ttk.Label(settings_frame, text="Parallel Workers:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        ttk.Spinbox(settings_frame, from_=1, to=mp.cpu_count()*2, 
                   textvariable=self.workers_var, width=20).grid(row=1, column=1, padx=10, pady=5)
        
        # Options
        options_frame = ttk.LabelFrame(self, text="Options")
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.save_intermediate_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Save intermediate results",
                       variable=self.save_intermediate_var).pack(anchor=tk.W, padx=10, pady=5)
        
        self.generate_report_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Generate summary report",
                       variable=self.generate_report_var).pack(anchor=tk.W, padx=10, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=20)
        
        ttk.Button(button_frame, text="Start Processing",
                  command=self._start_processing).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _browse_input(self):
        """Browse for input folder"""
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder_var.set(folder)
    
    def _browse_output(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
    
    def _start_processing(self):
        """Start batch processing"""
        if not self.input_folder_var.get():
            messagebox.showwarning("No Input", "Please select input folder")
            return
        
        if not self.output_folder_var.get():
            # Default to input_folder_results
            output = self.input_folder_var.get() + "_results"
            self.output_folder_var.set(output)
        
        # Create batch configuration
        config = BatchConfig(
            input_directory=self.input_folder_var.get(),
            output_directory=self.output_folder_var.get(),
            processing_profile=self.profile_var.get(),
            max_workers=self.workers_var.get(),
            save_intermediate=self.save_intermediate_var.get(),
            generate_summary_report=self.generate_report_var.get()
        )
        
        # Run batch processing in separate thread
        self.destroy()
        
        # Show progress window
        progress_window = BatchProgressWindow(self.master, config)
        progress_window.start_processing()


class BatchProgressWindow(tk.Toplevel):
    """Window showing batch processing progress"""
    
    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config
        self.title("Batch Processing Progress")
        self.geometry("600x400")
        
        # Progress display
        self.text = tk.Text(self, wrap=tk.WORD)
        scroll = ttk.Scrollbar(self, command=self.text.yview)
        self.text.config(yscrollcommand=scroll.set)
        
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(self, textvariable=self.status_var).pack(pady=5)
        
        # Cancel button
        self.cancel_btn = ttk.Button(self, text="Cancel", command=self._cancel)
        self.cancel_btn.pack(pady=10)
    
    def start_processing(self):
        """Start batch processing in thread"""
        self.processing_thread = threading.Thread(target=self._run_batch, daemon=True)
        self.processing_thread.start()
    
    def _run_batch(self):
        """Run batch processing"""
        try:
            # Create processor
            processor = BatchProcessor(self.config)
            
            # Redirect logging to text widget
            handler = TextHandler(self.text)
            processor.logger.addHandler(handler)
            
            # Run processing
            self.status_var.set("Processing images...")
            stats = processor.process_batch()
            
            # Show results
            self.status_var.set("Processing complete!")
            self.progress_var.set(100)
            
            # Summary
            self.text.insert(tk.END, "\n" + "="*50 + "\n")
            self.text.insert(tk.END, "PROCESSING COMPLETE\n")
            self.text.insert(tk.END, "="*50 + "\n")
            self.text.insert(tk.END, f"Total images: {stats['total_images']}\n")
            self.text.insert(tk.END, f"Processed: {stats['processed']}\n")
            self.text.insert(tk.END, f"Failed: {stats['failed']}\n")
            self.text.insert(tk.END, f"Total time: {stats['total_time']:.1f} seconds\n")
            
            self.cancel_btn.config(text="Close")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Processing Error", str(e))
    
    def _cancel(self):
        """Cancel or close"""
        if self.cancel_btn['text'] == "Cancel":
            # TODO: Implement cancellation
            pass
        self.destroy()


class TextHandler(logging.Handler):
    """Custom logging handler that writes to a Text widget"""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
    
    def emit(self, record):
        msg = self.format(record)
        
        def append():
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
        
        # GUI operations must be done in main thread
        self.text_widget.after(0, append)


class StatisticsViewer(tk.Toplevel):
    """View detailed statistics from database"""
    
    def __init__(self, parent, database):
        super().__init__(parent)
        self.database = database
        self.title("Inspection Statistics")
        self.geometry("900x600")
        
        # Date range selection
        date_frame = ttk.Frame(self)
        date_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(date_frame, text="Date Range:").pack(side=tk.LEFT, padx=5)
        
        # TODO: Add date pickers
        
        ttk.Button(date_frame, text="Refresh", command=self._refresh_stats).pack(side=tk.LEFT, padx=20)
        ttk.Button(date_frame, text="Export", command=self._export_stats).pack(side=tk.LEFT, padx=5)
        
        # Statistics display
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Overview tab
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="Overview")
        self._create_overview_tab(overview_frame)
        
        # Trends tab
        trends_frame = ttk.Frame(self.notebook)
        self.notebook.add(trends_frame, text="Trends")
        self._create_trends_tab(trends_frame)
        
        # Defects tab
        defects_frame = ttk.Frame(self.notebook)
        self.notebook.add(defects_frame, text="Defect Analysis")
        self._create_defects_tab(defects_frame)
        
        # Load initial data
        self._refresh_stats()
    
    def _create_overview_tab(self, parent):
        """Create overview statistics tab"""
        self.overview_text = tk.Text(parent, wrap=tk.WORD, font=('Courier', 10))
        scroll = ttk.Scrollbar(parent, command=self.overview_text.yview)
        self.overview_text.config(yscrollcommand=scroll.set)
        
        self.overview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_trends_tab(self, parent):
        """Create trends visualization tab"""
        self.trends_fig = Figure(figsize=(10, 6), dpi=100)
        self.trends_canvas = FigureCanvasTkAgg(self.trends_fig, master=parent)
        self.trends_canvas.draw()
        self.trends_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_defects_tab(self, parent):
        """Create defect analysis tab"""
        self.defects_fig = Figure(figsize=(10, 6), dpi=100)
        self.defects_canvas = FigureCanvasTkAgg(self.defects_fig, master=parent)
        self.defects_canvas.draw()
        self.defects_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _refresh_stats(self):
        """Refresh statistics from database"""
        # Get statistics
        stats = self.database.get_statistics()
        
        # Update overview
        self._update_overview(stats)
        
        # Update visualizations
        self._update_trends()
        self._update_defects()
    
    def _update_overview(self, stats):
        """Update overview text"""
        self.overview_text.delete(1.0, tk.END)
        
        self.overview_text.insert(tk.END, "INSPECTION STATISTICS OVERVIEW\n", 'heading')
        self.overview_text.insert(tk.END, "="*50 + "\n\n")
        
        self.overview_text.insert(tk.END, f"Total Inspections: {stats.get('total_inspections', 0)}\n")
        self.overview_text.insert(tk.END, f"Passed: {stats.get('passed', 0)}\n")
        self.overview_text.insert(tk.END, f"Failed: {stats.get('failed', 0)}\n")
        
        pass_rate = (stats['passed'] / stats['total_inspections'] * 100) if stats['total_inspections'] > 0 else 0
        self.overview_text.insert(tk.END, f"Pass Rate: {pass_rate:.1f}%\n\n")
        
        self.overview_text.insert(tk.END, f"Average Quality: {stats.get('average_quality', 0):.1f}/100\n")
        self.overview_text.insert(tk.END, f"Total Defects: {stats.get('total_defects', 0)}\n")
        self.overview_text.insert(tk.END, f"Average Processing Time: {stats.get('average_processing_time', 0):.2f}s\n")
        
        self.overview_text.tag_config('heading', font=('Arial', 12, 'bold'))
    
    def _update_trends(self):
        """Update trends visualization"""
        # This would query database for time-series data
        # For now, create dummy data
        self.trends_fig.clear()
        
        ax = self.trends_fig.add_subplot(111)
        
        # Dummy trend data
        days = pd.date_range(end=pd.Timestamp.now(), periods=30)
        pass_rates = np.random.normal(85, 5, 30)
        
        ax.plot(days, pass_rates, marker='o', linestyle='-', markersize=4)
        ax.set_xlabel('Date')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title('Pass Rate Trend (Last 30 Days)')
        ax.grid(True, alpha=0.3)
        
        # Format dates
        self.trends_fig.autofmt_xdate()
        
        self.trends_canvas.draw()
    
    def _update_defects(self):
        """Update defect analysis"""
        # This would query database for defect statistics
        self.defects_fig.clear()
        
        # Defect type distribution
        ax1 = self.defects_fig.add_subplot(121)
        defect_types = ['Scratch', 'Pit', 'Contamination', 'Particle', 'Other']
        counts = np.random.randint(10, 100, len(defect_types))
        
        ax1.pie(counts, labels=defect_types, autopct='%1.1f%%')
        ax1.set_title('Defect Type Distribution')
        
        # Defects by region
        ax2 = self.defects_fig.add_subplot(122)
        regions = ['Core', 'Cladding', 'Ferrule']
        region_counts = np.random.randint(5, 50, len(regions))
        
        ax2.bar(regions, region_counts, color=['red', 'green', 'blue'])
        ax2.set_ylabel('Defect Count')
        ax2.set_title('Defects by Region')
        
        self.defects_fig.tight_layout()
        self.defects_canvas.draw()
    
    def _export_stats(self):
        """Export statistics to file"""
        file_path = filedialog.asksaveasfilename(
            title="Export Statistics",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("Excel files", "*.xlsx")]
        )
        
        if file_path:
            # TODO: Implement export functionality
            messagebox.showinfo("Export", "Statistics export functionality to be implemented")


monitor_instance = None

def get_script_info():
    """Returns information about the script, its status, and exposed parameters."""
    if monitor_instance:
        return monitor_instance.get_script_info()
    return {"name": "Real-time Monitor GUI", "status": "not_initialized", "parameters": {}}

def set_script_parameter(key, value):
    """Sets a specific parameter for the script and updates shared_config."""
    if monitor_instance:
        return monitor_instance.set_script_parameter(key, value)
    return False

# Main application entry point
def main():
    """Run the real-time monitoring application"""
    global monitor_instance

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run monitor
    monitor_instance = RealTimeMonitor()
    monitor_instance.run()


if __name__ == "__main__":
    main()
