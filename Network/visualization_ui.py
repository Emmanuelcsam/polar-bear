#!/usr/bin/env python3
"""
Real-time Visualization UI for Fiber Optics Neural Network
Compatible with Ubuntu Linux Wayland system
Shows neural network processing in real-time with parameter tweaking
"""

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, GLib, GdkPixbuf, Gdk

import torch
import numpy as np
import cv2
from datetime import datetime
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import our modules
from config_loader import ConfigManager, get_config
from logger import get_logger
from integrated_network import EnhancedIntegratedNetwork
from tensor_processor import TensorProcessor


class NeuralNetworkVisualizer(Gtk.Window):
    """
    Real-time visualization UI for neural network processing
    "I want to see the real time visual process of the neural network working"
    """
    
    def __init__(self):
        """Initialize the visualization UI"""
        super().__init__(title="Fiber Optics Neural Network Visualizer")
        
        print(f"[{datetime.now()}] Initializing NeuralNetworkVisualizer")
        
        self.logger = get_logger("Visualizer")
        self.config = get_config()
        
        # Initialize components
        self.tensor_processor = TensorProcessor()
        self.model = None
        self.current_image = None
        self.processing_thread = None
        self.update_queue = queue.Queue()
        
        # Processing state
        self.is_processing = False
        self.current_intensity = 0.0
        self.intensity_step = 0.01
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Setup UI
        self._setup_ui()
        self._setup_model()
        
        # Start update timer
        GLib.timeout_add(16, self._update_ui)  # ~60 FPS
        
        self.logger.info("Visualization UI initialized")
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Set window properties
        self.set_default_size(
            self.config.visualization.window_width,
            self.config.visualization.window_height
        )
        self.set_border_width(10)
        
        # Main container
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.add(main_box)
        
        # Left panel - Original and processed images
        left_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.pack_start(left_panel, True, True, 0)
        
        # Original image frame
        original_frame = Gtk.Frame(label="Original Image")
        left_panel.pack_start(original_frame, True, True, 0)
        self.original_image = Gtk.Image()
        original_frame.add(self.original_image)
        
        # Processed images grid
        processed_frame = Gtk.Frame(label="Neural Network Processing")
        left_panel.pack_start(processed_frame, True, True, 0)
        
        processed_grid = Gtk.Grid()
        processed_grid.set_row_spacing(5)
        processed_grid.set_column_spacing(5)
        processed_frame.add(processed_grid)
        
        # Create image widgets for different processing stages
        self.stage_images = {}
        stages = [
            ("features", "Feature Extraction"),
            ("segmentation", "Segmentation"),
            ("anomaly", "Anomaly Detection"),
            ("reference", "Reference Matching"),
            ("reconstruction", "Reconstruction"),
            ("final", "Final Output")
        ]
        
        for i, (key, label) in enumerate(stages):
            frame = Gtk.Frame(label=label)
            image = Gtk.Image()
            frame.add(image)
            self.stage_images[key] = image
            processed_grid.attach(frame, i % 3, i // 3, 1, 1)
        
        # Right panel - Controls and parameters
        right_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_box.pack_start(right_panel, False, False, 0)
        
        # Control buttons
        control_frame = Gtk.Frame(label="Controls")
        right_panel.pack_start(control_frame, False, False, 0)
        
        control_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        control_frame.add(control_box)
        
        # Load image button
        self.load_button = Gtk.Button(label="Load Image")
        self.load_button.connect("clicked", self._on_load_image)
        control_box.pack_start(self.load_button, False, False, 0)
        
        # Start/Stop processing button
        self.process_button = Gtk.Button(label="Start Processing")
        self.process_button.connect("clicked", self._on_toggle_processing)
        control_box.pack_start(self.process_button, False, False, 0)
        
        # Intensity control
        intensity_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        control_box.pack_start(intensity_box, False, False, 0)
        
        intensity_label = Gtk.Label(label="Intensity:")
        intensity_box.pack_start(intensity_label, False, False, 0)
        
        self.intensity_adjustment = Gtk.Adjustment(
            value=0.0, lower=0.0, upper=1.0, 
            step_increment=0.01, page_increment=0.1
        )
        self.intensity_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=self.intensity_adjustment
        )
        self.intensity_scale.set_digits(2)
        self.intensity_scale.set_draw_value(True)
        self.intensity_scale.connect("value-changed", self._on_intensity_changed)
        intensity_box.pack_start(self.intensity_scale, True, True, 0)
        
        # Parameter controls
        param_frame = Gtk.Frame(label="Neural Network Parameters")
        right_panel.pack_start(param_frame, True, True, 0)
        
        param_scroll = Gtk.ScrolledWindow()
        param_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        param_frame.add(param_scroll)
        
        self.param_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        param_scroll.add(self.param_box)
        
        # Add parameter controls
        self._create_parameter_controls()
        
        # Statistics display
        stats_frame = Gtk.Frame(label="Statistics")
        right_panel.pack_start(stats_frame, False, False, 0)
        
        stats_grid = Gtk.Grid()
        stats_grid.set_row_spacing(5)
        stats_grid.set_column_spacing(10)
        stats_frame.add(stats_grid)
        
        # Create statistics labels
        self.stats_labels = {}
        stats = [
            ("fps", "FPS:"),
            ("similarity", "Similarity:"),
            ("anomaly_score", "Anomaly Score:"),
            ("computation_time", "Computation Time:"),
            ("memory_usage", "Memory Usage:")
        ]
        
        for i, (key, label) in enumerate(stats):
            label_widget = Gtk.Label(label=label)
            value_widget = Gtk.Label(label="0")
            self.stats_labels[key] = value_widget
            stats_grid.attach(label_widget, 0, i, 1, 1)
            stats_grid.attach(value_widget, 1, i, 1, 1)
    
    def _create_parameter_controls(self):
        """Create parameter adjustment controls"""
        # Equation coefficients
        coeff_label = Gtk.Label(label="<b>Equation Coefficients (I=Ax1+Bx2+...)</b>")
        coeff_label.set_use_markup(True)
        self.param_box.pack_start(coeff_label, False, False, 0)
        
        self.coeff_controls = {}
        for coeff in ['A', 'B', 'C', 'D', 'E']:
            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            self.param_box.pack_start(box, False, False, 0)
            
            label = Gtk.Label(label=f"{coeff}:")
            label.set_size_request(30, -1)
            box.pack_start(label, False, False, 0)
            
            adjustment = Gtk.Adjustment(
                value=self.config.equation.coefficients[coeff],
                lower=self.config.equation.min_coefficient,
                upper=self.config.equation.max_coefficient,
                step_increment=0.01,
                page_increment=0.1
            )
            
            spin = Gtk.SpinButton(adjustment=adjustment, digits=2)
            spin.connect("value-changed", 
                        lambda w, c=coeff: self._on_coefficient_changed(c, w.get_value()))
            box.pack_start(spin, True, True, 0)
            
            self.coeff_controls[coeff] = spin
        
        # Loss weights
        loss_label = Gtk.Label(label="<b>Loss Weights</b>")
        loss_label.set_use_markup(True)
        self.param_box.pack_start(loss_label, False, False, 0)
        
        self.loss_controls = {}
        for loss_type in ['segmentation', 'anomaly', 'contrastive', 
                         'perceptual', 'wasserstein', 'reconstruction']:
            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            self.param_box.pack_start(box, False, False, 0)
            
            label = Gtk.Label(label=f"{loss_type.capitalize()}:")
            label.set_size_request(100, -1)
            box.pack_start(label, False, False, 0)
            
            adjustment = Gtk.Adjustment(
                value=self.config.loss.weights[loss_type],
                lower=0.0, upper=1.0,
                step_increment=0.01, page_increment=0.1
            )
            
            scale = Gtk.Scale(
                orientation=Gtk.Orientation.HORIZONTAL,
                adjustment=adjustment
            )
            scale.set_digits(2)
            scale.set_draw_value(True)
            scale.connect("value-changed",
                         lambda w, t=loss_type: self._on_loss_weight_changed(t, w.get_value()))
            box.pack_start(scale, True, True, 0)
            
            self.loss_controls[loss_type] = scale
        
        # Thresholds
        threshold_label = Gtk.Label(label="<b>Thresholds</b>")
        threshold_label.set_use_markup(True)
        self.param_box.pack_start(threshold_label, False, False, 0)
        
        # Similarity threshold
        sim_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.param_box.pack_start(sim_box, False, False, 0)
        
        sim_label = Gtk.Label(label="Similarity:")
        sim_label.set_size_request(100, -1)
        sim_box.pack_start(sim_label, False, False, 0)
        
        self.similarity_threshold = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.01)
        self.similarity_threshold.set_value(self.config.similarity.threshold)
        self.similarity_threshold.connect("value-changed", self._on_similarity_threshold_changed)
        sim_box.pack_start(self.similarity_threshold, True, True, 0)
        
        # Anomaly threshold
        anom_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.param_box.pack_start(anom_box, False, False, 0)
        
        anom_label = Gtk.Label(label="Anomaly:")
        anom_label.set_size_request(100, -1)
        anom_box.pack_start(anom_label, False, False, 0)
        
        self.anomaly_threshold = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.01)
        self.anomaly_threshold.set_value(self.config.anomaly.threshold)
        self.anomaly_threshold.connect("value-changed", self._on_anomaly_threshold_changed)
        anom_box.pack_start(self.anomaly_threshold, True, True, 0)
    
    def _setup_model(self):
        """Initialize the neural network model"""
        try:
            self.model = EnhancedIntegratedNetwork()
            self.model.eval()
            
            # Move to GPU if available
            device = self.config.get_device()
            self.model = self.model.to(device)
            
            self.logger.info("Neural network model loaded")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self._show_error_dialog(f"Failed to load model: {e}")
    
    def _on_load_image(self, widget):
        """Handle load image button click"""
        dialog = Gtk.FileChooserDialog(
            title="Select Image",
            parent=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )
        
        # Add filter for image files
        filter_images = Gtk.FileFilter()
        filter_images.set_name("Image files")
        filter_images.add_mime_type("image/png")
        filter_images.add_mime_type("image/jpeg")
        filter_images.add_mime_type("image/bmp")
        dialog.add_filter(filter_images)
        
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            self._load_image(filename)
        
        dialog.destroy()
    
    def _load_image(self, filename: str):
        """Load and display image"""
        try:
            # Load image
            self.current_image = cv2.imread(filename)
            if self.current_image is None:
                raise ValueError("Failed to load image")
            
            # Convert BGR to RGB
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            self._update_image_widget(self.original_image, self.current_image)
            
            self.logger.info(f"Loaded image: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            self._show_error_dialog(f"Error loading image: {e}")
    
    def _on_toggle_processing(self, widget):
        """Toggle processing on/off"""
        if self.is_processing:
            self._stop_processing()
        else:
            self._start_processing()
    
    def _start_processing(self):
        """Start neural network processing"""
        if self.current_image is None:
            self._show_error_dialog("Please load an image first")
            return
        
        self.is_processing = True
        self.process_button.set_label("Stop Processing")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Started processing")
    
    def _stop_processing(self):
        """Stop neural network processing"""
        self.is_processing = False
        self.process_button.set_label("Start Processing")
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        self.logger.info("Stopped processing")
    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        while self.is_processing:
            try:
                start_time = time.time()
                
                # Apply current intensity to image
                adjusted_image = self._apply_intensity(
                    self.current_image, 
                    self.current_intensity
                )
                
                # Process through neural network
                results = self._process_image(adjusted_image)
                
                # Queue results for UI update
                self.update_queue.put(('results', results))
                
                # Update intensity for next iteration
                self.current_intensity += self.intensity_step
                if self.current_intensity > 1.0:
                    self.current_intensity = 0.0
                
                # Update intensity slider
                GLib.idle_add(
                    self.intensity_adjustment.set_value,
                    self.current_intensity
                )
                
                # Calculate timing
                process_time = time.time() - start_time
                
                # Maintain target FPS
                target_time = 1.0 / self.config.visualization.fps
                if process_time < target_time:
                    time.sleep(target_time - process_time)
                
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                self.update_queue.put(('error', str(e)))
                break
    
    def _apply_intensity(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply intensity adjustment to image"""
        # Simple intensity adjustment - can be made more sophisticated
        adjusted = image.astype(np.float32)
        adjusted = adjusted * (0.5 + intensity)
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    def _process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Process image through neural network"""
        # Convert to tensor
        image_tensor = self.tensor_processor.image_to_tensor(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Convert outputs to visualizable format
        results = {
            'image': image,
            'features': self._visualize_features(outputs['multi_scale_features']),
            'segmentation': self._visualize_segmentation(outputs['segmentation']),
            'anomaly': self._visualize_anomaly(outputs['anomaly_map']),
            'reference': self._visualize_reference(outputs),
            'reconstruction': self._tensor_to_image(outputs['reconstruction']),
            'final': self._create_final_visualization(outputs),
            'stats': {
                'similarity': outputs['final_similarity'][0].item(),
                'anomaly_score': outputs['anomaly_map'][0].mean().item(),
                'computation_time': outputs.get('computation_time', 0),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
        }
        
        return results
    
    def _visualize_features(self, features: List[torch.Tensor]) -> np.ndarray:
        """Visualize multi-scale features"""
        # Take first scale features
        feat = features[0][0]  # [C, H, W]
        
        # Average across channels
        feat_avg = feat.mean(dim=0).cpu().numpy()
        
        # Normalize to 0-255
        feat_norm = (feat_avg - feat_avg.min()) / (feat_avg.max() - feat_avg.min() + 1e-8)
        feat_vis = (feat_norm * 255).astype(np.uint8)
        
        # Apply colormap
        feat_colored = cv2.applyColorMap(feat_vis, cv2.COLORMAP_JET)
        
        return feat_colored
    
    def _visualize_segmentation(self, segmentation: torch.Tensor) -> np.ndarray:
        """Visualize segmentation results"""
        # Get class predictions
        seg_pred = segmentation[0].argmax(dim=0).cpu().numpy()
        
        # Create colored segmentation
        h, w = seg_pred.shape
        seg_colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colors from config
        colors = self.config.visualization.segmentation_colors
        seg_colored[seg_pred == 0] = colors.core
        seg_colored[seg_pred == 1] = colors.cladding
        seg_colored[seg_pred == 2] = colors.ferrule
        
        return seg_colored
    
    def _visualize_anomaly(self, anomaly_map: torch.Tensor) -> np.ndarray:
        """Visualize anomaly detection results"""
        # Get anomaly map
        anomaly = anomaly_map[0].cpu().numpy()
        
        # Normalize
        anomaly_norm = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)
        anomaly_vis = (anomaly_norm * 255).astype(np.uint8)
        
        # Apply colormap
        colormap_name = self.config.visualization.anomaly_colormap
        if colormap_name == 'hot':
            colormap = cv2.COLORMAP_HOT
        elif colormap_name == 'jet':
            colormap = cv2.COLORMAP_JET
        elif colormap_name == 'viridis':
            colormap = cv2.COLORMAP_VIRIDIS
        else:
            colormap = cv2.COLORMAP_PLASMA
        
        anomaly_colored = cv2.applyColorMap(anomaly_vis, colormap)
        
        return anomaly_colored
    
    def _visualize_reference(self, outputs: Dict) -> np.ndarray:
        """Visualize reference matching"""
        # Create simple visualization showing similarity score
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        
        similarity = outputs['final_similarity'][0].item()
        meets_threshold = outputs['meets_threshold'][0].item()
        
        # Draw similarity bar
        bar_height = int(similarity * 200)
        color = (0, 255, 0) if meets_threshold else (255, 0, 0)
        cv2.rectangle(img, (50, 250 - bar_height), (206, 250), color, -1)
        
        # Add text
        text = f"Similarity: {similarity:.2f}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 2)
        
        return img
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable image"""
        img = tensor[0].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        return img
    
    def _create_final_visualization(self, outputs: Dict) -> np.ndarray:
        """Create final combined visualization"""
        # Overlay anomalies on original image
        img = self._tensor_to_image(outputs['reconstruction'])
        anomaly = outputs['anomaly_map'][0].cpu().numpy()
        
        # Threshold anomalies
        threshold = self.config.anomaly.threshold
        anomaly_mask = anomaly > threshold
        
        # Highlight anomalies in red
        img[anomaly_mask] = [255, 0, 0]
        
        return img
    
    def _update_ui(self):
        """Update UI with processing results"""
        try:
            while not self.update_queue.empty():
                msg_type, data = self.update_queue.get_nowait()
                
                if msg_type == 'results':
                    # Update stage images
                    for key, image_widget in self.stage_images.items():
                        if key in data:
                            self._update_image_widget(image_widget, data[key])
                    
                    # Update statistics
                    if 'stats' in data:
                        self._update_statistics(data['stats'])
                
                elif msg_type == 'error':
                    self._show_error_dialog(data)
        
        except queue.Empty:
            pass
        
        # Update FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.stats_labels['fps'].set_text(f"{fps:.1f}")
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        return True  # Continue timer
    
    def _update_image_widget(self, widget: Gtk.Image, image: np.ndarray):
        """Update GTK image widget with numpy array"""
        # Resize if needed
        max_size = 300
        h, w = image.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Convert to GdkPixbuf
        h, w = image.shape[:2]
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            image.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False, 8, w, h, w * 3
        )
        
        widget.set_from_pixbuf(pixbuf)
    
    def _update_statistics(self, stats: Dict):
        """Update statistics display"""
        self.stats_labels['similarity'].set_text(f"{stats['similarity']:.3f}")
        self.stats_labels['anomaly_score'].set_text(f"{stats['anomaly_score']:.3f}")
        self.stats_labels['computation_time'].set_text(f"{stats['computation_time']*1000:.1f}ms")
        
        memory_mb = stats['memory_usage'] / (1024 * 1024)
        self.stats_labels['memory_usage'].set_text(f"{memory_mb:.1f}MB")
    
    def _on_intensity_changed(self, widget):
        """Handle intensity slider change"""
        self.current_intensity = widget.get_value()
    
    def _on_coefficient_changed(self, coeff: str, value: float):
        """Handle equation coefficient change"""
        self.config.equation.coefficients[coeff] = value
        if self.model:
            # Update model coefficients
            coeffs = [self.config.equation.coefficients[c] for c in ['A', 'B', 'C', 'D', 'E']]
            self.model.set_equation_coefficients(coeffs)
        
        self.logger.info(f"Updated coefficient {coeff} = {value}")
    
    def _on_loss_weight_changed(self, loss_type: str, value: float):
        """Handle loss weight change"""
        self.config.loss.weights[loss_type] = value
        self.logger.info(f"Updated loss weight {loss_type} = {value}")
    
    def _on_similarity_threshold_changed(self, widget):
        """Handle similarity threshold change"""
        value = widget.get_value()
        self.config.similarity.threshold = value
        self.logger.info(f"Updated similarity threshold = {value}")
    
    def _on_anomaly_threshold_changed(self, widget):
        """Handle anomaly threshold change"""
        value = widget.get_value()
        self.config.anomaly.threshold = value
        self.logger.info(f"Updated anomaly threshold = {value}")
    
    def _show_error_dialog(self, message: str):
        """Show error dialog"""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Error"
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()


def main():
    """Main entry point"""
    print(f"[{datetime.now()}] Starting Fiber Optics Neural Network Visualizer")
    
    # Initialize configuration
    config_manager = ConfigManager()
    config_manager.initialize()
    
    # Create and run application
    win = NeuralNetworkVisualizer()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    
    Gtk.main()


if __name__ == "__main__":
    main()