# realtime_visualizer.py
# Realtime Video Visualizer for Fiber Optic End-Face CNN
# Uses Pylon SDK for camera integration and real-time processing

import os
import sys
import time
import threading
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Pylon imports
try:
    from pypylon import pylon
except ImportError:
    print("Warning: Pylon SDK not found. Install with: pip install pypylon")
    pylon = None

from model import EndfaceNet
from dataset import build_default_transforms

class RealtimeVisualizer:
    """Realtime video visualizer for fiber optic end-face analysis."""
    
    def __init__(self, model_path: str, device: str = 'cuda', camera_index: int = 0):
        self.device = device
        self.camera_index = camera_index
        self.model = None
        self.transforms = None
        self.camera = None
        self.is_running = False
        self.is_grabbing = False
        self.current_frame = None
        self.current_results = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Load model
        self.load_model(model_path)
        
        # Setup transforms
        self.transforms = build_default_transforms(train=False, img_size=256)
        
        # Initialize camera
        self.setup_camera()
        
        # Setup visualization
        self.setup_visualization()
    
    def load_model(self, model_path: str):
        """Load the trained CNN model."""
        print(f"Loading model from {model_path}...")
        
        self.model = EndfaceNet(num_classes=40)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def setup_camera(self):
        """Setup Pylon camera for video capture."""
        if pylon is None:
            print("Error: Pylon SDK not available. Using webcam fallback.")
            self.camera = cv2.VideoCapture(self.camera_index)
            return
        
        try:
            # Get the transport layer factory
            tl_factory = pylon.TlFactory.GetInstance()
            
            # Get all attached devices and sort them by serial number
            devices = tl_factory.EnumerateDevices()
            
            if len(devices) == 0:
                print("No Pylon cameras found. Using webcam fallback.")
                self.camera = cv2.VideoCapture(self.camera_index)
                return
            
            # Use the first available camera
            camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
            
            # Open camera
            camera.Open()
            
            # Configure camera settings for real-time processing
            if camera.IsOpen():
                # Try to set pixel format to RGB8, fallback to Mono8 if not supported
                try:
                    camera.PixelFormat.SetValue("RGB8")
                    print("  Set pixel format to RGB8")
                except Exception as e:
                    print(f"  Camera doesn't support RGB8, using current format: {camera.PixelFormat.GetValue()}")
                
                # Set exposure time (adjust as needed)
                camera.ExposureTime.SetValue(10000)  # 10ms
                
                # Set gain (adjust as needed)
                camera.Gain.SetValue(0)
                
                # Enable continuous acquisition
                camera.AcquisitionMode.SetValue("Continuous")
                
                # Set trigger mode to software
                camera.TriggerMode.SetValue("Off")
                
                # Start grabbing
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                self.is_grabbing = True
                
                print(f"Pylon camera initialized: {camera.GetDeviceInfo().GetModelName()}")
                self.camera = camera
            else:
                print("Failed to open Pylon camera. Using webcam fallback.")
                self.camera = cv2.VideoCapture(self.camera_index)
                
        except Exception as e:
            print(f"Error setting up Pylon camera: {e}")
            print("Using webcam fallback.")
            self.camera = cv2.VideoCapture(self.camera_index)
    
    def setup_visualization(self):
        """Setup matplotlib visualization window."""
        plt.ion()  # Turn on interactive mode
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Fiber Optic End-Face Realtime Analysis', fontsize=16)
        
        # Initialize plots
        self.axes[0, 0].set_title('Live Camera Feed')
        self.axes[0, 1].set_title('Core Region Detection')
        self.axes[0, 2].set_title('Cladding Region Detection')
        self.axes[1, 0].set_title('Ferrule Region Detection')
        self.axes[1, 1].set_title('Defect Probabilities')
        self.axes[1, 2].set_title('Processing Info')
        
        # Hide axes for image displays
        for ax in self.axes.flat:
            ax.axis('off')
        
        plt.tight_layout()
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera."""
        if self.camera is None:
            return None
        
        try:
            if hasattr(self.camera, 'RetrieveResult'):  # Pylon camera
                # Grab one image
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    # Convert to numpy array
                    image = grab_result.Array
                    grab_result.Release()
                    
                    # Convert monochrome to RGB if needed
                    if len(image.shape) == 2:  # Monochrome
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    
                    return image
                else:
                    return None
            else:  # OpenCV camera
                ret, frame = self.camera.read()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return None
                
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame through the CNN model."""
        if frame is None:
            return None
        
        try:
            # Apply transforms
            sample = self.transforms(image=frame)
            tensor = sample["image"].unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                mask_logits, defect_logits, stat_feats = self.model(tensor)
            
            # Process predictions
            mask_probs = torch.sigmoid(mask_logits)
            defect_probs = torch.sigmoid(defect_logits)
            
            # Convert to numpy
            mask_probs = mask_probs.cpu().numpy()[0]  # [3, H, W]
            defect_probs = defect_probs.cpu().numpy()[0]  # [num_classes]
            stat_feats = stat_feats.cpu().numpy()[0]  # [88]
            
            # Create results
            results = {
                'region_masks': {
                    'core': mask_probs[0],
                    'cladding': mask_probs[1],
                    'ferrule': mask_probs[2]
                },
                'defect_probabilities': defect_probs,
                'statistical_features': stat_feats,
                'defects_detected': [],
                'confidence_scores': []
            }
            
            # Identify defects (threshold-based)
            defect_threshold = 0.5
            defect_names = [
                'scratch', 'dig', 'blob', 'contamination', 'crack',
                'chip', 'pit', 'discoloration', 'roughness', 'waviness',
                'eccentricity', 'concentricity', 'roundness', 'surface_finish',
                'edge_defect', 'center_defect', 'peripheral_defect',
                'structural_defect', 'optical_defect', 'mechanical_defect',
                'thermal_defect', 'chemical_defect', 'environmental_defect',
                'manufacturing_defect', 'handling_defect', 'storage_defect',
                'transport_defect', 'installation_defect', 'operation_defect',
                'maintenance_defect', 'inspection_defect', 'calibration_defect',
                'alignment_defect', 'focus_defect', 'illumination_defect',
                'imaging_defect', 'processing_defect', 'analysis_defect',
                'reporting_defect', 'documentation_defect', 'quality_defect'
            ]
            
            for i, prob in enumerate(defect_probs):
                if prob > defect_threshold:
                    results['defects_detected'].append(defect_names[i])
                    results['confidence_scores'].append(float(prob))
            
            return results
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
    
    def update_visualization(self, frame: np.ndarray, results: Dict):
        """Update the visualization with new frame and results."""
        if frame is None or results is None:
            return
        
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
            ax.axis('off')
        
        # Original frame
        self.axes[0, 0].imshow(frame)
        self.axes[0, 0].set_title('Live Camera Feed')
        
        # Region masks
        core_mask = results['region_masks']['core']
        cladding_mask = results['region_masks']['cladding']
        ferrule_mask = results['region_masks']['ferrule']
        
        self.axes[0, 1].imshow(core_mask, cmap='Reds', alpha=0.7)
        self.axes[0, 1].imshow(frame, alpha=0.3)
        self.axes[0, 1].set_title('Core Region Detection')
        
        self.axes[0, 2].imshow(cladding_mask, cmap='Blues', alpha=0.7)
        self.axes[0, 2].imshow(frame, alpha=0.3)
        self.axes[0, 2].set_title('Cladding Region Detection')
        
        self.axes[1, 0].imshow(ferrule_mask, cmap='Greens', alpha=0.7)
        self.axes[1, 0].imshow(frame, alpha=0.3)
        self.axes[1, 0].set_title('Ferrule Region Detection')
        
        # Defect probabilities (top 10)
        defect_probs = results['defect_probabilities']
        top_indices = np.argsort(defect_probs)[-10:]
        top_probs = defect_probs[top_indices]
        
        defect_names = [
            'scratch', 'dig', 'blob', 'contamination', 'crack',
            'chip', 'pit', 'discoloration', 'roughness', 'waviness',
            'eccentricity', 'concentricity', 'roundness', 'surface_finish',
            'edge_defect', 'center_defect', 'peripheral_defect',
            'structural_defect', 'optical_defect', 'mechanical_defect',
            'thermal_defect', 'chemical_defect', 'environmental_defect',
            'manufacturing_defect', 'handling_defect', 'storage_defect',
            'transport_defect', 'installation_defect', 'operation_defect',
            'maintenance_defect', 'inspection_defect', 'calibration_defect',
            'alignment_defect', 'focus_defect', 'illumination_defect',
            'imaging_defect', 'processing_defect', 'analysis_defect',
            'reporting_defect', 'documentation_defect', 'quality_defect'
        ]
        
        top_names = [defect_names[i] for i in top_indices]
        
        y_pos = np.arange(len(top_names))
        self.axes[1, 1].barh(y_pos, top_probs)
        self.axes[1, 1].set_yticks(y_pos)
        self.axes[1, 1].set_yticklabels(top_names)
        self.axes[1, 1].set_xlabel('Probability')
        self.axes[1, 1].set_title('Top Defect Probabilities')
        
        # Processing info
        info_text = f"""
        FPS: {self.fps:.1f}
        Frame Count: {self.frame_count}
        Defects Detected: {len(results['defects_detected'])}
        Device: {self.device}
        """
        
        self.axes[1, 2].text(0.1, 0.5, info_text, transform=self.axes[1, 2].transAxes,
                             fontsize=10, verticalalignment='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.axes[1, 2].set_title('Processing Info')
        
        # Update display
        plt.draw()
        plt.pause(0.001)
    
    def run(self):
        """Main loop for realtime processing."""
        print("Starting realtime visualization...")
        print("Press 'q' to quit, 's' to save current frame")
        
        self.is_running = True
        
        while self.is_running:
            # Capture frame
            frame = self.capture_frame()
            
            if frame is not None:
                # Process frame
                results = self.process_frame(frame)
                
                # Update visualization
                self.update_visualization(frame, results)
                
                # Update FPS calculation
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time
                
                # Store current frame and results
                self.current_frame = frame
                self.current_results = results
            
            # Check for key presses
            if plt.waitforbuttonpress(timeout=0.001):
                key = plt.gcf().canvas.get_current_key()
                if key == 'q':
                    self.is_running = False
                elif key == 's':
                    self.save_current_frame()
        
        self.cleanup()
    
    def save_current_frame(self):
        """Save the current frame and results."""
        if self.current_frame is not None and self.current_results is not None:
            timestamp = int(time.time())
            
            # Save frame
            frame_path = f"captured_frame_{timestamp}.png"
            cv2.imwrite(frame_path, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            
            # Save results
            results_path = f"captured_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(self.current_results, f, indent=2)
            
            print(f"Saved frame and results: {frame_path}, {results_path}")
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        if self.camera is not None:
            if hasattr(self.camera, 'Close'):  # Pylon camera
                if self.is_grabbing:
                    self.camera.StopGrabbing()
                self.camera.Close()
            else:  # OpenCV camera
                self.camera.release()
        
        plt.close('all')
        print("Cleanup complete.")

def main():
    parser = argparse.ArgumentParser(description="Realtime Fiber Optic End-Face Visualizer")
    parser.add_argument('--weights', required=True, help='Path to trained model weights')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = RealtimeVisualizer(
        model_path=args.weights,
        device=args.device,
        camera_index=args.camera
    )
    
    # Run realtime processing
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.cleanup()

if __name__ == "__main__":
    main() 