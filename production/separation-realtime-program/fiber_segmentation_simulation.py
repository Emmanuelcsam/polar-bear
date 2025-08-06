"""
Simulation version of the fiber segmentation real-time system.
Generates synthetic camera frames for testing without requiring a physical camera.
"""

import time
import threading
import queue
import cv2
import numpy as np
from pathlib import Path
import tempfile
import json
import os
import sys
import subprocess
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import matplotlib for visualizations
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualizations will be skipped")

# Import scipy components for enhanced processing
try:
    from scipy.ndimage import median_filter, gaussian_filter
    from scipy.ndimage import binary_opening, binary_closing
    HAS_SCIPY_FULL = True
except ImportError:
    HAS_SCIPY_FULL = False
    print("Warning: Some scipy components not available, using basic processing")

# Import the main system components
from fiber_segmentation_realtime import (
    NumpyEncoder, SegmentationResult, EnhancedConsensusSystem,
    UnifiedSegmentationSystem, RealtimeSegmentationProcessor
)


class SimulatedFrameGrabber(threading.Thread):
    """Simulated camera that generates synthetic frames for testing."""
    
    def __init__(self, frame_rate: int = 30):
        super().__init__(name="SimulatedGrabber")
        self.daemon = True
        self.latest_frame = None
        self.is_running = threading.Event()
        self.lock = threading.Lock()
        self.frame_rate = frame_rate
        self.frame_counter = 0
        
        # Simulation parameters
        self.fiber_center = (320, 240)
        self.core_radius = 50
        self.cladding_radius = 100
        self.noise_level = 0.1
        self.movement_speed = 2.0
        
    def run(self):
        """Generate synthetic frames continuously."""
        logging.info("SimulatedFrameGrabber thread started.")
        
        self.is_running.set()
        logging.info("Simulated camera started generating frames.")
        
        while self.is_running.is_set():
            # Generate a synthetic frame
            frame = self._generate_synthetic_frame()
            
            with self.lock:
                self.latest_frame = frame.copy()
            
            # Control frame rate
            time.sleep(1.0 / self.frame_rate)
        
        logging.info("SimulatedFrameGrabber thread finished.")
    
    def _generate_synthetic_frame(self):
        """Generate a synthetic fiber optic image."""
        # Create base image
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add some background variation
        noise = np.random.normal(0, 20, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add fiber structure
        y, x = np.ogrid[:480, :640]
        
        # Calculate distances from center
        dist_from_center = np.sqrt((x - self.fiber_center[0])**2 + 
                                 (y - self.fiber_center[1])**2)
        
        # Create core (bright center)
        core_mask = dist_from_center <= self.core_radius
        frame[core_mask] = [255, 255, 255]  # White core
        
        # Create cladding (medium brightness ring)
        cladding_mask = ((dist_from_center > self.core_radius) & 
                        (dist_from_center <= self.cladding_radius))
        frame[cladding_mask] = [180, 180, 180]  # Gray cladding
        
        # Add some realistic variations
        # Add some defects occasionally
        if np.random.random() < 0.1:  # 10% chance of defect
            defect_x = np.random.randint(0, 640)
            defect_y = np.random.randint(0, 480)
            defect_size = np.random.randint(5, 15)
            cv2.circle(frame, (defect_x, defect_y), defect_size, (50, 50, 50), -1)
        
        # Add some noise
        noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        # Ensure values are in valid range
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Simulate slight movement
        self.fiber_center = (
            int(self.fiber_center[0] + np.random.normal(0, self.movement_speed)),
            int(self.fiber_center[1] + np.random.normal(0, self.movement_speed))
        )
        
        # Keep center within bounds
        self.fiber_center = (
            max(100, min(540, self.fiber_center[0])),
            max(100, min(380, self.fiber_center[1]))
        )
        
        self.frame_counter += 1
        return frame
    
    def read(self):
        """Returns the most recent frame."""
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()
    
    def stop(self):
        """Signals the thread to stop."""
        logging.info("Stopping SimulatedFrameGrabber thread.")
        self.is_running.clear()


class SimulatedRealtimeSegmentationProcessor(RealtimeSegmentationProcessor):
    """Real-time processor with simulated camera for testing."""
    
    def __init__(self, methods_dir: str = "zones_methods", buffer_size: int = 10):
        # Initialize parent class
        super().__init__(methods_dir, buffer_size)
        
        # Replace camera with simulation
        self.frame_grabber = SimulatedFrameGrabber(frame_rate=30)
        
        print(f"✓ Simulated real-time processor initialized")
    
    def start_camera(self) -> bool:
        """Start the simulated camera."""
        try:
            self.frame_grabber.start()
            
            # Wait for camera to start generating frames
            timeout = 5  # 5 second timeout
            start_time = time.time()
            while not self.frame_grabber.is_running.is_set():
                if time.time() - start_time > timeout:
                    print("ERROR: Simulated camera startup timeout")
                    return False
                time.sleep(0.1)
            
            print("✓ Simulated camera started successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start simulated camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the simulated camera."""
        if self.frame_grabber:
            self.frame_grabber.stop()
            if self.frame_grabber.is_alive():
                self.frame_grabber.join(timeout=5)
            print("✓ Simulated camera stopped")


def main():
    """Main function to run the simulated real-time segmentation system."""
    import sys
    
    methods_dir = sys.argv[1] if len(sys.argv) > 1 else "zones_methods"
    
    print("🎮 Starting SIMULATED real-time segmentation system...")
    print("This version uses synthetic camera frames for testing.")
    print("No physical camera required.")
    
    # Create and start the simulated real-time processor
    processor = SimulatedRealtimeSegmentationProcessor(methods_dir)
    
    try:
        success = processor.start()
        if not success:
            print("Failed to start simulated real-time processor")
            return
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        processor.stop()
        
        # Print final statistics
        stats = processor.get_performance_stats()
        if stats:
            print("\n📊 Final Performance Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 