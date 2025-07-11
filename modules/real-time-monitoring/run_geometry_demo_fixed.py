#!/usr/bin/env python3
"""
Fixed geometry demo launcher that handles Qt conflicts
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Fix Qt environment before importing OpenCV
import fix_qt_env

# Fix matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt conflicts

# Now safe to import
import cv2
import numpy as np
from src.core.integrated_geometry_system import GeometryDetectionSystem, GeometryDetector, Config, setup_logging
from datetime import datetime
import logging
import shared_config # Import the shared configuration module

class SimpleGeometryDemo:
    """Simplified geometry demo without matplotlib dashboard"""
    
    def __init__(self, camera_source=None):
        self.logger = logging.getLogger("SimpleGeometryDemo")
        
        # Load initial configuration from shared_config
        self.current_config = shared_config.get_config()
        self.camera_source = self.current_config.get("camera_source", camera_source if camera_source is not None else 0)
        self.show_info = self.current_config.get("show_info_overlay", True)
        
        self.detector = GeometryDetector()
        self.frame_count = 0
        self.running = False
        self.paused = False
        self.status = "initialized"

    def get_script_info(self):
        return {
            "name": "Simple Geometry Demo (Fixed)",
            "status": self.status,
            "parameters": {
                "camera_source": self.camera_source,
                "show_info_overlay": self.show_info,
                "log_level": self.current_config.get("log_level"),
                "data_source": self.current_config.get("data_source"),
                "processing_enabled": self.current_config.get("processing_enabled"),
                "threshold_value": self.current_config.get("threshold_value")
            },
            "frame_count": self.frame_count
        }

    def set_script_parameter(self, key, value):
        if key in self.current_config:
            self.current_config[key] = value
            shared_config.set_config_value(key, value)
            
            if key == "camera_source":
                self.camera_source = value
                # Note: Re-initializing camera mid-run is complex, might require restart
            elif key == "show_info_overlay":
                self.show_info = value
            
            self.status = f"parameter '{key}' updated"
            return True
        return False

    def process_frame(self, frame):
        """Process frame and draw results"""
        # Detect shapes
        shapes = self.detector.detect_shapes(frame)
        
        # Draw shapes
        output = frame.copy()
        shape_counts = {}
        
        for shape in shapes:
            # Count shapes by type
            shape_type = shape.shape_type.name
            shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
            
            # Draw shape
            color = {
                'CIRCLE': (0, 255, 0),
                'RECTANGLE': (255, 0, 0),
                'TRIANGLE': (0, 0, 255),
                'POLYGON': (255, 255, 0)
            }.get(shape_type, (255, 255, 255))
            
            # Draw contour
            cv2.drawContours(output, [shape.contour], -1, color, 2)
            
            # Draw center
            cx, cy = shape.center
            cv2.circle(output, (int(cx), int(cy)), 3, color, -1)
            
            # Draw label
            label = f"{shape_type} ({shape.confidence:.2f})"
            cv2.putText(output, label, (int(cx-30), int(cy-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw info overlay
        if self.show_info:
            self._draw_info(output, shape_counts)
            
        return output
    
    def _draw_info(self, image, shape_counts):
        """Draw information overlay"""
        h, w = image.shape[:2]
        
        # Semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw text
        y = 30
        cv2.putText(image, "Geometry Detection Demo", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y += 30
        cv2.putText(image, f"Frame: {self.frame_count}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 25
        cv2.putText(image, "Detected Shapes:", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for shape_type, count in shape_counts.items():
            y += 20
            cv2.putText(image, f"  {shape_type}: {count}", (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(image, "Press 'q' to quit, 'i' to toggle info", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Run the demo"""
        print("Starting Simple Geometry Demo...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'i' - Toggle info overlay")
        print("  's' - Save screenshot")
        print("-" * 50)
        
        # Try V4L2 backend first for Linux
        cap = cv2.VideoCapture(self.camera_source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_source)
            
        if not cap.isOpened():
            print("ERROR: Failed to open camera!")
            self.status = "camera_error"
            return
            
        # Get camera info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera opened: {width}x{height} @ {fps} FPS")
        
        self.running = True
        self.status = "running"

        try:
            while self.running:
                if self.paused:
                    cv2.waitKey(50)
                    continue

                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame!")
                    break
                    
                self.frame_count += 1
                
                # Process and display only if processing is enabled
                if self.current_config.get("processing_enabled", True):
                    output = self.process_frame(frame)
                else:
                    output = frame.copy() # Display original frame if processing is disabled

                cv2.imshow('Geometry Detection Demo', output)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('i'):
                    self.show_info = not self.show_info
                elif key == ord('s'):
                    filename = f"geometry_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(filename, output)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('p'): # Add pause functionality
                    self.paused = not self.paused
                    self.status = "paused" if self.paused else "running"
                    print(f"Demo {'paused' if self.paused else 'resumed'}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.status = "stopped"
            print("Demo finished")

demo_instance = None

def get_script_info():
    """Returns information about the script, its status, and exposed parameters."""
    if demo_instance:
        return demo_instance.get_script_info()
    return {"name": "Simple Geometry Demo (Fixed)", "status": "not_initialized", "parameters": {}}

def set_script_parameter(key, value):
    """Sets a specific parameter for the script and updates shared_config."""
    if demo_instance:
        return demo_instance.set_script_parameter(key, value)
    return False

def main():
    """Main entry point"""
    global demo_instance

    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Create and run demo
    demo_instance = SimpleGeometryDemo()
    demo_instance.run()

if __name__ == "__main__":
    main()