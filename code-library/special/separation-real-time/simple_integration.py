"""
Simple integration script to get real-time segmentation working quickly.
This script provides the essential functions to connect Pylon grabber with separation.py
"""

import time
import threading
import queue
import cv2
import numpy as np
from pathlib import Path
import tempfile
import json

# Import your existing modules
from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE
from separation import UnifiedSegmentationSystem

class SimpleRealtimeSegmentation:
    """
    Simplified real-time segmentation that focuses on core functionality.
    """
    
    def __init__(self, methods_dir="zones_methods"):
        print("🚀 Initializing Simple Real-time Segmentation...")
        
        # Core components
        self.segmentation_system = UnifiedSegmentationSystem(methods_dir)
        self.camera = PylonFrameGrabber() if PYLON_AVAILABLE else None
        
        # Simple control flags
        self.running = False
        self.processing_enabled = True
        
        # Frame processing
        self.frame_count = 0
        self.processed_count = 0
        self.process_every_n_frames = 30  # Process every 30th frame
        
        # Results storage
        self.latest_result = None
        self.results_history = []
        
        print("✅ Initialization complete")
    
    def start_camera(self):
        """Start the Pylon camera."""
        if not PYLON_AVAILABLE:
            print("❌ Error: Pylon SDK not available")
            return False
        
        try:
            print("📷 Starting camera...")
            self.camera.start()
            
            # Wait for camera to be ready
            timeout = 10
            start_time = time.time()
            while not self.camera.is_running.is_set():
                if time.time() - start_time > timeout:
                    print("❌ Camera startup timeout")
                    return False
                time.sleep(0.1)
            
            print("✅ Camera started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera."""
        if self.camera:
            self.camera.stop()
            print("🛑 Camera stopped")
    
    def process_frame(self, frame):
        """Process a single frame using the segmentation system."""
        try:
            # Create temporary file for the frame
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
                cv2.imwrite(str(temp_path), frame)
            
            # Process using segmentation system
            with tempfile.TemporaryDirectory() as temp_output:
                result = self.segmentation_system.process_image(temp_path, temp_output)
            
            # Cleanup
            temp_path.unlink()
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return None
    
    def create_display_frame(self, frame, result):
        """Create display frame with segmentation overlay."""
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Add frame counter
        cv2.putText(display_frame, f"Frame: {self.frame_count} | Processed: {self.processed_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add processing status
        status = "ON" if self.processing_enabled else "OFF"
        color = (0, 255, 0) if self.processing_enabled else (0, 0, 255)
        cv2.putText(display_frame, f"Processing: {status}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add segmentation overlay if available
        if result and result.get('center') and result.get('core_radius'):
            center = (int(result['center'][0]), int(result['center'][1]))
            core_radius = int(result['core_radius'])
            cladding_radius = int(result['cladding_radius'])
            
            # Draw circles
            cv2.circle(display_frame, center, core_radius, (0, 255, 0), 2)  # Green core
            cv2.circle(display_frame, center, cladding_radius, (255, 0, 0), 2)  # Blue cladding
            cv2.circle(display_frame, center, 3, (0, 0, 255), -1)  # Red center
            
            # Add radius info
            info = f"Core: {core_radius}px | Cladding: {cladding_radius}px"
            cv2.putText(display_frame, info, (10, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add controls
        cv2.putText(display_frame, "Q=Quit | P=Pause/Resume | S=Save", 
                   (width - 350, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def save_current_result(self, frame, result):
        """Save current frame and result."""
        timestamp = int(time.time())
        save_dir = Path(f"realtime_save_{timestamp}")
        save_dir.mkdir(exist_ok=True)
        
        # Save frame
        cv2.imwrite(str(save_dir / "frame.png"), frame)
        
        # Save result if available
        if result:
            with open(save_dir / "result.json", 'w') as f:
                json.dump(result, f, indent=4, default=str)
            
            # Save masks if available
            if result.get('masks'):
                masks = result['masks']
                cv2.imwrite(str(save_dir / "core_mask.png"), masks['core'] * 255)
                cv2.imwrite(str(save_dir / "cladding_mask.png"), masks['cladding'] * 255)
                cv2.imwrite(str(save_dir / "ferrule_mask.png"), masks['ferrule'] * 255)
        
        print(f"💾 Saved to {save_dir}")
    
    def run(self):
        """Main processing loop."""
        if not self.start_camera():
            return
        
        # Setup display window
        cv2.namedWindow('Real-time Fiber Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Fiber Segmentation', 1200, 800)
        
        self.running = True
        print("🔄 Starting main processing loop...")
        print("Controls: Q=Quit, P=Pause/Resume Processing, S=Save Current Result")
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                
                # Decide whether to process this frame
                should_process = (
                    self.processing_enabled and 
                    (self.frame_count % self.process_every_n_frames == 0)
                )
                
                if should_process:
                    print(f"\n🔄 Processing frame {self.frame_count}...")
                    start_time = time.time()
                    
                    result = self.process_frame(frame)
                    
                    processing_time = time.time() - start_time
                    
                    if result:
                        self.latest_result = result
                        self.processed_count += 1
                        print(f"✅ Frame processed in {processing_time:.2f}s")
                        
                        # Store in history
                        self.results_history.append({
                            'frame_number': self.frame_count,
                            'processing_time': processing_time,
                            'result': result
                        })
                        
                        # Keep history manageable
                        if len(self.results_history) > 50:
                            self.results_history = self.results_history[-25:]
                    else:
                        print(f"❌ Frame processing failed")
                
                # Create and show display frame
                display_frame = self.create_display_frame(frame, self.latest_result)
                cv2.imshow('Real-time Fiber Segmentation', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quit requested")
                    break
                elif key == ord('p'):
                    self.processing_enabled = not self.processing_enabled
                    status = "enabled" if self.processing_enabled else "disabled"
                    print(f"🔄 Processing {status}")
                elif key == ord('s'):
                    if self.latest_result:
                        self.save_current_result(frame, self.latest_result)
                    else:
                        print("❌ No result to save")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.running = False
            cv2.destroyAllWindows()
            self.stop_camera()
            
            # Print final statistics
            print("\n📊 Final Statistics:")
            print(f"  Total frames: {self.frame_count}")
            print(f"  Processed frames: {self.processed_count}")
            if self.processed_count > 0:
                success_rate = self.processed_count / (self.frame_count // self.process_every_n_frames)
                print(f"  Success rate: {success_rate:.2%}")
            
            if self.results_history:
                avg_time = sum(r['processing_time'] for r in self.results_history) / len(self.results_history)
                print(f"  Average processing time: {avg_time:.2f}s")


# Easy-to-use function
def run_realtime_segmentation(methods_dir="zones_methods"):
    """
    Simple function to start real-time segmentation.
    
    Args:
        methods_dir: Directory containing your segmentation method scripts
    """
    
    if not PYLON_AVAILABLE:
        print("❌ Error: Pylon SDK not available")
        print("Please install pypylon: pip install pypylon")
        return
    
    # Create and run the system
    system = SimpleRealtimeSegmentation(methods_dir)
    system.run()


# Advanced function with configuration
def run_advanced_realtime_segmentation(methods_dir="zones_methods", 
                                     process_interval=30,
                                     display_size=(1200, 800)):
    """
    Advanced real-time segmentation with configuration options.
    
    Args:
        methods_dir: Directory containing segmentation methods
        process_interval: Process every Nth frame (higher = less frequent processing)
        display_size: Window size for display (width, height)
    """
    
    if not PYLON_AVAILABLE:
        print("❌ Error: Pylon SDK not available")
        return
    
    system = SimpleRealtimeSegmentation(methods_dir)
    system.process_every_n_frames = process_interval
    
    # Override display setup
    original_run = system.run
    def enhanced_run():
        if not system.start_camera():
            return
        
        cv2.namedWindow('Real-time Fiber Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Fiber Segmentation', *display_size)
        
        # Add performance monitoring
        fps_counter = 0
        fps_start_time = time.time()
        
        system.running = True
        print(f"🔄 Advanced mode: processing every {process_interval} frames")
        
        try:
            while system.running:
                frame = system.camera.read()
                if frame is None:
                    continue
                
                system.frame_count += 1
                fps_counter += 1
                
                # Calculate FPS every second
                if fps_counter % 30 == 0:
                    current_time = time.time()
                    if current_time - fps_start_time >= 1.0:
                        fps = fps_counter / (current_time - fps_start_time)
                        print(f"📊 Display FPS: {fps:.1f}")
                        fps_counter = 0
                        fps_start_time = current_time
                
                # Process frame if needed
                if (system.processing_enabled and 
                    system.frame_count % system.process_every_n_frames == 0):
                    
                    print(f"🔄 Processing frame {system.frame_count}...")
                    result = system.process_frame(frame)
                    
                    if result:
                        system.latest_result = result
                        system.processed_count += 1
                        print("✅ Success")
                
                # Display
                display_frame = system.create_display_frame(frame, system.latest_result)
                cv2.imshow('Real-time Fiber Segmentation', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    system.processing_enabled = not system.processing_enabled
                    status = "ON" if system.processing_enabled else "OFF"
                    print(f"🔄 Processing: {status}")
                elif key == ord('s'):
                    if system.latest_result:
                        system.save_current_result(frame, system.latest_result)
                elif key == ord('+'):
                    system.process_every_n_frames = max(5, system.process_every_n_frames - 5)
                    print(f"📈 Processing interval: {system.process_every_n_frames}")
                elif key == ord('-'):
                    system.process_every_n_frames = min(120, system.process_every_n_frames + 5)
                    print(f"📉 Processing interval: {system.process_every_n_frames}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        finally:
            system.running = False
            cv2.destroyAllWindows()
            system.stop_camera()
    
    # Replace run method
    system.run = enhanced_run
    system.run()


if __name__ == "__main__":
    import sys
    
    # Simple usage
    if len(sys.argv) == 1:
        run_realtime_segmentation()
    else:
        methods_dir = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2] == "advanced":
            run_advanced_realtime_segmentation(methods_dir)
        else:
            run_realtime_segmentation(methods_dir)
