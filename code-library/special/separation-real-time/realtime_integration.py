"""
Real-time Pylon camera integration with separation.py
Enables continuous learning and segmentation on live camera frames
"""

import time
import threading
import queue
import cv2
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Optional, Dict, Any
import logging

# Import your existing modules
from pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE
from separation import UnifiedSegmentationSystem, SegmentationResult

class RealTimeSegmentationProcessor:
    """
    Real-time processor that combines Pylon camera with segmentation system.
    Uses producer-consumer pattern for efficient frame processing.
    """
    
    def __init__(self, methods_dir: str = "zones_methods", buffer_size: int = 10):
        """
        Initialize the real-time processor.
        
        Args:
            methods_dir: Directory containing segmentation method scripts
            buffer_size: Maximum number of frames to buffer
        """
        # Core components
        self.segmentation_system = UnifiedSegmentationSystem(methods_dir)
        self.frame_grabber = PylonFrameGrabber() if PYLON_AVAILABLE else None
        
        # Threading and synchronization
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.display_thread = None
        
        # Control flags
        self.is_running = threading.Event()
        self.should_process = threading.Event()
        self.frame_counter = 0
        self.processed_counter = 0
        
        # Configuration
        self.process_every_n_frames = 30  # Process every 30th frame initially
        self.adaptive_processing = True
        self.min_processing_interval = 10  # Minimum frames between processing
        self.max_processing_interval = 120  # Maximum frames between processing
        
        # Performance tracking
        self.last_process_time = 0
        self.avg_processing_time = 2.0  # Initial estimate
        self.target_fps = 30  # Target display FPS
        
        # Results storage
        self.latest_consensus = None
        self.latest_masks = None
        self.performance_history = []
        
        print(f"✓ Real-time processor initialized")
        if not PYLON_AVAILABLE:
            print("WARNING: Pylon not available - running in simulation mode")

    def start_camera(self) -> bool:
        """Start the camera and frame grabbing."""
        if not PYLON_AVAILABLE:
            print("ERROR: Pylon SDK not available")
            return False
            
        try:
            self.frame_grabber.start()
            
            # Wait for camera to start grabbing
            timeout = 10  # 10 second timeout
            start_time = time.time()
            while not self.frame_grabber.is_running.is_set():
                if time.time() - start_time > timeout:
                    print("ERROR: Camera startup timeout")
                    return False
                time.sleep(0.1)
            
            print("✓ Camera started successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start camera: {e}")
            return False

    def stop_camera(self):
        """Stop the camera and cleanup."""
        if self.frame_grabber:
            self.frame_grabber.stop()
            if self.frame_grabber.is_alive():
                self.frame_grabber.join(timeout=5)
            print("✓ Camera stopped")

    def frame_producer_thread(self):
        """Thread function that continuously grabs frames from camera."""
        print("📹 Frame producer thread started")
        
        while self.is_running.is_set():
            if not self.frame_grabber or not self.frame_grabber.is_running.is_set():
                time.sleep(0.1)
                continue
                
            # Get latest frame from grabber
            frame = self.frame_grabber.read()
            if frame is None:
                time.sleep(0.01)  # Small delay if no frame available
                continue
                
            self.frame_counter += 1
            
            # Decide whether to queue this frame for processing
            should_queue = False
            
            if self.adaptive_processing:
                # Adaptive processing based on performance
                frames_since_last = self.frame_counter - self.processed_counter
                if frames_since_last >= self.process_every_n_frames:
                    should_queue = True
            else:
                # Fixed interval processing
                if self.frame_counter % self.process_every_n_frames == 0:
                    should_queue = True
            
            if should_queue and self.should_process.is_set():
                try:
                    # Try to put frame in queue (non-blocking)
                    self.frame_queue.put_nowait({
                        'frame': frame.copy(),
                        'frame_number': self.frame_counter,
                        'timestamp': time.time()
                    })
                except queue.Full:
                    # Queue is full, skip this frame
                    pass
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)
        
        print("📹 Frame producer thread finished")

    def frame_processor_thread(self):
        """Thread function that processes queued frames."""
        print("🔧 Frame processor thread started")
        
        while self.is_running.is_set() or not self.frame_queue.empty():
            try:
                # Get frame from queue with timeout
                frame_data = self.frame_queue.get(timeout=1.0)
                
                if not self.should_process.is_set():
                    continue
                
                print(f"\n🔄 Processing frame {frame_data['frame_number']} "
                      f"(queue size: {self.frame_queue.qsize()})")
                
                # Process the frame
                start_time = time.time()
                consensus = self._process_single_frame(
                    frame_data['frame'], 
                    frame_data['frame_number']
                )
                processing_time = time.time() - start_time
                
                # Update performance tracking
                self.avg_processing_time = (
                    0.8 * self.avg_processing_time + 0.2 * processing_time
                )
                
                # Adaptive processing interval adjustment
                if self.adaptive_processing:
                    self._adjust_processing_interval(processing_time)
                
                # Store results
                if consensus:
                    self.latest_consensus = consensus
                    self.latest_masks = consensus['masks']
                    
                    # Put result in result queue for display
                    self.result_queue.put({
                        'frame_number': frame_data['frame_number'],
                        'consensus': consensus,
                        'processing_time': processing_time,
                        'timestamp': time.time()
                    })
                
                self.processed_counter = frame_data['frame_number']
                self.performance_history.append({
                    'frame_number': frame_data['frame_number'],
                    'processing_time': processing_time,
                    'timestamp': time.time()
                })
                
                # Keep only recent history
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-50:]
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"ERROR in frame processor: {e}")
                continue
        
        print("🔧 Frame processor thread finished")

    def _process_single_frame(self, frame: np.ndarray, frame_number: int) -> Optional[Dict]:
        """Process a single frame using the segmentation system."""
        try:
            # Create temporary file for frame
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
                cv2.imwrite(str(temp_path), frame)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_output:
                output_dir = Path(temp_output)
                
                # Process using segmentation system
                consensus = self.segmentation_system.process_image(temp_path, str(output_dir))
                
                # Cleanup temporary file
                temp_path.unlink()
                
                return consensus
                
        except Exception as e:
            print(f"ERROR processing frame {frame_number}: {e}")
            return None

    def _adjust_processing_interval(self, processing_time: float):
        """Dynamically adjust processing interval based on performance."""
        # Calculate target interval based on processing time and target FPS
        target_interval = max(
            int(processing_time * self.target_fps * 2),  # 2x safety factor
            self.min_processing_interval
        )
        target_interval = min(target_interval, self.max_processing_interval)
        
        # Smooth adjustment
        self.process_every_n_frames = int(
            0.7 * self.process_every_n_frames + 0.3 * target_interval
        )
        
        print(f"📊 Adjusted processing interval to {self.process_every_n_frames} frames "
              f"(processing time: {processing_time:.2f}s)")

    def display_thread_func(self):
        """Thread function for displaying live feed with overlay."""
        print("🖥️ Display thread started")
        
        cv2.namedWindow('Real-time Fiber Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Fiber Segmentation', 1200, 800)
        
        last_display_frame = None
        last_result = None
        
        while self.is_running.is_set():
            # Get latest frame for display
            if self.frame_grabber:
                current_frame = self.frame_grabber.read()
                if current_frame is not None:
                    last_display_frame = current_frame.copy()
            
            # Check for new processing results
            try:
                result = self.result_queue.get_nowait()
                last_result = result
            except queue.Empty:
                pass
            
            # Create display frame
            if last_display_frame is not None:
                display_frame = self._create_display_frame(last_display_frame, last_result)
                cv2.imshow('Real-time Fiber Segmentation', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested by user")
                self.stop()
                break
            elif key == ord('p'):
                # Toggle processing
                if self.should_process.is_set():
                    self.should_process.clear()
                    print("⏸️ Processing paused")
                else:
                    self.should_process.set()
                    print("▶️ Processing resumed")
            elif key == ord('s'):
                # Save current results
                if last_result:
                    self._save_current_results(last_display_frame, last_result)
            
            time.sleep(0.033)  # ~30 FPS display
        
        cv2.destroyAllWindows()
        print("🖥️ Display thread finished")

    def _create_display_frame(self, frame: np.ndarray, result: Optional[Dict]) -> np.ndarray:
        """Create display frame with overlays."""
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Add status overlay
        status_text = f"Frame: {self.frame_counter} | Processed: {self.processed_counter}"
        status_text += f" | Queue: {self.frame_queue.qsize()}"
        status_text += f" | Interval: {self.process_every_n_frames}"
        
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add processing status
        if self.should_process.is_set():
            cv2.putText(display_frame, "PROCESSING: ON", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "PROCESSING: OFF", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add segmentation overlay if available
        if result and result.get('consensus'):
            consensus = result['consensus']
            
            # Draw center and radii
            if consensus.get('center') and consensus.get('core_radius'):
                center = (int(consensus['center'][0]), int(consensus['center'][1]))
                core_radius = int(consensus['core_radius'])
                cladding_radius = int(consensus['cladding_radius'])
                
                # Draw circles
                cv2.circle(display_frame, center, core_radius, (0, 255, 0), 2)  # Green for core
                cv2.circle(display_frame, center, cladding_radius, (255, 0, 0), 2)  # Blue for cladding
                cv2.circle(display_frame, center, 3, (0, 0, 255), -1)  # Red center point
                
                # Add text info
                info_text = f"Core: {core_radius}px | Cladding: {cladding_radius}px"
                cv2.putText(display_frame, info_text, (10, height - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                methods_text = f"Methods: {', '.join(consensus.get('contributing_methods', []))}"
                cv2.putText(display_frame, methods_text, (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add controls help
        help_text = "Controls: Q=Quit, P=Pause/Resume, S=Save"
        cv2.putText(display_frame, help_text, (width - 400, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame

    def _save_current_results(self, frame: np.ndarray, result: Dict):
        """Save current frame and segmentation results."""
        timestamp = int(time.time())
        save_dir = Path(f"realtime_results_{timestamp}")
        save_dir.mkdir(exist_ok=True)
        
        # Save original frame
        cv2.imwrite(str(save_dir / "frame.png"), frame)
        
        # Save consensus results if available
        if result.get('consensus'):
            consensus = result['consensus']
            
            # Save JSON report
            with open(save_dir / "consensus_report.json", 'w') as f:
                json.dump(consensus, f, indent=4, cls=self.segmentation_system.NumpyEncoder)
            
            # Save masks if available
            masks = consensus.get('masks')
            if masks:
                cv2.imwrite(str(save_dir / "mask_core.png"), masks['core'] * 255)
                cv2.imwrite(str(save_dir / "mask_cladding.png"), masks['cladding'] * 255)
                cv2.imwrite(str(save_dir / "mask_ferrule.png"), masks['ferrule'] * 255)
        
        print(f"💾 Results saved to {save_dir}")

    def start(self):
        """Start the real-time processing system."""
        if not PYLON_AVAILABLE:
            print("ERROR: Cannot start - Pylon SDK not available")
            return False
        
        print("🚀 Starting real-time segmentation system...")
        
        # Start camera
        if not self.start_camera():
            return False
        
        # Set running flag
        self.is_running.set()
        self.should_process.set()  # Start with processing enabled
        
        # Start threads
        self.processing_thread = threading.Thread(
            target=self.frame_processor_thread, 
            name="FrameProcessor"
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start frame producer thread
        producer_thread = threading.Thread(
            target=self.frame_producer_thread,
            name="FrameProducer"
        )
        producer_thread.daemon = True
        producer_thread.start()
        
        # Start display thread
        self.display_thread = threading.Thread(
            target=self.display_thread_func,
            name="DisplayThread"
        )
        self.display_thread.daemon = True
        self.display_thread.start()
        
        print("✅ Real-time system started successfully")
        print("\nControls:")
        print("  Q - Quit application")
        print("  P - Pause/Resume processing")
        print("  S - Save current results")
        print("\nWaiting for threads to finish...")
        
        # Wait for threads to complete
        try:
            self.display_thread.join()
            self.processing_thread.join()
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.stop()
        
        return True

    def stop(self):
        """Stop the real-time processing system."""
        print("🛑 Stopping real-time system...")
        
        # Clear running flags
        self.is_running.clear()
        self.should_process.clear()
        
        # Stop camera
        self.stop_camera()
        
        print("✅ Real-time system stopped")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_history:
            return {}
        
        recent_times = [p['processing_time'] for p in self.performance_history[-10:]]
        
        return {
            'total_frames': self.frame_counter,
            'processed_frames': self.processed_counter,
            'processing_rate': self.processed_counter / max(self.frame_counter, 1),
            'avg_processing_time': self.avg_processing_time,
            'recent_avg_time': sum(recent_times) / len(recent_times) if recent_times else 0,
            'current_interval': self.process_every_n_frames,
            'queue_size': self.frame_queue.qsize()
        }


def main():
    """Main function to run the real-time segmentation system."""
    import sys
    
    methods_dir = sys.argv[1] if len(sys.argv) > 1 else "zones_methods"
    
    # Create and start the real-time processor
    processor = RealTimeSegmentationProcessor(methods_dir)
    
    try:
        success = processor.start()
        if not success:
            print("Failed to start real-time processor")
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
