#!/usr/bin/env python3
"""
Real-Time Integration Controller

This is the main integration controller that connects your Enhanced Pylon Frame Grabber
with the Real-Time Detector using a producer-consumer pattern with threading queues.

Key Features:
- Producer-consumer threading architecture
- Real-time frame processing pipeline
- Configurable processing rates
- Live visualization and monitoring
- Result logging and alerts
- Graceful shutdown handling
"""

import time
import threading
import queue
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Callable
import json
from dataclasses import asdict
import cv2
import numpy as np

# Import our custom modules
try:
    from enhanced_pylon_grabber import EnhancedPylonFrameGrabber
    from realtime_detector import RealTimeDetector, RealTimeConfig, DetectionResult
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Required modules not available: {e}")
    MODULES_AVAILABLE = False


class RealTimeController:
    """
    Main controller for real-time defect detection system.
    
    Orchestrates the interaction between:
    - Enhanced Pylon Frame Grabber (producer)
    - Real-Time Detector (processor) 
    - Result Handler (consumer)
    - Visualization System
    """
    
    def __init__(self, 
                 reference_image_path: str,
                 processing_fps: float = 10.0,
                 visualization: bool = True,
                 save_results: bool = True,
                 output_dir: str = "realtime_output"):
        
        self.reference_image_path = reference_image_path
        self.processing_fps = processing_fps
        self.enable_visualization = visualization
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # System components
        self.frame_grabber = None
        self.detector = None
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=5)  # Limited buffer
        self.result_queue = queue.Queue(maxsize=20)
        self.processing_thread = None
        self.result_thread = None
        self.visualization_thread = None
        
        # Control flags
        self.running = threading.Event()
        self.shutdown_requested = threading.Event()
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'defects_detected': 0,
            'average_processing_time': 0,
            'start_time': None
        }
        
        # Alert callback
        self.defect_alert_callback = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Configure logging system."""
        log_file = self.output_dir / "realtime_detection.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Real-time controller initialized - Output: {self.output_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def initialize(self) -> bool:
        """Initialize all system components."""
        if not MODULES_AVAILABLE:
            self.logger.error("Required modules not available")
            return False
        
        try:
            self.logger.info("Initializing real-time detection system...")
            
            # Initialize frame grabber
            self.logger.info("Initializing frame grabber...")
            self.frame_grabber = EnhancedPylonFrameGrabber(
                buffer_size=3,  # Small buffer for real-time
                grab_strategy="LatestImageOnly"
            )
            
            # Initialize camera
            if not self.frame_grabber.initialize_camera(exposure_time=10000):  # 10ms exposure
                self.logger.error("Failed to initialize camera")
                return False
            
            # Initialize detector
            self.logger.info("Initializing detector...")
            detector_config = RealTimeConfig(
                reference_image_path=self.reference_image_path,
                anomaly_threshold=2.0,
                ssim_threshold=0.85,
                enable_fast_mode=True,
                resize_factor=1.0,  # Full resolution
                min_defect_area=25,
                max_defect_area=5000,
                enable_visualization=self.enable_visualization,
                output_dir=str(self.output_dir)
            )
            
            self.detector = RealTimeDetector(detector_config)
            
            self.logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start the real-time detection system."""
        if not self.initialize():
            return False
        
        try:
            self.logger.info("Starting real-time detection system...")
            self.stats['start_time'] = time.time()
            
            # Start frame grabber
            self.frame_grabber.start()
            if not self.frame_grabber.wait_for_initialization(timeout=10):
                self.logger.error("Frame grabber initialization timeout")
                return False
            
            # Start processing threads
            self.running.set()
            
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="ProcessingThread",
                daemon=True
            )
            self.processing_thread.start()
            
            self.result_thread = threading.Thread(
                target=self._result_loop,
                name="ResultThread", 
                daemon=True
            )
            self.result_thread.start()
            
            if self.enable_visualization:
                self.visualization_thread = threading.Thread(
                    target=self._visualization_loop,
                    name="VisualizationThread",
                    daemon=True
                )
                self.visualization_thread.start()
            
            # Start main monitoring loop
            self._main_loop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.shutdown()
            return False
    
    def _main_loop(self):
        """Main system monitoring loop."""
        self.logger.info("Real-time detection system started successfully")
        self.logger.info("Press Ctrl+C to stop...")
        
        last_stats_time = time.time()
        
        try:
            while self.running.is_set() and not self.shutdown_requested.is_set():
                # Get latest frame from grabber
                frame, metadata = self.frame_grabber.read_latest_frame()
                
                if frame is not None:
                    self.stats['frames_captured'] += 1
                    
                    # Add frame to processing queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait((frame.copy(), metadata.copy()))
                    except queue.Full:
                        # Queue is full, skip this frame
                        pass
                
                # Print statistics every 5 seconds
                if time.time() - last_stats_time >= 5.0:
                    self._print_statistics()
                    last_stats_time = time.time()
                
                # Control processing rate
                time.sleep(1.0 / self.processing_fps)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        
        finally:
            self.shutdown()
    
    def _processing_loop(self):
        """Processing thread loop."""
        self.logger.info("Processing thread started")
        frame_id = 0
        
        while self.running.is_set():
            try:
                # Get frame from queue with timeout
                frame, metadata = self.frame_queue.get(timeout=1.0)
                frame_id += 1
                
                # Process frame
                start_time = time.time()
                result = self.detector.detect_defects(frame, frame_id)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.stats['frames_processed'] += 1
                if result.is_anomalous:
                    self.stats['defects_detected'] += 1
                    
                    # Trigger alert callback if registered
                    if self.defect_alert_callback:
                        try:
                            self.defect_alert_callback(result, frame)
                        except Exception as e:
                            self.logger.warning(f"Alert callback error: {e}")
                
                # Update average processing time
                total_frames = self.stats['frames_processed']
                current_avg = self.stats['average_processing_time']
                self.stats['average_processing_time'] = (
                    (current_avg * (total_frames - 1) + processing_time) / total_frames
                )
                
                # Send to result queue
                try:
                    self.result_queue.put_nowait((result, frame.copy(), metadata))
                except queue.Full:
                    # Result queue is full, skip
                    pass
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
        
        self.logger.info("Processing thread finished")
    
    def _result_loop(self):
        """Result handling thread loop."""
        self.logger.info("Result thread started")
        
        while self.running.is_set():
            try:
                # Get result from queue
                result, frame, metadata = self.result_queue.get(timeout=1.0)
                
                # Save results if enabled
                if self.save_results:
                    self._save_result(result, frame, metadata)
                
                # Log significant detections
                if result.is_anomalous and result.confidence > 0.7:
                    self.logger.warning(
                        f"DEFECT DETECTED - Frame {result.frame_id}: "
                        f"Confidence={result.confidence:.3f}, "
                        f"Defects={result.defect_count}, "
                        f"SSIM={result.ssim_score:.3f}"
                    )
                
                self.result_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Result handling error: {e}")
        
        self.logger.info("Result thread finished")
    
    def _visualization_loop(self):
        """Visualization thread loop."""
        self.logger.info("Visualization thread started")
        
        # Create window
        cv2.namedWindow("Real-Time Defect Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-Time Defect Detection", 1200, 800)
        
        while self.running.is_set():
            try:
                # Get result from queue (non-blocking)
                try:
                    result, frame, metadata = self.result_queue.get_nowait()
                except queue.Empty:
                    # Show last frame or continue
                    time.sleep(0.03)  # ~30 FPS display rate
                    continue
                
                # Create visualization
                vis_frame = self.detector.visualize_result(frame, result)
                
                # Add system statistics overlay
                self._add_stats_overlay(vis_frame)
                
                # Display frame
                cv2.imshow("Real-Time Defect Detection", vis_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.logger.info("Visualization window closed")
                    self.shutdown_requested.set()
                    break
                
            except Exception as e:
                self.logger.error(f"Visualization error: {e}")
        
        cv2.destroyAllWindows()
        self.logger.info("Visualization thread finished")
    
    def _add_stats_overlay(self, frame: np.ndarray):
        """Add statistics overlay to visualization frame."""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 120), (400, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Statistics text
        stats_text = [
            f"Captured: {self.stats['frames_captured']}",
            f"Processed: {self.stats['frames_processed']}",
            f"Defects: {self.stats['defects_detected']}",
            f"Avg Time: {self.stats['average_processing_time']*1000:.1f}ms",
            f"FPS: {1.0/max(self.stats['average_processing_time'], 0.001):.1f}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (15, height - 100 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _save_result(self, result: DetectionResult, frame: np.ndarray, metadata: dict):
        """Save detection result and frame."""
        timestamp = int(result.timestamp)
        
        # Save result JSON
        result_file = self.output_dir / f"result_{timestamp}_{result.frame_id:06d}.json"
        with open(result_file, 'w') as f:
            json.dump({
                'result': result.to_dict(),
                'metadata': metadata
            }, f, indent=2)
        
        # Save frame if defect detected
        if result.is_anomalous and result.confidence > 0.5:
            frame_file = self.output_dir / f"defect_{timestamp}_{result.frame_id:06d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            
            # Save visualization
            vis_frame = self.detector.visualize_result(frame, result)
            vis_file = self.output_dir / f"defect_vis_{timestamp}_{result.frame_id:06d}.jpg"
            cv2.imwrite(str(vis_file), vis_frame)
    
    def _print_statistics(self):
        """Print current system statistics."""
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        # Get additional statistics
        grabber_stats = self.frame_grabber.get_statistics() if self.frame_grabber else {}
        detector_stats = self.detector.get_statistics() if self.detector else {}
        
        self.logger.info(f"=== STATISTICS (Runtime: {runtime:.1f}s) ===")
        self.logger.info(f"Capture FPS: {grabber_stats.get('fps', 0):.1f}")
        self.logger.info(f"Processing FPS: {1.0/max(self.stats['average_processing_time'], 0.001):.1f}")
        self.logger.info(f"Frames Captured: {self.stats['frames_captured']}")
        self.logger.info(f"Frames Processed: {self.stats['frames_processed']}")
        self.logger.info(f"Defects Detected: {self.stats['defects_detected']}")
        self.logger.info(f"Detection Rate: {self.stats['defects_detected']/max(self.stats['frames_processed'], 1)*100:.1f}%")
        self.logger.info(f"Queue Sizes - Frame: {self.frame_queue.qsize()}, Result: {self.result_queue.qsize()}")
    
    def register_defect_alert(self, callback: Callable[[DetectionResult, np.ndarray], None]):
        """Register a callback function for defect alerts."""
        self.defect_alert_callback = callback
        self.logger.info("Defect alert callback registered")
    
    def shutdown(self):
        """Shutdown the system gracefully."""
        self.logger.info("Initiating system shutdown...")
        
        # Stop processing
        self.running.clear()
        
        # Stop frame grabber
        if self.frame_grabber:
            self.frame_grabber.stop()
        
        # Wait for threads to finish
        threads_to_wait = [
            self.processing_thread,
            self.result_thread,
            self.visualization_thread
        ]
        
        for thread in threads_to_wait:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Final statistics
        self._print_statistics()
        
        self.logger.info("System shutdown completed")


# Example usage and main function
def defect_alert_handler(result: DetectionResult, frame: np.ndarray):
    """Example defect alert handler."""
    print(f"🚨 DEFECT ALERT! Frame {result.frame_id} - Confidence: {result.confidence:.3f}")
    # Add your custom alert logic here (email, alarm, etc.)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python realtime_controller.py <reference_image_path>")
        print("Example: python realtime_controller.py reference_image.jpg")
        sys.exit(1)
    
    reference_image = sys.argv[1]
    
    if not Path(reference_image).exists():
        print(f"Error: Reference image not found: {reference_image}")
        sys.exit(1)
    
    print("🎥 Real-Time Defect Detection System")
    print("====================================")
    print(f"Reference Image: {reference_image}")
    print("Press Ctrl+C to stop or 'q' in visualization window")
    print()
    
    # Create controller
    controller = RealTimeController(
        reference_image_path=reference_image,
        processing_fps=10.0,  # Process 10 frames per second
        visualization=True,
        save_results=True,
        output_dir="realtime_detection_output"
    )
    
    # Register alert handler
    controller.register_defect_alert(defect_alert_handler)
    
    # Start system
    try:
        success = controller.start()
        if not success:
            print("❌ Failed to start real-time detection system")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Shutdown requested by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()