#!/usr/bin/env python3
"""
Enhanced Pylon Frame Grabber for Real-Time Integration

This enhanced version of your PylonFrameGrabber provides better performance
and integration capabilities for real-time defect detection.

Key Improvements:
- Optimized frame buffer management
- Better error handling and recovery
- Frame metadata tracking
- Performance monitoring
- Configurable grab strategies
"""

import time
import threading
import logging
import queue
from collections import deque
import numpy as np

# Pylon SDK availability check (from your original code)
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
    print("INFO: Pylon SDK found. Enhanced Basler camera support is enabled.")
    
    try:
        from genicam import GenericException
    except ImportError:
        class GenericException(Exception):
            pass
            
except ImportError:
    print("WARNING: Pylon SDK not found. Cannot use Basler camera.")
    print("Please install pypylon: pip install pypylon")


class EnhancedPylonFrameGrabber(threading.Thread):
    """
    Enhanced real-time frame grabber with improved performance and monitoring.
    
    Features:
    - Real-time frame statistics
    - Configurable buffer management
    - Frame metadata tracking
    - Improved error recovery
    - Performance optimization for detection integration
    """
    
    def __init__(self, camera_index=0, buffer_size=5, grab_strategy="LatestImageOnly"):
        super().__init__(name="EnhancedPylonGrabber")
        self.daemon = True
        
        # Camera configuration
        self.camera_index = camera_index
        self.camera = None
        self.grab_strategy = grab_strategy
        
        # Frame management
        self.latest_frame = None
        self.frame_metadata = {}
        self.lock = threading.RLock()  # Reentrant lock for better performance
        
        # Buffer management
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Control flags
        self.is_running = threading.Event()
        self.is_initialized = threading.Event()
        
        # Performance monitoring
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.processing_times = deque(maxlen=100)  # Track processing performance
        
        # Error handling
        self.error_count = 0
        self.last_error = None
        self.max_errors = 10
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        if PYLON_AVAILABLE:
            self._setup_converter()
    
    def _setup_converter(self):
        """Initialize image format converter for OpenCV compatibility."""
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    def initialize_camera(self, exposure_time=None, gain=None):
        """
        Initialize camera with optional parameters.
        
        Args:
            exposure_time (float): Exposure time in microseconds
            gain (float): Camera gain value
            
        Returns:
            bool: True if initialization successful
        """
        if not PYLON_AVAILABLE:
            self.logger.error("Pylon SDK not available")
            return False
        
        try:
            # Create and configure camera
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            
            self.camera.Open()
            self.logger.info(f"Camera initialized: {self.camera.GetDeviceInfo().GetModelName()}")
            
            # Configure camera parameters
            if exposure_time is not None:
                self.camera.ExposureTime.SetValue(exposure_time)
                self.logger.info(f"Exposure time set to: {exposure_time} μs")
            
            if gain is not None:
                self.camera.Gain.SetValue(gain)
                self.logger.info(f"Gain set to: {gain}")
            
            # Set buffer count for optimal performance
            self.camera.MaxNumBuffer = self.buffer_size + 2
            
            # Configure grab strategy
            if self.grab_strategy == "LatestImageOnly":
                grab_strategy = pylon.GrabStrategy_LatestImageOnly
            else:
                grab_strategy = pylon.GrabStrategy_OneByOne
            
            self.camera.StartGrabbing(grab_strategy)
            self.is_initialized.set()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            self.last_error = str(e)
            return False
    
    def run(self):
        """Enhanced main grabbing loop with performance monitoring."""
        self.logger.info("Enhanced frame grabber thread started")
        
        if not PYLON_AVAILABLE:
            self.logger.critical("Pylon SDK not available - cannot run")
            return
        
        # Wait for initialization
        if not self.is_initialized.wait(timeout=10):
            self.logger.error("Camera initialization timeout")
            return
        
        self.is_running.set()
        self.last_fps_time = time.time()
        
        try:
            while self.is_running.is_set() and self.camera.IsGrabbing():
                try:
                    grab_start_time = time.time()
                    
                    # Retrieve frame with timeout
                    grab_result = self.camera.RetrieveResult(
                        1000, pylon.TimeoutHandling_Return
                    )
                    
                    if grab_result and grab_result.GrabSucceeded():
                        self._process_frame(grab_result, grab_start_time)
                        self.error_count = 0  # Reset error count on success
                    else:
                        self._handle_grab_failure(grab_result)
                    
                    if grab_result:
                        grab_result.Release()
                
                except GenericException as e:
                    self._handle_error(f"GenICam error: {e}")
                except Exception as e:
                    self._handle_error(f"Unexpected error: {e}")
                
                # Update FPS calculation
                self._update_fps()
        
        finally:
            self._cleanup()
    
    def _process_frame(self, grab_result, grab_start_time):
        """Process successfully grabbed frame."""
        try:
            # Convert frame format
            image = self.converter.Convert(grab_result)
            frame_array = image.GetArray().copy()  # Create copy for thread safety
            
            # Create frame metadata
            processing_time = time.time() - grab_start_time
            metadata = {
                'timestamp': time.time(),
                'frame_number': grab_result.GetBlockID(),
                'processing_time': processing_time,
                'camera_timestamp': grab_result.GetTimeStamp(),
                'frame_size': frame_array.shape
            }
            
            # Thread-safe frame update
            with self.lock:
                self.latest_frame = frame_array
                self.frame_metadata = metadata
                self.frame_buffer.append((frame_array.copy(), metadata.copy()))
                self.frame_count += 1
                self.processing_times.append(processing_time)
        
        except Exception as e:
            self.logger.warning(f"Frame processing error: {e}")
            self.dropped_frames += 1
    
    def _handle_grab_failure(self, grab_result):
        """Handle failed frame grab."""
        if grab_result:
            error_msg = f"Grab failed: {grab_result.GetErrorDescription()}"
        else:
            error_msg = "Grab result is None (timeout)"
        
        self._handle_error(error_msg)
    
    def _handle_error(self, error_msg):
        """Centralized error handling."""
        self.error_count += 1
        self.last_error = error_msg
        
        if self.error_count <= 3:  # Log first few errors
            self.logger.warning(f"Error {self.error_count}: {error_msg}")
        elif self.error_count >= self.max_errors:
            self.logger.critical(f"Too many errors ({self.error_count}). Stopping grabber.")
            self.stop()
        
        time.sleep(0.01)  # Brief pause to prevent tight error loops
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            time_diff = current_time - self.last_fps_time
            self.current_fps = self.frame_count / time_diff if time_diff > 0 else 0
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _cleanup(self):
        """Clean up camera resources."""
        try:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
                self.logger.info("Camera stopped grabbing")
            
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
                self.logger.info("Camera closed")
        
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
        
        finally:
            self.is_running.clear()
            self.logger.info("Enhanced frame grabber thread finished")
    
    def read_latest_frame(self):
        """
        Get the most recent frame with metadata.
        
        Returns:
            tuple: (frame_array, metadata) or (None, None) if no frame available
        """
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.frame_metadata.copy()
            return None, None
    
    def read_buffered_frames(self, count=1):
        """
        Get multiple recent frames from buffer.
        
        Args:
            count (int): Number of frames to retrieve
            
        Returns:
            list: List of (frame_array, metadata) tuples
        """
        with self.lock:
            if not self.frame_buffer:
                return []
            
            # Return the most recent 'count' frames
            return list(self.frame_buffer)[-count:]
    
    def get_statistics(self):
        """
        Get current performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        with self.lock:
            avg_processing_time = (
                np.mean(self.processing_times) 
                if self.processing_times else 0
            )
            
            return {
                'fps': self.current_fps,
                'total_frames': sum(self.processing_times.__len__() for _ in [self.processing_times]),
                'dropped_frames': self.dropped_frames,
                'error_count': self.error_count,
                'last_error': self.last_error,
                'avg_processing_time_ms': avg_processing_time * 1000,
                'buffer_size': len(self.frame_buffer),
                'is_running': self.is_running.is_set(),
                'is_initialized': self.is_initialized.is_set()
            }
    
    def stop(self):
        """Stop the frame grabber gracefully."""
        self.logger.info("Stopping enhanced frame grabber...")
        self.is_running.clear()
    
    def wait_for_initialization(self, timeout=10):
        """
        Wait for camera initialization to complete.
        
        Args:
            timeout (float): Maximum wait time in seconds
            
        Returns:
            bool: True if initialized successfully
        """
        return self.is_initialized.wait(timeout)
    
    def is_healthy(self):
        """
        Check if the grabber is running healthily.
        
        Returns:
            bool: True if grabber is healthy
        """
        return (
            self.is_running.is_set() and 
            self.error_count < self.max_errors and
            (time.time() - self.frame_metadata.get('timestamp', 0)) < 5.0  # Recent frame
        )


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create enhanced grabber
    grabber = EnhancedPylonFrameGrabber(
        buffer_size=10,
        grab_strategy="LatestImageOnly"
    )
    
    try:
        # Initialize camera
        if grabber.initialize_camera(exposure_time=10000):  # 10ms exposure
            print("Camera initialized successfully")
            
            # Start grabbing
            grabber.start()
            
            if grabber.wait_for_initialization():
                print("Frame grabber started successfully")
                
                # Monitor for 10 seconds
                for i in range(10):
                    time.sleep(1)
                    stats = grabber.get_statistics()
                    print(f"FPS: {stats['fps']:.1f}, "
                          f"Errors: {stats['error_count']}, "
                          f"Avg proc time: {stats['avg_processing_time_ms']:.2f}ms")
                    
                    # Get latest frame
                    frame, metadata = grabber.read_latest_frame()
                    if frame is not None:
                        print(f"Frame shape: {frame.shape}, "
                              f"Timestamp: {metadata['timestamp']:.3f}")
        else:
            print("Camera initialization failed")
    
    finally:
        grabber.stop()
        if grabber.is_alive():
            grabber.join(timeout=5)
        print("Enhanced frame grabber test completed")