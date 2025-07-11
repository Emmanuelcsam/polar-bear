#!/usr/bin/env python3
"""
Real-time Video Processing - Standalone Module
Extracted from fiber optic defect detection system
Provides real-time camera feed processing with live analysis
"""

import cv2
import numpy as np
import os
import json
import argparse
import threading
import queue
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass


@dataclass
class FrameResult:
    """Result from processing a single frame"""
    frame_id: int
    timestamp: float
    original_frame: np.ndarray
    processed_frame: np.ndarray
    analysis_results: Dict[str, Any]
    processing_time: float
    fps: float


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class FrameBuffer:
    """Thread-safe frame buffer for smooth video processing"""
    
    def __init__(self, max_size: int = 30):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.frame_counter = 0
    
    def put(self, frame: np.ndarray) -> int:
        """Add frame to buffer, returns frame ID"""
        with self.lock:
            self.frame_counter += 1
            frame_id = self.frame_counter
            self.buffer.append((frame_id, time.time(), frame))
            return frame_id
    
    def get(self) -> Optional[tuple]:
        """Get oldest frame from buffer"""
        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            return None
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()


class VideoCapture:
    """Enhanced video capture with error handling"""
    
    def __init__(self, source: int = 0):
        self.source = source
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self._setup_capture()
    
    def _setup_capture(self):
        """Setup video capture"""
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source}")
        
        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
        
        # Try to set higher resolution if available
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Get actual properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {self.width}x{self.height} @ {self.fps:.1f} FPS")
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the camera"""
        with self.lock:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                return frame if ret else None
            return None
    
    def release(self):
        """Release the camera"""
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None


class SimpleFrameProcessor:
    """Simple frame processor for real-time analysis"""
    
    def __init__(self):
        self.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and extract basic information.
        
        Args:
            frame (np.ndarray): Input frame (BGR or grayscale)
            
        Returns:
            dict: Analysis results
        """
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Basic image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        min_intensity = np.min(gray)
        max_intensity = np.max(gray)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Simple circle detection
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=100, param2=30, minRadius=10, maxRadius=200
        )
        
        num_circles = len(circles[0]) if circles is not None else 0
        
        # Find contours for shape analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        
        # Calculate largest contour properties if available
        largest_contour_area = 0
        largest_contour_circularity = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour_area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                largest_contour_circularity = 4 * np.pi * largest_contour_area / (perimeter**2)
        
        processing_time = time.time() - start_time
        
        # Update stats
        self.processing_stats['frames_processed'] += 1
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['avg_processing_time'] = (
            self.processing_stats['total_processing_time'] / 
            self.processing_stats['frames_processed']
        )
        
        return {
            'frame_stats': {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'min_intensity': int(min_intensity),
                'max_intensity': int(max_intensity),
                'contrast': float(std_intensity / mean_intensity) if mean_intensity > 0 else 0.0,
                'sharpness': float(laplacian_var),
                'edge_density': float(edge_density)
            },
            'shape_analysis': {
                'num_circles': int(num_circles),
                'num_contours': int(num_contours),
                'largest_contour_area': float(largest_contour_area),
                'largest_contour_circularity': float(largest_contour_circularity)
            },
            'processing_stats': self.processing_stats.copy(),
            'processing_time': processing_time
        }


class RealTimeProcessor:
    """Main real-time video processor"""
    
    def __init__(self, camera_source: int = 0, max_fps: int = 30):
        self.camera_source = camera_source
        self.max_fps = max_fps
        self.frame_interval = 1.0 / max_fps
        
        # Components
        self.camera = None
        self.frame_buffer = FrameBuffer()
        self.processor = SimpleFrameProcessor()
        
        # Threading
        self.capture_thread = None
        self.processing_thread = None
        self.display_thread = None
        self.running = False
        
        # Results
        self.result_queue = queue.Queue(maxsize=10)
        self.latest_result = None
        
        # Statistics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Recording
        self.recording = False
        self.video_writer = None
        self.recording_path = None
    
    def start(self):
        """Start the real-time processing"""
        try:
            # Initialize camera
            self.camera = VideoCapture(self.camera_source)
            self.running = True
            
            # Start threads
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            
            self.capture_thread.start()
            self.processing_thread.start()
            
            print("Real-time processing started")
            print("Controls:")
            print("  'q' - Quit")
            print("  's' - Save current frame")
            print("  'r' - Start/stop recording")
            print("  'p' - Pause/resume processing")
            print("  ' ' (space) - Take screenshot")
            
        except Exception as e:
            print(f"Failed to start real-time processing: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the real-time processing"""
        self.running = False
        
        # Stop recording if active
        if self.recording:
            self._stop_recording()
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        print("Real-time processing stopped")
    
    def _capture_loop(self):
        """Capture frames from camera"""
        last_frame_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Limit frame rate
            if current_time - last_frame_time < self.frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            frame = self.camera.read()
            if frame is not None:
                self.frame_buffer.put(frame)
                last_frame_time = current_time
                
                # Update FPS counter
                self.fps_counter += 1
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
    
    def _processing_loop(self):
        """Process frames"""
        frame_id_counter = 0
        
        while self.running:
            frame_data = self.frame_buffer.get()
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            frame_id, timestamp, frame = frame_data
            
            try:
                # Process frame
                analysis_results = self.processor.process_frame(frame)
                
                # Create annotated frame
                annotated_frame = self._create_annotated_frame(frame, analysis_results)
                
                # Create frame result
                result = FrameResult(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    original_frame=frame,
                    processed_frame=annotated_frame,
                    analysis_results=analysis_results,
                    processing_time=analysis_results.get('processing_time', 0),
                    fps=self.current_fps
                )
                
                # Update latest result
                self.latest_result = result
                
                # Add to result queue (non-blocking)
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Remove oldest result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                print(f"Processing error: {e}")
    
    def _create_annotated_frame(self, frame: np.ndarray, analysis_results: Dict) -> np.ndarray:
        """Create annotated frame with analysis overlay"""
        annotated = frame.copy()
        
        # Get frame stats
        frame_stats = analysis_results.get('frame_stats', {})
        shape_analysis = analysis_results.get('shape_analysis', {})
        
        # Add text overlay
        y_offset = 30
        line_height = 25
        
        # Frame statistics
        cv2.putText(annotated, f"FPS: {self.current_fps:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        cv2.putText(annotated, f"Mean: {frame_stats.get('mean_intensity', 0):.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        cv2.putText(annotated, f"Contrast: {frame_stats.get('contrast', 0):.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        cv2.putText(annotated, f"Sharpness: {frame_stats.get('sharpness', 0):.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        cv2.putText(annotated, f"Circles: {shape_analysis.get('num_circles', 0)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        # Recording indicator
        if self.recording:
            cv2.circle(annotated, (annotated.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(annotated, "REC", (annotated.shape[1] - 60, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return annotated
    
    def display_loop(self):
        """Main display loop with user interaction"""
        paused = False
        
        while self.running:
            try:
                if not paused and self.latest_result:
                    # Display the processed frame
                    cv2.imshow('Real-time Fiber Analysis', self.latest_result.processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit
                    break
                elif key == ord('s'):
                    # Save current frame
                    self._save_current_frame()
                elif key == ord('r'):
                    # Toggle recording
                    self._toggle_recording()
                elif key == ord('p'):
                    # Toggle pause
                    paused = not paused
                    print(f"Processing {'paused' if paused else 'resumed'}")
                elif key == ord(' '):
                    # Take screenshot
                    self._take_screenshot()
                
            except KeyboardInterrupt:
                break
        
        cv2.destroyAllWindows()
        self.stop()
    
    def _save_current_frame(self):
        """Save the current frame"""
        if self.latest_result:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{timestamp_str}.jpg"
            cv2.imwrite(filename, self.latest_result.original_frame)
            print(f"Frame saved: {filename}")
    
    def _take_screenshot(self):
        """Take a screenshot of the annotated frame"""
        if self.latest_result:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp_str}.jpg"
            cv2.imwrite(filename, self.latest_result.processed_frame)
            print(f"Screenshot saved: {filename}")
    
    def _toggle_recording(self):
        """Toggle video recording"""
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _start_recording(self):
        """Start video recording"""
        if self.camera:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            self.recording_path = f"recording_{timestamp_str}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                self.recording_path, fourcc, self.max_fps, 
                (self.camera.width, self.camera.height)
            )
            
            self.recording = True
            print(f"Recording started: {self.recording_path}")
    
    def _stop_recording(self):
        """Stop video recording"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print(f"Recording stopped: {self.recording_path}")


def main():
    """Command line interface for real-time processing"""
    parser = argparse.ArgumentParser(description='Real-time Fiber Optic Analysis')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera source index (default: 0)')
    parser.add_argument('--max-fps', type=int, default=30,
                       help='Maximum FPS for processing (default: 30)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display (headless mode)')
    
    args = parser.parse_args()
    
    try:
        # Create and start processor
        processor = RealTimeProcessor(
            camera_source=args.camera,
            max_fps=args.max_fps
        )
        
        processor.start()
        
        if not args.no_display:
            # Run display loop
            processor.display_loop()
        else:
            # Headless mode - just run for a while
            print("Running in headless mode. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
                    if processor.latest_result:
                        stats = processor.latest_result.analysis_results.get('frame_stats', {})
                        print(f"FPS: {processor.current_fps:.1f}, "
                              f"Mean: {stats.get('mean_intensity', 0):.1f}, "
                              f"Circles: {processor.latest_result.analysis_results.get('shape_analysis', {}).get('num_circles', 0)}")
            except KeyboardInterrupt:
                print("\nStopping...")
                processor.stop()
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
