"""
Real-time Video Processing Module
- Live camera feed processing
- Real-time defect detection
- Performance-optimized pipeline
- No argparse
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque
from dataclasses import dataclass
import threading
import queue
import time
from datetime import datetime

from config_manager import get_config
from enhanced_logging import get_logger, log_execution, log_performance
from process import EnhancedProcessor
from separation import EnhancedSeparator
from detection import EnhancedDetector, Defect

logger = get_logger(__name__)


@dataclass
class FrameResult:
    """Result from processing a single frame"""
    frame_id: int
    timestamp: float
    original_frame: np.ndarray
    processed_frame: np.ndarray
    zones: Dict[str, np.ndarray]
    defects: List[Defect]
    processing_time: float
    fps: float


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
    
    def get(self) -> Optional[Tuple[int, float, np.ndarray]]:
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
        
        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
        
        # Try to set higher resolution if available
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Get actual properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        logger.info(f"Video capture initialized", 
                   width=self.width, height=self.height, fps=self.fps)
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame from capture"""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None
            
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None
    
    def release(self):
        """Release video capture"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


class OptimizedPipeline:
    """Optimized processing pipeline for real-time performance"""
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize components with lightweight settings
        self._setup_lightweight_components()
        
        # Cache for reusable data
        self.cache = {}
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
    
    def _setup_lightweight_components(self):
        """Setup components with real-time optimizations"""
        # Reduce number of variations for real-time
        original_variations = self.config.processing.num_variations
        self.config.processing.num_variations = min(10, original_variations)
        
        # Disable parallel processing to reduce overhead
        self.config.processing.parallel_processing = False
        self.config.separation.parallel_execution = False
        
        # Initialize components
        self.processor = EnhancedProcessor()
        self.separator = EnhancedSeparator()
        self.detector = EnhancedDetector()
        
        # Restore config
        self.config.processing.num_variations = original_variations
    
    @log_performance
    def process_frame(self, frame: np.ndarray, frame_id: int) -> FrameResult:
        """Process a single frame"""
        start_time = time.time()
        
        # Resize if needed for performance
        if self.config.processing.image_resize_factor != 1.0:
            h, w = frame.shape[:2]
            new_h = int(h * self.config.processing.image_resize_factor)
            new_w = int(w * self.config.processing.image_resize_factor)
            resized_frame = cv2.resize(frame, (new_w, new_h))
        else:
            resized_frame = frame
        
        # Quick preprocessing (limited variations)
        variations = self._quick_preprocess(resized_frame)
        
        # Zone separation (use cached if similar)
        zones = self._quick_separation(resized_frame, variations)
        
        # Defect detection
        defects = self.detector.detect_defects(resized_frame, zones, variations)
        
        # Scale defects back if resized
        if self.config.processing.image_resize_factor != 1.0:
            scale = 1.0 / self.config.processing.image_resize_factor
            for defect in defects:
                defect.location = (
                    int(defect.location[0] * scale),
                    int(defect.location[1] * scale)
                )
                defect.bbox = (
                    int(defect.bbox[0] * scale),
                    int(defect.bbox[1] * scale),
                    int(defect.bbox[2] * scale),
                    int(defect.bbox[3] * scale)
                )
        
        # Create visualization
        processed_frame = self._create_visualization(frame, zones, defects)
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        avg_time = np.mean(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return FrameResult(
            frame_id=frame_id,
            timestamp=time.time(),
            original_frame=frame,
            processed_frame=processed_frame,
            zones=zones,
            defects=defects,
            processing_time=processing_time,
            fps=fps
        )
    
    def _quick_preprocess(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Quick preprocessing with essential variations only"""
        variations = {"original": frame}
        
        # Essential variations for real-time
        essential_transforms = [
            ("blur_gaussian", lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
            ("enhance_clahe", self._apply_clahe),
            ("edge_canny", lambda img: cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)),
            ("threshold_otsu", self._apply_otsu)
        ]
        
        for name, transform in essential_transforms:
            try:
                variations[name] = transform(frame)
            except Exception as e:
                logger.warning(f"Transform {name} failed: {e}")
        
        return variations
    
    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE enhancement"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def _apply_otsu(self, img: np.ndarray) -> np.ndarray:
        """Apply Otsu thresholding"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result
    
    def _quick_separation(self, frame: np.ndarray, variations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Quick zone separation using cached or simplified method"""
        # Check cache for similar frame
        frame_hash = self._compute_frame_hash(frame)
        
        if frame_hash in self.cache:
            logger.debug("Using cached separation")
            return self.cache[frame_hash]
        
        # Simplified separation for speed
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        # Use Hough circles for quick detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=int(min(center) * 0.1),
            maxRadius=int(min(center) * 0.9)
        )
        
        zones = {}
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Sort by radius
            circles = sorted(circles, key=lambda x: x[2])
            
            # Create masks based on detected circles
            if len(circles) >= 1:
                # Core (smallest circle)
                core_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(core_mask, (circles[0][0], circles[0][1]), circles[0][2], 255, -1)
                zones['core'] = core_mask
            
            if len(circles) >= 2:
                # Cladding (between first and second circle)
                cladding_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(cladding_mask, (circles[1][0], circles[1][1]), circles[1][2], 255, -1)
                cv2.circle(cladding_mask, (circles[0][0], circles[0][1]), circles[0][2], 0, -1)
                zones['cladding'] = cladding_mask
            
            if len(circles) >= 3:
                # Ferrule (between second and third circle)
                ferrule_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(ferrule_mask, (circles[2][0], circles[2][1]), circles[2][2], 255, -1)
                cv2.circle(ferrule_mask, (circles[1][0], circles[1][1]), circles[1][2], 0, -1)
                zones['ferrule'] = ferrule_mask
        
        # Fallback to default zones if Hough fails
        if not zones:
            zones = self._create_default_zones(h, w)
        
        # Cache result
        self.cache[frame_hash] = zones
        
        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entries
            for key in list(self.cache.keys())[:50]:
                del self.cache[key]
        
        return zones
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute hash for frame similarity"""
        # Downsample for faster hashing
        small = cv2.resize(frame, (64, 64))
        return hash(small.tobytes())
    
    def _create_default_zones(self, h: int, w: int) -> Dict[str, np.ndarray]:
        """Create default circular zones"""
        center = (w // 2, h // 2)
        radius = min(center) * 0.8
        
        zones = {}
        
        # Core
        core_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core_mask, center, int(radius * 0.2), 255, -1)
        zones['core'] = core_mask
        
        # Cladding
        cladding_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(cladding_mask, center, int(radius * 0.6), 255, -1)
        cv2.circle(cladding_mask, center, int(radius * 0.2), 0, -1)
        zones['cladding'] = cladding_mask
        
        # Ferrule
        ferrule_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ferrule_mask, center, int(radius), 255, -1)
        cv2.circle(ferrule_mask, center, int(radius * 0.6), 0, -1)
        zones['ferrule'] = ferrule_mask
        
        return zones
    
    def _create_visualization(self, frame: np.ndarray, zones: Dict[str, np.ndarray], 
                            defects: List[Defect]) -> np.ndarray:
        """Create real-time visualization"""
        overlay = frame.copy()
        
        # Draw zone boundaries with transparency
        alpha = 0.3
        zone_colors = {
            'core': (0, 255, 0),
            'cladding': (255, 255, 0),
            'ferrule': (255, 0, 0)
        }
        
        for zone_name, mask in zones.items():
            if mask is not None:
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours
                color = zone_colors.get(zone_name, (255, 255, 255))
                cv2.drawContours(overlay, contours, -1, color, 2)
        
        # Draw defects
        for defect in defects:
            color = self.config.visualization.defect_colors.get(
                defect.type,
                self.config.visualization.defect_colors['unknown']
            )
            
            # Bounding box
            x, y, w, h = defect.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Severity indicator
            severity_radius = int(3 + defect.severity * 7)
            cv2.circle(overlay, defect.location, severity_radius, color, -1)
            
            # Label (simplified for performance)
            label = f"{defect.type[:3]}"
            cv2.putText(overlay, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return overlay


class RealtimeProcessor:
    """Main real-time processing controller"""
    
    def __init__(self):
        self.config = get_config()
        self.capture = None
        self.frame_buffer = FrameBuffer()
        self.result_queue = queue.Queue()
        self.pipeline = OptimizedPipeline()
        
        self.is_running = False
        self.capture_thread = None
        self.process_thread = None
        
        # Performance metrics
        self.metrics = {
            'frames_processed': 0,
            'frames_dropped': 0,
            'avg_fps': 0,
            'avg_processing_time': 0
        }
    
    def start(self, source: int = 0):
        """Start real-time processing"""
        logger.info("Starting real-time processor")
        
        # Initialize capture
        self.capture = VideoCapture(source)
        
        # Start threads
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        self.capture_thread.start()
        self.process_thread.start()
        
        logger.info("Real-time processor started")
    
    def stop(self):
        """Stop real-time processing"""
        logger.info("Stopping real-time processor")
        
        self.is_running = False
        
        # Wait for threads
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.process_thread:
            self.process_thread.join(timeout=2)
        
        # Release resources
        if self.capture:
            self.capture.release()
        
        # Log metrics
        logger.info("Real-time processor stopped", **self.metrics)
    
    def _capture_loop(self):
        """Capture frames from video source"""
        logger.info("Capture loop started")
        
        while self.is_running:
            try:
                frame = self.capture.read()
                
                if frame is not None:
                    # Add to buffer
                    frame_id = self.frame_buffer.put(frame)
                    
                    # Check buffer size
                    if self.frame_buffer.size() > 20:
                        logger.warning("Frame buffer growing, possible processing lag")
                        self.metrics['frames_dropped'] += 1
                
                # Control capture rate
                time.sleep(1.0 / self.config.processing.max_fps)
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)
        
        logger.info("Capture loop ended")
    
    def _process_loop(self):
        """Process frames from buffer"""
        logger.info("Process loop started")
        
        while self.is_running or self.frame_buffer.size() > 0:
            try:
                # Get frame from buffer
                frame_data = self.frame_buffer.get()
                
                if frame_data is None:
                    time.sleep(0.01)
                    continue
                
                frame_id, timestamp, frame = frame_data
                
                # Process frame
                result = self.pipeline.process_frame(frame, frame_id)
                
                # Update metrics
                self.metrics['frames_processed'] += 1
                self.metrics['avg_fps'] = result.fps
                self.metrics['avg_processing_time'] = result.processing_time
                
                # Add to result queue
                self.result_queue.put(result)
                
                # Log performance periodically
                if self.metrics['frames_processed'] % 30 == 0:
                    logger.info("Performance update", **self.metrics)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
        
        logger.info("Process loop ended")
    
    def get_result(self, timeout: float = 0.1) -> Optional[FrameResult]:
        """Get processed result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def display_loop(self):
        """Display processed frames (blocking)"""
        logger.info("Starting display loop")
        
        cv2.namedWindow("Fiber Optic Defect Detection", cv2.WINDOW_NORMAL)
        
        try:
            while self.is_running:
                result = self.get_result()
                
                if result is not None:
                    # Add performance overlay
                    display_frame = self._add_performance_overlay(result)
                    
                    # Show frame
                    cv2.imshow("Fiber Optic Defect Detection", display_frame)
                    
                    # Save frame if defects detected
                    if result.defects and self.config.visualization.save_format:
                        self._save_defect_frame(result)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    # Save current frame
                    if result:
                        self._save_frame(result)
                elif key == ord('p'):
                    # Pause/resume
                    self.is_running = not self.is_running
                    logger.info(f"Processing {'resumed' if self.is_running else 'paused'}")
        
        finally:
            cv2.destroyAllWindows()
            self.stop()
    
    def _add_performance_overlay(self, result: FrameResult) -> np.ndarray:
        """Add performance metrics to frame"""
        frame = result.processed_frame.copy()
        h, w = frame.shape[:2]
        
        # Create overlay background
        overlay_h = 120
        overlay = np.zeros((overlay_h, w, 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)
        
        texts = [
            f"FPS: {result.fps:.1f}",
            f"Processing: {result.processing_time*1000:.1f}ms",
            f"Defects: {len(result.defects)}",
            f"Frame: {result.frame_id}"
        ]
        
        y = 25
        for text in texts:
            cv2.putText(overlay, text, (10, y), font, 0.6, color, 1)
            y += 25
        
        # Add defect summary
        if result.defects:
            defect_types = {}
            for d in result.defects:
                defect_types[d.type] = defect_types.get(d.type, 0) + 1
            
            summary = ", ".join(f"{k}: {v}" for k, v in defect_types.items())
            cv2.putText(overlay, summary, (10, y), font, 0.5, (255, 255, 0), 1)
        
        # Combine with frame
        combined = np.vstack([overlay, frame])
        
        return combined
    
    def _save_frame(self, result: FrameResult):
        """Save current frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}_{result.frame_id}.jpg"
        
        output_dir = self.config.output_dir / "realtime_captures"
        output_dir.mkdir(exist_ok=True)
        
        path = output_dir / filename
        cv2.imwrite(str(path), result.processed_frame)
        logger.info(f"Saved frame to {path}")
    
    def _save_defect_frame(self, result: FrameResult):
        """Save frame with detected defects"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        defect_count = len(result.defects)
        filename = f"defects_{timestamp}_count{defect_count}_frame{result.frame_id}.jpg"
        
        output_dir = self.config.output_dir / "detected_defects"
        output_dir.mkdir(exist_ok=True)
        
        path = output_dir / filename
        cv2.imwrite(str(path), result.processed_frame)
        
        # Save defect metadata
        metadata = {
            'frame_id': result.frame_id,
            'timestamp': result.timestamp,
            'defects': [d.to_dict() for d in result.defects],
            'processing_time': result.processing_time,
            'fps': result.fps
        }
        
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Main function for real-time processing"""
    config = get_config()
    
    # Check if real-time is enabled
    if not config.processing.realtime_enabled:
        logger.warning("Real-time processing not enabled in config")
        print("\nReal-time processing is not enabled.")
        enable = input("Would you like to enable it now? [yes/no]: ")
        
        if enable.lower() in ['yes', 'y']:
            config.processing.realtime_enabled = True
        else:
            return
    
    # Get camera source
    camera_source = config.processing.camera_index
    print(f"\nUsing camera source: {camera_source}")
    print("Press 'q' to quit, 's' to save frame, 'p' to pause/resume")
    
    # Create processor
    processor = RealtimeProcessor()
    
    try:
        # Start processing
        processor.start(camera_source)
        
        # Run display loop (blocking)
        processor.display_loop()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Real-time processing failed: {e}")
    finally:
        processor.stop()


if __name__ == "__main__":
    main()