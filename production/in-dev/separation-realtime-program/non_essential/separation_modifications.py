"""
Modifications to separation.py for real-time processing compatibility.
Add these methods to your existing UnifiedSegmentationSystem class.
"""

import threading
import queue
import time
from typing import Optional, Callable

class UnifiedSegmentationSystem:
    # ... existing code ...
    
    def __init__(self, methods_dir: str = "zones_methods"):
        # ... existing initialization code ...
        
        # Add real-time specific attributes
        self.realtime_mode = False
        self.realtime_callback = None
        self.frame_processing_lock = threading.Lock()
        self.continuous_learning = True
        
    def enable_realtime_mode(self, callback: Optional[Callable] = None):
        """Enable real-time processing mode with optional result callback."""
        self.realtime_mode = True
        self.realtime_callback = callback
        print("✓ Real-time mode enabled")
    
    def disable_realtime_mode(self):
        """Disable real-time processing mode."""
        self.realtime_mode = False
        self.realtime_callback = None
        print("✓ Real-time mode disabled")
    
    def process_frame_realtime(self, frame: np.ndarray, frame_id: int = None) -> Optional[Dict]:
        """
        Process a single frame in real-time mode.
        Optimized for speed and thread safety.
        """
        with self.frame_processing_lock:
            try:
                # Create temporary file for frame
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = Path(tmp_file.name)
                    cv2.imwrite(str(temp_path), frame)
                
                # Create temporary output directory
                with tempfile.TemporaryDirectory() as temp_output:
                    output_dir = Path(temp_output)
                    
                    # Process using existing pipeline but skip heavy I/O
                    consensus = self._process_image_lightweight(temp_path, output_dir, frame.shape[:2])
                    
                    # Cleanup
                    temp_path.unlink()
                    
                    # Trigger callback if provided
                    if self.realtime_callback and consensus:
                        self.realtime_callback(consensus, frame_id)
                    
                    return consensus
                    
            except Exception as e:
                print(f"ERROR in real-time frame processing: {e}")
                return None
    
    def _process_image_lightweight(self, image_path: Path, output_dir: Path, image_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        Lightweight version of process_image for real-time processing.
        Skips heavy visualization and file I/O operations.
        """
        print(f"🔄 Processing frame: {image_path.name}")
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            return None
        
        # Skip defect detection in real-time for speed (optional)
        # You can enable this if needed: inpainted_img, defect_mask = self.detect_and_inpaint_anomalies(original_img)
        
        # Run subset of most reliable methods for speed
        priority_methods = self._get_priority_methods_for_realtime()
        
        all_results = []
        for method_name in priority_methods:
            print(f"Running {method_name}...")
            result = self.run_method(method_name, image_path, image_shape)
            all_results.append(result)
            
            if result.error:
                print(f"  ✗ Failed: {result.error}")
            else:
                print(f"  ✓ Success - Confidence: {result.confidence:.2f}")
        
        # Generate consensus
        consensus = self.consensus_system.generate_consensus(
            all_results, 
            {name: info['score'] for name, info in self.methods.items()}, 
            image_shape
        )
        
        if consensus:
            print("✓ Real-time consensus achieved.")
            if self.continuous_learning:
                self.update_learning(consensus, all_results)
        
        return consensus
    
    def _get_priority_methods_for_realtime(self) -> List[str]:
        """
        Get priority methods for real-time processing.
        Returns fastest, most reliable methods first.
        """
        # Sort methods by score (reliability) and prefer faster ones
        method_priorities = {
            'geometric_approach': 0.9,  # Usually fast and reliable
            'threshold_separation': 0.8,  # Fast
            'hough_separation': 0.7,  # Reasonable speed
            'unified_core_cladding_detector': 0.6,
            'adaptive_intensity': 0.5,  # Can be slow but accurate
        }
        
        available_methods = []
        for method_name in self.methods.keys():
            if method_name in method_priorities:
                priority = method_priorities[method_name] * self.methods[method_name]['score']
                available_methods.append((method_name, priority))
        
        # Sort by priority and return top methods
        available_methods.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3-5 methods for speed
        return [method[0] for method in available_methods[:4]]
    
    def update_realtime_performance(self, method_name: str, execution_time: float, success: bool):
        """Update method performance metrics for real-time optimization."""
        if method_name not in self.dataset_stats.get('method_performance', {}):
            self.dataset_stats.setdefault('method_performance', {})[method_name] = {
                'avg_time': execution_time,
                'success_rate': 1.0 if success else 0.0,
                'total_calls': 1
            }
        else:
            perf = self.dataset_stats['method_performance'][method_name]
            total_calls = perf['total_calls'] + 1
            
            # Exponential moving average for time
            perf['avg_time'] = 0.8 * perf['avg_time'] + 0.2 * execution_time
            
            # Update success rate
            perf['success_rate'] = ((perf['success_rate'] * perf['total_calls']) + (1.0 if success else 0.0)) / total_calls
            perf['total_calls'] = total_calls
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time processing statistics."""
        return {
            'realtime_mode': self.realtime_mode,
            'method_performance': self.dataset_stats.get('method_performance', {}),
            'method_scores': self.dataset_stats.get('method_scores', {}),
            'continuous_learning': self.continuous_learning
        }


# Additional utility functions for real-time processing

class FrameBuffer:
    """
    Thread-safe circular buffer for frames with automatic overflow handling.
    """
    
    def __init__(self, maxsize: int = 50):
        self.maxsize = maxsize
        self.buffer = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        self.dropped_frames = 0
    
    def put_frame(self, frame: np.ndarray, metadata: Dict = None) -> bool:
        """
        Add frame to buffer. Returns True if successful, False if dropped.
        """
        frame_data = {
            'frame': frame.copy(),
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        try:
            self.buffer.put_nowait(frame_data)
            return True
        except queue.Full:
            # Buffer full, drop oldest frame and add new one
            try:
                self.buffer.get_nowait()  # Remove oldest
                self.buffer.put_nowait(frame_data)  # Add new
                self.dropped_frames += 1
                return True
            except queue.Empty:
                return False
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get frame from buffer with timeout."""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest_frame(self) -> Optional[Dict]:
        """Get the most recent frame, discarding older ones."""
        latest = None
        while True:
            try:
                latest = self.buffer.get_nowait()
            except queue.Empty:
                break
        return latest
    
    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer.qsize()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.buffer.empty()


class AdaptiveProcessingController:
    """
    Controls processing rate based on performance and system load.
    """
    
    def __init__(self, target_fps: float = 30.0, initial_interval: int = 30):
        self.target_fps = target_fps
        self.processing_interval = initial_interval
        self.min_interval = 5
        self.max_interval = 120
        
        self.performance_history = []
        self.max_history = 20
        
        self.adaptive_enabled = True
    
    def update_performance(self, processing_time: float, success: bool):
        """Update performance metrics."""
        self.performance_history.append({
            'processing_time': processing_time,
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
        
        if self.adaptive_enabled:
            self._adjust_interval()
    
    def _adjust_interval(self):
        """Adjust processing interval based on recent performance."""
        if len(self.performance_history) < 5:
            return
        
        recent_times = [p['processing_time'] for p in self.performance_history[-5:]]
        recent_success = [p['success'] for p in self.performance_history[-5:]]
        
        avg_time = sum(recent_times) / len(recent_times)
        success_rate = sum(recent_success) / len(recent_success)
        
        # Calculate target interval based on processing time and target FPS
        target_interval = max(
            int(avg_time * self.target_fps * 1.5),  # 1.5x safety factor
            self.min_interval
        )
        target_interval = min(target_interval, self.max_interval)
        
        # Adjust based on success rate
        if success_rate < 0.8:
            target_interval = int(target_interval * 1.2)  # Slow down if failing
        elif success_rate > 0.95 and avg_time < 1.0:
            target_interval = int(target_interval * 0.9)  # Speed up if doing well
        
        # Smooth adjustment
        self.processing_interval = int(
            0.7 * self.processing_interval + 0.3 * target_interval
        )
        
        print(f"📊 Adaptive controller: interval={self.processing_interval}, "
              f"avg_time={avg_time:.2f}s, success_rate={success_rate:.2f}")
    
    def should_process_frame(self, frame_number: int) -> bool:
        """Determine if a frame should be processed."""
        return frame_number % self.processing_interval == 0
    
    def get_current_interval(self) -> int:
        """Get current processing interval."""
        return self.processing_interval


class RealtimeResultsManager:
    """
    Manages real-time segmentation results with history and statistics.
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.results_history = []
        self.current_consensus = None
        self.lock = threading.Lock()
        
        # Statistics
        self.total_processed = 0
        self.successful_consensus = 0
        self.method_contribution_stats = {}
    
    def add_result(self, frame_number: int, consensus: Optional[Dict], processing_time: float):
        """Add a new processing result."""
        with self.lock:
            result_entry = {
                'frame_number': frame_number,
                'consensus': consensus,
                'processing_time': processing_time,
                'timestamp': time.time(),
                'success': consensus is not None
            }
            
            self.results_history.append(result_entry)
            
            # Keep history size manageable
            if len(self.results_history) > self.max_history:
                self.results_history = self.results_history[-self.max_history:]
            
            # Update current consensus
            if consensus:
                self.current_consensus = consensus
                self.successful_consensus += 1
                
                # Update method contribution statistics
                contributing_methods = consensus.get('contributing_methods', [])
                for method in contributing_methods:
                    self.method_contribution_stats[method] = \
                        self.method_contribution_stats.get(method, 0) + 1
            
            self.total_processed += 1
    
    def get_latest_consensus(self) -> Optional[Dict]:
        """Get the most recent successful consensus."""
        with self.lock:
            return self.current_consensus.copy() if self.current_consensus else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self.lock:
            if not self.results_history:
                return {}
            
            recent_results = self.results_history[-20:]  # Last 20 results
            recent_times = [r['processing_time'] for r in recent_results]
            recent_success = [r['success'] for r in recent_results]
            
            return {
                'total_processed': self.total_processed,
                'successful_consensus': self.successful_consensus,
                'success_rate': self.successful_consensus / max(self.total_processed, 1),
                'avg_processing_time': sum(recent_times) / len(recent_times) if recent_times else 0,
                'recent_success_rate': sum(recent_success) / len(recent_success) if recent_success else 0,
                'method_contributions': self.method_contribution_stats.copy(),
                'results_history_size': len(self.results_history)
            }
    
    def export_results_summary(self, filepath: str):
        """Export results summary to JSON file."""
        with self.lock:
            summary = {
                'statistics': self.get_statistics(),
                'recent_results': self.results_history[-50:],  # Last 50 results
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=4, default=str)
            
            print(f"📄 Results summary exported to {filepath}")


# Enhanced real-time processing integration function
def integrate_pylon_with_separation(methods_dir: str = "zones_methods", 
                                  buffer_size: int = 20,
                                  processing_interval: int = 30) -> RealTimeSegmentationProcessor:
    """
    Factory function to create integrated real-time segmentation processor.
    
    Args:
        methods_dir: Directory containing segmentation method scripts
        buffer_size: Maximum number of frames to buffer
        processing_interval: Initial interval between processed frames
    
    Returns:
        Configured RealTimeSegmentationProcessor instance
    """
    
    # Create processor with custom configuration
    processor = RealTimeSegmentationProcessor(methods_dir, buffer_size)
    
    # Configure adaptive processing
    processor.process_every_n_frames = processing_interval
    processor.adaptive_processing = True
    
    # Add enhanced error handling
    def enhanced_error_handler(error_msg: str, frame_number: int):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        error_log = f"[{timestamp}] Frame {frame_number}: {error_msg}\n"
        
        # Log to file
        with open("realtime_errors.log", "a") as f:
            f.write(error_log)
        
        print(f"🚨 {error_log.strip()}")
    
    # Enhanced result callback
    def enhanced_result_callback(consensus: Dict, frame_number: int):
        print(f"✅ Frame {frame_number} processed successfully")
        if consensus.get('contributing_methods'):
            methods = ', '.join(consensus['contributing_methods'])
            print(f"   Contributing methods: {methods}")
        
        # Optional: Save periodic results
        if frame_number % 1000 == 0:  # Every 1000th frame
            processor._save_current_results(None, {'consensus': consensus})
    
    return processor


# Additional utility for method performance optimization
class MethodPerformanceOptimizer:
    """
    Optimizes method selection and execution for real-time performance.
    """
    
    def __init__(self, segmentation_system: UnifiedSegmentationSystem):
        self.system = segmentation_system
        self.method_timings = {}
        self.method_accuracy = {}
        self.optimization_enabled = True
    
    def optimize_method_selection(self, target_time: float = 2.0) -> List[str]:
        """
        Select optimal methods based on time budget and accuracy.
        
        Args:
            target_time: Target total processing time in seconds
        
        Returns:
            List of method names to use
        """
        if not self.optimization_enabled:
            return list(self.system.methods.keys())
        
        available_methods = []
        
        for method_name, method_info in self.system.methods.items():
            avg_time = self.method_timings.get(method_name, 1.0)
            accuracy = self.method_accuracy.get(method_name, method_info['score'])
            
            # Calculate efficiency score (accuracy per second)
            efficiency = accuracy / max(avg_time, 0.1)
            
            available_methods.append({
                'name': method_name,
                'time': avg_time,
                'accuracy': accuracy,
                'efficiency': efficiency
            })
        
        # Sort by efficiency
        available_methods.sort(key=lambda x: x['efficiency'], reverse=True)
        
        # Select methods within time budget
        selected_methods = []
        total_time = 0
        
        for method in available_methods:
            if total_time + method['time'] <= target_time:
                selected_methods.append(method['name'])
                total_time += method['time']
            
            # Always include at least 2 methods for consensus
            if len(selected_methods) >= 2 and total_time > target_time * 0.8:
                break
        
        # Ensure minimum methods for consensus
        if len(selected_methods) < 2:
            selected_methods = [m['name'] for m in available_methods[:3]]
        
        print(f"🎯 Optimized method selection: {selected_methods} "
              f"(estimated time: {total_time:.2f}s)")
        
        return selected_methods
    
    def update_method_performance(self, method_name: str, execution_time: float, 
                                accuracy: float):
        """Update performance metrics for a method."""
        # Exponential moving average for timing
        if method_name in self.method_timings:
            self.method_timings[method_name] = (
                0.8 * self.method_timings[method_name] + 0.2 * execution_time
            )
        else:
            self.method_timings[method_name] = execution_time
        
        # Update accuracy
        self.method_accuracy[method_name] = accuracy
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all methods."""
        summary = {}
        for method_name in self.system.methods.keys():
            summary[method_name] = {
                'avg_time': self.method_timings.get(method_name, 'N/A'),
                'accuracy': self.method_accuracy.get(method_name, 'N/A'),
                'efficiency': (
                    self.method_accuracy.get(method_name, 0) / 
                    max(self.method_timings.get(method_name, 1), 0.1)
                    if method_name in self.method_timings else 'N/A'
                )
            }
        return summary