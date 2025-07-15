#!/usr/bin/env python3
"""
Master Orchestrator - The Ultimate Control System for Polar Bear
Orchestrates all components with complete understanding of the entire ecosystem
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import cv2
import pandas as pd
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

# Import all major components
try:
    from ultimate_mega_connector import UltimateMegaConnector
    ULTIMATE_AVAILABLE = True
except ImportError:
    ULTIMATE_AVAILABLE = False

try:
    from polar_bear_brain import PolarBearBrain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False

try:
    from streaming_data_pipeline import StreamProcessor
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

@dataclass
class ProcessingTask:
    """Processing task definition"""
    id: str
    type: str  # 'image', 'video', 'stream', 'batch'
    source: Any
    priority: int = 5
    config: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = 'pending'
    result: Optional[Dict] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    tasks_processed: int = 0
    tasks_failed: int = 0
    average_processing_time: float = 0.0
    throughput: float = 0.0  # tasks per second
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    active_workers: int = 0

class MasterOrchestrator:
    """Master orchestrator for the entire Polar Bear system"""
    
    def __init__(self, config_file: str = "master_config.json"):
        # Initialize logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_configuration(config_file)
        
        # Initialize components
        self.components = self.initialize_components()
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        
        # Worker pools
        self.worker_pool = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 10))
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.get('max_processes', 4))
        
        # Metrics and monitoring
        self.metrics = SystemMetrics()
        self.performance_history = deque(maxlen=1000)
        
        # State management
        self.running = False
        self.learning_mode = False
        self.optimization_mode = 'balanced'
        
        # Initialize subsystems
        self.initialize_subsystems()
        
        self.logger.info("Master Orchestrator initialized successfully")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = Path("logs/master")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("MasterOrchestrator")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_dir / "master_orchestrator.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def load_configuration(self, config_file: str) -> Dict:
        """Load or create configuration"""
        config_path = Path(config_file)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_file}")
        else:
            # Create default configuration
            config = self.create_default_configuration()
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Created default configuration at {config_file}")
        
        return config
    
    def create_default_configuration(self) -> Dict:
        """Create default system configuration"""
        return {
            'system': {
                'name': 'Polar Bear Master System',
                'version': '3.0',
                'optimization_mode': 'balanced',
                'auto_scaling': True
            },
            'processing': {
                'max_workers': min(10, os.cpu_count() * 2),
                'max_processes': os.cpu_count(),
                'task_timeout': 300,  # 5 minutes
                'batch_size': 10,
                'cache_results': True
            },
            'components': {
                'use_ultimate_connector': True,
                'use_brain': True,
                'use_streaming': True,
                'use_gpu': True
            },
            'monitoring': {
                'metrics_interval': 10,  # seconds
                'health_check_interval': 60,
                'alert_thresholds': {
                    'cpu_usage': 90,
                    'memory_usage': 85,
                    'task_queue_size': 100,
                    'failure_rate': 0.1
                }
            },
            'learning': {
                'enabled': True,
                'update_interval': 3600,  # 1 hour
                'min_samples': 100,
                'model_update_threshold': 0.05
            },
            'storage': {
                'results_dir': 'results',
                'cache_dir': 'cache',
                'models_dir': 'models',
                'logs_dir': 'logs'
            }
        }
    
    def initialize_components(self) -> Dict:
        """Initialize all system components"""
        components = {}
        
        # Ultimate Connector
        if ULTIMATE_AVAILABLE and self.config['components']['use_ultimate_connector']:
            try:
                components['ultimate'] = UltimateMegaConnector(
                    optimization_profile=self.config['system']['optimization_mode']
                )
                self.logger.info("Ultimate Mega Connector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Ultimate Connector: {e}")
        
        # Polar Bear Brain
        if BRAIN_AVAILABLE and self.config['components']['use_brain']:
            try:
                components['brain'] = PolarBearBrain()
                self.logger.info("Polar Bear Brain initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Brain: {e}")
        
        # Streaming Pipeline
        if STREAMING_AVAILABLE and self.config['components']['use_streaming']:
            try:
                components['streaming'] = StreamProcessor()
                self.logger.info("Streaming Pipeline initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Streaming: {e}")
        
        return components
    
    def initialize_subsystems(self):
        """Initialize all subsystems"""
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start task processor
        self.processor_thread = threading.Thread(target=self._task_processor_loop, daemon=True)
        self.processor_thread.start()
        
        # Start learning system if enabled
        if self.config['learning']['enabled']:
            self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
            self.learning_thread.start()
    
    def add_task(self, task_type: str, source: Any, priority: int = 5, config: Dict = None) -> str:
        """Add a processing task to the queue"""
        task = ProcessingTask(
            id=f"{task_type}_{int(time.time()*1000)}",
            type=task_type,
            source=source,
            priority=priority,
            config=config or {}
        )
        
        self.task_queue.put((priority, task.created_at, task))
        self.logger.debug(f"Added task {task.id} with priority {priority}")
        
        return task.id
    
    def process_image(self, image_path: str, analysis_type: str = 'comprehensive') -> Dict:
        """Process a single image"""
        task_id = self.add_task('image', image_path, priority=5, config={'analysis_type': analysis_type})
        
        # Wait for result (with timeout)
        start_time = time.time()
        timeout = self.config['processing']['task_timeout']
        
        while time.time() - start_time < timeout:
            if task_id in [task.id for task in self.completed_tasks]:
                for task in self.completed_tasks:
                    if task.id == task_id:
                        return task.result
            time.sleep(0.1)
        
        return {'success': False, 'error': 'Task timeout'}
    
    def process_batch(self, image_paths: List[str], parallel: bool = True) -> List[Dict]:
        """Process a batch of images"""
        if parallel:
            # Add all tasks with same priority
            task_ids = []
            for path in image_paths:
                task_id = self.add_task('image', path, priority=3)
                task_ids.append(task_id)
            
            # Wait for all results
            results = []
            timeout = self.config['processing']['task_timeout'] * len(image_paths)
            start_time = time.time()
            
            while len(results) < len(task_ids) and time.time() - start_time < timeout:
                for task in self.completed_tasks:
                    if task.id in task_ids and task.id not in [r.get('task_id') for r in results]:
                        results.append(task.result)
                time.sleep(0.1)
            
            return results
        else:
            # Process sequentially
            return [self.process_image(path) for path in image_paths]
    
    def start_video_stream(self, source: Union[str, int], callback: Callable = None) -> str:
        """Start processing a video stream"""
        stream_id = f"stream_{int(time.time())}"
        
        # Create stream processing task
        config = {
            'stream_id': stream_id,
            'callback': callback,
            'frame_skip': self.config.get('stream_frame_skip', 5)
        }
        
        task_id = self.add_task('stream', source, priority=8, config=config)
        
        self.logger.info(f"Started video stream {stream_id} from {source}")
        return stream_id
    
    def _task_processor_loop(self):
        """Main task processing loop"""
        self.running = True
        
        while self.running:
            try:
                # Get next task (with timeout to allow checking running status)
                try:
                    priority, created_at, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Update task status
                task.status = 'processing'
                self.active_tasks[task.id] = task
                
                # Process based on task type
                future = self.worker_pool.submit(self._process_task, task)
                
                # Handle result
                def handle_result(f):
                    try:
                        result = f.result()
                        task.result = result
                        task.status = 'completed'
                        self.completed_tasks.append(task)
                        self.metrics.tasks_processed += 1
                    except Exception as e:
                        task.result = {'success': False, 'error': str(e)}
                        task.status = 'failed'
                        self.completed_tasks.append(task)
                        self.metrics.tasks_failed += 1
                        self.logger.error(f"Task {task.id} failed: {e}")
                    finally:
                        del self.active_tasks[task.id]
                
                future.add_done_callback(handle_result)
                
            except Exception as e:
                self.logger.error(f"Task processor error: {e}")
    
    def _process_task(self, task: ProcessingTask) -> Dict:
        """Process a single task"""
        start_time = time.time()
        
        try:
            if task.type == 'image':
                result = self._process_image_task(task)
            elif task.type == 'video':
                result = self._process_video_task(task)
            elif task.type == 'stream':
                result = self._process_stream_task(task)
            elif task.type == 'batch':
                result = self._process_batch_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.type}")
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task processing error: {e}")
            self._update_metrics(time.time() - start_time, success=False)
            raise
    
    def _process_image_task(self, task: ProcessingTask) -> Dict:
        """Process an image analysis task"""
        if 'ultimate' in self.components:
            # Use Ultimate Connector for most comprehensive analysis
            return self.components['ultimate'].analyze_image_comprehensive(
                task.source,
                task.config
            )
        elif 'brain' in self.components:
            # Use Brain for processing
            return self.components['brain'].pipeline.process_image(task.source)
        else:
            # Fallback to basic processing
            return {'success': False, 'error': 'No processing components available'}
    
    def _process_video_task(self, task: ProcessingTask) -> Dict:
        """Process a video file task"""
        video_path = task.source
        results = []
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Process frames at intervals
        frame_interval = task.config.get('frame_interval', 30)  # Process every 30 frames
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Process frame
                temp_path = f"/tmp/frame_{frame_idx}.jpg"
                cv2.imwrite(temp_path, frame)
                
                frame_result = self._process_image_task(
                    ProcessingTask(
                        id=f"{task.id}_frame_{frame_idx}",
                        type='image',
                        source=temp_path,
                        config=task.config
                    )
                )
                
                frame_result['frame_index'] = frame_idx
                frame_result['timestamp'] = frame_idx / fps if fps > 0 else 0
                results.append(frame_result)
                
                # Clean up
                os.remove(temp_path)
            
            frame_idx += 1
        
        cap.release()
        
        return {
            'success': True,
            'video_path': video_path,
            'frame_count': frame_count,
            'fps': fps,
            'frames_processed': len(results),
            'results': results
        }
    
    def _process_stream_task(self, task: ProcessingTask) -> Dict:
        """Process a streaming video task"""
        if 'streaming' in self.components:
            # Use streaming pipeline
            return self.components['streaming'].process_stream(
                task.source,
                task.config
            )
        else:
            # Fallback to frame-by-frame processing
            return self._process_video_stream_fallback(task)
    
    def _process_video_stream_fallback(self, task: ProcessingTask) -> Dict:
        """Fallback video stream processing"""
        stream_id = task.config.get('stream_id', 'unknown')
        callback = task.config.get('callback')
        frame_skip = task.config.get('frame_skip', 5)
        
        cap = cv2.VideoCapture(task.source)
        
        frame_idx = 0
        results = []
        
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                # Process frame
                result = self._quick_frame_analysis(frame)
                
                if callback:
                    callback(frame, result)
                
                results.append(result)
            
            frame_idx += 1
            
            # Limit history
            if len(results) > 100:
                results.pop(0)
        
        cap.release()
        
        return {
            'success': True,
            'stream_id': stream_id,
            'frames_processed': frame_idx,
            'final_results': results[-10:] if results else []
        }
    
    def _quick_frame_analysis(self, frame: np.ndarray) -> Dict:
        """Quick analysis for real-time processing"""
        # Use fastest available method
        if 'ultimate' in self.components:
            # Use real-time optimized pipeline
            optimizer = self.components['ultimate']
            algo = optimizer.select_optimal_algorithm('general_defects', {
                'max_time': 0.1,  # 100ms max
                'image_size': frame.shape[:2]
            })
            
            # Execute selected algorithm
            # This is simplified - actual implementation would execute the algorithm
            return {
                'timestamp': time.time(),
                'algorithm_used': algo['name'],
                'defects_detected': 0  # Placeholder
            }
        else:
            # Basic analysis
            return {
                'timestamp': time.time(),
                'mean_intensity': np.mean(frame),
                'std_intensity': np.std(frame)
            }
    
    def _process_batch_task(self, task: ProcessingTask) -> Dict:
        """Process a batch of items"""
        items = task.source
        results = []
        
        # Process in parallel using process pool for better performance
        futures = []
        for item in items:
            future = self.process_pool.submit(
                self._process_single_item,
                item,
                task.config
            )
            futures.append(future)
        
        # Collect results
        for future in futures:
            try:
                result = future.result(timeout=self.config['processing']['task_timeout'])
                results.append(result)
            except Exception as e:
                results.append({'success': False, 'error': str(e)})
        
        return {
            'success': True,
            'batch_size': len(items),
            'successful': sum(1 for r in results if r.get('success', False)),
            'results': results
        }
    
    def _process_single_item(self, item: Any, config: Dict) -> Dict:
        """Process a single item (for multiprocessing)"""
        # This runs in a separate process, so we need to create components
        if isinstance(item, str) and os.path.isfile(item):
            # Image file
            if ULTIMATE_AVAILABLE:
                from ultimate_mega_connector import create_ultimate_connector
                connector = create_ultimate_connector()
                return connector.analyze_image_comprehensive(item, config)
        
        return {'success': False, 'error': 'Unsupported item type'}
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update system metrics"""
        # Update average processing time
        if self.metrics.tasks_processed > 0:
            self.metrics.average_processing_time = (
                (self.metrics.average_processing_time * (self.metrics.tasks_processed - 1) + 
                 processing_time) / self.metrics.tasks_processed
            )
        else:
            self.metrics.average_processing_time = processing_time
        
        # Update throughput
        self.metrics.throughput = self.metrics.tasks_processed / (time.time() - self.start_time)
        
        # Record in history
        self.performance_history.append({
            'timestamp': time.time(),
            'processing_time': processing_time,
            'success': success,
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks)
        })
    
    def _monitoring_loop(self):
        """System monitoring loop"""
        self.start_time = time.time()
        
        while self.running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Log metrics periodically
                if len(self.performance_history) % 10 == 0:
                    self.logger.info(f"System metrics: {self.metrics}")
                
                # Sleep for monitoring interval
                time.sleep(self.config['monitoring']['metrics_interval'])
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    def _update_system_metrics(self):
        """Update system resource metrics"""
        try:
            import psutil
            
            # CPU usage
            self.metrics.cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.memory_usage = memory.percent
            
            # Active workers
            self.metrics.active_workers = len(self.active_tasks)
            
            # GPU usage (if available)
            if self.config['components']['use_gpu']:
                self._update_gpu_metrics()
                
        except ImportError:
            pass  # psutil not available
    
    def _update_gpu_metrics(self):
        """Update GPU metrics if available"""
        try:
            import torch
            if torch.cuda.is_available():
                # Simple GPU memory usage
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    self.metrics.gpu_usage = (memory_allocated / memory_reserved * 100 
                                             if memory_reserved > 0 else 0)
        except:
            pass
    
    def _check_alerts(self):
        """Check for system alerts"""
        alerts = self.config['monitoring']['alert_thresholds']
        
        # CPU alert
        if self.metrics.cpu_usage > alerts['cpu_usage']:
            self.logger.warning(f"High CPU usage: {self.metrics.cpu_usage}%")
        
        # Memory alert
        if self.metrics.memory_usage > alerts['memory_usage']:
            self.logger.warning(f"High memory usage: {self.metrics.memory_usage}%")
        
        # Queue size alert
        queue_size = self.task_queue.qsize()
        if queue_size > alerts['task_queue_size']:
            self.logger.warning(f"Large task queue: {queue_size} tasks")
        
        # Failure rate alert
        if self.metrics.tasks_processed > 0:
            failure_rate = self.metrics.tasks_failed / self.metrics.tasks_processed
            if failure_rate > alerts['failure_rate']:
                self.logger.warning(f"High failure rate: {failure_rate:.2%}")
    
    def _learning_loop(self):
        """Continuous learning loop"""
        while self.running:
            try:
                # Wait for update interval
                time.sleep(self.config['learning']['update_interval'])
                
                # Check if we have enough samples
                if len(self.completed_tasks) >= self.config['learning']['min_samples']:
                    self._update_learning_models()
                
            except Exception as e:
                self.logger.error(f"Learning loop error: {e}")
    
    def _update_learning_models(self):
        """Update learning models based on recent results"""
        self.logger.info("Updating learning models...")
        
        # Analyze recent results
        recent_tasks = list(self.completed_tasks)[-self.config['learning']['min_samples']:]
        
        # Extract patterns
        success_rate = sum(1 for t in recent_tasks if t.status == 'completed') / len(recent_tasks)
        avg_time = np.mean([t.result.get('duration', 0) for t in recent_tasks if t.result])
        
        # Update optimization strategy
        if success_rate < 0.9:
            self.logger.info("Adjusting optimization for higher accuracy")
            self.optimization_mode = 'accuracy'
        elif avg_time > 5.0:
            self.logger.info("Adjusting optimization for speed")
            self.optimization_mode = 'speed'
        else:
            self.optimization_mode = 'balanced'
        
        # Update component configurations
        if 'ultimate' in self.components:
            # This would update the optimization profile
            pass
    
    def get_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'system': {
                'running': self.running,
                'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                'optimization_mode': self.optimization_mode,
                'learning_enabled': self.learning_mode
            },
            'components': {
                name: {'available': comp is not None, 'type': type(comp).__name__}
                for name, comp in self.components.items()
            },
            'tasks': {
                'queued': self.task_queue.qsize(),
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks),
                'total_processed': self.metrics.tasks_processed,
                'total_failed': self.metrics.tasks_failed
            },
            'performance': {
                'average_processing_time': self.metrics.average_processing_time,
                'throughput': self.metrics.throughput,
                'success_rate': (self.metrics.tasks_processed - self.metrics.tasks_failed) / 
                               max(1, self.metrics.tasks_processed)
            },
            'resources': {
                'cpu_usage': self.metrics.cpu_usage,
                'memory_usage': self.metrics.memory_usage,
                'gpu_usage': self.metrics.gpu_usage,
                'active_workers': self.metrics.active_workers
            }
        }
        
        # Add component-specific status
        if 'ultimate' in self.components:
            status['ultimate_connector'] = self.components['ultimate'].get_system_health()
        
        return status
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down Master Orchestrator...")
        
        # Stop processing
        self.running = False
        
        # Wait for active tasks to complete
        timeout = 30  # 30 seconds
        start = time.time()
        while self.active_tasks and time.time() - start < timeout:
            time.sleep(0.5)
        
        # Shutdown worker pools
        self.worker_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Shutdown components
        for name, component in self.components.items():
            if hasattr(component, 'shutdown'):
                component.shutdown()
        
        self.logger.info("Master Orchestrator shutdown complete")

# Global instance
_master = None

def get_master() -> MasterOrchestrator:
    """Get or create master orchestrator instance"""
    global _master
    if _master is None:
        _master = MasterOrchestrator()
    return _master

# CLI Interface
def main():
    """Main entry point with CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Polar Bear Master Orchestrator")
    parser.add_argument('command', choices=['start', 'status', 'process', 'batch', 'stop'],
                       help='Command to execute')
    parser.add_argument('--image', help='Image path for processing')
    parser.add_argument('--images', nargs='+', help='Multiple image paths for batch processing')
    parser.add_argument('--video', help='Video path for processing')
    parser.add_argument('--stream', help='Stream source (camera index or URL)')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        print("Starting Master Orchestrator...")
        master = MasterOrchestrator(config_file=args.config or "master_config.json")
        
        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            master.shutdown()
    
    elif args.command == 'status':
        master = get_master()
        status = master.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == 'process':
        if not args.image:
            print("Error: --image required for process command")
            sys.exit(1)
        
        master = get_master()
        result = master.process_image(args.image)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == 'batch':
        if not args.images:
            print("Error: --images required for batch command")
            sys.exit(1)
        
        master = get_master()
        results = master.process_batch(args.images)
        
        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\nBatch processing complete: {successful}/{len(results)} successful")
        
        # Save detailed results
        output_file = f"batch_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Detailed results saved to: {output_file}")
    
    elif args.command == 'stop':
        master = get_master()
        master.shutdown()
        print("Master Orchestrator stopped")

if __name__ == "__main__":
    main()