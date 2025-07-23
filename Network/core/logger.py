#!/usr/bin/env python3
"""
Enhanced Logger module for Fiber Optics Neural Network
"the program will log all of its processes in a log file as soon as they happen, 
it will log verbosely and with time stamps so I can see exactly what is happening when it happens"
Includes run/cycle tracking, verbose logging for every process, and unit test support
"""

import logging
import sys
import os
import traceback
import inspect
import functools
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import json
import uuid
from contextlib import contextmanager

class RunLogger:
    """Manages individual run log files"""
    
    def __init__(self, run_id: str, run_dir: Path):
        self.run_id = run_id
        self.run_dir = run_dir
        self.run_start = datetime.now()
        self.cycle_count = 0
        
        # Create run-specific log file
        self.log_file = run_dir / f"run_{run_id}_{self.run_start.strftime('%Y%m%d_%H%M%S')}.log"
        self.metadata_file = run_dir / f"run_{run_id}_metadata.json"
        
        # Initialize metadata
        self.metadata = {
            'run_id': run_id,
            'start_time': self.run_start.isoformat(),
            'cycles': [],
            'processes': [],
            'errors': [],
            'warnings': [],
            'performance_metrics': {}
        }
        
        self._save_metadata()
    
    def start_cycle(self) -> int:
        """Start a new cycle and return cycle number"""
        self.cycle_count += 1
        cycle_info = {
            'cycle_number': self.cycle_count,
            'start_time': datetime.now().isoformat(),
            'processes': []
        }
        self.metadata['cycles'].append(cycle_info)
        self._save_metadata()
        return self.cycle_count
    
    def end_cycle(self, cycle_number: int, metrics: Dict[str, Any] = None):
        """End a cycle and record metrics"""
        if cycle_number <= len(self.metadata['cycles']):
            cycle = self.metadata['cycles'][cycle_number - 1]
            cycle['end_time'] = datetime.now().isoformat()
            if metrics:
                cycle['metrics'] = metrics
            self._save_metadata()
    
    def add_process(self, process_name: str, details: Dict[str, Any] = None):
        """Add a process to the current run"""
        process_info = {
            'name': process_name,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.metadata['processes'].append(process_info)
        
        # Also add to current cycle if one is active
        if self.metadata['cycles']:
            self.metadata['cycles'][-1]['processes'].append(process_info)
        
        self._save_metadata()
    
    def add_error(self, error: str, exception: Exception = None):
        """Add an error to the run metadata"""
        error_info = {
            'message': error,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc() if exception else None
        }
        self.metadata['errors'].append(error_info)
        self._save_metadata()
    
    def add_warning(self, warning: str):
        """Add a warning to the run metadata"""
        warning_info = {
            'message': warning,
            'timestamp': datetime.now().isoformat()
        }
        self.metadata['warnings'].append(warning_info)
        self._save_metadata()
    
    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics for the run"""
        self.metadata['performance_metrics'].update(metrics)
        self._save_metadata()
    
    def finalize(self):
        """Finalize the run and save final metadata"""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['duration_seconds'] = (datetime.now() - self.run_start).total_seconds()
        self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

class FiberOpticsLogger(logging.Logger):
    """Enhanced logger with verbose output and run/cycle tracking"""
    
    def __init__(self, name: str, config=None):
        super().__init__(name)
        
        print(f"[{datetime.now()}] Initializing FiberOpticsLogger for {name}")
        print(f"[{datetime.now()}] Previous script: config.py")
        
        self.name = name
        self.process_stack = []  # Track nested processes
        self.function_call_depth = 0  # Track function call depth
        
        # Get configuration
        if config is None:
            import importlib
            config_module = importlib.import_module("core.config_loader")
            get_config = getattr(config_module, "get_config")
            config = get_config()
        self.config = config
        
        # Set up run directory
        self.runs_dir = Path(self.config.system.logs_path if hasattr(self.config.system, 'logs_path') else 'logs') / 'runs'
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize run logger
        self.run_id = str(uuid.uuid4())[:8]
        self.run_logger = RunLogger(self.run_id, self.runs_dir)
        
        # Set logging level
        self.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        self.handlers = []
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] [RUN:%(run_id)s] [%(name)s] [%(levelname)s] '
            '[%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.simple_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Set up handlers
        self._setup_handlers()
        
        # Log initialization
        self.info(f"Enhanced Logger initialized for {name}")
        self.info(f"Run ID: {self.run_id}")
        self.info(f"Run log file: {self.run_logger.log_file}")
        
        print(f"[{datetime.now()}] FiberOpticsLogger initialized successfully")
    
    def _setup_handlers(self):
        """Set up logging handlers"""
        # Main log file handler
        main_log_file = Path(self.config.system.log_file_path if hasattr(self.config, 'system') else Path(__file__).parent.parent / 'logs/fiber_optics.log')
        main_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        main_handler = logging.FileHandler(main_log_file)
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(self.detailed_formatter)
        self.addHandler(main_handler)
        
        # Run-specific log file handler
        run_handler = logging.FileHandler(self.run_logger.log_file)
        run_handler.setLevel(logging.DEBUG)
        run_handler.setFormatter(self.detailed_formatter)
        self.addHandler(run_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.simple_formatter)
        self.addHandler(console_handler)
        
        # Error file handler
        error_log_file = self.runs_dir.parent / 'errors.log'
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.detailed_formatter)
        self.addHandler(error_handler)
    
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        """Override to add run_id to all records"""
        if extra is None:
            extra = {}
        extra['run_id'] = self.run_id
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)
    
    # Cycle management
    def start_cycle(self, cycle_name: str = None) -> int:
        """Start a new cycle"""
        cycle_num = self.run_logger.start_cycle()
        self.info(f"{'='*80}")
        self.info(f"STARTING CYCLE {cycle_num}" + (f": {cycle_name}" if cycle_name else ""))
        self.info(f"{'='*80}")
        return cycle_num
    
    def end_cycle(self, cycle_number: int, metrics: Dict[str, Any] = None):
        """End a cycle"""
        self.run_logger.end_cycle(cycle_number, metrics)
        self.info(f"{'='*80}")
        self.info(f"COMPLETED CYCLE {cycle_number}")
        if metrics:
            self.info(f"Cycle metrics: {json.dumps(metrics, indent=2)}")
        self.info(f"{'='*80}")
    
    # Enhanced process tracking
    @contextmanager
    def track_process(self, process_name: str, **details):
        """Context manager for tracking processes"""
        self.log_process_start(process_name, **details)
        start_time = time.time()
        
        try:
            yield
        except Exception as e:
            self.log_error(f"Process '{process_name}' failed", e)
            raise
        finally:
            duration = time.time() - start_time
            self.log_process_end(process_name, success=True, duration=duration)
    
    def log_process_start(self, process_name: str, **details):
        """Log start of a process with details"""
        self.process_stack.append(process_name)
        indent = "  " * (len(self.process_stack) - 1)
        
        self.info(f"{indent}{'='*60}")
        self.info(f"{indent}Starting process: {process_name}")
        if details:
            self.debug(f"{indent}Process details: {json.dumps(details, indent=2)}")
        self.info(f"{indent}Process stack: {' -> '.join(self.process_stack)}")
        self.info(f"{indent}{'='*60}")
        
        self.run_logger.add_process(process_name, details)
    
    def log_process_end(self, process_name: str, success: bool = True, duration: float = None):
        """Log end of a process"""
        if self.process_stack and self.process_stack[-1] == process_name:
            self.process_stack.pop()
        
        indent = "  " * len(self.process_stack)
        status = "successfully" if success else "with errors"
        
        self.info(f"{indent}Process '{process_name}' completed {status}")
        if duration is not None:
            self.info(f"{indent}Duration: {duration:.3f} seconds")
        self.info(f"{indent}{'='*60}")
    
    # Function tracking with decorators
    def track_function(self, func: Callable) -> Callable:
        """Decorator to track function calls"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.log_function_entry(func.__name__, args, kwargs)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.log_function_exit(func.__name__, result, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                self.log_function_error(func.__name__, e, duration)
                raise
        
        return wrapper
    
    def log_function_entry(self, func_name: str, args: tuple = None, kwargs: dict = None):
        """Log entry into a function with parameters"""
        self.function_call_depth += 1
        indent = "  " * self.function_call_depth
        
        self.debug(f"{indent}→ Entering function: {func_name}")
        
        # Log parameters (be careful with large objects)
        if args or kwargs:
            params = []
            if args:
                params.extend([f"arg{i}={repr(arg)[:100]}" for i, arg in enumerate(args)])
            if kwargs:
                params.extend([f"{k}={repr(v)[:100]}" for k, v in kwargs.items()])
            self.debug(f"{indent}  Parameters: {', '.join(params)}")
    
    def log_function_exit(self, func_name: str, result=None, duration: float = None):
        """Log exit from a function with result"""
        indent = "  " * self.function_call_depth
        self.function_call_depth = max(0, self.function_call_depth - 1)
        
        self.debug(f"{indent}← Exiting function: {func_name}")
        if result is not None:
            self.debug(f"{indent}  Result: {repr(result)[:200]}")
        if duration is not None:
            self.debug(f"{indent}  Duration: {duration:.6f} seconds")
    
    def log_function_error(self, func_name: str, exception: Exception, duration: float = None):
        """Log function error"""
        indent = "  " * self.function_call_depth
        self.function_call_depth = max(0, self.function_call_depth - 1)
        
        self.error(f"{indent}✗ Function '{func_name}' failed: {str(exception)}")
        if duration is not None:
            self.error(f"{indent}  Duration before error: {duration:.6f} seconds")
        self.debug(f"{indent}  Traceback: {traceback.format_exc()}")
    
    # Class and module tracking
    def log_class_init(self, class_name: str, **kwargs):
        """Log class initialization with verbose details"""
        self.info(f"Initializing class: {class_name}")
        
        # Log initialization parameters
        if kwargs:
            self.debug(f"Initialization parameters for {class_name}:")
            for key, value in kwargs.items():
                self.debug(f"  {key}: {repr(value)[:200]}")
        
        # Log caller information
        frame = inspect.currentframe().f_back
        self.debug(f"Called from: {frame.f_code.co_filename}:{frame.f_lineno}")
    
    def log_module_import(self, module_name: str, from_module: str = None):
        """Log module imports"""
        if from_module:
            self.debug(f"Importing {module_name} from {from_module}")
        else:
            self.debug(f"Importing module: {module_name}")
    
    # Tensor and model tracking
    def log_tensor_info(self, tensor_name: str, tensor, detailed: bool = True):
        """Log detailed tensor information"""
        if hasattr(tensor, 'shape'):
            basic_info = f"Tensor '{tensor_name}': shape={tensor.shape}, dtype={tensor.dtype}"
            
            if hasattr(tensor, 'device'):
                basic_info += f", device={tensor.device}"
            
            self.debug(basic_info)
            
            if detailed and hasattr(tensor, 'numel'):
                self.debug(f"  Elements: {tensor.numel():,}")
                if hasattr(tensor, 'mean'):
                    try:
                        self.debug(f"  Stats: mean={tensor.mean().item():.6f}, "
                                 f"std={tensor.std().item():.6f}, "
                                 f"min={tensor.min().item():.6f}, "
                                 f"max={tensor.max().item():.6f}")
                    except:
                        pass
    
    def log_model_info(self, model_name: str, model=None, num_params: int = None):
        """Log detailed model information"""
        if model and hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.info(f"Model '{model_name}':")
            self.info(f"  Total parameters: {total_params:,}")
            self.info(f"  Trainable parameters: {trainable_params:,}")
            self.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
            
            # Log layer information
            if hasattr(model, 'named_modules'):
                self.debug("Model architecture:")
                for name, module in model.named_modules():
                    if name:  # Skip the root module
                        self.debug(f"  {name}: {module.__class__.__name__}")
        elif num_params is not None:
            self.info(f"Model '{model_name}' initialized with {num_params:,} parameters")
    
    # Training and evaluation tracking
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log start of training epoch"""
        self.info(f"\n{'='*80}")
        self.info(f"EPOCH {epoch}/{total_epochs} STARTED")
        self.info(f"{'='*80}")
        self.run_logger.add_process(f"epoch_{epoch}", {'total_epochs': total_epochs})
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log end of training epoch with metrics"""
        self.info(f"EPOCH {epoch} COMPLETED")
        self.info("Epoch metrics:")
        for key, value in metrics.items():
            self.info(f"  {key}: {value:.6f}")
        self.info(f"{'='*80}\n")
    
    def log_batch_progress(self, batch_idx: int, total_batches: int, loss: float, **metrics):
        """Log detailed batch training progress"""
        progress = (batch_idx + 1) / total_batches * 100
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * (batch_idx + 1) // total_batches)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        self.info(f"Batch [{batch_idx+1:>4}/{total_batches}] |{bar}| {progress:>5.1f}% - Loss: {loss:.6f}")
        
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.debug(f"  Additional metrics: {metrics_str}")
    
    # Analysis tracking
    def log_similarity_check(self, similarity: float, threshold: float, reference_id: str, details: Dict = None):
        """Log detailed similarity comparison"""
        meets_threshold = similarity > threshold
        status = "PASS" if meets_threshold else "FAIL"
        
        self.info(f"Similarity check [{status}]: {similarity:.6f} (threshold: {threshold})")
        self.info(f"  Reference: {reference_id}")
        
        if details:
            self.debug("  Similarity components:")
            for key, value in details.items():
                self.debug(f"    {key}: {value:.6f}")
        
        if not meets_threshold:
            self.warning(f"Similarity below threshold! {similarity:.6f} < {threshold}")
            self.run_logger.add_warning(f"Low similarity for reference {reference_id}: {similarity:.6f}")
    
    def log_anomaly_detection(self, num_anomalies: int, locations: list, severity: Dict[str, Any] = None):
        """Log detailed anomaly detection results"""
        if num_anomalies > 0:
            self.warning(f"Detected {num_anomalies} anomalies")
            self.debug(f"Anomaly locations: {locations}")
            
            if severity:
                self.debug("Anomaly severity analysis:")
                for key, value in severity.items():
                    self.debug(f"  {key}: {value}")
            
            self.run_logger.add_warning(f"Detected {num_anomalies} anomalies")
        else:
            self.info("No anomalies detected")
    
    def log_region_classification(self, region_probs: dict, confidence_scores: dict = None):
        """Log detailed region classification results"""
        self.info("Region classification probabilities:")
        for region, prob in region_probs.items():
            self.info(f"  {region}: {prob:.6f}")
        
        if confidence_scores:
            self.debug("Classification confidence scores:")
            for metric, score in confidence_scores.items():
                self.debug(f"  {metric}: {score:.6f}")
    
    # Performance tracking
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        self.info("Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.6f}")
            else:
                self.info(f"  {key}: {value}")
        
        self.run_logger.update_performance_metrics(metrics)
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            import torch
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            self.debug(f"Memory usage: RSS={memory_info.rss / 1024**2:.2f} MB, "
                      f"VMS={memory_info.vms / 1024**2:.2f} MB")
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**2
                    reserved = torch.cuda.memory_reserved(i) / 1024**2
                    self.debug(f"GPU {i} memory: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")
        except ImportError:
            pass
    
    # Script transitions
    def log_script_transition(self, current_script: str, next_script: str, data_passed: Dict = None):
        """Log detailed script transitions"""
        self.info(f"\n{'='*80}")
        self.info(f"SCRIPT TRANSITION")
        self.info(f"  From: {current_script}")
        self.info(f"  To: {next_script}")
        
        if data_passed:
            self.debug("  Data passed:")
            for key, value in data_passed.items():
                self.debug(f"    {key}: {type(value).__name__}")
        
        self.info(f"{'='*80}\n")
        
        print(f"[{datetime.now()}] Transitioning from {current_script} to {next_script}")
    
    # Error handling
    def log_error(self, error_msg: str, exception: Optional[Exception] = None, critical: bool = False):
        """Log detailed error with optional exception"""
        if critical:
            self.critical(f"CRITICAL ERROR: {error_msg}")
        else:
            self.error(f"ERROR: {error_msg}")
        
        if exception:
            self.error(f"Exception type: {type(exception).__name__}")
            self.error(f"Exception message: {str(exception)}")
            self.debug("Full traceback:")
            self.debug(traceback.format_exc())
            
            self.run_logger.add_error(error_msg, exception)
    
    # Finalization
    def finalize_run(self):
        """Finalize the current run"""
        self.info("Finalizing run...")
        self.log_memory_usage()
        self.run_logger.finalize()
        self.info(f"Run {self.run_id} completed. Logs saved to: {self.run_logger.log_file}")


# Global logger instances cache
_loggers = {}

def get_logger(name: str = "FiberOptics") -> FiberOpticsLogger:
    """Get or create an enhanced logger instance"""
    if name not in _loggers:
        _loggers[name] = FiberOpticsLogger(name)
    return _loggers[name]


# Decorator for automatic function logging
def log_function(logger_name: str = "FiberOptics"):
    """Decorator to automatically log function calls"""
    def decorator(func):
        logger = get_logger(logger_name)
        return logger.track_function(func)
    return decorator


# Context manager for process tracking
@contextmanager
def track_process(process_name: str, logger_name: str = "FiberOptics", **details):
    """Context manager for tracking processes"""
    logger = get_logger(logger_name)
    with logger.track_process(process_name, **details):
        yield


# Test the logger
if __name__ == "__main__":
    logger = get_logger("LoggerTest")
    
    # Test run and cycle management
    cycle1 = logger.start_cycle("Test Cycle 1")
    
    # Test process tracking
    with track_process("Logger Module Test", test_param="value1"):
        # Test function tracking
        @log_function("LoggerTest")
        def test_function(param1, param2=42):
            logger.log_tensor_info("test_tensor", type('MockTensor', (), {
                'shape': (3, 256, 256), 
                'dtype': 'float32', 
                'device': 'cpu',
                'numel': lambda: 3 * 256 * 256,
                'mean': lambda: type('mean', (), {'item': lambda: 0.5})(),
                'std': lambda: type('std', (), {'item': lambda: 0.1})(),
                'min': lambda: type('min', (), {'item': lambda: 0.0})(),
                'max': lambda: type('max', (), {'item': lambda: 1.0})()
            })())
            
            # Test similarity checks
            logger.log_similarity_check(0.65, 0.7, "ref_001", {
                'structural': 0.7,
                'perceptual': 0.6,
                'pixel': 0.65
            })
            logger.log_similarity_check(0.85, 0.7, "ref_002")
            
            # Test anomaly detection
            logger.log_anomaly_detection(3, [(100, 150), (200, 250), (50, 75)], {
                'max_severity': 0.8,
                'avg_severity': 0.5,
                'types': ['scratch', 'contamination']
            })
            
            # Test region classification
            logger.log_region_classification(
                {'core': 0.8, 'cladding': 0.15, 'ferrule': 0.05},
                {'entropy': 0.2, 'margin': 0.65}
            )
            
            return "Success"
        
        # Call the test function
        result = test_function("value1", param2=100)
        
        # Test batch progress
        for i in range(10):
            logger.log_batch_progress(i, 10, 0.5 - i * 0.05, accuracy=0.8 + i * 0.02)
        
        # Test memory logging
        logger.log_memory_usage()
    
    # End cycle with metrics
    logger.end_cycle(cycle1, {
        'avg_loss': 0.25,
        'accuracy': 0.95,
        'processing_time': 10.5
    })
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error("Test error occurred", e)
    
    # Test script transition
    logger.log_script_transition("logger.py", "tensor_processor.py", {
        'tensors': 'List[Tensor]',
        'config': 'Config'
    })
    
    # Finalize run
    logger.finalize_run()
    
    print(f"[{datetime.now()}] Logger test completed")
    print(f"[{datetime.now()}] Check run logs in: {logger.runs_dir}")