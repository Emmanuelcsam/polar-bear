#!/usr/bin/env python3
"""
Logger module for Fiber Optics Neural Network
"the program will log all of its processes in a log file as soon as they happen, 
it will log verbosely and with time stamps so I can see exactly what is happening when it happens"
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class FiberOpticsLogger:
    """Custom logger for fiber optics analysis with verbose output"""
    
    def __init__(self, name: str, config=None):
        print(f"[{datetime.now()}] Initializing FiberOpticsLogger for {name}")
        print(f"[{datetime.now()}] Previous script: config.py")
        
        self.name = name
        
        # Get configuration
        if config is None:
            from fiber_config import get_config
            config = get_config()
        self.config = config
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler - verbose logging
        file_handler = logging.FileHandler(self.config.LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler - less verbose
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info(f"Logger initialized for {name}")
        self.logger.debug(f"Log file: {self.config.LOG_FILE}")
        
        print(f"[{datetime.now()}] FiberOpticsLogger initialized successfully")
    
    def log_function_entry(self, func_name: str, **kwargs):
        """Log entry into a function with parameters"""
        self.logger.debug(f"Entering function: {func_name}")
        if kwargs:
            self.logger.debug(f"Parameters: {kwargs}")
    
    def log_function_exit(self, func_name: str, result=None):
        """Log exit from a function with result"""
        self.logger.debug(f"Exiting function: {func_name}")
        if result is not None:
            self.logger.debug(f"Result: {result}")
    
    def log_class_init(self, class_name: str, **kwargs):
        """Log class initialization"""
        self.logger.info(f"Initializing class: {class_name}")
        if kwargs:
            self.logger.debug(f"Initialization parameters: {kwargs}")
    
    def log_process_start(self, process_name: str):
        """Log start of a process"""
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting process: {process_name}")
        self.logger.info(f"{'='*60}")
    
    def log_process_end(self, process_name: str, success: bool = True):
        """Log end of a process"""
        status = "successfully" if success else "with errors"
        self.logger.info(f"Process '{process_name}' completed {status}")
        self.logger.info(f"{'='*60}")
    
    def log_tensor_info(self, tensor_name: str, tensor):
        """Log tensor information"""
        if hasattr(tensor, 'shape'):
            self.logger.debug(f"Tensor '{tensor_name}': shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    
    def log_model_info(self, model_name: str, num_params: int):
        """Log model information"""
        self.logger.info(f"Model '{model_name}' initialized with {num_params:,} parameters")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log start of training epoch"""
        self.logger.info(f"Starting epoch {epoch}/{total_epochs}")
    
    def log_batch_progress(self, batch_idx: int, total_batches: int, loss: float, **metrics):
        """Log batch training progress"""
        progress = (batch_idx + 1) / total_batches * 100
        self.logger.info(f"Batch [{batch_idx+1}/{total_batches}] ({progress:.1f}%) - Loss: {loss:.4f}")
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.debug(f"Batch metrics: {metrics_str}")
    
    def log_similarity_check(self, similarity: float, threshold: float, reference_id: str):
        """Log similarity comparison - 'the program must achieve over .7'"""
        meets_threshold = similarity > threshold
        status = "PASS" if meets_threshold else "FAIL"
        self.logger.info(f"Similarity check [{status}]: {similarity:.4f} (threshold: {threshold}) for reference: {reference_id}")
        if not meets_threshold:
            self.logger.warning(f"Similarity below threshold! {similarity:.4f} < {threshold}")
    
    def log_anomaly_detection(self, num_anomalies: int, locations: list):
        """Log anomaly detection results"""
        if num_anomalies > 0:
            self.logger.warning(f"Detected {num_anomalies} anomalies")
            self.logger.debug(f"Anomaly locations: {locations}")
        else:
            self.logger.info("No anomalies detected")
    
    def log_region_classification(self, region_probs: dict):
        """Log region classification results"""
        self.logger.info("Region classification probabilities:")
        for region, prob in region_probs.items():
            self.logger.info(f"  {region}: {prob:.4f}")
    
    def log_script_transition(self, current_script: str, next_script: str):
        """Log transition between scripts"""
        self.logger.info(f"Completing script: {current_script}")
        self.logger.info(f"Next script: {next_script}")
        print(f"[{datetime.now()}] Transitioning from {current_script} to {next_script}")
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Log error with optional exception"""
        self.logger.error(f"ERROR: {error_msg}")
        if exception:
            self.logger.exception("Exception details:", exc_info=exception)
    
    # Convenience methods that mirror standard logging
    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def critical(self, msg): self.logger.critical(msg)

# Global logger instances cache
_loggers = {}

def get_logger(name: str = "FiberOptics") -> FiberOpticsLogger:
    """Get or create a logger instance"""
    if name not in _loggers:
        _loggers[name] = FiberOpticsLogger(name)
    return _loggers[name]

# Test the logger
if __name__ == "__main__":
    logger = get_logger("LoggerTest")
    
    # Test various logging functions
    logger.log_process_start("Logger Module Test")
    
    logger.log_function_entry("test_function", param1="value1", param2=42)
    logger.log_tensor_info("test_tensor", type('MockTensor', (), {'shape': (3, 256, 256), 'dtype': 'float32', 'device': 'cpu'})())
    logger.log_similarity_check(0.65, 0.7, "ref_001")
    logger.log_similarity_check(0.85, 0.7, "ref_002")
    logger.log_anomaly_detection(3, [(100, 150), (200, 250), (50, 75)])
    logger.log_region_classification({'core': 0.8, 'cladding': 0.15, 'ferrule': 0.05})
    logger.log_function_exit("test_function", result="Success")
    
    logger.log_process_end("Logger Module Test")
    logger.log_script_transition("logger.py", "tensor_processor.py")
    
    print(f"[{datetime.now()}] Logger test completed")
    print(f"[{datetime.now()}] Next script: tensor_processor.py")
