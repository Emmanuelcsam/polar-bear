import logging
import os
from datetime import datetime
from pathlib import Path


class FiberOpticsLogger:
    """
    Verbose logging system for fiber optics neural network processing.
    "the program will log all of its processes in a log file as soon as they happen, 
    it will log verbosely and with time stamps so I can see exactly what is happening when it happens"
    """
    
    def __init__(self, log_dir="logs", log_name=None):
        # Create log directory if it doesn't exist
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Generate log filename with timestamp
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"fiber_optics_nn_{timestamp}.log"
        
        self.log_path = self.log_dir / log_name
        
        # Configure logger
        self.logger = logging.getLogger("FiberOpticsNN")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # File handler - logs everything
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler - logs INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter with microsecond precision
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - [%(levelname)s] - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Initialized FiberOpticsLogger - Log file: {self.log_path}")
        
    def debug(self, message):
        """Log debug message with immediate flush"""
        self.logger.debug(message)
        self._flush_handlers()
        
    def info(self, message):
        """Log info message with immediate flush"""
        self.logger.info(message)
        self._flush_handlers()
        
    def warning(self, message):
        """Log warning message with immediate flush"""
        self.logger.warning(message)
        self._flush_handlers()
        
    def error(self, message, exc_info=False):
        """Log error message with immediate flush"""
        self.logger.error(message, exc_info=exc_info)
        self._flush_handlers()
        
    def critical(self, message, exc_info=False):
        """Log critical message with immediate flush"""
        self.logger.critical(message, exc_info=exc_info)
        self._flush_handlers()
        
    def _flush_handlers(self):
        """Force immediate write to log file"""
        for handler in self.logger.handlers:
            handler.flush()
    
    def log_tensor_operation(self, operation, tensor_shape, additional_info=""):
        """Special logging for tensor operations"""
        message = f"TENSOR_OP: {operation} | Shape: {tensor_shape}"
        if additional_info:
            message += f" | {additional_info}"
        self.info(message)
    
    def log_weight_update(self, layer_name, old_weight, new_weight, gradient):
        """Special logging for neural network weight updates"""
        message = (f"WEIGHT_UPDATE: Layer: {layer_name} | "
                  f"Old: {old_weight:.6f} | New: {new_weight:.6f} | "
                  f"Gradient: {gradient:.6f}")
        self.debug(message)
    
    def log_image_processing(self, image_path, operation, result):
        """Special logging for image processing operations"""
        message = f"IMAGE_PROC: {image_path} | Op: {operation} | Result: {result}"
        self.info(message)
    
    def log_defect_detection(self, image_id, defect_type, location, confidence):
        """Special logging for defect detection results"""
        message = (f"DEFECT_DETECTED: Image: {image_id} | Type: {defect_type} | "
                  f"Location: {location} | Confidence: {confidence:.4f}")
        self.warning(message)
    
    def log_performance_metric(self, metric_name, value, unit="ms"):
        """Log performance metrics for optimization tracking"""
        message = f"PERFORMANCE: {metric_name}: {value:.2f}{unit}"
        self.info(message)


# Global logger instance
_logger = None

def get_logger():
    """Get or create the global logger instance"""
    global _logger
    if _logger is None:
        _logger = FiberOpticsLogger()
    return _logger