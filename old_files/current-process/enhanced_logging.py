"""
Enhanced Logging System
Full logging by default, no argparse required
"""

import logging
import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
from functools import wraps
import threading
import queue
import atexit


class ColoredFormatter(logging.Formatter):
    """Colored output formatter for console"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class StructuredLogger:
    """Structured logging with JSON output option"""
    
    def __init__(self, name: str, structured: bool = False):
        self.logger = logging.getLogger(name)
        self.structured = structured
        self._context = {}
    
    def set_context(self, **kwargs):
        """Set persistent context for all logs"""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear persistent context"""
        self._context.clear()
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with structure support"""
        if self.structured:
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": message,
                "logger": self.logger.name,
                **self._context,
                **kwargs
            }
            self.logger.log(
                getattr(logging, level),
                json.dumps(log_data, default=str)
            )
        else:
            extra_str = ""
            if kwargs:
                extra_str = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.log(
                getattr(logging, level),
                f"{message}{extra_str}"
            )
    
    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log("CRITICAL", message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs["traceback"] = traceback.format_exc()
        self._log("ERROR", message, **kwargs)


class AsyncFileHandler(logging.Handler):
    """Asynchronous file handler to prevent I/O blocking"""
    
    def __init__(self, filename: str, mode='a', encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._writer)
        self.thread.daemon = True
        self.thread.start()
        atexit.register(self.close)
    
    def _writer(self):
        """Background thread for writing logs"""
        with open(self.filename, self.mode, encoding=self.encoding) as f:
            while True:
                record = self.queue.get()
                if record is None:
                    break
                try:
                    f.write(self.format(record) + '\n')
                    f.flush()
                except Exception:
                    pass
    
    def emit(self, record):
        """Queue the record for writing"""
        self.queue.put(record)
    
    def close(self):
        """Close the handler"""
        self.queue.put(None)
        self.thread.join()
        super().close()


class LogManager:
    """Central log management system"""
    
    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.handlers = {}
        self.setup_root_logger()
    
    def setup_root_logger(self):
        """Setup the root logger with default configuration"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        
        # Console handler with color
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)-20s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Main log file (async)
        main_log = self.log_dir / f"fiber_detection_{self.session_id}.log"
        file_handler = AsyncFileHandler(str(main_log))
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_log = self.log_dir / f"errors_{self.session_id}.log"
        error_handler = AsyncFileHandler(str(error_log))
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        self.handlers['console'] = console_handler
        self.handlers['file'] = file_handler
        self.handlers['error'] = error_handler
    
    def get_logger(self, name: str, structured: bool = False) -> StructuredLogger:
        """Get a logger instance"""
        return StructuredLogger(name, structured)
    
    def set_console_level(self, level: str):
        """Dynamically change console log level"""
        if 'console' in self.handlers:
            self.handlers['console'].setLevel(getattr(logging, level.upper()))
    
    def add_file_handler(self, name: str, filename: str, level: str = 'DEBUG'):
        """Add a custom file handler"""
        handler = AsyncFileHandler(str(self.log_dir / filename))
        handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        self.handlers[name] = handler
    
    def create_module_log(self, module_name: str):
        """Create a dedicated log file for a module"""
        log_file = f"{module_name}_{self.session_id}.log"
        self.add_file_handler(f"{module_name}_file", log_file)
        return self.get_logger(module_name)


# Global log manager instance
_log_manager = None


def get_log_manager() -> LogManager:
    """Get or create the log manager singleton"""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager


def get_logger(name: str, structured: bool = False) -> StructuredLogger:
    """Get a logger instance"""
    return get_log_manager().get_logger(name, structured)


# Decorators for function logging
def log_execution(func):
    """Decorator to log function execution"""
    logger = get_logger(func.__module__)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Entering {func_name}", args=str(args)[:100], kwargs=str(kwargs)[:100])
        
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Exiting {func_name}", duration_seconds=duration)
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.exception(
                f"Error in {func_name}",
                duration_seconds=duration,
                error_type=type(e).__name__
            )
            raise
    
    return wrapper


def log_performance(func):
    """Decorator to log function performance metrics"""
    logger = get_logger(f"{func.__module__}.performance")
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Before metrics
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            # After metrics
            cpu_after = process.cpu_percent()
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            logger.info(
                f"Performance: {func.__name__}",
                duration_seconds=duration,
                cpu_usage_percent=cpu_after,
                memory_delta_mb=mem_after - mem_before,
                memory_total_mb=mem_after
            )
            
            return result
        except Exception as e:
            logger.error(f"Performance tracking failed for {func.__name__}: {e}")
            raise
    
    return wrapper


class ProgressLogger:
    """Logger for tracking progress of long operations"""
    
    def __init__(self, name: str, total: int, update_interval: int = 10):
        self.logger = get_logger(name)
        self.total = total
        self.current = 0
        self.update_interval = update_interval
        self.start_time = datetime.now()
        self.last_update = 0
    
    def update(self, increment: int = 1, message: str = ""):
        """Update progress"""
        self.current += increment
        
        if self.current - self.last_update >= self.update_interval or self.current >= self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            self.logger.info(
                f"Progress: {message}" if message else "Progress update",
                current=self.current,
                total=self.total,
                percent=round(100 * self.current / self.total, 1),
                rate_per_second=round(rate, 2),
                eta_seconds=round(eta, 1)
            )
            
            self.last_update = self.current
    
    def complete(self, message: str = "Operation completed"):
        """Mark as complete"""
        duration = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            message,
            total_processed=self.total,
            duration_seconds=round(duration, 2),
            rate_per_second=round(self.total / duration if duration > 0 else 0, 2)
        )


# Setup logging on import
get_log_manager()


# Convenience functions
def debug(message: str, **kwargs):
    """Quick debug logging"""
    get_logger("quick").debug(message, **kwargs)


def info(message: str, **kwargs):
    """Quick info logging"""
    get_logger("quick").info(message, **kwargs)


def warning(message: str, **kwargs):
    """Quick warning logging"""
    get_logger("quick").warning(message, **kwargs)


def error(message: str, **kwargs):
    """Quick error logging"""
    get_logger("quick").error(message, **kwargs)


def exception(message: str, **kwargs):
    """Quick exception logging"""
    get_logger("quick").exception(message, **kwargs)