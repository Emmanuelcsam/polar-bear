#!/usr/bin/env python3
"""
Enterprise-Grade Multi-Dimensional Logging System
Provides comprehensive logging with multiple channels, real-time streaming, and advanced analytics
"""

import os
import sys
import json
import time
import logging
import threading
import queue
import datetime
import traceback
import inspect
import socket
import platform
try:
    import psutil
except ImportError:
    psutil = None
import colorama
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import gzip
import pickle


# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)


class LogLevel(Enum):
    """Enhanced log levels with additional granularity"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    FATAL = 60


class LogChannel(Enum):
    """Different logging channels for categorization"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATA_FLOW = "data_flow"
    ERRORS = "errors"
    BUSINESS = "business"
    SYSTEM = "system"
    NETWORK = "network"
    DEBUG = "debug"
    NEURAL = "neural"
    MODULE = "module"
    SYNAPSE = "synapse"


@dataclass
class LogEntry:
    """Structured log entry with comprehensive metadata"""
    timestamp: float
    level: LogLevel
    channel: LogChannel
    module: str
    function: str
    line: int
    message: str
    data: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None
    thread_id: str = ""
    process_id: int = 0
    hostname: str = ""
    user: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['level'] = self.level.name
        result['channel'] = self.channel.value
        return result


class LogFormatter:
    """Custom formatter with color support and structured output"""
    
    COLORS = {
        LogLevel.TRACE: colorama.Fore.LIGHTBLACK_EX,
        LogLevel.DEBUG: colorama.Fore.CYAN,
        LogLevel.INFO: colorama.Fore.WHITE,
        LogLevel.SUCCESS: colorama.Fore.GREEN,
        LogLevel.WARNING: colorama.Fore.YELLOW,
        LogLevel.ERROR: colorama.Fore.RED,
        LogLevel.CRITICAL: colorama.Fore.LIGHTRED_EX,
        LogLevel.FATAL: colorama.Fore.LIGHTRED_EX + colorama.Style.BRIGHT
    }
    
    ICONS = {
        LogLevel.TRACE: "🔍",
        LogLevel.DEBUG: "🐛",
        LogLevel.INFO: "ℹ️",
        LogLevel.SUCCESS: "✅",
        LogLevel.WARNING: "⚠️",
        LogLevel.ERROR: "❌",
        LogLevel.CRITICAL: "🚨",
        LogLevel.FATAL: "💀"
    }
    
    def __init__(self, use_color: bool = True, use_icons: bool = True, detailed: bool = False):
        self.use_color = use_color
        self.use_icons = use_icons
        self.detailed = detailed
    
    def format(self, entry: LogEntry) -> str:
        """Format log entry for console output"""
        # Timestamp
        dt = datetime.datetime.fromtimestamp(entry.timestamp)
        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Level with color and icon
        level_str = entry.level.name.ljust(8)
        if self.use_icons:
            level_str = f"{self.ICONS[entry.level]} {level_str}"
        
        if self.use_color:
            level_str = f"{self.COLORS[entry.level]}{level_str}{colorama.Style.RESET_ALL}"
        
        # Channel
        channel_str = f"[{entry.channel.value}]"
        
        # Location info
        location = f"{entry.module}:{entry.function}:{entry.line}"
        
        # Build message
        parts = [timestamp, level_str, channel_str]
        
        if self.detailed:
            parts.extend([
                f"PID:{entry.process_id}",
                f"TID:{entry.thread_id[-8:]}",
                location
            ])
        
        parts.append(entry.message)
        
        # Add data if present
        if entry.data:
            data_str = json.dumps(entry.data, indent=2) if self.detailed else json.dumps(entry.data)
            parts.append(f"\n  Data: {data_str}")
        
        # Add traceback if present
        if entry.traceback:
            parts.append(f"\n{entry.traceback}")
        
        return " | ".join(parts[:4]) + " | " + " ".join(parts[4:])


class LogHandler:
    """Base class for log handlers"""
    
    def handle(self, entry: LogEntry):
        raise NotImplementedError


class ConsoleLogHandler(LogHandler):
    """Handler for console output with formatting"""
    
    def __init__(self, formatter: LogFormatter, min_level: LogLevel = LogLevel.INFO):
        self.formatter = formatter
        self.min_level = min_level
    
    def handle(self, entry: LogEntry):
        if entry.level.value >= self.min_level.value:
            print(self.formatter.format(entry))


class FileLogHandler(LogHandler):
    """Handler for file output with rotation and compression"""
    
    def __init__(self, filename: str, max_size: int = 100 * 1024 * 1024, 
                 max_files: int = 10, compress: bool = True):
        self.filename = filename
        self.max_size = max_size
        self.max_files = max_files
        self.compress = compress
        self.current_size = 0
        self.file = None
        self.lock = threading.Lock()
        self._open_file()
    
    def _open_file(self):
        """Open log file for writing"""
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.file = open(self.filename, 'a', encoding='utf-8')
        self.current_size = os.path.getsize(self.filename) if os.path.exists(self.filename) else 0
    
    def _rotate(self):
        """Rotate log files"""
        with self.lock:
            self.file.close()
            
            # Rotate existing files
            for i in range(self.max_files - 1, 0, -1):
                old_name = f"{self.filename}.{i}"
                new_name = f"{self.filename}.{i + 1}"
                if self.compress:
                    old_name += ".gz"
                    new_name += ".gz"
                
                if os.path.exists(old_name):
                    if i == self.max_files - 1:
                        os.remove(old_name)
                    else:
                        os.rename(old_name, new_name)
            
            # Move current file
            if self.compress:
                with open(self.filename, 'rb') as f_in:
                    with gzip.open(f"{self.filename}.1.gz", 'wb') as f_out:
                        f_out.writelines(f_in)
                os.remove(self.filename)
            else:
                os.rename(self.filename, f"{self.filename}.1")
            
            self._open_file()
    
    def handle(self, entry: LogEntry):
        """Write entry to file"""
        with self.lock:
            line = json.dumps(entry.to_dict()) + "\n"
            self.file.write(line)
            self.file.flush()
            
            self.current_size += len(line.encode('utf-8'))
            if self.current_size >= self.max_size:
                self._rotate()


class DatabaseLogHandler(LogHandler):
    """Handler for database storage with indexing and querying"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                level TEXT NOT NULL,
                channel TEXT NOT NULL,
                module TEXT NOT NULL,
                function TEXT NOT NULL,
                line INTEGER NOT NULL,
                message TEXT NOT NULL,
                data TEXT,
                traceback TEXT,
                thread_id TEXT,
                process_id INTEGER,
                hostname TEXT,
                user TEXT
            )
        ''')
        
        # Create indexes separately
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_level ON logs(level)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_channel ON logs(channel)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_module ON logs(module)')
        self.conn.commit()
    
    def handle(self, entry: LogEntry):
        """Store entry in database"""
        with self.lock:
            self.conn.execute('''
                INSERT INTO logs (
                    timestamp, level, channel, module, function, line,
                    message, data, traceback, thread_id, process_id,
                    hostname, user
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.timestamp,
                entry.level.name,
                entry.channel.value,
                entry.module,
                entry.function,
                entry.line,
                entry.message,
                json.dumps(entry.data) if entry.data else None,
                entry.traceback,
                entry.thread_id,
                entry.process_id,
                entry.hostname,
                entry.user
            ))
            self.conn.commit()


class NeuralNetworkLogger:
    """Main logger class for the neural network system"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logger if not already initialized"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.handlers: List[LogHandler] = []
        self.queue = queue.Queue(maxsize=10000)
        self.running = True
        self.min_level = LogLevel.TRACE
        
        # System info
        self.hostname = socket.gethostname()
        self.user = os.getenv('USER', os.getenv('USERNAME', 'unknown'))
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        # Default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default logging handlers"""
        # Console handler
        console_formatter = LogFormatter(use_color=True, use_icons=True, detailed=False)
        self.add_handler(ConsoleLogHandler(console_formatter, LogLevel.INFO))
        
        # File handlers for different channels
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Main log file
        self.add_handler(FileLogHandler(
            str(log_dir / "neural_network.log"),
            max_size=50 * 1024 * 1024,
            compress=True
        ))
        
        # Error log file
        error_handler = FileLogHandler(
            str(log_dir / "errors.log"),
            max_size=10 * 1024 * 1024,
            compress=True
        )
        self.add_handler(lambda entry: error_handler.handle(entry) 
                        if entry.level.value >= LogLevel.ERROR.value else None)
        
        # Database handler
        self.add_handler(DatabaseLogHandler(str(log_dir / "logs.db")))
    
    def add_handler(self, handler: LogHandler):
        """Add a log handler"""
        self.handlers.append(handler)
    
    def _worker(self):
        """Worker thread for processing log entries"""
        while self.running:
            try:
                entry = self.queue.get(timeout=1)
                for handler in self.handlers:
                    try:
                        if callable(handler):
                            handler(entry)
                        else:
                            handler.handle(entry)
                    except Exception as e:
                        # Print to stderr if handler fails
                        print(f"Log handler error: {e}", file=sys.stderr)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logger worker error: {e}", file=sys.stderr)
    
    def _create_entry(self, level: LogLevel, channel: LogChannel, message: str,
                     data: Optional[Dict[str, Any]] = None, 
                     exc_info: Optional[bool] = False) -> LogEntry:
        """Create a log entry with metadata"""
        # Get caller info
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_frame = frame.f_back.f_back
            module = os.path.basename(caller_frame.f_code.co_filename)
            function = caller_frame.f_code.co_name
            line = caller_frame.f_lineno
        else:
            module = "unknown"
            function = "unknown"
            line = 0
        
        # Get exception info if requested
        tb = None
        if exc_info and sys.exc_info()[0] is not None:
            tb = traceback.format_exc()
        
        return LogEntry(
            timestamp=time.time(),
            level=level,
            channel=channel,
            module=module,
            function=function,
            line=line,
            message=message,
            data=data,
            traceback=tb,
            thread_id=threading.current_thread().name,
            process_id=os.getpid(),
            hostname=self.hostname,
            user=self.user
        )
    
    def log(self, level: LogLevel, channel: LogChannel, message: str,
            data: Optional[Dict[str, Any]] = None, exc_info: Optional[bool] = False):
        """Log a message"""
        if level.value >= self.min_level.value:
            entry = self._create_entry(level, channel, message, data, exc_info)
            try:
                self.queue.put_nowait(entry)
            except queue.Full:
                # If queue is full, process synchronously
                for handler in self.handlers:
                    try:
                        if callable(handler):
                            handler(entry)
                        else:
                            handler.handle(entry)
                    except:
                        pass
    
    # Convenience methods
    def trace(self, channel: LogChannel, message: str, **kwargs):
        self.log(LogLevel.TRACE, channel, message, kwargs if kwargs else None)
    
    def debug(self, channel: LogChannel, message: str, **kwargs):
        self.log(LogLevel.DEBUG, channel, message, kwargs if kwargs else None)
    
    def info(self, channel: LogChannel, message: str, **kwargs):
        self.log(LogLevel.INFO, channel, message, kwargs if kwargs else None)
    
    def success(self, channel: LogChannel, message: str, **kwargs):
        self.log(LogLevel.SUCCESS, channel, message, kwargs if kwargs else None)
    
    def warning(self, channel: LogChannel, message: str, **kwargs):
        self.log(LogLevel.WARNING, channel, message, kwargs if kwargs else None)
    
    def error(self, channel: LogChannel, message: str, **kwargs):
        self.log(LogLevel.ERROR, channel, message, kwargs if kwargs else None, exc_info=True)
    
    def critical(self, channel: LogChannel, message: str, **kwargs):
        self.log(LogLevel.CRITICAL, channel, message, kwargs if kwargs else None, exc_info=True)
    
    def fatal(self, channel: LogChannel, message: str, **kwargs):
        self.log(LogLevel.FATAL, channel, message, kwargs if kwargs else None, exc_info=True)
    
    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        data = {"duration_ms": duration * 1000, "operation": operation}
        data.update(kwargs)
        self.log(LogLevel.INFO, LogChannel.PERFORMANCE, f"Performance: {operation}", data)
    
    def shutdown(self):
        """Shutdown the logger gracefully"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        # Close all handlers
        for handler in self.handlers:
            if hasattr(handler, 'close'):
                handler.close()


# Global logger instance
logger = NeuralNetworkLogger()


# Convenience functions
def log_trace(message: str, channel: LogChannel = LogChannel.DEBUG, **kwargs):
    logger.trace(channel, message, **kwargs)

def log_debug(message: str, channel: LogChannel = LogChannel.DEBUG, **kwargs):
    logger.debug(channel, message, **kwargs)

def log_info(message: str, channel: LogChannel = LogChannel.SYSTEM, **kwargs):
    logger.info(channel, message, **kwargs)

def log_success(message: str, channel: LogChannel = LogChannel.SYSTEM, **kwargs):
    logger.success(channel, message, **kwargs)

def log_warning(message: str, channel: LogChannel = LogChannel.SYSTEM, **kwargs):
    logger.warning(channel, message, **kwargs)

def log_error(message: str, channel: LogChannel = LogChannel.ERRORS, **kwargs):
    logger.error(channel, message, **kwargs)

def log_critical(message: str, channel: LogChannel = LogChannel.ERRORS, **kwargs):
    logger.critical(channel, message, **kwargs)

def log_performance(operation: str, duration: float, **kwargs):
    logger.performance(operation, duration, **kwargs)


# Context manager for timing operations
class TimedOperation:
    """Context manager for timing and logging operations"""
    
    def __init__(self, operation: str, channel: LogChannel = LogChannel.PERFORMANCE):
        self.operation = operation
        self.channel = channel
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.trace(self.channel, f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            logger.performance(self.operation, duration)
        else:
            logger.error(self.channel, f"Operation failed: {self.operation}", 
                        duration_ms=duration * 1000, error=str(exc_val))


if __name__ == "__main__":
    # Demo the logging system
    print("🚀 Neural Network Logger Demo")
    print("=" * 50)
    
    # Test different log levels
    log_info("Logger initialized successfully")
    log_debug("This is a debug message", module="test", function="main")
    log_success("Operation completed successfully", operation="test", result="passed")
    log_warning("This is a warning", severity="medium")
    
    # Test performance logging
    with TimedOperation("test_operation"):
        time.sleep(0.1)
    
    # Test error logging
    try:
        1 / 0
    except Exception:
        log_error("Division by zero error occurred")
    
    print("\n✅ Logger demo complete! Check the logs directory for output files.")