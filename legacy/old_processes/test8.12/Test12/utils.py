from datetime import datetime
import time
from typing import Optional
from data_structures import ImageResult

def _log_message(message: str, level: str = "INFO"):
    """Prints a timestamped log message to the console."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{current_time}] [{level.upper()}] {message}")

def _start_timer() -> float:
    """Returns the current time to start a timer."""
    return time.perf_counter()

def _log_duration(operation_name: str, start_time: float, image_result: Optional[ImageResult] = None):
    """Logs the duration of an operation and stores it."""
    duration = time.perf_counter() - start_time
    _log_message(f"Operation '{operation_name}' completed in {duration:.4f} seconds.")
    if image_result and hasattr(image_result, "timing_log"):
        image_result.timing_log[operation_name] = duration
    return duration
