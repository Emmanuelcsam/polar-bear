from datetime import datetime
import time
from typing import Optional
from data_structures.image_result import ImageResult

def _log_message(message: str, level: str = "INFO"):
    """Prints a timestamped log message to the console."""
    # Get current time in a specific format.
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # Print the formatted log message.
    print(f"[{current_time}] [{level.upper()}] {message}")

def _start_timer() -> float:
    """Returns the current time to start a timer."""
    # Get high-resolution performance counter time.
    return time.perf_counter()

def _log_duration(operation_name: str, start_time: float, image_result: Optional[ImageResult] = None):
    """Logs the duration of an operation."""
    # Calculate the elapsed time.
    duration = time.perf_counter() - start_time
    # Log the duration message.
    _log_message(f"Operation '{operation_name}' completed in {duration:.4f} seconds.")
    # Placeholder for storing timing in ImageResult if needed later
    # if image_result and hasattr(image_result, 'timing_log'):
    #     image_result.timing_log[operation_name] = duration
    # Return the duration.
    return duration

if __name__ == '__main__':
    # Example of using the logging utility functions
    _log_message("This is an info message.")
    _log_message("This is a warning message.", level="WARNING")
    
    start = _start_timer()
    time.sleep(0.1)
    _log_duration("Test Operation", start)
