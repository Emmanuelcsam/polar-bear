

import time
from datetime import datetime
from typing import Optional

# It's often better to import specific classes rather than entire modules
# to avoid circular dependencies, especially as the project grows.
# from image_result import ImageResult 
# For this example, we'll use a placeholder to avoid the dependency for now.
from typing import Any, Dict
ImageResult = Any 

def start_timer() -> float:
    """Returns the current time to start a timer."""
    # Get high-resolution performance counter time.
    return time.perf_counter()

def log_duration(operation_name: str, start_time: float, image_result: Optional[ImageResult] = None):
    """Logs the duration of an operation."""
    # Calculate the elapsed time.
    duration = time.perf_counter() - start_time
    
    # A simple logger for demonstration purposes
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{current_time}] [INFO] Operation '{operation_name}' completed in {duration:.4f} seconds.")
    
    # If an ImageResult object is provided and has a timing_log dictionary, store the duration.
    if image_result and hasattr(image_result, 'timing_log') and isinstance(image_result.timing_log, dict):
        image_result.timing_log[operation_name] = duration
    
    return duration

if __name__ == '__main__':
    # Example of how to use the timer utility functions

    # Mock ImageResult class for demonstration
    class MockImageResult:
        def __init__(self):
            self.timing_log: Dict[str, float] = {}
        def __repr__(self):
            return f"MockImageResult(timing_log={self.timing_log})"

    mock_result = MockImageResult()

    print("--- Running timing example ---")
    
    # 1. Time a simple operation
    op1_start_time = start_timer()
    time.sleep(0.5) # Simulate work
    log_duration("First Operation", op1_start_time)

    # 2. Time another operation and store the result in the mock object
    op2_start_time = start_timer()
    time.sleep(0.7) # Simulate more work
    log_duration("Second Operation (with result logging)", op2_start_time, mock_result)

    print(f"\nFinal state of mock_result: {mock_result}")
    print("--- Timing example finished ---")

