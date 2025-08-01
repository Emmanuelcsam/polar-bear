#!/usr/bin/env python3
"""
Performance Timer Decorator Module
=================================
A reusable decorator for measuring and logging function execution time.
Perfect for performance optimization and debugging.
"""

import time
import logging
from functools import wraps
from typing import Any, Callable

def performance_timer(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Wrapper function that measures execution time
        
    Example:
        @performance_timer
        def my_slow_function():
            time.sleep(1)
            return "Done"
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

def time_function(func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """
    Times a function call and returns both result and execution time.
    
    Args:
        func: Function to time
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Tuple of (result, execution_time_seconds)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_time = time.perf_counter() - start_time
    return result, elapsed_time

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Test the decorator
    @performance_timer
    def test_function():
        time.sleep(0.1)
        return "Test completed"
    
    # Test function timing
    def another_test():
        time.sleep(0.05)
        return "Another test"
    
    print("Testing performance timer decorator:")
    result1 = test_function()
    print(f"Result: {result1}")
    
    print("\nTesting function timing utility:")
    result2, exec_time = time_function(another_test)
    print(f"Result: {result2}, Time: {exec_time:.4f}s")
