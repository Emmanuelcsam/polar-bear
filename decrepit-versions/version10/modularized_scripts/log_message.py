
from datetime import datetime

def log_message(message: str, level: str = "INFO"):
    """Prints a timestamped log message to the console."""
    # Get current time in a specific format.
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # Print the formatted log message.
    print(f"[{current_time}] [{level.upper()}] {message}")

if __name__ == '__main__':
    # Example of how to use the log_message function
    
    log_message("This is a standard information message.")
    log_message("This is a debug message.", level="DEBUG")
    log_message("This is a warning.", level="WARNING")
    log_message("This is a critical error!", level="ERROR")
