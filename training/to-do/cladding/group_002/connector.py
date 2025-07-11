
import os
import logging
import sys

# --- Configuration ---
LOG_FILE = "connector.log"

# --- Setup Logging ---
# Ensure the logger is configured from scratch for each script
logger = logging.getLogger(os.path.abspath(__file__))
logger.setLevel(logging.INFO)

# Prevent logging from propagating to the root logger
logger.propagate = False

# Remove any existing handlers to avoid duplicate logs
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

# File Handler
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except (IOError, OSError) as e:
    # Fallback to console if file logging fails
    print(f"Could not write to log file {LOG_FILE}: {e}", file=sys.stderr)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


def main():
    """Main function for the connector script."""
    logger.info(f"--- Connector Script Initialized in {os.getcwd()} ---")
    logger.info(f"This script is responsible for connecting the modules in this directory to the main control script.")
    
    # List files in the current directory
    try:
        files = os.listdir()
        if files:
            logger.info("Files in this directory:")
            for file in files:
                logger.info(f"- {file}")
        else:
            logger.info("No files found in this directory.")
    except OSError as e:
        logger.error(f"Error listing files: {e}")

    logger.info("Connector script finished.")

if __name__ == "__main__":
    main()
