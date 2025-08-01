
import os
import logging
import sys
from connector_interface import ConnectorInterface, setup_connector

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
    
    # Setup connector interface
    connector = setup_connector("connector.py")
    
    # Register this script's capabilities
    if connector.is_connected:
        logger.info("Connected to hivemind system")
        connector.register_parameter("scan_interval", 60, "Interval in seconds between directory scans")
        connector.register_callback("list_files", list_directory_files)
        connector.register_callback("get_file_info", get_file_info)
    else:
        logger.info("Running in standalone mode")
    
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
    
    # Send status to hivemind if connected
    if connector.is_connected:
        status = {
            "directory": os.getcwd(),
            "file_count": len(files) if 'files' in locals() else 0,
            "status": "active"
        }
        connector.send_status(status)

    logger.info("Connector script finished.")

def list_directory_files():
    """List all files in the current directory"""
    try:
        files = os.listdir()
        return {"files": files, "count": len(files)}
    except OSError as e:
        return {"error": str(e)}

def get_file_info(filename):
    """Get information about a specific file"""
    try:
        if os.path.exists(filename):
            stat = os.stat(filename)
            return {
                "filename": filename,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "is_directory": os.path.isdir(filename)
            }
        else:
            return {"error": "File not found"}
    except OSError as e:
        return {"error": str(e)}

if __name__ == "__main__":
    main()
