import os
import subprocess
import sys
import logging
from datetime import datetime

# --- Configuration ---
LOG_FILE = "main_control.log"
CONNECTOR_SCRIPT_NAME = "connector.py"

# --- Setup Logging ---
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

def check_and_install_packages(packages):
    """Checks if packages are installed and installs them if they are not."""
    for package in packages:
        try:
            __import__(package)
            logger.info(f"'{package}' is already installed.")
        except ImportError:
            logger.info(f"'{package}' not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])
                logger.info(f"Successfully installed '{package}'.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install '{package}'. Error: {e}")
                # Exit or handle the error as needed
                sys.exit(1)

def find_connector_scripts(root_dirs):
    """Finds all connector scripts in the specified directories."""
    connector_scripts = []
    for root_dir in root_dirs:
        for subdir, _, files in os.walk(root_dir):
            if CONNECTOR_SCRIPT_NAME in files:
                connector_scripts.append(os.path.join(subdir, CONNECTOR_SCRIPT_NAME))
    return connector_scripts

def main():
    """Main function to control the execution."""
    logger.info("--- Main Control Script Initialized ---")

    # Set log level
    logger.setLevel(logging.INFO)

    # Check for essential packages
    required_packages = ["requests"] # Add any other required packages here
    check_and_install_packages(required_packages)

    # Define the directories to scan for connector scripts
    target_dirs = [
        r"C:\Users\Saem1001\Documents\GitHub\polar-bear\ruleset",
        r"C:\Users\Saem1001\Documents\GitHub\polar-bear\modules",
        r"C:\Users\Saem1001\Documents\GitHub\polar-bear\training",
        r"C:\Users\Saem1001\Documents\GitHub\polar-bear\static",
        r"C:\Users\Saem1001\Documents\GitHub\polar-bear\templates"
    ]

    logger.info(f"Scanning for connector scripts in: {target_dirs}")
    connector_scripts = find_connector_scripts(target_dirs)
    logger.info(f"Found {len(connector_scripts)} connector scripts.")

    for script in connector_scripts:
        logger.info(f"Executing connector script: {script}")
        try:
            # Execute each connector script
            result = subprocess.run([sys.executable, script], capture_output=True, text=True, check=True)
            logger.info(f"Output from {script}:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"Errors from {script}:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing {script}: {e}")
            logger.error(f"Output:\n{e.stdout}")
            logger.error(f"Errors:\n{e.stderr}")

    logger.info("--- Main Control Script Finished ---")

if __name__ == "__main__":
    main()