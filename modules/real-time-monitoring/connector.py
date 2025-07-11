
import os
import logging
import sys
import shared_config # Import the shared configuration module

# Import all scripts that need to be controlled
import live_fiber_analyzer
import live_monitoring_dashboard
import live_video_processor
import location_tracking_pipeline
import realtime_monitor
import run_calibration
import run_circle_detector
import run_geometry_demo_fixed
import run_geometry_demo
import src.applications.example_application as example_application
import src.applications.realtime_circle_detector as realtime_circle_detector_src
import src.core.integrated_geometry_system as integrated_geometry_system
import src.core.python313_fix as python313_fix
import src.tools.performance_benchmark_tool as performance_benchmark_tool
import src.tools.realtime_calibration_tool as realtime_calibration_tool_src
import src.tools.setup_installer as setup_installer
import src.tools.uv_compatible_setup as uv_compatible_setup

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


# Dictionary to hold references to all controllable scripts
controllable_scripts = {
    "live_fiber_analyzer": live_fiber_analyzer,
    "live_monitoring_dashboard": live_monitoring_dashboard,
    "live_video_processor": live_video_processor,
    "location_tracking_pipeline": location_tracking_pipeline,
    "realtime_monitor": realtime_monitor,
    "run_calibration": run_calibration,
    "run_circle_detector": run_circle_detector,
    "run_geometry_demo_fixed": run_geometry_demo_fixed,
    "run_geometry_demo": run_geometry_demo,
    "example_application": example_application,
    "realtime_circle_detector_src": realtime_circle_detector_src,
    "integrated_geometry_system": integrated_geometry_system,
    "python313_fix": python313_fix,
    "performance_benchmark_tool": performance_benchmark_tool,
    "realtime_calibration_tool_src": realtime_calibration_tool_src,
    "setup_installer": setup_installer,
    "uv_compatible_setup": uv_compatible_setup,
}

def get_all_script_info():
    """
    Retrieves information from all integrated scripts.
    Returns a dictionary where keys are script names and values are their info.
    """
    all_info = {}
    for name, module in controllable_scripts.items():
        try:
            if hasattr(module, 'get_script_info') and callable(module.get_script_info):
                info = module.get_script_info()
                all_info[name] = info
            else:
                all_info[name] = {"status": "not_controllable", "message": "Module does not expose get_script_info"}
        except Exception as e:
            logger.error(f"Error getting info from {name}: {e}")
            all_info[name] = {"status": "error", "message": str(e)}
    return all_info

def set_script_parameter_for_module(module_name: str, key: str, value: any):
    """
    Sets a parameter for a specific script.
    :param module_name: The name of the module (key in controllable_scripts).
    :param key: The parameter key to set.
    :param value: The new value for the parameter.
    :return: True if successful, False otherwise.
    """
    if module_name in controllable_scripts:
        module = controllable_scripts[module_name]
        try:
            if hasattr(module, 'set_script_parameter') and callable(module.set_script_parameter):
                success = module.set_script_parameter(key, value)
                if success:
                    logger.info(f"Successfully set parameter '{key}' to '{value}' for '{module_name}'")
                else:
                    logger.warning(f"Failed to set parameter '{key}' for '{module_name}'. Parameter might not be supported or value is invalid.")
                return success
            else:
                logger.warning(f"Module '{module_name}' does not expose set_script_parameter.")
                return False
        except Exception as e:
            logger.error(f"Error setting parameter for {module_name}: {e}")
            return False
    else:
        logger.error(f"Module '{module_name}' not found in controllable_scripts.")
        return False

def main():
    """Main function for the connector script."""
    logger.info(f"--- Connector Script Initialized in {os.getcwd()} ---")
    logger.info(f"This script is responsible for connecting the modules in this directory to the main control script.")
    
    # Example usage: Get info from all scripts
    logger.info("\n--- Current Status of All Scripts ---")
    all_info = get_all_script_info()
    for script_name, info in all_info.items():
        logger.info(f"Script: {script_name}")
        logger.info(f"  Status: {info.get('status', 'N/A')}")
        if 'parameters' in info:
            logger.info(f"  Parameters: {info['parameters']}")
        if 'message' in info:
            logger.info(f"  Message: {info['message']}")
        logger.info("-" * 20)

    # Example usage: Set a parameter for a specific script
    # This assumes 'live_fiber_analyzer' has a 'min_frame_interval' parameter
    # and 'shared_config' has 'log_level'
    logger.info("\n--- Attempting to change shared_config log_level to DEBUG ---")
    set_script_parameter_for_module("live_fiber_analyzer", "log_level", "DEBUG")
    
    logger.info("\n--- Re-checking status after parameter change ---")
    all_info_after_change = get_all_script_info()
    if "live_fiber_analyzer" in all_info_after_change:
        logger.info(f"Live Fiber Analyzer log_level after change: {all_info_after_change['live_fiber_analyzer']['parameters'].get('log_level')}")

    logger.info("\nConnector script finished.")

if __name__ == "__main__":
    main()
