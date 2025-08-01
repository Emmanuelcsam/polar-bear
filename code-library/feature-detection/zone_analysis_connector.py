

import os
import sys
import logging
import subprocess
import json

# --- Configuration ---
LOG_FILE = "zone_analysis.log"
CONFIG_FILE = "zone_analysis_config.json"
REQUIREMENTS = ["numpy", "opencv-python", "scikit-image", "scipy", "matplotlib"]

# --- Logger Setup ---
def setup_logger():
    """Sets up a logger that outputs to both console and a file."""
    logger = logging.getLogger("ZoneAnalysisConnector")
    logger.setLevel(logging.DEBUG)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

# --- Configuration Management ---
class ConfigurationManager:
    """Handles loading, saving, and editing of configuration parameters."""
    def __init__(self, config_file):
        self.config_file = Path(config_file)
        self.config = self.get_default_config()
        self.load_config()

    def get_default_config():
        """Defines the default tunable parameters for all modules."""
        return {
            "robust-mask-generator": {
                "cladding_core_ratio": 13.88,  # 125 / 9
                "ferrule_buffer_ratio": 1.2,
                "hough_circles_params": {
                    "param1": 50,
                    "param2": 30,
                    "dp": 1.0
                }
            },
            "fiber-zone-locator": {
                "hough_circles_param1": 120,
                "hough_circles_param2": 50
            },
            "zone-mask-creator-v3": {
                "um_per_px": 0.7,
                "zones_def": {
                    "core": {"r_min": 0, "r_max": 30},
                    "cladding": {"r_min": 30, "r_max": 62.5},
                    "ferrule": {"r_min": 62.5, "r_max": 125}
                }
            }
        }

    def load_config(self):
        """Loads the configuration from the JSON file, or creates it if it doesn't exist."""
        if self.config_file.exists():
            logger.info(f"Loading configuration from {self.config_file}")
            with open(self.config_file, 'r') as f:
                try:
                    loaded_config = json.load(f)
                    # Update default config with loaded values, keeping defaults for new keys
                    for key, value in loaded_config.items():
                        if key in self.config:
                            self.config[key].update(value)
                        else:
                            self.config[key] = value
                except json.JSONDecodeError:
                    logger.error(f"Could not decode {self.config_file}. Using default configuration.")
        else:
            logger.info("No config file found. Creating one with default values.")
            self.save_config()

    def save_config(self):
        """Saves the current configuration to the JSON file."""
        logger.info(f"Saving configuration to {self.config_file}")
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get(self, key, default=None):
        """Gets a configuration value."""
        return self.config.get(key, default)

def show_config_menu(config_manager):
    """Displays a menu to view and edit configuration."""
    logger.info("Displaying configuration menu.")
    
    while True:
        print("\n--- Configuration Menu ---")
        print("Current Configuration:")
        print(json.dumps(config_manager.config, indent=4))
        
        print("\n[1] Edit a parameter")
        print("[2] Restore defaults")
        print("[3] Back to Main Menu")
        
        choice = input("Select an option: ")
        
        if choice == '1':
            logger.info("User chose to edit a parameter.")
            module_key = input("Enter the module key to edit (e.g., 'robust-mask-generator'): ")
            if module_key in config_manager.config:
                param_key = input(f"Enter the parameter key in '{module_key}' to edit: ")
                if param_key in config_manager.config[module_key]:
                    new_value = input(f"Enter the new value for '{param_key}': ")
                    try:
                        # Attempt to convert to original type (int, float, or dict)
                        original_value = config_manager.config[module_key][param_key]
                        if isinstance(original_value, (int, float)):
                            new_value = type(original_value)(new_value)
                        elif isinstance(original_value, dict):
                            new_value = json.loads(new_value) # For editing complex dicts
                        config_manager.config[module_key][param_key] = new_value
                        config_manager.save_config()
                        logger.info(f"Updated '{module_key}.{param_key}' to '{new_value}'.")
                    except (ValueError, json.JSONDecodeError) as e:
                        logger.error(f"Invalid value or format: {e}")
                else:
                    logger.warning(f"Parameter key '{param_key}' not found in '{module_key}'.")
            else:
                logger.warning(f"Module key '{module_key}' not found in configuration.")
        
        elif choice == '2':
            logger.info("User chose to restore default configuration.")
            confirm = input("Are you sure you want to restore all settings to their defaults? (y/n): ").lower()
            if confirm == 'y':
                config_manager.config = config_manager.get_default_config()
                config_manager.save_config()
                logger.info("Configuration restored to defaults.")
            else:
                logger.info("Restore operation cancelled.")

        elif choice == '3':
            logger.info("Returning to main menu.")
            break
        else:
            logger.warning(f"Invalid option '{choice}' selected.")

# --- Dependency Management ---
def check_and_install_dependencies():
    """Checks for required Python packages and prompts the user to install them if missing."""
    logger.info("Checking for required dependencies...")
    missing_packages = []
    for package_name in REQUIREMENTS:
        try:
            # Use a more reliable check than just importlib.util.find_spec
            # This handles cases where the import name differs from the package name (e.g., opencv-python -> cv2)
            if package_name == "opencv-python":
                importlib.import_module("cv2")
            elif package_name == "scikit-image":
                importlib.import_module("skimage")
            else:
                 importlib.import_module(package_name)
            logger.debug(f"'{package_name}' is already installed.")
        except ImportError:
            missing_packages.append(package_name)

    if not missing_packages:
        logger.info("All required dependencies are satisfied.")
        return True

    logger.warning(f"The following required packages are missing: {', '.join(missing_packages)}")
    
    try:
        prompt = input(f"Would you like to install them now? (y/n): ").lower()
        if prompt == 'y':
            logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
            for package in missing_packages:
                try:
                    # Using subprocess to install the package
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logger.info(f"Successfully installed '{package}'.")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install '{package}'. Please install it manually using 'pip install {package}'.")
                    logger.error(f"Error: {e}")
                    return False
            logger.info("All missing dependencies have been installed.")
            return True
        else:
            logger.error("Dependencies not installed. The application may not function correctly.")
            return False
    except KeyboardInterrupt:
        logger.error("\nInstallation cancelled by user.")
        return False

# --- Module Loading ---
def load_modules():
    """Dynamically loads all .py files in the current directory as modules."""
    loaded_modules = {}
    current_dir = Path(__file__).parent
    for file_path in current_dir.glob("*.py"):
        module_name = file_path.stem
        if module_name == "zone_analysis_connector":
            continue  # Skip loading self

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded_modules[module_name] = module
            logger.debug(f"Successfully loaded module: {module_name}")
        except Exception as e:
            logger.error(f"Failed to load module '{module_name}': {e}")
    
    logger.info(f"Loaded {len(loaded_modules)} modules: {', '.join(loaded_modules.keys())}")
    return loaded_modules

# --- Tool Execution ---
def run_tool(module, config_manager):
    """
    Executes the main function of a selected module, passing configuration if possible.
    """
    logger.info(f"Attempting to run module: '{module.__name__}'")
    
    # Check for module-specific configuration
    module_config = config_manager.get(module.__name__)

    # Example of passing config to a tool that supports it
    if module.__name__ == "robust-mask-generator" and hasattr(module, 'RobustMaskGenerator'):
        logger.info("Running 'robust-mask-generator' with custom configuration.")
        try:
            # Instantiate with config values
            generator = module.RobustMaskGenerator(
                cladding_core_ratio=module_config.get('cladding_core_ratio', 13.88),
                ferrule_buffer_ratio=module_config.get('ferrule_buffer_ratio', 1.2)
            )
            # The demo function would need to be adapted to use this instance
            # For now, we just call the demo as is.
            module.demo_robust_mask_generation()
        except Exception as e:
            logger.error(f"An error occurred while running '{module.__name__}': {e}")
        return

    # Fallback to generic main/demo execution
    if hasattr(module, 'main'):
        logger.info(f"Executing main function of '{module.__name__}'...")
        try:
            module.main()
            logger.info(f"Finished executing '{module.__name__}'.")
        except Exception as e:
            logger.error(f"An error occurred while running '{module.__name__}': {e}")
    elif hasattr(module, 'demo_robust_mask_generation'):
        logger.info(f"Executing demo function of '{module.__name__}'...")
        try:
            module.demo_robust_mask_generation()
            logger.info(f"Finished executing demo for '{module.__name__}'.")
        except Exception as e:
            logger.error(f"An error occurred while running demo for '{module.__name__}': {e}")
    else:
        logger.warning(f"Module '{module.__name__}' has no 'main' or 'demo' function to run.")
        print(f"Sorry, I don't know how to run '{module.__name__}' automatically yet.")

def show_tool_menu(modules, config_manager):
    """Displays a menu of available tools and handles user selection."""
    logger.info("Displaying tool menu.")
    
    tool_names = sorted(modules.keys())
    
    while True:
        print("\n--- Available Tools ---")
        for i, name in enumerate(tool_names, 1):
            print(f"[{i}] {name}")
        print(f"[{len(tool_names) + 1}] Back to Main Menu")
        
        try:
            choice = int(input("Select a tool to run: "))
            if 1 <= choice <= len(tool_names):
                selected_tool_name = tool_names[choice - 1]
                logger.info(f"User selected tool: '{selected_tool_name}'")
                run_tool(modules[selected_tool_name], config_manager)
                input("Press Enter to continue...")
                break
            elif choice == len(tool_names) + 1:
                logger.info("Returning to main menu.")
                break
            else:
                logger.warning(f"Invalid tool selection: {choice}")
        except ValueError:
            logger.warning("Invalid input, please enter a number.")

# --- Main Application ---
def main_menu(modules, config_manager):
    """Displays the main menu and handles user interaction."""
    logger.info("Initializing Zone Analysis Connector...")
    
    print("\n" + "="*50)
    print("    Zone Analysis Neural Network Connector")
    print("="*50)
    
    while True:
        print("\n[1] Run a tool from the zone analysis suite")
        print("[2] Configure parameters")
        print("[3] Exit")
        
        choice = input("Please select an option: ")
        
        if choice == '1':
            logger.info("User selected 'Run a tool'.")
            show_tool_menu(modules, config_manager)
        elif choice == '2':
            logger.info("User selected 'Configure parameters'.")
            show_config_menu(config_manager)
        elif choice == '3':
            logger.info("Exiting application.")
            break
        else:
            logger.warning(f"Invalid option '{choice}' selected.")
            print("Invalid choice, please try again.")

if __name__ == '__main__':
    if check_and_install_dependencies():
        config_manager = ConfigurationManager(CONFIG_FILE)
        modules = load_modules()
        main_menu(modules, config_manager)
    else:
        logger.error("Could not satisfy dependencies. Exiting.")
        sys.exit(1)

