import json
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("neural_connector.config_manager")

class ConfigManager:
    """
    Manages the configuration for the Neural Connector.

    Handles loading, saving, and interactively creating configuration files.
    It automatically creates a default config in non-interactive environments.
    """
    def __init__(self, config_path: str = "neural_connector_config.json"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Loads the configuration from the specified file.

        If the file does not exist, it creates a default one in non-interactive
        environments or launches a wizard in interactive ones.
        """
        if self.config_path.exists():
            logger.info("Loading configuration from %s", self.config_path)
            try:
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
                return self.config
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON from %s. Recreating config.", self.config_path)
                return self._create_or_ask_for_config()
        else:
            logger.warning("Configuration file not found at %s.", self.config_path)
            return self._create_or_ask_for_config()

    def _create_or_ask_for_config(self) -> Dict[str, Any]:
        """Decides whether to run the interactive wizard or create a default config."""
        # sys.stdout.isatty() checks if the script is running in an interactive terminal
        if sys.stdout.isatty():
            logger.info("Interactive terminal detected. Launching configuration wizard.")
            return self.run_configuration_wizard()
        else:
            logger.warning("Non-interactive environment detected. Creating default configuration.")
            return self.create_default_config()

    def create_default_config(self) -> Dict[str, Any]:
        """Creates and saves a default configuration file."""
        self.config = {
            "general": {
                "log_level": "INFO"
            },
            "module_loader": {
                "scripts_to_ignore": ["neural_connector"]
            },
            "tunable_parameters": {}
        }
        self.save_config()
        return self.config

    def save_config(self):
        """Saves the current configuration to the file."""
        logger.info("Saving configuration to %s", self.config_path)
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error("Failed to save configuration file: %s", e)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieves a value from the configuration."""
        return self.config.get(key, default)

    def run_configuration_wizard(self) -> Dict[str, Any]:
        """
        Runs an interactive command-line wizard to generate a new configuration.
        """
        print("\n--- Neural Connector Configuration Wizard ---")
        print("Please answer the following questions to set up the system.")

        print("\n[General Settings]")
        log_level = self._ask_question(
            "Enter the desired logging level (DEBUG, INFO, WARNING, ERROR):",
            default="INFO",
            validator=lambda x: x.upper() in ["DEBUG", "INFO", "WARNING", "ERROR"]
        ).upper()

        print("\n[Module Settings]")
        ignore_list_str = self._ask_question(
            "Enter a comma-separated list of scripts to ignore (e.g., test.py,old_script.py):",
            default="neural_connector"
        )
        scripts_to_ignore = [item.strip() for item in ignore_list_str.split(',') if item.strip()]

        self.config = {
            "general": {"log_level": log_level},
            "module_loader": {"scripts_to_ignore": scripts_to_ignore},
            "tunable_parameters": {}
        }

        print("\n--- Configuration Preview ---")
        print(json.dumps(self.config, indent=4))
        print("-" * 29)

        if self._ask_question("Save this configuration? (yes/no):", default="yes").lower() == "yes":
            self.save_config()
            print("Configuration saved successfully.")
        else:
            print("Configuration not saved. Using temporary settings for this session.")
            
        return self.config

    def _ask_question(self, question: str, default: Optional[str] = None, validator: Optional[callable] = None) -> str:
        """Helper function to ask a question and get a validated response."""
        while True:
            prompt = f"{question} "
            if default:
                prompt += f"[{default}] "
            
            response = input(prompt).strip()
            if not response and default is not None:
                response = default
            
            if validator:
                if validator(response):
                    return response
                else:
                    print("Invalid input. Please try again.")
            else:
                return response

if __name__ == '__main__':
    from logging_config import setup_logging
    setup_logging()
    
    config_manager = ConfigManager("test_config.json")
    config = config_manager.load_config()
    
    print("\n--- Loaded Configuration ---")
    print(json.dumps(config, indent=4))
    
    if Path("test_config.json").exists():
        Path("test_config.json").unlink()