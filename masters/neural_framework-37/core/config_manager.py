import json
import os
from .logger import log

class ConfigManager:
    def __init__(self, config_name="config.json"):
        self.config_path = os.path.join("neural_framework/config", config_name)
        self.config = {}
        self.load_config()

    def load_config(self):
        """Loads the configuration from a JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                log.info(f"Configuration loaded from {self.config_path}")
            else:
                log.info("No configuration file found. A new one will be created.")
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        except Exception as e:
            log.error(f"Could not load configuration: {e}")

    def get(self, key, default=None, prompt=""):
        """
        Gets a value from the configuration.
        If the key is not found, it prompts the user for the value.
        """
        value = self.config.get(key)
        if value is not None:
            return value
        
        if prompt:
            try:
                value = input(prompt)
                self.config[key] = value
                self.save_config()
                return value
            except EOFError:
                log.warning("EOFError received, cannot prompt for input. Using default.")
                return default

        return default

    def set(self, key, value):
        """Sets a configuration value and saves it."""
        self.config[key] = value
        self.save_config()

    def save_config(self):
        """Saves the configuration to a JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4, sort_keys=True)
            log.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            log.error(f"Could not save configuration: {e}")

if __name__ == '__main__':
    # Example Usage
    log.info("Running ConfigManager example...")
    config = ConfigManager("test_config.json")
    
    # Get a value that doesn't exist
    user_name = config.get("user_name", prompt="What is your name? ")
    log.info(f"Hello, {user_name}")

    # Get a value with a default
    favorite_color = config.get("favorite_color", default="blue")
    log.info(f"Favorite color is {favorite_color}")

    # Set a value
    config.set("project_name", "NeuralFramework")
    
    # Verify it was saved
    reloaded_config = ConfigManager("test_config.json")
    log.info(f"Reloaded project name: {reloaded_config.get('project_name')}")

    # Clean up the dummy config file
    try:
        os.remove(os.path.join("neural_framework/config", "test_config.json"))
        log.info("Cleaned up test config file.")
    except OSError as e:
        log.error(f"Error removing test config file: {e}")
