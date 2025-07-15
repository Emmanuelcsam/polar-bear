
import json
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "paths": {
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "model_weights": "models",
        "dataset_output": "dataset"
    },
    "training_parameters": {
        "default_epochs": 20,
        "default_batch_size": 32,
        "default_learning_rate": 1e-3
    },
    "segmentation_settings": {
        "n_classes": 4,
        "default_weights": "segmenter_best.pth"
    },
    "anomaly_settings": {
        "reconstruction_loss": "l2",
        "default_weights": "cae_last.pth"
    }
}

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = {}

    def load_config(self) -> dict:
        """
        Loads configuration from the file. If the file doesn't exist,
        it runs the interactive setup.
        """
        if self.config_path.exists():
            logger.info(f"Loading configuration from {self.config_path}")
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            logger.warning(f"Configuration file not found at {self.config_path}.")
            self.interactive_setup()
        
        return self.config

    def interactive_setup(self):
        """
        Guides the user through an interactive setup process to create the
        configuration file.
        """
        print("--- Configuration Setup ---")
        print("Please provide the following configuration details.")
        print("Press Enter to accept the default value in brackets.")

        config = DEFAULT_CONFIG.copy()

        try:
            for section, settings in DEFAULT_CONFIG.items():
                if isinstance(settings, dict):
                    for key, value in settings.items():
                        prompt = f"Enter value for '{key}' in section '{section}' [{value}]: "
                        user_input = input(prompt).strip()
                        if user_input:
                            # Attempt to cast to the original type (int, float)
                            if isinstance(value, int):
                                config[section][key] = int(user_input)
                            elif isinstance(value, float):
                                config[section][key] = float(user_input)
                            else:
                                config[section][key] = user_input
        except (EOFError, KeyboardInterrupt):
            print("\n\nConfiguration setup aborted by user.")
            sys.exit(0)


        self.config = config
        self.save_config()

    def save_config(self):
        """
        Saves the current configuration to the file.
        """
        logger.info(f"Saving configuration to {self.config_path}")
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved successfully to {self.config_path}")

    def get(self, key_path: str, default=None):
        """
        Retrieves a value from the config using a dot-separated path.
        e.g., "paths.raw_data"
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            logger.warning(f"Configuration key '{key_path}' not found.")
            return default

if __name__ == '__main__':
    # This is for standalone testing of the config manager
    from logging_manager import setup_logging
    setup_logging()
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    print("\n--- Current Configuration ---")
    print(json.dumps(config, indent=4))
    
    print(f"\nExample of getting a specific value:")
    print(f"Raw data path: {config_manager.get('paths.raw_data')}")
