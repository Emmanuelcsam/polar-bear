# shared_config.py
# Centralized configuration for real-time monitoring scripts.

# Example configuration parameters:
# You can add more parameters here as needed.
CONFIG = {
    "log_level": "INFO",
    "data_source": "default_camera",
    "processing_enabled": True,
    "threshold_value": 0.75
}

def get_config():
    """Returns the current shared configuration."""
    return CONFIG

def set_config_value(key, value):
    """Sets a specific configuration value."""
    if key in CONFIG:
        CONFIG[key] = value
        return True
    return False

def update_config(new_config_dict):
    """Updates multiple configuration values from a dictionary."""
    for key, value in new_config_dict.items():
        if key in CONFIG:
            CONFIG[key] = value
    return True
