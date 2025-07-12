# Hivemind Connector Integration Guide

## Overview

All scripts in this directory have been integrated with the hivemind connector system. This allows them to:
- Run independently (standalone mode)
- Participate in the hivemind collective
- Be controlled remotely via the connector system
- Share status and receive parameters from the hivemind

## Architecture

### Core Components

1. **connector_interface.py** - The interface module that all scripts use to communicate with the hivemind
2. **hivemind_connector.py** - The main connector that manages communication between scripts
3. **connector.py** - A simple connector for basic directory operations

### Integration Pattern

Each script follows this pattern:

```python
from connector_interface import setup_connector, get_hivemind_parameter, send_hivemind_status

def main():
    # Setup connector
    connector = setup_connector("script_name.py")
    
    if connector.is_connected:
        # Register parameters
        connector.register_parameter("param_name", default_value, "description")
        
        # Register callbacks
        connector.register_callback("function_name", callback_function)
    
    # Script logic here (works with or without connector)
    # ...
```

## Integrated Scripts

### 1. auto_installer_refactored.py
- **Purpose**: Automatically installs required Python packages
- **Hivemind Parameters**:
  - `libraries`: List of libraries to install
  - `upgrade`: Whether to upgrade existing libraries
- **Callbacks**:
  - `install`: Install libraries
  - `check`: Check if a library is installed
  - `get_installed`: Get installation status of all libraries

### 2. batch_processor_refactored.py
- **Purpose**: Processes batches of images for classification
- **Hivemind Parameters**:
  - `batch_dir`: Directory containing images
  - `batch_size`: Number of images to process
  - `output_file`: Where to save results
- **Callbacks**:
  - `process`: Process a batch of images
  - `get_stats`: Get current statistics
  - `list_images`: List available images

### 3. config-wizard.py
- **Purpose**: Interactive configuration wizard
- **Status Updates**: Sends configuration progress and completion status

### 4. correlation_analyzer_refactored.py
- **Purpose**: Analyzes image correlations and classifications
- **Hivemind Parameters**:
  - `comparisons`: Number of comparisons for analysis
  - `learning_rate`: Learning rate for weight updates
- **Status Updates**: Loading, analysis, batch processing, weight updates

### 5. demo_system.py
- **Purpose**: Demonstrates the image classification system
- **Hivemind Parameters**:
  - `sample_size`: Number of samples to use
  - `comparisons`: Number of comparisons per analysis

### 6. learning-optimizer.py
- **Purpose**: Optimizes classification weights based on feedback
- **Hivemind Parameters**:
  - `boost_threshold`: Confidence threshold for boosting
  - `reduce_threshold`: Confidence threshold for reduction

### 7. live-monitor.py
- **Purpose**: Live monitoring of image processing
- **Hivemind Parameters**:
  - `sample_points`: Number of sample points
  - `refresh_rate`: Refresh rate in seconds

### 8. main-controller.py
- **Purpose**: Main control interface for the system
- **Status Updates**: System ready, module launches, shutdown

### 9. pixel_sampler_refactored.py
- **Purpose**: Samples pixels from images for database building
- **Hivemind Parameters**:
  - `sample_size`: Number of pixels to sample per image

### 10. self_reviewer_refactored.py
- **Purpose**: Reviews and validates classification results
- **Hivemind Parameters**:
  - `confidence_threshold`: Minimum confidence for acceptance
  - `std_dev_threshold`: Maximum standard deviation allowed

### 11. stats-viewer.py
- **Purpose**: Displays statistics about the classification system
- **Status Updates**: Sends comprehensive statistics to hivemind

## Usage Examples

### Running Scripts Standalone
```bash
python3 script_name.py
```

### Running with Hivemind
1. Start the hivemind connector:
```bash
python3 hivemind_connector.py
```

2. Scripts will automatically connect if the hivemind is running

### Controlling Scripts via Hivemind

Send commands to the hivemind connector to control scripts:

```python
import socket
import json

# Connect to hivemind
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 10314))

# Send command
command = {
    'command': 'control_script',
    'script': 'batch_processor_refactored.py',
    'command': {
        'type': 'set_parameter',
        'parameter': 'batch_size',
        'value': 100
    }
}
sock.send(json.dumps(command).encode())
```

## Benefits

1. **Centralized Control**: Control all scripts from a single point
2. **Parameter Management**: Dynamically adjust parameters without restarting
3. **Status Monitoring**: Real-time visibility into script operations
4. **Collaborative Processing**: Scripts can work together via the hivemind
5. **Backward Compatibility**: Scripts still work independently when needed

## Testing

Run the integration test to verify everything is working:
```bash
python3 test_integration.py
```

This will verify:
- Scripts can run standalone
- Connector interface is functional
- All scripts are properly integrated