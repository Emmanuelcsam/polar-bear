# Lab Framework Integration Summary

## Overview
This lab framework has been fully integrated with bidirectional communication capabilities between scripts and connectors. All scripts maintain their ability to run independently while also supporting full control and monitoring through the connector system.

## Key Features

### 1. Bidirectional Communication
- **ScriptConnectorInterface**: Base class that enables scripts to expose parameters, variables, and methods to connectors
- **ConnectorClient**: Allows connectors to control scripts, set parameters, read variables, and call methods
- **Real-time Updates**: Scripts can notify connectors of variable changes

### 2. Enhanced Connectors
- **connector.py**: Main connector with module discovery, control, and workflow execution capabilities
- **hivemind_connector.py**: Network-based connector supporting script registration and remote control
- **connector_interface.py**: Core infrastructure for bidirectional communication

### 3. Script Independence
All scripts can run independently without requiring connectors:
- Direct import and function calls work normally
- Legacy interfaces maintained for backward compatibility
- No external dependencies required for basic operation

### 4. Full Integration Capabilities
Connectors can:
- Discover all available modules and scripts
- Read script information and capabilities
- Set parameters with callback support
- Monitor variable values in real-time
- Execute methods exposed by scripts
- Run complex workflows across multiple modules

## Updated Scripts

### Core Infrastructure
- `core/connector_interface.py`: New bidirectional communication interface
- `connector.py`: Enhanced with LabFrameworkConnector class
- `hivemind_connector.py`: Added script registration and control

### Module Updates
- `modules/random_pixel.py`: Fully integrated with connector interface
- `modules/anomaly_detector.py`: Fixed imports for independent operation
- `modules/batch_processor.py`: Fixed imports for independent operation
- `modules/realtime_processor.py`: Fixed imports for independent operation
- `modules/hpc.py`: Fixed imports for independent operation

### Testing and Troubleshooting
- `test_integration.py`: Comprehensive integration tests
- `troubleshoot_all.py`: Automatic diagnostic and repair tool

## Usage Examples

### 1. Independent Script Usage
```python
from modules import random_pixel
img = random_pixel.gen()  # Works without connectors
```

### 2. Connector Control
```python
from connector import LabFrameworkConnector
connector = LabFrameworkConnector()
connector.control_module('random_pixel', 'gen')
```

### 3. Workflow Execution
```python
workflow = [
    {'module': 'random_pixel', 'action': 'gen'},
    {'module': 'cv_module', 'action': 'batch', 'params': {'folder': 'data'}},
    {'module': 'intensity_reader', 'action': 'learn'}
]
results = connector.execute_workflow(workflow)
```

### 4. Parameter Control (when connector interface is active)
```python
from core.connector_interface import ConnectorClient
client = ConnectorClient()
client.set_script_parameter('random_pixel', 'size', 64)
client.set_script_parameter('random_pixel', 'color_mode', 'rgb')
```

## Testing

Run the following commands to verify the integration:

1. **Basic functionality test**: `python test_basic.py`
2. **Full integration test**: `python test_integration.py`
3. **Troubleshooting**: `python troubleshoot_all.py`

## Requirements

- Python 3.6+
- numpy
- opencv-python
- scikit-learn
- torch (optional, for HPC features)

## Architecture

```
iteration6-lab-framework/
├── connector.py              # Enhanced main connector
├── hivemind_connector.py     # Network-based connector
├── core/
│   ├── connector_interface.py # Bidirectional communication
│   ├── config.py
│   ├── datastore.py
│   └── logger.py
├── modules/
│   ├── random_pixel.py       # Fully integrated example
│   ├── cv_module.py
│   ├── intensity_reader.py
│   └── ... (other modules)
└── tests/
    ├── test_integration.py
    └── troubleshoot_all.py
```

## Status

✅ All systems fully integrated and operational
- Scripts work independently
- Connectors can control all scripts
- Bidirectional communication established
- All tests passing

The system is ready for both standalone operation and full connector-based control.