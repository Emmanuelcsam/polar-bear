# Analysis/Reporting Module Connector API Documentation

## Overview

The Analysis/Reporting module now features fully integrated connectors that provide comprehensive control over all scripts in the directory. The system consists of:

1. **connector.py** - Enhanced connector with script management capabilities
2. **hivemind_connector.py** - Hivemind-compatible connector with parameter control and monitoring
3. **script_interface.py** - Core interface for script discovery and management
4. **script_wrappers.py** - Wrapper functions for controlled script execution

## Architecture

```
┌─────────────────────┐     ┌────────────────────┐
│  External Client    │────▶│   connector.py     │
└─────────────────────┘     │   (Port: 12000)    │
                            └────────┬───────────┘
                                     │
┌─────────────────────┐              ▼
│  Hivemind System    │     ┌────────────────────┐
│                     │────▶│ hivemind_connector │
└─────────────────────┘     │   (Port: 10109)    │
                            └────────┬───────────┘
                                     │
                                     ▼
                            ┌────────────────────┐
                            │  script_interface  │
                            │   (ScriptManager)  │
                            └────────┬───────────┘
                                     │
                                     ▼
                            ┌────────────────────┐
                            │  script_wrappers   │
                            └────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │  Analysis/Reporting Scripts │
                        └────────────────────────────┘
```

## Connector Commands

### Basic Commands (Both Connectors)

#### 1. Status
Get connector status and system information.

```json
Request:
{
    "command": "status"
}

Response:
{
    "status": "active",
    "port": 12000,
    "directory": "/path/to/analysis-reporting",
    "scripts": {
        "total": 25,
        "running": 1,
        "completed": 10,
        "failed": 0
    }
}
```

#### 2. List Files
List all files in the module directory.

```json
Request:
{
    "command": "list_files"
}

Response:
{
    "files": [
        {"name": "analysis_engine.py", "type": "file", "size": 15234},
        {"name": "config.json", "type": "file", "size": 2048}
    ]
}
```

### Script Management Commands

#### 3. List Scripts
Get information about all available scripts.

```json
Request:
{
    "command": "list_scripts"
}

Response:
{
    "scripts": [
        {
            "name": "analysis_engine.py",
            "path": "/path/to/analysis_engine.py",
            "module_name": "analysis_engine",
            "main_function": "run_full_pipeline",
            "description": "Main analysis orchestration engine",
            "status": "idle"
        }
    ]
}
```

#### 4. Get Script Info
Get detailed information about a specific script.

```json
Request:
{
    "command": "get_script_info",
    "script": "defect-analyzer.py"
}

Response:
{
    "name": "defect-analyzer.py",
    "main_function": "analyze_defects",
    "description": "Analyzes defects in refined masks",
    "parameters": {
        "refined_masks": {"type": "dict", "required": true},
        "pixels_per_micron": {"type": "float", "default": 0.5}
    }
}
```

#### 5. Execute Script
Execute a script with parameters.

```json
Request:
{
    "command": "execute_script",
    "script": "quality-metrics-calculator.py",
    "parameters": {
        "image": "path/to/image.png",
        "save_report": true
    },
    "async": false
}

Response:
{
    "status": "executed",
    "success": true,
    "result": {
        "quality_score": 0.92,
        "metrics": {...}
    }
}
```

### Parameter Control Commands

#### 6. Get Parameter
Get current value of a script parameter.

```json
Request:
{
    "command": "get_parameter",
    "script": "defect-analyzer.py",
    "parameter": "pixels_per_micron"
}

Response:
{
    "parameter": "pixels_per_micron",
    "value": 0.5
}
```

#### 7. Update Parameter
Update a script parameter value.

```json
Request:
{
    "command": "update_parameter",
    "script": "defect-analyzer.py",
    "parameter": "pixels_per_micron",
    "value": 0.6
}

Response:
{
    "status": "updated",
    "parameter": "pixels_per_micron"
}
```

### Configuration Commands

#### 8. Read Config
Read the current configuration file.

```json
Request:
{
    "command": "read_config"
}

Response:
{
    "config": {
        "shared_parameters": {
            "pixels_per_micron": 0.5,
            "min_defect_area_px": 5
        },
        ...
    }
}
```

#### 9. Update Config
Update configuration values.

```json
Request:
{
    "command": "update_config",
    "config": {
        "shared_parameters": {
            "pixels_per_micron": 0.6
        }
    }
}

Response:
{
    "status": "config_updated"
}
```

### Hivemind-Specific Commands

#### 10. List Parameters
List all parameters for a script (Hivemind only).

```json
Request:
{
    "command": "list_parameters",
    "script": "report-generator.py"
}

Response:
{
    "script": "report-generator.py",
    "parameters": {
        "image": {"type": "numpy.ndarray", "required": true},
        "output_path": {"type": "str", "required": true}
    },
    "overrides": {}
}
```

#### 11. Get History
Get execution history (Hivemind only).

```json
Request:
{
    "command": "get_history",
    "limit": 5
}

Response:
{
    "history": [
        {
            "script": "analysis_engine.py",
            "timestamp": 1234567890,
            "status": "completed",
            "parameters": {...}
        }
    ],
    "total_executions": 42
}
```

#### 12. Monitor
Get comprehensive monitoring data (Hivemind only).

```json
Request:
{
    "command": "monitor"
}

Response:
{
    "connector_id": "499cbf4c",
    "uptime": 3600.5,
    "scripts_available": 25,
    "execution_stats": {
        "total": 42,
        "successful": 40,
        "failed": 2
    },
    "script_states": {
        "idle": 24,
        "running": 1,
        "completed": 0,
        "failed": 0
    }
}
```

## Script Parameters Reference

### analysis_engine.py
- `image_path` (str, required): Path to input image
- `output_dir` (str): Output directory (default: ".")
- `config` (dict): Configuration overrides
- `verbose` (bool): Verbose output (default: False)

### defect-analyzer.py
- `refined_masks` (dict, required): Dictionary of refined masks
- `pixels_per_micron` (float): Calibration factor (default: 0.5)

### report-generator.py
- `image` (numpy.ndarray, required): Input image (BGR)
- `results` (dict, required): Analysis results
- `localization` (dict, required): Fiber localization data
- `zone_masks` (dict, required): Zone segmentation masks
- `output_path` (str, required): Output image path

### quality-metrics-calculator.py
- `image` (numpy.ndarray, required): Grayscale image
- `defect_mask` (numpy.ndarray): Binary defect mask
- `roi_mask` (numpy.ndarray): Region of interest mask

## Usage Examples

### Python Client Example

```python
import socket
import json

def send_command(command, host='localhost', port=12000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.send(json.dumps(command).encode())
    response = sock.recv(4096).decode()
    sock.close()
    return json.loads(response)

# Get status
status = send_command({"command": "status"})
print(f"Connector status: {status['status']}")

# List scripts
scripts = send_command({"command": "list_scripts"})
print(f"Available scripts: {len(scripts['scripts'])}")

# Execute a script
result = send_command({
    "command": "execute_script",
    "script": "quality-metrics-calculator.py",
    "parameters": {
        "image": "sample.png"
    }
})
print(f"Execution result: {result}")
```

### Running the Connectors

1. **Start the enhanced connector:**
   ```bash
   python connector.py [port]
   # Default port: 12000
   ```

2. **Start the hivemind connector:**
   ```bash
   python hivemind_connector.py
   # Default port: 10109
   ```

3. **Test the integration:**
   ```bash
   python test_integration.py
   ```

## Features

1. **Full Script Control**: Both connectors can control all scripts in the directory
2. **Parameter Management**: Dynamic parameter updates without script modification
3. **Execution Monitoring**: Track script execution status and history
4. **Configuration Persistence**: Changes are saved to config.json
5. **Error Handling**: Comprehensive error reporting and recovery
6. **Async/Sync Execution**: Support for both execution modes
7. **Script Discovery**: Automatic detection of new scripts
8. **Wrapper Support**: Enhanced control through script wrappers

## Security Considerations

1. Connectors only accept connections from localhost by default
2. No authentication is implemented - add if needed for production
3. Scripts run with the same permissions as the connector process
4. Parameter validation should be implemented for production use

## Troubleshooting

1. **Connection refused**: Ensure the connector is running on the correct port
2. **Script not found**: Check that the script exists in the directory
3. **Parameter errors**: Verify parameter types match script requirements
4. **Execution timeouts**: Default timeout is 30 seconds, adjust if needed

## Future Enhancements

1. Add authentication and authorization
2. Implement WebSocket support for real-time updates
3. Add script scheduling capabilities
4. Implement distributed execution support
5. Add comprehensive logging and audit trails