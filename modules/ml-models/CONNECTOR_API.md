# ML Models Connector System API Documentation

## Overview

The ML Models Connector System provides a unified interface for controlling, monitoring, and enabling collaboration between machine learning scripts. All scripts in this module can now be controlled through the connector system, allowing for parameter adjustments, real-time monitoring, and inter-script communication.

## Architecture

### Components

1. **Script Interface (`script_interface.py`)**
   - Base class for all connector-enabled scripts
   - Provides parameter registration and control
   - Handles variable monitoring
   - Enables script-to-script communication

2. **Enhanced Connector (`connector.py`)**
   - Interactive control interface
   - Script execution management
   - Parameter control
   - Results monitoring

3. **Hivemind Connector (`hivemind_connector.py`)**
   - Distributed system integration
   - Parent-child connector communication
   - Script discovery and registration
   - Collaboration request handling

4. **Script Wrappers (`script_wrappers.py`)**
   - Wrapper classes for legacy scripts
   - Automatic parameter discovery
   - Connector integration for non-modified scripts

## Script Integration

### Method 1: Direct Integration (Recommended)

Modify your script to inherit from `ScriptInterface`:

```python
from script_interface import ScriptInterface, ConnectorClient

class MyMLScript(ScriptInterface):
    def __init__(self):
        super().__init__("my_script_name", "Description of my script")
        
        # Register parameters
        self.register_parameter("learning_rate", 0.001, [0.0001, 0.001, 0.01])
        self.register_parameter("epochs", 10, range(1, 101))
        
        # Register variables
        self.register_variable("current_loss", 0.0)
        self.register_variable("accuracy", 0.0)
        
        # Initialize connector client
        self.client = ConnectorClient(self)
        
    def run(self):
        """Main execution method"""
        # Your ML code here
        # Update variables during execution
        self.set_variable("current_loss", loss_value)
        self.update_results("final_accuracy", accuracy)

# Main execution
if __name__ == "__main__":
    script = MyMLScript()
    if "--with-connector" in sys.argv:
        script.run_with_connector()
    else:
        script.run()
```

### Method 2: Using Wrappers

For scripts that cannot be modified:

```python
from script_wrappers import create_wrapper_for_script
from pathlib import Path

# Create wrapper for existing script
wrapper = create_wrapper_for_script(Path("legacy_script.py"))
wrapper.run_with_connector()
```

## Connector Commands

### Status Commands

- **`status`**: Get connector status and statistics
- **`scan`**: Rescan directory for scripts
- **`troubleshoot`**: Run diagnostics

### Script Control

- **`execute`**: Execute a script
  ```json
  {
    "command": "execute",
    "script": "script_name",
    "with_connector": true
  }
  ```

- **`stop`**: Stop a running script
  ```json
  {
    "command": "stop",
    "script": "script_name"
  }
  ```

- **`control`**: Send control command to script
  ```json
  {
    "command": "control",
    "script": "script_name",
    "control_command": {
      "command": "set_parameter",
      "parameter": "learning_rate",
      "value": 0.01
    }
  }
  ```

### Information Queries

- **`get_scripts_info`**: Get detailed information about all scripts
- **`list_scripts`**: Get list of available scripts

## Script-to-Script Communication

### Broadcasting Data

```python
# In your script
self.client.broadcast_data({
    "type": "results_update",
    "data": {
        "accuracy": 0.95,
        "loss": 0.05
    }
})
```

### Requesting Collaboration

```python
# Request specific collaboration
response = self.client.request_collaboration("target_script", {
    "type": "parameter_sync",
    "parameters": {
        "learning_rate": self.get_parameter("learning_rate")
    }
})
```

### Receiving Collaboration Requests

```python
# Check for incoming requests
response = self.send_to_connector({
    "command": "get_collaborations",
    "script_name": self.script_name
})

for request in response.get("requests", []):
    source = request["source"]
    data = request["data"]
    # Process collaboration data
```

## Parameter Control

### Registering Parameters

```python
# Basic registration
self.register_parameter("name", default_value, valid_values)

# Examples
self.register_parameter("threshold", 3.0, [2.0, 2.5, 3.0, 3.5, 4.0])
self.register_parameter("method", "auto", ["auto", "manual", "hybrid"])
self.register_parameter("iterations", 100, range(10, 1001))
```

### Parameter Callbacks

```python
def on_threshold_change(self, new_value):
    self.logger.info(f"Threshold changed to {new_value}")
    # Adjust algorithm based on new threshold
    
self.register_callback("threshold", on_threshold_change)
```

## Variable Monitoring

### Registering Variables

```python
# Read-write variable
self.register_variable("current_epoch", 0)

# Read-only variable
self.register_variable("model_accuracy", 0.0, read_only=True)
```

### Updating Variables

```python
# Update during execution
self.set_variable("current_epoch", epoch)
self.set_variable("model_accuracy", accuracy)
```

## Results Management

### Updating Results

```python
# Single result
self.update_results("final_accuracy", 0.95)

# Multiple results
self.update_results("metrics", {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.96,
    "f1_score": 0.94
})
```

## Running Scripts

### Standalone Mode

```bash
python my_script.py
```

### With Connector Integration

```bash
# Start the connector system
python hivemind_connector.py &
python connector.py

# Run script with connector
python my_script.py --with-connector
```

### Using the Connector Menu

1. Start the connector: `python connector.py`
2. Select option 2: "Execute script"
3. Choose your script
4. Select "Run with connector integration"

## Best Practices

1. **Parameter Registration**: Register all configurable parameters in `__init__`
2. **Variable Updates**: Update variables regularly during execution
3. **Error Handling**: Use try-except blocks in collaboration handlers
4. **Resource Cleanup**: Properly stop scripts when done
5. **Logging**: Use the built-in logger for debugging

## Troubleshooting

### Running Diagnostics

```bash
python troubleshoot_all.py
```

### Common Issues

1. **Port conflicts**: Ensure ports 10117-10118 are available
2. **Import errors**: Check that `script_interface.py` is in the same directory
3. **Connection refused**: Start the hivemind connector first
4. **Script not found**: Ensure script is in the ml-models directory

## Examples

### Example 1: Simple Parameter Control

```python
class SimpleMLScript(ScriptInterface):
    def __init__(self):
        super().__init__("simple_ml", "Simple ML Example")
        self.register_parameter("learning_rate", 0.01, [0.001, 0.01, 0.1])
        
    def run(self):
        lr = self.get_parameter("learning_rate")
        print(f"Training with learning rate: {lr}")
```

### Example 2: Collaborative Training

```python
class CollaborativeTrainer(ScriptInterface):
    def __init__(self):
        super().__init__("collab_trainer", "Collaborative Training")
        self.client = ConnectorClient(self)
        
    def run(self):
        # Share training progress
        for epoch in range(10):
            loss = train_epoch()
            self.client.broadcast_data({
                "type": "training_update",
                "epoch": epoch,
                "loss": loss
            })
```

### Example 3: Monitoring Long-Running Process

```python
class LongRunningAnalysis(ScriptInterface):
    def __init__(self):
        super().__init__("long_analysis", "Long Running Analysis")
        self.register_variable("progress", 0.0)
        self.register_variable("eta_seconds", 0)
        
    def run(self):
        total_steps = 1000
        for step in range(total_steps):
            # Update progress
            progress = (step / total_steps) * 100
            self.set_variable("progress", progress)
            self.set_variable("eta_seconds", (total_steps - step) * 0.1)
            
            # Do work
            process_step(step)
```

## API Reference

### ScriptInterface Methods

- `register_parameter(name, default, valid_values=None)`
- `register_variable(name, value, read_only=False)`
- `get_parameter(name) -> Any`
- `set_parameter(name, value) -> bool`
- `get_variable(name) -> Any`
- `set_variable(name, value) -> bool`
- `update_results(key, value)`
- `register_callback(parameter_name, callback)`
- `send_to_connector(message) -> dict`
- `notify_connector(event, data=None)`

### ConnectorClient Methods

- `register_script() -> dict`
- `request_collaboration(target_script, data) -> dict`
- `broadcast_data(data)`
- `get_script_list() -> List[str]`

## Version History

- v1.0: Initial connector system with basic parameter control
- v1.1: Added script-to-script collaboration
- v1.2: Enhanced monitoring and results management
- v1.3: Added wrapper support for legacy scripts