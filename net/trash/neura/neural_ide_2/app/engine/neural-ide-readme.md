# Neural Script IDE - Production Version

A sophisticated IDE for building neural networks from interconnected Python scripts. This tool allows you to create complex systems where each script acts as a node in a neural network, communicating through a robust message-passing system.

## Features

### Core Features
- **Multi-Script Management**: Run and manage multiple Python scripts simultaneously
- **Inter-Script Communication**: Built-in message broker for script-to-script communication
- **Dependency Visualization**: Real-time graph showing script relationships
- **Advanced Debugging**: Breakpoints, variable inspection, and step-through debugging
- **Performance Monitoring**: CPU, memory, and message throughput tracking
- **Script Orchestration**: Automatic execution order based on dependencies

### Advanced Features
- **Logic Analysis**: Beyond syntax checking - detects infinite loops, unreachable code
- **Message Inspector**: Monitor and filter inter-script messages
- **Performance Profiling**: Identify bottlenecks in your script network
- **Script Templates**: Pre-built templates for common patterns
- **Project Import/Export**: Save and share complete neural network configurations

## Installation

### Method 1: Using the Launcher (Recommended)
1. Save all three files in the same directory:
   - `launch_neural_ide.py` (launcher)
   - `neural_script_ide.py` (main application)
   - Example scripts (optional)

2. Run the launcher:
   ```bash
   python launch_neural_ide.py
   ```

3. The launcher will:
   - Check Python version (3.7+ required)
   - Install required dependencies
   - Launch the IDE

### Method 2: Manual Installation
```bash
# Install required packages
pip install tkinter pylint networkx matplotlib psutil numpy pyyaml websockets

# Run the IDE directly
python neural_script_ide.py
```

## Quick Start

### 1. Create Your First Neural Network

1. Launch the IDE
2. Create multiple script tabs (Ctrl+N)
3. Use the provided templates or write your own scripts
4. Each script should use the `comm` object for communication:
   ```python
   # Send a message
   comm.send_message('target_script', data)
   
   # Broadcast to all scripts
   comm.broadcast(data)
   
   # Log metrics
   comm.log_metric('accuracy', 0.95)
   ```

### 2. Set Up Dependencies

1. Click "Show Dependencies" to visualize script relationships
2. Dependencies are automatically detected from your code
3. Manual dependency configuration available in script properties

### 3. Run Your Network

1. Click "Run All" to execute scripts in dependency order
2. Monitor execution in real-time:
   - Console output for each script
   - Message flow visualization
   - Performance metrics
   - System resource usage

## Example Project: Simple Neural Network

The included example demonstrates a 3-layer neural network:

```
Input Layer → Hidden Layer 1 → Output Layer
           ↘ Hidden Layer 2 ↗
```

### Running the Example:

1. Load the example scripts into separate tabs
2. Run all scripts (F5)
3. Enter data in the input layer: `5.1,3.5,1.4,0.2`
4. Watch data flow through the network
5. See predictions in the output layer

## Key Concepts

### Message Types
- **DATA**: Regular data passing between scripts
- **CONTROL**: Start/stop/pause commands
- **STATUS**: Script state updates
- **ERROR**: Error notifications
- **METRIC**: Performance metrics
- **HEARTBEAT**: Keep-alive signals

### Script Lifecycle
1. **Idle**: Script loaded but not running
2. **Running**: Actively executing
3. **Completed**: Finished successfully
4. **Error**: Terminated with error

### Communication Patterns

#### Direct Messaging
```python
comm.send_message('specific_script', {'data': [1, 2, 3]})
```

#### Broadcasting
```python
comm.broadcast({'event': 'training_complete'})
```

#### Request-Response
```python
# Script A
comm.send_message('script_b', {'request': 'get_weights'})

# Script B (receives and responds)
weights = self.get_current_weights()
comm.send_message('script_a', {'response': weights})
```

## Advanced Usage

### Custom Message Handlers
```python
def process_message(message):
    if message.message_type == MessageType.DATA:
        result = process_data(message.payload)
        comm.send_message(message.sender_id, result)
```

### Performance Optimization
- Use batch processing for large datasets
- Implement message queuing for high-throughput scenarios
- Monitor metrics to identify bottlenecks

### Debugging Complex Networks
1. Set breakpoints in critical paths
2. Use the message inspector to trace data flow
3. Enable performance profiling for bottleneck analysis
4. Check dependency graph for circular dependencies

## Keyboard Shortcuts

- **Ctrl+N**: New script tab
- **Ctrl+O**: Open script
- **Ctrl+S**: Save current script
- **Ctrl+Shift+S**: Save all scripts
- **Ctrl+W**: Close current tab
- **Ctrl+Tab**: Next tab
- **F5**: Run current/all scripts
- **Shift+F5**: Stop all scripts
- **F9**: Toggle breakpoint
- **F10**: Start debugging
- **F11**: Step over

## Configuration

Settings are stored in `~/.neural_script_ide_config.json`:

```json
{
  "auto_save": true,
  "auto_analyze": true,
  "performance_monitoring": true,
  "message_history_size": 1000,
  "theme": "dark"
}
```

## Troubleshooting

### Scripts not communicating
- Check script IDs match in send_message calls
- Verify all scripts are running
- Use message inspector to trace messages

### High CPU usage
- Check for infinite loops in scripts
- Reduce message frequency
- Enable performance monitoring

### Dependency errors
- View dependency graph for circular dependencies
- Ensure all referenced scripts exist
- Check script IDs are unique

## Best Practices

1. **Modular Design**: Keep scripts focused on single responsibilities
2. **Error Handling**: Always handle exceptions in message processing
3. **Resource Management**: Clean up resources in script termination
4. **Documentation**: Comment your message formats and protocols
5. **Testing**: Test scripts individually before integration

## Contributing

To extend the IDE:

1. Add new message types in `MessageType` enum
2. Implement custom analyzers in `ScriptDebugger`
3. Create new visualization modes
4. Add script templates for common patterns

## License

This software is provided as-is for educational and development purposes.

## Support

For issues and questions:
- Check the built-in documentation (Help menu)
- Review example projects
- Examine script templates
- Enable debug mode for detailed logging

---

Happy neural network building! 🧠🚀