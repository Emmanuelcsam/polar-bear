# Monitoring Systems - Full Integration Guide

## Overview
This monitoring system provides comprehensive real-time monitoring, analysis, and control for fiber optic inspection systems. All scripts are fully integrated with bidirectional connectors that allow:

- **Remote parameter control**: Change script parameters in real-time
- **Live metrics monitoring**: Track performance metrics across all components
- **Status management**: Start, stop, pause, and resume scripts remotely
- **Unified control**: Manage all scripts from a central connector

## Architecture

### Core Components

1. **connector.py** - Main monitoring connector
   - Manages all script connections
   - Provides centralized control interface
   - Tracks script states and metrics
   - Port: 10130

2. **hivemind_connector.py** - Hierarchical connector system
   - Manages distributed script execution
   - Supports parent-child connector relationships
   - Enables scalable monitoring architecture

3. **connector_interface.py** - Integration library
   - Provides easy integration for any Python script
   - Supports parameter registration and control
   - Real-time metric updates
   - Bidirectional communication

### Monitoring Scripts

1. **advanced_monitoring_system.py**
   - Real-time performance monitoring
   - Anomaly detection
   - Model drift detection
   - Automated retraining pipeline
   
2. **realtime_fiber_system.py**
   - Real-time video stream processing
   - WebSocket communication
   - Queue-based processing
   - Edge device integration

3. **streaming_data_pipeline.py**
   - Apache Kafka integration
   - Stream processing with Faust
   - Real-time analytics
   - Data lake integration

4. **comprehensive_testing_suite.py**
   - Unit and integration tests
   - Performance benchmarks
   - Load testing capabilities
   - Chaos engineering tests

5. **main-application.py**
   - Main entry point for fiber inspection
   - User interface for batch processing
   - Integrated with connector system

## Quick Start

### 1. Start the Monitoring System
```bash
./start_monitoring_system.sh
```

### 2. Run the Troubleshooting Tool
```bash
python troubleshoot_all.py
```

### 3. Test Connector Integration
```bash
# Run the test script
python test_connector_integration.py

# In another terminal, test control
python test_connector_integration.py --control-test
```

### 4. Stop the System
```bash
./stop_monitoring_system.sh
```

## Integration Guide

### Adding Connector Support to Your Script

1. **Simple Integration with Decorator**:
```python
from connector_interface import connector_enabled

@connector_enabled(connector_port=10130)
def main():
    # Your script logic here
    pass
```

2. **Manual Integration**:
```python
from connector_interface import ConnectorInterface

# Initialize connector
connector = ConnectorInterface('my_script.py')

# Register parameters
connector.register_parameter(
    'threshold',
    0.5,
    'float',
    'Detection threshold',
    min_value=0.0,
    max_value=1.0,
    callback=lambda old, new: print(f"Threshold changed: {old} -> {new}")
)

# Update metrics
connector.update_metric('processed_items', 100)
connector.update_metric('error_rate', 0.01)

# Set status
connector.set_status('running')
```

## Controllable Parameters

Each script exposes different parameters that can be controlled:

### advanced_monitoring_system.py
- `accuracy_threshold` (float): Minimum acceptable model accuracy
- `drift_threshold` (float): Maximum acceptable distribution drift
- `monitoring_interval` (int): Performance monitoring interval in seconds
- `alert_enabled` (bool): Enable/disable alerting system
- `retraining_enabled` (bool): Enable/disable automated retraining

### realtime_fiber_system.py
- `processing_fps` (int): Target processing FPS
- `quality_threshold` (float): Quality detection threshold
- `max_queue_size` (int): Maximum frame queue size
- `enable_recording` (bool): Enable video recording of anomalies
- `websocket_enabled` (bool): Enable WebSocket streaming

### streaming_data_pipeline.py
- `kafka_broker` (str): Kafka broker URL
- `batch_size` (int): Batch processing size
- `window_size` (int): Aggregation window size in seconds
- `anomaly_threshold` (float): Anomaly detection threshold
- `enable_enrichment` (bool): Enable stream enrichment

## API Reference

### Connector Commands

Send JSON commands to the connector on port 10130:

```python
# Get all scripts status
{
    "command": "get_scripts"
}

# Control a script
{
    "command": "control_script",
    "script": "script_name.py",
    "action": "start|stop|pause|resume",
    "params": {}
}

# Set parameter
{
    "command": "set_parameter",
    "script": "script_name.py",
    "parameter": "param_name",
    "value": new_value
}

# Get script state
{
    "command": "get_script_state",
    "script": "script_name.py"
}
```

## Troubleshooting

### Common Issues

1. **Connector not starting**
   - Check if port 10130 is available
   - Ensure no other instance is running
   - Check logs in `connector.log`

2. **Scripts not connecting**
   - Verify connector is running
   - Check script has connector_interface import
   - Ensure correct port configuration

3. **Missing dependencies**
   - Run `python troubleshoot_all.py` to identify missing modules
   - Install with: `pip install -r requirements.txt`

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- Each script maintains its own connection to the connector
- Metrics updates are asynchronous and non-blocking
- Parameter changes trigger callbacks immediately
- The connector can handle multiple simultaneous connections

## Security Notes

- Connector only accepts local connections (localhost)
- No authentication implemented (add if needed for production)
- Parameter validation prevents invalid values
- Scripts validate all incoming commands

## Future Enhancements

- [ ] Web UI for monitoring dashboard
- [ ] Historical metrics storage
- [ ] Alert integration (email, Slack, etc.)
- [ ] Distributed connector mesh network
- [ ] Authentication and authorization
- [ ] Metric aggregation and analytics

## Support

For issues or questions:
1. Run the troubleshooting script: `python troubleshoot_all.py`
2. Check the generated report and logs
3. Ensure all dependencies are installed
4. Verify connector is running and accessible