# Core Detection Node.js Orchestrator

This document describes the Node.js integration that enables running the three main Python scripts (`auto-core-detection.py`, `live_feed.py`, and `main.py`) as parallel processes with real-time monitoring and control.

## 🚀 Quick Start

### Prerequisites
- Node.js (v14 or higher)
- Python 3.8+ with required packages (see `requirements.txt`)
- All Python scripts and `config.json` in place

### Installation

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Test the orchestrator:**
   ```bash
   npm test
   ```

3. **Start the parallel processes:**
   ```bash
   npm start
   ```

4. **Start with web monitoring interface:**
   ```bash
   node monitor.js
   ```
   Then open http://localhost:3000 in your browser

## 📁 File Structure

```
version12/
├── orchestrator.js          # Main Node.js orchestrator
├── monitor.js              # Web monitoring interface
├── test_orchestrator.js    # Test script
├── package.json            # Node.js dependencies
├── public/
│   └── index.html         # Web monitoring UI
├── config.json            # Centralized configuration
├── auto-core-detection.py # Enhanced core detection
├── live_feed.py          # Live camera feed
├── main.py               # Main orchestration script
└── config_manager.py     # Configuration manager
```

## 🔧 Core Components

### 1. CoreDetectionOrchestrator (`orchestrator.js`)

The main orchestrator class that manages parallel Python processes:

**Key Features:**
- Automatic Python path detection
- Process lifecycle management (start/stop/restart)
- Real-time output capture and logging
- Event-driven architecture
- Graceful shutdown handling

**Methods:**
- `startAllProcesses()` - Starts all three Python scripts in parallel
- `stopAllProcesses()` - Gracefully stops all processes
- `restartProcess(scriptName)` - Restarts a specific process
- `getProcessStatus()` - Returns current status of all processes
- `getAllProcessOutputs()` - Returns captured stdout/stderr

### 2. Web Monitoring Interface (`monitor.js`)

Express.js server with Socket.IO for real-time monitoring:

**Features:**
- Real-time process status updates
- Web-based control panel
- Live output streaming
- Process restart capabilities
- Connection status monitoring

**Endpoints:**
- `GET /` - Main monitoring interface
- `GET /api/status` - Current process status
- `POST /api/start` - Start all processes
- `POST /api/stop` - Stop all processes
- `POST /api/restart/:script` - Restart specific process

### 3. Test Suite (`test_orchestrator.js`)

Comprehensive testing of the orchestrator functionality:

**Tests:**
- Configuration file validation
- Python script availability
- Python path detection
- Process management methods
- Event system functionality

## 🎯 Usage Examples

### Basic Usage

```javascript
const orchestrator = require('./orchestrator');

// Start all processes
await orchestrator.startAllProcesses();

// Get current status
const status = orchestrator.getProcessStatus();
console.log(status);

// Stop all processes
orchestrator.stopAllProcesses();
```

### Event Handling

```javascript
orchestrator.on('allProcessesStarted', () => {
    console.log('🎉 All processes are running!');
});

orchestrator.on('processOutput', ({ script, output, type }) => {
    console.log(`[${script}] ${output}`);
});

orchestrator.on('processClosed', ({ script, code }) => {
    console.log(`Process ${script} exited with code ${code}`);
});
```

### Web Interface

```bash
# Start the monitoring server
node monitor.js

# Access the web interface
# Open http://localhost:3000 in your browser
```

## 🔄 Process Management

### Parallel Process Architecture

```
Node.js Orchestrator
├── auto-core-detection.py (Process 1)
│   ├── Multi-scale circle detection
│   ├── PyTorch learning integration
│   └── Real-time confidence calculation
├── live_feed.py (Process 2)
│   ├── Camera feed processing
│   ├── Frame analysis
│   └── Output generation
└── main.py (Process 3)
    ├── Main orchestration
    ├── Result aggregation
    └── System coordination
```

### Inter-Process Communication

- **Shared Configuration**: All processes use the centralized `config.json`
- **Output Capture**: Node.js captures and logs all stdout/stderr
- **Status Monitoring**: Real-time process status tracking
- **Error Handling**: Automatic error detection and logging

## 📊 Monitoring Features

### Real-Time Dashboard

The web interface provides:

1. **Process Status Cards**
   - Running/Stopped status indicators
   - Process ID (PID) display
   - Exit code monitoring
   - Real-time output streaming

2. **Control Panel**
   - Start All Processes button
   - Stop All Processes button
   - Individual process restart
   - Status refresh

3. **System Overview**
   - Total process count
   - Running process count
   - System status
   - Connection status

### Logging and Error Handling

- **Error Logging**: All stderr output is logged to `orchestrator_errors.log`
- **Connection Status**: Real-time WebSocket connection monitoring
- **Process Recovery**: Automatic process restart capabilities
- **Graceful Shutdown**: Proper cleanup on system termination

## ⚙️ Configuration

The orchestrator uses the same `config.json` file as the Python scripts:

```json
{
  "auto_core_detection": {
    "detection": { ... },
    "hough_circles": { ... },
    "preprocessing": { ... }
  },
  "live_feed": {
    "camera": { ... },
    "processing": { ... }
  },
  "main": {
    "orchestration": { ... }
  }
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Python Not Found**
   ```
   Error: Python executable not found
   ```
   **Solution**: Ensure Python is installed and in PATH, or update the `findPythonPath()` method in `orchestrator.js`

2. **Script Not Found**
   ```
   Error: Failed to start auto-core-detection
   ```
   **Solution**: Verify all Python scripts exist in the project directory

3. **Port Already in Use**
   ```
   Error: listen EADDRINUSE :::3000
   ```
   **Solution**: Change the port in `monitor.js` or kill the existing process

4. **Permission Denied**
   ```
   Error: spawn EACCES
   ```
   **Solution**: Ensure proper file permissions and Python executable access

### Debug Mode

Enable detailed logging:

```javascript
// In orchestrator.js, add:
process.env.DEBUG = 'orchestrator:*';
```

## 🔧 Advanced Configuration

### Custom Python Path

Update the `findPythonPath()` method in `orchestrator.js`:

```javascript
findPythonPath() {
    const possiblePaths = [
        '/path/to/your/python',
        'python',
        'python3',
        'py'
    ];
    // ... rest of the method
}
```

### Custom Process Arguments

Modify the `startProcess()` method to pass custom arguments:

```javascript
const scripts = [
    { name: 'auto-core-detection', path: 'auto-core-detection.py', args: ['--debug'] },
    { name: 'live-feed', path: 'live_feed.py', args: ['--camera=1'] },
    { name: 'main', path: 'main.py', args: ['--verbose'] }
];
```

## 📈 Performance Considerations

### Resource Management

- **Memory Usage**: Each Python process runs independently
- **CPU Utilization**: Processes run in parallel for optimal performance
- **I/O Handling**: Non-blocking I/O with event-driven architecture
- **Error Recovery**: Automatic restart on process failure

### Scalability

- **Process Isolation**: Each script runs in its own process space
- **Load Distribution**: Parallel execution reduces total processing time
- **Monitoring Overhead**: Minimal impact on Python script performance
- **Extensibility**: Easy to add more processes or monitoring features

## 🔮 Future Enhancements

### Planned Features

1. **Process Communication**
   - Inter-process message passing
   - Shared memory for data exchange
   - Result aggregation and analysis

2. **Advanced Monitoring**
   - Performance metrics collection
   - Resource usage monitoring
   - Alert system for failures

3. **Configuration Management**
   - Hot-reload configuration changes
   - Dynamic parameter tuning
   - A/B testing capabilities

4. **Deployment Options**
   - Docker containerization
   - Kubernetes orchestration
   - Cloud deployment support

## 📝 API Reference

### Orchestrator Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `startAllProcesses()` | Start all Python scripts | Promise |
| `stopAllProcesses()` | Stop all processes | void |
| `restartProcess(name)` | Restart specific process | void |
| `getProcessStatus()` | Get current status | Object |
| `getAllProcessOutputs()` | Get all outputs | Object |

### Events

| Event | Description | Data |
|-------|-------------|------|
| `allProcessesStarted` | All processes started | - |
| `allProcessesStopped` | All processes stopped | - |
| `processOutput` | Process output received | `{script, output, type}` |
| `processClosed` | Process exited | `{script, code}` |
| `processError` | Process error occurred | `{script, error}` |

## 🤝 Contributing

To contribute to the Node.js orchestrator:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Test with all Python scripts
5. Ensure backward compatibility

## 📄 License

This Node.js integration is part of the core detection system and follows the same license as the main project. 