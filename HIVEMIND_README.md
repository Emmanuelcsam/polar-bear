# Polar Bear Hivemind System

A comprehensive deep recursive connector network that deploys intelligent connector scripts to ALL subdirectories throughout your project.

## Overview

The Polar Bear Hivemind System creates a network of interconnected Python scripts (connectors) in every subdirectory of your project. Each connector can:
- Communicate with parent and child connectors
- Scan and monitor its directory
- Execute scripts in its directory
- Report status and health information
- Troubleshoot connection issues

## System Architecture

```
Root Hivemind (port 10000)
├── Static Connector (port 10001)
│   ├── CSS Connector (port 10002)
│   └── JS Connector (port 10003)
├── Templates Connector (port 10004)
├── Modules Connector (port 10005)
│   ├── [137 sub-connectors across multiple levels]
│   └── ...
├── Training Connector
│   ├── [Multiple sub-connectors]
│   └── ...
└── Ruleset Connector
    └── [Sub-connectors]
```

## Features

- **Deep Recursive Deployment**: Automatically deploys to ALL subdirectories (up to 10 levels deep by default)
- **Smart Directory Filtering**: Automatically skips virtual environments, cache directories, and version control folders
- **Hierarchical Communication**: Parent-child relationships maintained throughout the network
- **TCP Socket Based**: Each connector listens on its own port (starting from 10000)
- **Health Monitoring**: Built-in heartbeat system and health checks
- **Interactive Control**: Command-line interface for managing the entire network
- **Comprehensive Logging**: Both terminal and file logging with timestamps

## Installation & Usage

### Quick Start
```bash
python launch_hivemind.py
```

### With Options
```bash
# Skip confirmation prompts
python launch_hivemind.py --quick

# Run diagnostics
python test_hivemind.py
```

### Direct Execution
```bash
python polar_bear_hivemind.py
```

## Commands

When the hivemind is running, you can use these commands:

- `status` - Show system status and active connectors
- `tree` - Display hierarchical connector tree
- `scan` - Trigger rescan of all directories
- `check` - Check health of all connectors
- `reload` - Reload configuration
- `exit` - Gracefully shutdown the hivemind

## Configuration

The system creates a `hivemind_config.json` file with these options:
- `project_name`: Name of your project
- `auto_install_requirements`: Auto-install missing packages
- `scan_interval`: Directory scan interval in seconds
- `max_connector_depth`: Maximum directory depth for deployment
- `connector_timeout`: Timeout for connector operations
- `log_level`: Logging verbosity level

## Directory Structure After Deployment

```
your-project/
├── polar_bear_hivemind.py (main control script)
├── launch_hivemind.py (startup utility)
├── test_hivemind.py (diagnostic tool)
├── hivemind_config.json (configuration)
├── hivemind_logs/ (log files)
├── modules/
│   ├── hivemind_connector.py
│   ├── submodule1/
│   │   ├── hivemind_connector.py
│   │   └── deeper/
│   │       └── hivemind_connector.py
│   └── ...
└── [other directories with connectors]
```

## Connector Capabilities

Each connector can:
1. **Monitor** its directory for changes
2. **Execute** Python scripts in its directory
3. **Communicate** with parent/child connectors
4. **Report** status and statistics
5. **Troubleshoot** connection issues

## API Example

Connectors respond to JSON commands:

```python
# Status request
{"command": "status"}

# Scan directory
{"command": "scan"}

# Execute script
{"command": "execute", "script": "example.py"}

# Troubleshoot
{"command": "troubleshoot"}
```

## Troubleshooting

1. **Port conflicts**: The system starts at port 10000. Ensure ports 10000-11000 are available.
2. **Permission errors**: Ensure write permissions in all directories.
3. **Missing dependencies**: Only watchdog is optional. psutil, colorama, and requests are required.
4. **Too many connectors**: Adjust `max_connector_depth` in configuration if needed.

## Statistics

Based on your project:
- Total connectors deployed: 324
- Maximum depth reached: 6 levels
- Directories skipped: Virtual environments and cache folders
- Port range used: 10000-10324

## Security Notes

- Connectors only listen on localhost (127.0.0.1)
- No external network access by default
- Script execution requires explicit commands
- All operations are logged

## Performance

The system is designed to be lightweight:
- Each connector runs in its own thread
- Minimal CPU usage when idle
- Efficient parent-child communication
- Automatic cleanup on shutdown

## Future Enhancements

The modular design allows for easy extension:
- File watching with watchdog (when installed)
- Custom command plugins
- Web dashboard interface
- Distributed deployment across machines
- Integration with CI/CD pipelines