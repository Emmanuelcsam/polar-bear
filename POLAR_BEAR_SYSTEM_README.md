# Polar Bear System

A comprehensive Python connector system for the Polar Bear project, providing centralized control and communication between different components.

## Features

- **Centralized Control**: Main control script manages all connectors
- **Auto-deployment**: Automatically creates connector scripts in each directory
- **Inter-connector Communication**: All connectors can communicate with each other
- **Comprehensive Logging**: Detailed logs to both terminal and file
- **Auto-requirements**: Detects and installs missing dependencies
- **Interactive Configuration**: Easy setup through command-line questions
- **Health Monitoring**: Tracks connector status with heartbeat system

## Quick Start

1. **Launch the system**:
   ```bash
   python launch_polar_bear.py
   ```

2. **Quick launch (skip checks)**:
   ```bash
   python launch_polar_bear.py --quick
   ```

3. **Direct system start**:
   ```bash
   python polar_bear_system.py
   ```

## System Architecture

```
polar-bear/
├── polar_bear_system.py          # Main control script
├── launch_polar_bear.py          # Launcher utility
├── polar_bear_requirements.txt   # System requirements
├── polar_bear_config.json        # System configuration (auto-created)
├── polar_bear_system.log         # System log file
│
├── ruleset/
│   └── ruleset_connector.py      # Auto-generated connector
├── modules/
│   └── modules_connector.py      # Auto-generated connector
├── training/
│   └── training_connector.py     # Auto-generated connector
├── static/
│   └── static_connector.py       # Auto-generated connector
└── templates/
    └── templates_connector.py    # Auto-generated connector
```

## Configuration

On first run, the system will guide you through configuration:

- Project name
- Log level (DEBUG/INFO/WARNING/ERROR)
- Enable/disable specific connectors
- Advanced settings (timeouts, auto-restart, etc.)

Configuration is saved to `polar_bear_config.json`.

## Interactive Commands

Once running, use these commands:

- `status` - Get status from all connectors
- `scan` - Scan all connector directories
- `send` - Send command to specific connector
- `list` - List connected connectors
- `config` - Reconfigure system
- `restart` - Restart a connector
- `exit` - Exit system

## Communication Protocol

Connectors communicate using JSON messages over TCP sockets:

```python
# Message format
{
    "type": "message_type",
    "connector": "sender_name",
    "timestamp": 1234567890.123,
    "data": {...}
}
```

### Message Types

- `identify` - Connector identification
- `heartbeat` - Health check
- `scan` - Directory scan request
- `scan_results` - Scan response
- `status` - Status request
- `status_update` - Status response
- `execute` - Command execution
- `disconnect` - Graceful disconnect

## Connector Ports

- Root Controller: 9000
- Ruleset Connector: 9001
- Modules Connector: 9002
- Training Connector: 9003
- Static Connector: 9004
- Templates Connector: 9005

## Logging

Logs are written to:
- Terminal (with color coding)
- `polar_bear_system.log` (detailed file logs)

Log format:
```
2025-07-11 10:30:45 | ModuleName | INFO | Message
```

## Extending the System

To add custom functionality to a connector:

1. Edit the generated connector script
2. Add handlers in the `handle_message()` method
3. Implement new message types
4. Update the root controller to handle responses

## Troubleshooting

**Connectors not connecting:**
- Check if ports 9000-9005 are available
- Ensure no firewall blocking localhost connections
- Check logs for specific error messages

**Requirements installation fails:**
- Run with appropriate permissions
- Use virtual environment if needed
- Install requirements manually if auto-install fails

**Configuration issues:**
- Delete `polar_bear_config.json` to reset
- Run configuration setup again

## Development

To run in development mode:
```bash
# Set log level to DEBUG
python polar_bear_system.py
# In interactive mode, use 'config' command
# Set log level to DEBUG
```

## Requirements

- Python 3.7+
- See `polar_bear_requirements.txt` for dependencies

## Notes

- Connectors are automatically skipped in virtual environment directories
- The system respects `.gitignore` patterns when scanning
- All communication is on localhost only (security)
- Graceful shutdown with Ctrl+C

## Version

Current version: 1.0.0