#!/bin/bash

echo "========================================"
echo "   Core Detection Orchestrator"
echo "   with Pylon Viewer Integration"
echo "========================================"
echo

# Check Node.js installation
echo "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check Python installation
echo "Checking Python installation..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v py &> /dev/null; then
    PYTHON_CMD="py"
else
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python from https://python.org/"
    exit 1
fi

echo "Found Python at: $PYTHON_CMD"

# Check required files
echo "Checking required files..."
if [ ! -f "config.json" ]; then
    echo "ERROR: config.json not found"
    exit 1
fi

if [ ! -f "auto-core-detection.py" ]; then
    echo "ERROR: auto-core-detection.py not found"
    exit 1
fi

if [ ! -f "live_feed.py" ]; then
    echo "ERROR: live_feed.py not found"
    exit 1
fi

if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found"
    exit 1
fi

if [ ! -f "circle_overlay.py" ]; then
    echo "ERROR: circle_overlay.py not found"
    exit 1
fi

if [ ! -f "pylon_viewer_integration.py" ]; then
    echo "ERROR: pylon_viewer_integration.py not found"
    exit 1
fi

echo "Installing Node.js dependencies..."
npm install

echo
echo "========================================"
echo "Starting Core Detection Orchestrator..."
echo "========================================"
echo
echo "The web monitoring interface will be available at:"
echo "http://localhost:3000"
echo
echo "Features enabled:"
echo "  - Pylon Viewer integration"
echo "  - Circle overlay controls (WASD, Q/E)"
echo "  - Auto core detection"
echo "  - Live feed processing"
echo "  - Web monitoring interface"
echo
echo "Press Ctrl+C to stop the orchestrator"
echo

# Check if port 3000 is available
echo "Checking port availability..."
if netstat -ano | grep -q ":3000"; then
    echo "WARNING: Port 3000 is already in use. Attempting to free it..."
    # Try to kill any process using port 3000
    for pid in $(netstat -ano | grep ":3000" | awk '{print $5}' | sort -u); do
        if [ ! -z "$pid" ] && [ "$pid" != "0" ]; then
            echo "Killing process $pid using port 3000..."
            cmd //c "taskkill /F /PID $pid" 2>/dev/null || true
        fi
    done
    # Wait a moment for the port to be freed
    sleep 2
fi

# Start the main orchestrator with Pylon Viewer integration
echo "Starting integrated core detection system..."
node monitor.js

echo "Orchestrator stopped." 