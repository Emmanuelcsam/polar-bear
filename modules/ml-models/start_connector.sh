#!/bin/bash
# Start the ML Models Connector System

echo "========================================="
echo "Starting ML Models Connector System"
echo "========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Check if connectors are already running
if check_port 10117; then
    echo "Warning: Port 10117 is already in use (Hivemind Connector may be running)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start Hivemind Connector in background
echo "Starting Hivemind Connector..."
python3 hivemind_connector.py > hivemind_connector.log 2>&1 &
HIVEMIND_PID=$!
echo "Hivemind Connector started (PID: $HIVEMIND_PID)"

# Wait a moment for startup
sleep 2

# Check if it started successfully
if ! kill -0 $HIVEMIND_PID 2>/dev/null; then
    echo "Error: Hivemind Connector failed to start"
    echo "Check hivemind_connector.log for details"
    exit 1
fi

# Create stop script
cat > stop_connector.sh << 'EOF'
#!/bin/bash
echo "Stopping ML Models Connector System..."

# Find and kill hivemind connector
HIVEMIND_PID=$(lsof -ti:10117)
if [ ! -z "$HIVEMIND_PID" ]; then
    kill $HIVEMIND_PID
    echo "Stopped Hivemind Connector (PID: $HIVEMIND_PID)"
fi

# Find and kill script control server
CONTROL_PID=$(lsof -ti:10118)
if [ ! -z "$CONTROL_PID" ]; then
    kill $CONTROL_PID
    echo "Stopped Script Control Server (PID: $CONTROL_PID)"
fi

# Kill any running test scripts
pkill -f "test_script_"

echo "Connector system stopped"
EOF

chmod +x stop_connector.sh

echo ""
echo "========================================="
echo "Connector System Started Successfully!"
echo "========================================="
echo ""
echo "Available Commands:"
echo "1. Run a script with connector:"
echo "   python3 <script_name.py> --with-connector"
echo ""
echo "2. Start the interactive control menu:"
echo "   python3 connector.py"
echo ""
echo "3. Run integration tests:"
echo "   python3 test_integration.py"
echo ""
echo "4. Run diagnostics:"
echo "   python3 troubleshoot_all.py"
echo ""
echo "5. Stop the connector system:"
echo "   ./stop_connector.sh"
echo ""
echo "Logs are saved to: hivemind_connector.log"
echo "========================================="

# Optional: Start the interactive connector
read -p "Start interactive connector menu? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 connector.py
fi