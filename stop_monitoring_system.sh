#!/bin/bash
# Stop the monitoring system

echo "Stopping Monitoring System..."

# Stop connector
if [ -f "connector.pid" ]; then
    CONNECTOR_PID=$(cat connector.pid)
    echo "Stopping connector (PID: $CONNECTOR_PID)..."
    kill $CONNECTOR_PID 2>/dev/null
    rm connector.pid
fi

# Stop hivemind connector
if [ -f "hivemind_connector.pid" ]; then
    HIVEMIND_PID=$(cat hivemind_connector.pid)
    echo "Stopping hivemind connector (PID: $HIVEMIND_PID)..."
    kill $HIVEMIND_PID 2>/dev/null
    rm hivemind_connector.pid
fi

# Kill any remaining python processes
pkill -f "python connector.py"
pkill -f "python hivemind_connector.py"

echo "Monitoring system stopped."