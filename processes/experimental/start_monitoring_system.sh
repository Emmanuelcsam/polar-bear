#!/bin/bash
# Start the monitoring system with connectors

echo "Starting Monitoring System..."

# Start the main connector
echo "Starting main connector..."
python connector.py &
CONNECTOR_PID=$!
echo "Connector started with PID: $CONNECTOR_PID"

# Wait for connector to initialize
sleep 3

# Start the hivemind connector if needed
if [ -f "hivemind_connector.py" ]; then
    echo "Starting hivemind connector..."
    python hivemind_connector.py &
    HIVEMIND_PID=$!
    echo "Hivemind connector started with PID: $HIVEMIND_PID"
fi

# Create PID file
echo $CONNECTOR_PID > connector.pid
if [ ! -z "$HIVEMIND_PID" ]; then
    echo $HIVEMIND_PID > hivemind_connector.pid
fi

echo "Monitoring system started successfully!"
echo "To stop: ./stop_monitoring_system.sh"