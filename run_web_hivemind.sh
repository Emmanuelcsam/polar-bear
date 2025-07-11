#!/bin/bash

echo "========================================"
echo "Neural Hivemind Web Interface Launcher"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements_web.txt

# Create necessary directories
mkdir -p uploads
mkdir -p hivemind_logs

# Launch the web interface
echo ""
echo "Starting Neural Hivemind Web Interface..."
echo "========================================"
echo "Access the interface at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python neural_hivemind_web.py