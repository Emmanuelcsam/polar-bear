#!/bin/bash

# TensorBoard Start Script for Fiber Optic Neural Networks
# Automatically starts TensorBoard and opens the website in your default browser.

echo "============================================================"
echo "🔍 TensorBoard Starter for Fiber Optic Neural Networks"
echo "============================================================"

# Check if runs directory exists
if [ ! -d "runs" ]; then
    echo "❌ Error: Log directory 'runs' does not exist."
    echo "   Please run one of the neural network scripts first:"
    echo "   - python neural-network--with-visual-1.py"
    echo "   - python neural-network--with-visual-2.py"
    exit 1
fi

# Check if TensorBoard is installed
if ! command -v tensorboard &> /dev/null; then
    echo "❌ Error: TensorBoard is not installed."
    echo "   Install it with: pip install tensorboard"
    exit 1
fi

echo "🚀 Starting TensorBoard..."
echo "   Log directory: runs"
echo "   Port: 6006"
echo "   URL: http://localhost:6006"

# Start TensorBoard in background
tensorboard --logdir=runs --port=6006 --host=localhost --reload_multifile=true &
TENSORBOARD_PID=$!

# Wait a moment for TensorBoard to start
sleep 3

# Check if TensorBoard is running
if kill -0 $TENSORBOARD_PID 2>/dev/null; then
    echo "✅ TensorBoard started successfully!"
    echo "🌐 Opening browser..."
    
    # Open browser based on OS
    if command -v xdg-open &> /dev/null; then
        # Linux
        xdg-open http://localhost:6006
    elif command -v open &> /dev/null; then
        # macOS
        open http://localhost:6006
    else
        # Fallback
        echo "   Please manually open: http://localhost:6006"
    fi
    
    echo ""
    echo "============================================================"
    echo "📊 TensorBoard is now running!"
    echo "   URL: http://localhost:6006"
    echo "   Press Ctrl+C to stop TensorBoard"
    echo "============================================================"
    
    # Wait for user to stop
    echo ""
    echo "💡 Tips:"
    echo "   - If browser doesn't open automatically, manually visit: http://localhost:6006"
    echo "   - To use a different port, edit this script and change --port=6006"
    echo "   - To stop TensorBoard, press Ctrl+C"
    echo ""
    
    # Wait for interrupt
    trap "echo '🛑 Stopping TensorBoard...'; kill $TENSORBOARD_PID; echo '✅ TensorBoard stopped.'; exit" INT
    wait $TENSORBOARD_PID
else
    echo "❌ Failed to start TensorBoard"
    exit 1
fi 