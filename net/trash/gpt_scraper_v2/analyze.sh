#!/bin/bash
# One-click ChatGPT Analyzer for Mac/Linux
# Just run: ./analyze.sh

echo "====================================================="
echo "   ChatGPT Analyzer - Easy Mode!"
echo "====================================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating Python environment..."
    source venv/bin/activate
else
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python 3 is not installed!"
        echo ""
        echo "Please install Python 3 first:"
        echo "  Mac: brew install python3"
        echo "  Ubuntu/Debian: sudo apt-get install python3"
        echo "  Or run ./install_deps.sh first"
        echo ""
        exit 1
    fi
fi

# Check if main script exists
if [ ! -f "gptscraper.py" ]; then
    echo "ERROR: gptscraper.py not found!"
    echo ""
    echo "Make sure you're running this from the ChatGPT Analyzer folder"
    echo ""
    exit 1
fi

# Run the analyzer in interactive mode
echo "Starting ChatGPT Analyzer..."
echo ""
python3 gptscraper.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "====================================================="
    echo "There was an error. Check the message above."
    read -p "Press Enter to close..."
fi
