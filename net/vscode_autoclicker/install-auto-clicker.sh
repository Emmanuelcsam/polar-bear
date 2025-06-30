#!/bin/bash
# VS Code Auto-Clicker Installation Script

set -e

echo "VS Code Copilot Auto-Clicker Installer"
echo "======================================"

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Installing Python dependencies..."
pip3 install opencv-python pyautogui pytesseract python-mss pillow numpy

# Install Tesseract based on OS
if [ "$OS" == "linux" ]; then
    echo "Installing Tesseract OCR..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr python3-tk python3-dev scrot
    elif command -v yum &> /dev/null; then
        sudo yum install -y tesseract
    else
        echo "Warning: Could not install Tesseract automatically. Please install manually."
    fi
elif [ "$OS" == "macos" ]; then
    if command -v brew &> /dev/null; then
        echo "Installing Tesseract OCR..."
        brew install tesseract
    else
        echo "Warning: Homebrew not found. Please install Tesseract manually."
    fi
fi

# Create directory
INSTALL_DIR="$HOME/vscode-auto-clicker"
mkdir -p "$INSTALL_DIR"

# Download the main script
echo "Setting up auto-clicker script..."
cat > "$INSTALL_DIR/vscode_auto_clicker.py" << 'SCRIPT_CONTENT'
[INSERT THE MAIN PYTHON SCRIPT HERE]
SCRIPT_CONTENT

chmod +x "$INSTALL_DIR/vscode_auto_clicker.py"

# Create convenience scripts
echo "Creating launcher scripts..."

# Create start script
cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 vscode_auto_clicker.py
EOF
chmod +x "$INSTALL_DIR/start.sh"

# Create test script
cat > "$INSTALL_DIR/test.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 vscode_auto_clicker.py --test
EOF
chmod +x "$INSTALL_DIR/test.sh"

# Platform specific setup
if [ "$OS" == "macos" ]; then
    echo ""
    echo "IMPORTANT for macOS:"
    echo "1. Grant screen recording permission:"
    echo "   System Preferences → Security & Privacy → Privacy → Screen Recording"
    echo "   Add Terminal or your Python executable"
    echo ""
    echo "2. Grant accessibility permission:"
    echo "   System Preferences → Security & Privacy → Privacy → Accessibility"
    echo "   Add Terminal or your Python executable"
fi

echo ""
echo "Installation complete!"
echo ""
echo "Usage:"
echo "  Start auto-clicker: $INSTALL_DIR/start.sh"
echo "  Test detection:     $INSTALL_DIR/test.sh"
echo "  View logs:          ls ~/.vscode-auto-clicker/"
echo ""
echo "To stop: Press Ctrl+C or move mouse to top-left corner"
echo ""

# Offer to test
read -p "Would you like to test the detection now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$INSTALL_DIR"
    python3 vscode_auto_clicker.py --test
fi