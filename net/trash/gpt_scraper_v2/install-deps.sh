#!/bin/bash
# ChatGPT Analyzer - Dependency Installation Script for Linux/Mac
# Production-ready installation with error handling

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Header
echo "======================================================"
echo "   🤖 ChatGPT Analyzer - Dependency Installer"
echo "======================================================"
echo ""

# Check if running with appropriate permissions
if [[ $EUID -eq 0 ]]; then
   print_warning "This script is running as root. This is not recommended."
   print_warning "Consider running as a regular user with sudo when needed."
   echo ""
fi

# Detect OS
OS="Unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    # Detect distribution
    if [ -f /etc/debian_version ]; then
        DISTRO="Debian/Ubuntu"
    elif [ -f /etc/redhat-release ]; then
        DISTRO="RedHat/CentOS"
    else
        DISTRO="Unknown"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS ${DISTRO:-}"
echo ""

# Check Python version
print_status "Checking Python installation..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    echo ""
    echo "Please install Python 3.8 or higher:"
    if [[ "$OS" == "Linux" ]]; then
        echo "  sudo apt-get install python3 python3-pip  # Debian/Ubuntu"
        echo "  sudo yum install python3 python3-pip      # RedHat/CentOS"
    elif [[ "$OS" == "macOS" ]]; then
        echo "  brew install python3"
    fi
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

print_success "Python $PYTHON_VERSION found"

# Check pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed!"
    echo "Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi

# Create virtual environment (recommended)
print_status "Setting up virtual environment..."

VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists. Using existing environment."
else
    python3 -m venv $VENV_DIR
    print_success "Virtual environment created"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Chrome and ChromeDriver if needed
print_status "Checking Chrome installation..."

install_chrome() {
    if [[ "$OS" == "Linux" ]]; then
        if [[ "$DISTRO" == "Debian/Ubuntu" ]]; then
            print_status "Installing Chrome for Debian/Ubuntu..."
            wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
            sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
            sudo apt-get update
            sudo apt-get install -y google-chrome-stable
        else
            print_warning "Please install Chrome manually for your distribution"
        fi
    elif [[ "$OS" == "macOS" ]]; then
        if command -v brew &> /dev/null; then
            print_status "Installing Chrome via Homebrew..."
            brew install --cask google-chrome
        else
            print_warning "Please install Chrome manually from https://www.google.com/chrome/"
        fi
    fi
}

# Check if Chrome is installed
CHROME_INSTALLED=false
if [[ "$OS" == "Linux" ]]; then
    if command -v google-chrome &> /dev/null || command -v google-chrome-stable &> /dev/null; then
        CHROME_INSTALLED=true
    fi
elif [[ "$OS" == "macOS" ]]; then
    if [ -d "/Applications/Google Chrome.app" ]; then
        CHROME_INSTALLED=true
    fi
fi

if [ "$CHROME_INSTALLED" = true ]; then
    print_success "Chrome is installed"
else
    print_warning "Chrome is not installed"
    read -p "Would you like to install Chrome? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_chrome
    else
        print_warning "Chrome is required for live mode. You can still use export mode."
    fi
fi

# Install system dependencies for OpenCV and PyAutoGUI
print_status "Installing system dependencies..."

if [[ "$OS" == "Linux" ]]; then
    if [[ "$DISTRO" == "Debian/Ubuntu" ]]; then
        print_status "Installing dependencies for Debian/Ubuntu..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-dev \
            python3-tk \
            libopencv-dev \
            tesseract-ocr \
            libtesseract-dev \
            scrot \
            xclip \
            xsel
    elif [[ "$DISTRO" == "RedHat/CentOS" ]]; then
        print_status "Installing dependencies for RedHat/CentOS..."
        sudo yum install -y \
            python3-devel \
            python3-tkinter \
            opencv-devel \
            tesseract \
            tesseract-devel \
            scrot
    fi
elif [[ "$OS" == "macOS" ]]; then
    if command -v brew &> /dev/null; then
        print_status "Installing dependencies via Homebrew..."
        brew install opencv tesseract
    else
        print_warning "Homebrew not found. Some features may not work without manual installation."
    fi
fi

# Install Python packages
echo ""
print_status "Installing Python packages..."
echo ""

# Core packages
print_status "Installing core dependencies..."
pip install selenium webdriver-manager beautifulsoup4 requests

# CAPTCHA handling
print_status "Installing CAPTCHA handler dependencies..."
pip install opencv-python pillow pyautogui numpy

# Optional but recommended
print_status "Installing optional dependencies..."
pip install pytesseract reportlab pandas

# Development tools
print_status "Installing development tools..."
pip install ipython black flake8 pytest

# Verify installations
echo ""
print_status "Verifying installations..."
echo ""

# Function to check package
check_package() {
    if python3 -c "import $1" 2>/dev/null; then
        VERSION=$(python3 -c "import $1; print($1.__version__ if hasattr($1, '__version__') else 'installed')")
        print_success "$1: $VERSION"
        return 0
    else
        print_error "$1: NOT INSTALLED"
        return 1
    fi
}

# Check all packages
PACKAGES=(selenium cv2 PIL pyautogui bs4 reportlab pytesseract pandas numpy)
FAILED_PACKAGES=()

for package in "${PACKAGES[@]}"; do
    if ! check_package $package; then
        FAILED_PACKAGES+=($package)
    fi
done

# Create run script
print_status "Creating run script..."

cat > run_analyzer.sh << 'EOF'
#!/bin/bash
# Activate virtual environment and run analyzer

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run the analyzer with all arguments passed to this script
python3 gptscraper.py "$@"

# Deactivate virtual environment
deactivate
EOF

chmod +x run_analyzer.sh
print_success "Created run_analyzer.sh"

# Create desktop entry for Linux
if [[ "$OS" == "Linux" ]]; then
    print_status "Creating desktop entry..."
    
    DESKTOP_FILE="$HOME/.local/share/applications/chatgpt-analyzer.desktop"
    mkdir -p "$HOME/.local/share/applications"
    
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ChatGPT Analyzer
Comment=Analyze and extract ChatGPT conversations
Exec=$PWD/run_analyzer.sh
Icon=$PWD/icon.png
Terminal=true
Categories=Utility;Development;
EOF
    
    chmod +x "$DESKTOP_FILE"
    print_success "Created desktop entry"
fi

# Summary
echo ""
echo "======================================================"
echo "   ✅ Installation Complete!"
echo "======================================================"
echo ""

if [ ${#FAILED_PACKAGES[@]} -eq 0 ]; then
    print_success "All packages installed successfully!"
else
    print_warning "Some packages failed to install: ${FAILED_PACKAGES[*]}"
    echo "You may need to install them manually."
fi

echo ""
echo "📋 Next steps:"
echo ""
echo "1. Make sure these files are in the same directory:"
echo "   - gptscraper.py (main script)"
echo "   - captcha_handler.py (CAPTCHA handler)"
echo ""
echo "2. Run the analyzer using the convenience script:"
echo "   ./run_analyzer.sh --help"
echo ""
echo "   Or activate the virtual environment manually:"
echo "   source venv/bin/activate"
echo "   python gptscraper.py --help"
echo ""
echo "3. For export mode (recommended for first use):"
echo "   a. Go to ChatGPT → Settings → Data Controls"
echo "   b. Export your data (you'll receive it via email)"
echo "   c. Extract the ZIP and locate conversations.json"
echo "   d. Run: ./run_analyzer.sh --mode export --file conversations.json"
echo ""
echo "4. For live mode (requires Chrome):"
echo "   ./run_analyzer.sh --mode live --email your@email.com"
echo ""

# Deactivate virtual environment
deactivate

print_success "Setup complete! Happy analyzing! 🎉"
