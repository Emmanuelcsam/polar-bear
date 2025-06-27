#!/bin/bash
# Install script for ChatGPT Analyzer with CAPTCHA support

echo "🚀 ChatGPT Analyzer - Dependency Installer"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python  is not installed. Please install Python 3 first."
    exit 1
fi

echo "📦 Installing required packages..."
echo ""

# Core dependencies
echo "1️⃣ Installing core dependencies..."
pip install selenium webdriver-manager

# CAPTCHA handling dependencies
echo ""
echo "2️⃣ Installing CAPTCHA handler dependencies..."
pip install opencv-python pillow pyautogui

# Optional but recommended
echo ""
echo "3️⃣ Installing optional dependencies..."
pip install pytesseract reportlab

# Additional useful packages
echo ""
echo "4️⃣ Installing additional utilities..."
pip install beautifulsoup4 requests

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Place both scripts in the same directory:"
echo "   - gptscraper.py (main script)"
echo "   - captcha_handler.py (CAPTCHA handler)"
echo ""
echo "2. Run the analyzer:"
echo "   python gptscraper.py"
echo ""
echo "3. Choose either:"
echo "   - Export mode (easier, no CAPTCHA issues)"
echo "   - Live mode (with automatic CAPTCHA handling)"
echo ""
echo "💡 For Export mode:"
echo "   Go to ChatGPT → Settings → Data Controls → Export data"
echo ""
echo "🤖 Happy analyzing!"
