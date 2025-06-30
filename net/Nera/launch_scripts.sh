#!/bin/bash
# launch.sh - Launch Neural Nexus IDE on Mac/Linux

echo "🚀 Launching Neural Nexus IDE..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Run the setup script
python3 setup.py --start

# --- Windows Batch File (save as launch.bat) ---
: '
@echo off
echo 🚀 Launching Neural Nexus IDE...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed.
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

:: Run the setup script
python setup.py --start

pause
'