@echo off
echo =========================================
echo    Unified Web Core Detection System
echo =========================================
echo.

REM Check if Python is available
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Installing dependencies...
py -m pip install opencv-python numpy flask

REM Start the unified web detector
echo.
echo Starting Unified Web Core Detector...
echo This will open your browser with the live camera feed
echo.
py start_unified_web.py

pause 