@echo off
echo =========================================
echo    Unified Core Detection System
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

REM Start the unified core detector
echo.
echo Starting Unified Core Detector...
echo This will show your camera with manual overlay and automatic detection
echo.
py start_unified_detector.py

pause 