@echo off
echo =========================================
echo    Visual Core Detection System
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

REM Start the web-based core detector
echo.
echo Starting Visual Core Detector...
echo The web interface will open in your browser
echo.
py start_web_detector.py

pause 