@echo off
echo =========================================
echo    Core Detection System
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
py -m pip install -r requirements_simple.txt

REM Start the core detector
echo.
echo Starting Core Detector...
py start_core_detector.py

pause 