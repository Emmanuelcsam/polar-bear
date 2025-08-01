@echo off
echo ========================================
echo    Core Detection Orchestrator
echo ========================================
echo.

echo Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Checking Python installation...
py --version >nul 2>&1
if %errorlevel% neq 0 (
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        python3 --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo ERROR: Python is not installed or not in PATH
            echo Please install Python from https://python.org/
            pause
            exit /b 1
        )
    )
)

echo Checking required files...
if not exist "config.json" (
    echo ERROR: config.json not found
    pause
    exit /b 1
)

if not exist "auto-core-detection.py" (
    echo ERROR: auto-core-detection.py not found
    pause
    exit /b 1
)

if not exist "live_feed.py" (
    echo ERROR: live_feed.py not found
    pause
    exit /b 1
)

if not exist "main.py" (
    echo ERROR: main.py not found
    pause
    exit /b 1
)

echo Installing Node.js dependencies...
npm install

echo.
echo ========================================
echo Starting Core Detection Orchestrator...
echo ========================================
echo.
echo The web monitoring interface will be available at:
echo http://localhost:3000
echo.
echo Starting Pylon Viewer integration...
echo Press Ctrl+C to stop the orchestrator
echo.

REM Start Pylon Viewer integration in background
start /B python pylon_viewer_integration.py

REM Start the main orchestrator
node monitor.js

REM Cleanup Pylon Viewer when orchestrator stops
taskkill /F /IM PylonViewerApp.exe >nul 2>&1

pause 