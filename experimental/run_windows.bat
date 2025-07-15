@echo off
REM Windows batch script to run the geometry detection applications

echo ========================================
echo Real-time Geometry Detection System
echo ========================================
echo.
echo Available applications:
echo 1. Circle Detector (Optimized for Basler cameras)
echo 2. Geometry Demo (General shape detection)
echo 3. Calibration Tool (Interactive calibration)
echo 4. Run Tests
echo 5. Install Dependencies
echo 6. Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo Starting Circle Detector...
    python run_circle_detector.py
) else if "%choice%"=="2" (
    echo Starting Geometry Demo...
    python run_geometry_demo.py
) else if "%choice%"=="3" (
    echo Starting Calibration Tool...
    python run_calibration.py
) else if "%choice%"=="4" (
    echo Running tests...
    python -m pytest src/tests/ -v
) else if "%choice%"=="5" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
    echo Installation complete!
    pause
    run_windows.bat
) else if "%choice%"=="6" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    pause
    run_windows.bat
)

pause