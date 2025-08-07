@echo off
REM Simple Interactive File Tree Crawler Batch Script
REM This script runs the Python file tree crawler in interactive mode

echo Starting Interactive File Tree Crawler...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.6 or higher
    pause
    exit /b 1
)

REM Run the file tree crawler in interactive mode
python file_tree_crawler.py --interactive

echo.
pause
