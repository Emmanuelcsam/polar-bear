@echo off
REM File Tree Crawler Batch Script
REM This script runs the Python file tree crawler with interactive prompts

echo ============================================
echo    INTERACTIVE FILE TREE CRAWLER
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.6 or higher
    pause
    exit /b 1
)

echo Using Python command: python
echo.

echo Let's configure your file tree crawler:
echo.

REM Root directory
echo 1. Root Directory
set /p root_dir="Enter the root directory to start crawling from (press Enter for current directory): "
if "%root_dir%"=="" set root_dir=.

REM Output file
echo.
echo 2. Output File
set /p output_file="Enter the output file name (press Enter for 'file_tree_structure.txt'): "
if "%output_file%"=="" set output_file=file_tree_structure.txt

REM Maximum depth
echo.
echo 3. Maximum Depth
set /p max_depth="Enter maximum depth to crawl (press Enter for unlimited): "

REM Include files
echo.
echo 4. File Inclusion
set /p include_files="Include files in the tree structure? (Y/n): "
if "%include_files%"=="" set include_files=Y

REM Include hidden files
echo.
echo 5. Hidden Files
set /p include_hidden="Include hidden files and directories? (y/N): "
if "%include_hidden%"=="" set include_hidden=N

REM Include virtual environments
echo.
echo 6. Virtual Environments
set /p include_venv="Include Python virtual environment directories? (y/N): "
if "%include_venv%"=="" set include_venv=N

REM Include statistics
echo.
echo 7. Statistics
set /p include_stats="Include directory statistics in the output? (Y/n): "
if "%include_stats%"=="" set include_stats=Y

REM Build command arguments
set ARGS=--root "%root_dir%" --output "%output_file%"

if not "%max_depth%"=="" set ARGS=%ARGS% --max-depth %max_depth%

if /i "%include_files%"=="n" set ARGS=%ARGS% --no-files

if /i "%include_hidden%"=="y" set ARGS=%ARGS% --include-hidden

if /i "%include_venv%"=="y" set ARGS=%ARGS% --include-venv

if /i "%include_stats%"=="y" set ARGS=%ARGS% --stats

REM Show configuration summary
echo.
echo ============================================
echo    CONFIGURATION SUMMARY
echo ============================================
echo Root directory: %root_dir%
echo Output file: %output_file%
if "%max_depth%"=="" (
    echo Max depth: unlimited
) else (
    echo Max depth: %max_depth%
)
echo Include files: %include_files%
echo Include hidden: %include_hidden%
echo Include virtual envs: %include_venv%
echo Include statistics: %include_stats%
echo ============================================
echo.

REM Confirm before running
set /p confirm="Run the file tree crawler with these settings? (Y/n): "
if "%confirm%"=="" set confirm=Y

if /i "%confirm%"=="y" (
    echo Running file tree crawler...
    echo Command: python file_tree_crawler.py %ARGS%
    echo.
    
    REM Run the file tree crawler with collected settings
    python file_tree_crawler.py %ARGS%
    
    echo.
    echo ============================================
    echo File tree generation complete!
    echo Check the '%output_file%' file for results.
    echo ============================================
) else (
    echo Operation cancelled.
)

echo.
pause
