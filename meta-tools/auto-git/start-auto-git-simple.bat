@echo off
echo ==================================
echo    GIT AUTO-PUSHER MENU
echo ==================================
echo.
echo Choose an option:
echo 1. Watch for changes (real-time)
echo 2. Check every 30 minutes
echo 3. Check every 30 seconds
echo 4. Run once now
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting real-time file watcher...
    powershell.exe -ExecutionPolicy Bypass -File "auto-git-watcher.ps1"
) else if "%choice%"=="2" (
    echo Starting 30-minute checker...
    call auto-git-loop.bat
) else if "%choice%"=="3" (
    echo Starting 30-second checker...
    call rapid-git-check.bat
) else if "%choice%"=="4" (
    echo Running once...
    call auto-git-batch.bat
    pause
) else (
    echo Invalid choice!
    pause
)
