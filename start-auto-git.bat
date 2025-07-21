@echo off
REM Startup script for Auto Git Push
REM Place this in Windows Startup folder

REM Hide the window after 5 seconds
echo Starting Auto Git Push in background...
echo This window will minimize in 5 seconds...
timeout /t 5

REM Start the loop script minimized
start /min "Auto Git Push" "C:\Users\Saem1001\Documents\GitHub\polar-bear\auto-git-loop.bat"

exit