@echo off
REM Rapid Git Change Checker - Checks every 30 seconds
REM Save as: rapid-git-check.bat

set REPO_PATH=C:\Users\Saem1001\Documents\GitHub\polar-bear
set CHECK_INTERVAL=30

echo Rapid Git Checker Started!
echo Checking every %CHECK_INTERVAL% seconds...
echo Press Ctrl+C to stop

:loop
cd /d "%REPO_PATH%"

REM Check for changes silently
git status --porcelain > temp_status.txt
set /p STATUS=<temp_status.txt
del temp_status.txt

if defined STATUS (
    echo.
    echo [%time%] Changes detected! Pushing...
    git add .
    git commit -m "Auto commit: %date% %time%"
    git push
    echo Pushed successfully!
) else (
    echo [%time%] No changes
)

timeout /t %CHECK_INTERVAL% /nobreak > nul
goto loop