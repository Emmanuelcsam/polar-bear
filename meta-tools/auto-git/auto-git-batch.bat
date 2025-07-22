@echo off
REM Auto Git Commit and Push Script
REM Save this as: auto-git-push.bat

REM Configuration - CHANGE THIS PATH
set REPO_PATH=C:\Users\Saem1001\Documents\GitHub\polar-bear

REM Navigate to repository
cd /d "%REPO_PATH%"

REM Check if git repository exists
if not exist .git (
    echo Error: Not a git repository!
    pause
    exit /b 1
)

REM Get current date and time for commit message
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YYYY=%dt:~0,4%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "Min=%dt:~10,2%"
set "Sec=%dt:~12,2%"
set "TIMESTAMP=%YYYY%-%MM%-%DD% %HH%:%Min%:%Sec%"

REM Stage all changes
git add .

REM Commit with timestamp
git commit -m "Auto commit: %TIMESTAMP%"

REM Push to remote
git push

echo.
echo Auto git push completed at %TIMESTAMP%
echo.