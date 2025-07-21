@echo off
REM Continuous Auto Git Push Script with Timer
REM Save this as: auto-git-loop.bat

REM Configuration
set REPO_PATH=C:\Users\Saem1001\Documents\GitHub\polar-bear
set INTERVAL_SECONDS=1800
REM 1800 = 30 minutes, 3600 = 1 hour, 7200 = 2 hours, 300 = 5 minutes

echo Starting Auto Git Push Loop...
echo Repository: %REPO_PATH%
echo Interval: %INTERVAL_SECONDS% seconds
echo.
echo Press Ctrl+C to stop
echo.

:loop
REM Navigate to repository
cd /d "%REPO_PATH%"

REM Check if git repository exists
if not exist .git (
    echo Error: Not a git repository!
    pause
    exit /b 1
)

REM Get current timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YYYY=%dt:~0,4%"
set "MM=%dt:~4,2%"
set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%"
set "Min=%dt:~10,2%"
set "Sec=%dt:~12,2%"
set "TIMESTAMP=%YYYY%-%MM%-%DD% %HH%:%Min%:%Sec%"

echo [%TIMESTAMP%] Checking for changes...

REM Check if there are changes
git status --porcelain > temp_status.txt
set /p STATUS=<temp_status.txt
del temp_status.txt

if defined STATUS (
    echo Changes detected! Committing and pushing...
    
    REM Stage all changes
    git add .
    
    REM Commit with timestamp
    git commit -m "Auto commit: %TIMESTAMP%"
    
    REM Push to remote
    git push
    
    echo Successfully pushed changes!
) else (
    echo No changes to commit.
)

echo.
echo Waiting %INTERVAL_SECONDS% seconds until next check...
echo.

REM Wait for the specified interval
timeout /t %INTERVAL_SECONDS% /nobreak

REM Loop back
goto loop