@echo off
REM Install script for ChatGPT Analyzer with CAPTCHA support (Windows)

echo ====================================================
echo   ChatGPT Analyzer - Windows Dependency Installer
echo ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [1/5] Installing core dependencies...
pip install selenium webdriver-manager

echo.
echo [2/5] Installing CAPTCHA handler dependencies...
pip install opencv-python pillow pyautogui

echo.
echo [3/5] Installing optional dependencies...
pip install pytesseract reportlab

echo.
echo [4/5] Installing additional utilities...
pip install beautifulsoup4 requests numpy

echo.
echo [5/5] Verifying installations...
python -c "import selenium; print('✓ Selenium:', selenium.__version__)"
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)"
python -c "import PIL; print('✓ Pillow:', PIL.__version__)"
python -c "import pyautogui; print('✓ PyAutoGUI:', pyautogui.__version__)"

echo.
echo ====================================================
echo   Installation Complete!
echo ====================================================
echo.
echo Next steps:
echo.
echo 1. Make sure these files are in the same folder:
echo    - gptscraper.py (main script)
echo    - captcha_handler.py (CAPTCHA handler)
echo.
echo 2. Run the analyzer:
echo    python gptscraper.py
echo.
echo 3. For Export mode (recommended):
echo    - Go to ChatGPT Settings
echo    - Data Controls - Export data
echo    - Download and extract the ZIP file
echo    - Use conversations.json with the script
echo.
echo Press any key to exit...
pause >nul
