#!/usr/bin/env python3
"""
ChatGPT Analyzer - Installation Test Script
Verifies that all components are properly installed and configured
"""

import sys
import os
import platform
import subprocess
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print a section header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def check_status(condition, success_msg, fail_msg):
    """Print status message based on condition"""
    if condition:
        print(f"{GREEN}✓{RESET} {success_msg}")
        return True
    else:
        print(f"{RED}✗{RESET} {fail_msg}")
        return False

def check_python_version():
    """Check Python version"""
    print_header("Python Version Check")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    is_valid = version.major >= 3 and version.minor >= 8
    
    check_status(
        is_valid,
        f"Python {version_str} - OK (3.8+ required)",
        f"Python {version_str} - Too old! (3.8+ required)"
    )
    
    return is_valid

def check_required_files():
    """Check if required files exist"""
    print_header("Required Files Check")
    
    required_files = [
        ('gptscraper.py', 'Main analyzer script'),
        ('captcha_handler.py', 'CAPTCHA handler module'),
        ('requirements.txt', 'Python dependencies list')
    ]
    
    all_present = True
    for filename, description in required_files:
        exists = os.path.exists(filename)
        all_present &= check_status(
            exists,
            f"{filename} - {description}",
            f"{filename} - MISSING! {description}"
        )
    
    return all_present

def check_python_packages():
    """Check if required Python packages are installed"""
    print_header("Python Packages Check")
    
    packages = {
        # Package name: (import name, description, required)
        'selenium': ('selenium', 'Web automation', True),
        'webdriver-manager': ('webdriver_manager', 'Chrome driver management', True),
        'beautifulsoup4': ('bs4', 'HTML parsing', True),
        'requests': ('requests', 'HTTP requests', True),
        'opencv-python': ('cv2', 'Computer vision for CAPTCHA', True),
        'Pillow': ('PIL', 'Image processing', True),
        'pyautogui': ('pyautogui', 'GUI automation', True),
        'numpy': ('numpy', 'Numerical computing', True),
        'pytesseract': ('pytesseract', 'OCR for CAPTCHA', False),
        'reportlab': ('reportlab', 'PDF generation', False),
        'pandas': ('pandas', 'Data analysis', False),
    }
    
    all_required_installed = True
    
    for package, (import_name, description, required) in packages.items():
        try:
            __import__(import_name)
            version = get_package_version(import_name)
            check_status(True, f"{package} {version} - {description}", "")
        except ImportError:
            status = check_status(
                False, 
                "", 
                f"{package} - MISSING! {description} {'(REQUIRED)' if required else '(optional)'}"
            )
            if required:
                all_required_installed = False
    
    return all_required_installed

def get_package_version(package_name):
    """Get version of installed package"""
    try:
        if package_name == 'cv2':
            import cv2
            return f"({cv2.__version__})"
        elif package_name == 'PIL':
            import PIL
            return f"({PIL.__version__})"
        else:
            module = __import__(package_name)
            if hasattr(module, '__version__'):
                return f"({module.__version__})"
    except:
        pass
    return ""

def check_chrome_installation():
    """Check if Chrome or Chromium is installed"""
    print_header("Chrome/Chromium Browser Check")
    
    chrome_found = False
    chromium_found = False
    chrome_path = None
    chromium_path = None
    
    if platform.system() == 'Windows':
        possible_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                chrome_found = True
                chrome_path = path
                break
    elif platform.system() == 'Darwin':  # macOS
        if os.path.exists('/Applications/Google Chrome.app'):
            chrome_found = True
            chrome_path = '/Applications/Google Chrome.app'
        if os.path.exists('/Applications/Chromium.app'):
            chromium_found = True
            chromium_path = '/Applications/Chromium.app'
    else:  # Linux
        # Check for Chrome
        try:
            result = subprocess.run(['which', 'google-chrome'], capture_output=True, text=True)
            if result.returncode == 0:
                chrome_found = True
                chrome_path = result.stdout.strip()
        except:
            pass
            
        # Check for Chromium
        for cmd in ['chromium-browser', 'chromium']:
            try:
                result = subprocess.run(['which', cmd], capture_output=True, text=True)
                if result.returncode == 0:
                    chromium_found = True
                    chromium_path = result.stdout.strip()
                    break
            except:
                pass
    
    if chrome_found:
        check_status(
            True,
            f"Chrome installed at: {chrome_path}",
            ""
        )
    
    if chromium_found:
        check_status(
            True,
            f"Chromium installed at: {chromium_path}",
            ""
        )
        
    if not chrome_found and not chromium_found:
        check_status(
            False,
            "",
            "Neither Chrome nor Chromium found - Required for live mode (export mode will work)"
        )
    
    return chrome_found or chromium_found

def check_chromedriver():
    """Check if ChromeDriver is available"""
    print_header("ChromeDriver Check")
    
    # Check if webdriver-manager is installed
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        check_status(
            True,
            "webdriver-manager installed - Will auto-download ChromeDriver",
            ""
        )
        return True
    except ImportError:
        print(f"{YELLOW}!{RESET} webdriver-manager not installed - Checking manual installation...")
    
    # Check manual installation
    chromedriver_found = False
    driver_path = None
    
    # Check for chromedriver and chromium-chromedriver
    for cmd in ['chromedriver', 'chromium-chromedriver']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                chromedriver_found = True
                version = result.stdout.strip()
                check_status(True, f"{cmd} in PATH: {version}", "")
                return True
        except:
            pass
    
    # Check environment variable
    env_path = os.getenv('CHROMEDRIVER_PATH')
    if env_path and os.path.exists(env_path):
        chromedriver_found = True
        check_status(True, f"ChromeDriver at: {env_path}", "")
        return True
    
    # Check common paths
    common_paths = [
        '/usr/lib/chromium-browser/chromedriver',
        '/usr/bin/chromedriver',
        '/usr/local/bin/chromedriver',
        '/snap/bin/chromium.chromedriver'
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            check_status(True, f"ChromeDriver found at: {path}", "")
            print(f"  {YELLOW}Tip:{RESET} Add to .env file: CHROMEDRIVER_PATH={path}")
            return True
    
    check_status(
        False,
        "",
        "ChromeDriver not found - Install webdriver-manager or download manually"
    )
    
    print(f"\n  {YELLOW}For Ubuntu with Chromium:{RESET}")
    print("    sudo apt-get install chromium-chromedriver")
    
    return False

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print_header("Tesseract OCR Check (Optional)")
    
    tesseract_found = False
    
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            tesseract_found = True
            version_line = result.stdout.split('\n')[0]
            check_status(True, f"Tesseract installed: {version_line}", "")
    except:
        pass
    
    if not tesseract_found:
        check_status(
            False,
            "",
            "Tesseract not installed - OCR features disabled (optional)"
        )
        print(f"  {YELLOW}Install with:{RESET}")
        if platform.system() == 'Windows':
            print("    Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        elif platform.system() == 'Darwin':
            print("    brew install tesseract")
        else:
            print("    sudo apt-get install tesseract-ocr  # Debian/Ubuntu")
            print("    sudo yum install tesseract          # RedHat/CentOS")
    
    return tesseract_found

def check_environment_variables():
    """Check environment variables and .env file"""
    print_header("Environment Configuration Check")
    
    # Check for .env file
    if os.path.exists('.env'):
        check_status(True, ".env file found", "")
        
        # Check if python-dotenv is installed
        try:
            import dotenv
            check_status(True, "python-dotenv installed - .env will be loaded", "")
        except ImportError:
            check_status(False, "", "python-dotenv not installed - .env won't be loaded")
            print(f"  {YELLOW}Install with:{RESET} pip install python-dotenv")
    else:
        print(f"{YELLOW}!{RESET} No .env file found")
        print(f"  {YELLOW}Create one with:{RESET} python setup_env.py")
        print(f"  {YELLOW}Or manually:{RESET} cp .env.example .env")
    
    print("")
    
    env_vars = {
        'CHATGPT_EMAIL': 'Email for live mode login',
        'CHATGPT_PASSWORD': 'Password for live mode login',
        'CHROMEDRIVER_PATH': 'Path to ChromeDriver (optional)',
        'OUTPUT_DIR': 'Default output directory',
        'DEFAULT_FORMATS': 'Default report formats'
    }
    
    any_set = False
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            masked_value = value[:3] + '*' * (len(value) - 3) if 'PASSWORD' in var else value
            print(f"{GREEN}✓{RESET} {var} = {masked_value} - {description}")
            any_set = True
        else:
            print(f"{YELLOW}!{RESET} {var} not set - {description}")
    
    return True

def test_imports():
    """Test importing the main modules"""
    print_header("Module Import Test")
    
    try:
        import gptscraper
        check_status(True, "gptscraper module imports successfully", "")
        
        # Check main classes
        classes = ['ChatGPTAnalyzer', 'ConversationType', 'ConversationMetadata']
        for class_name in classes:
            if hasattr(gptscraper, class_name):
                check_status(True, f"  - {class_name} class available", "")
            else:
                check_status(False, "", f"  - {class_name} class MISSING")
                
    except Exception as e:
        check_status(False, "", f"Failed to import gptscraper: {e}")
        return False
    
    try:
        import captcha_handler
        check_status(True, "captcha_handler module imports successfully", "")
        
        if hasattr(captcha_handler, 'CaptchaHandler'):
            check_status(True, "  - CaptchaHandler class available", "")
        if hasattr(captcha_handler, 'integrate_with_scraper'):
            check_status(True, "  - Integration function available", "")
            
    except Exception as e:
        check_status(False, "", f"Failed to import captcha_handler: {e}")
        return False
    
    return True

def check_permissions():
    """Check file permissions"""
    print_header("File Permissions Check")
    
    # Check if we can create output directory
    try:
        test_dir = 'test_output_dir'
        os.makedirs(test_dir, exist_ok=True)
        
        # Try to write a file
        test_file = os.path.join(test_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        # Clean up
        os.remove(test_file)
        os.rmdir(test_dir)
        
        check_status(True, "Can create directories and files", "")
        return True
    except Exception as e:
        check_status(False, "", f"Permission error: {e}")
        return False

def generate_summary(results):
    """Generate installation summary"""
    print_header("Installation Summary")
    
    all_good = all(results.values())
    
    if all_good:
        print(f"{GREEN}✅ All checks passed! The analyzer is ready to use.{RESET}\n")
        print("Next steps:")
        print("1. For export mode:")
        print("   python gptscraper.py --mode export --file conversations.json")
        print("\n2. For live mode:")
        print("   python gptscraper.py --mode live --email your@email.com")
    else:
        print(f"{YELLOW}⚠️  Some checks failed. The analyzer may have limited functionality.{RESET}\n")
        
        # Determine what will work
        if results['python_version'] and results['required_files'] and results['python_packages']:
            print(f"{GREEN}✓ Export mode should work{RESET}")
        else:
            print(f"{RED}✗ Export mode may not work properly{RESET}")
            
        if results['chrome'] and (results['chromedriver'] or results['python_packages']):
            print(f"{GREEN}✓ Live mode should work{RESET}")
        else:
            print(f"{YELLOW}! Live mode will not work without Chrome and ChromeDriver{RESET}")
            
    print("\nFor detailed setup instructions, see README.md")

def main():
    """Run all installation tests"""
    print("🔧 ChatGPT Analyzer - Installation Test")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    results = {
        'python_version': check_python_version(),
        'required_files': check_required_files(),
        'python_packages': check_python_packages(),
        'chrome': check_chrome_installation(),
        'chromedriver': check_chromedriver(),
        'tesseract': check_tesseract(),
        'env_vars': check_environment_variables(),
        'imports': test_imports(),
        'permissions': check_permissions()
    }
    
    generate_summary(results)
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
