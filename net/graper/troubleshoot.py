#!/usr/bin/env python3
"""
ChatGPT Analyzer - Troubleshooting Helper
Diagnoses common issues and provides solutions
"""

import os
import sys
import platform
import subprocess
import socket
from pathlib import Path


def check_port_availability(port=5000):
    """Check if the default port is available"""
    print(f"\n🔌 Checking port {port} availability...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result == 0:
        print(f"❌ Port {port} is already in use!")
        print("\nSolutions:")
        print(f"1. Kill the process using port {port}")
        print("2. Use a different port: PORT=8080 python app.py")
        
        # Try to find what's using the port
        if platform.system() != "Windows":
            try:
                output = subprocess.check_output(f"lsof -i :{port}", shell=True).decode()
                print(f"\nProcess using port {port}:")
                print(output)
            except:
                pass
        return False
    else:
        print(f"✅ Port {port} is available")
        return True


def check_chrome_installation():
    """Check if Chrome is installed"""
    print("\n🌐 Checking Chrome installation...")
    
    chrome_paths = {
        "Windows": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
        ],
        "Darwin": [  # macOS
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        ],
        "Linux": [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium"
        ]
    }
    
    system = platform.system()
    paths = chrome_paths.get(system, [])
    
    found = False
    for path in paths:
        if os.path.exists(path):
            print(f"✅ Chrome found at: {path}")
            found = True
            break
            
    if not found:
        print("❌ Chrome not found!")
        print("\nSolution: Download Chrome from https://www.google.com/chrome/")
        
        # Check for Chromium as alternative
        if system == "Linux":
            try:
                subprocess.check_output("which chromium", shell=True)
                print("✅ Chromium found (alternative to Chrome)")
                found = True
            except:
                pass
                
    return found


def check_chromedriver():
    """Check ChromeDriver availability"""
    print("\n🔧 Checking ChromeDriver...")
    
    # Check if webdriver-manager is installed
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        print("✅ webdriver-manager is installed (will download driver automatically)")
        return True
    except ImportError:
        print("⚠️  webdriver-manager not installed")
        
    # Check system PATH
    try:
        subprocess.check_output("chromedriver --version", shell=True)
        print("✅ ChromeDriver found in PATH")
        return True
    except:
        print("❌ ChromeDriver not found in PATH")
        print("\nSolutions:")
        print("1. Install webdriver-manager: pip install webdriver-manager")
        print("2. Download ChromeDriver: https://chromedriver.chromium.org/")
        return False


def check_permissions():
    """Check file permissions"""
    print("\n🔐 Checking file permissions...")
    
    issues = []
    
    # Check if we can write to current directory
    try:
        test_file = Path("test_write.tmp")
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Can write to current directory")
    except:
        issues.append("Cannot write to current directory")
        
    # Check Python scripts are readable
    for script in ['app.py', 'captcha_handler.py']:
        if Path(script).exists():
            if not os.access(script, os.R_OK):
                issues.append(f"Cannot read {script}")
                
    if issues:
        print("❌ Permission issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nSolution: Check file permissions or run from a different directory")
        return False
    else:
        print("✅ All permissions OK")
        return True


def check_virtual_environment():
    """Check if running in a virtual environment"""
    print("\n🐍 Checking virtual environment...")
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print("✅ Running in virtual environment")
        print(f"   Python: {sys.executable}")
    else:
        print("⚠️  Not running in virtual environment")
        print("\nRecommended: Create a virtual environment")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   venv\\Scripts\\activate   # Windows")
        
    return True


def check_dependencies_version():
    """Check versions of key dependencies"""
    print("\n📦 Checking dependency versions...")
    
    packages = {
        'flask': '3.0.0',
        'selenium': '4.0.0',
        'flask_sock': '0.5.0'
    }
    
    for package, min_version in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {package} version: {version}")
        except ImportError:
            print(f"❌ {package} not installed")
            
    return True


def test_chatgpt_connectivity():
    """Test if we can reach ChatGPT"""
    print("\n🌍 Testing ChatGPT connectivity...")
    
    try:
        import urllib.request
        response = urllib.request.urlopen('https://chat.openai.com', timeout=5)
        print("✅ Can reach chat.openai.com")
        return True
    except Exception as e:
        print("❌ Cannot reach chat.openai.com")
        print(f"   Error: {e}")
        print("\nPossible issues:")
        print("- No internet connection")
        print("- ChatGPT is down")
        print("- Firewall blocking connection")
        return False


def suggest_fixes():
    """Provide common fixes"""
    print("\n💡 Common Fixes:")
    print("\n1. **Reinstall dependencies:**")
    print("   pip install -r requirements.txt --force-reinstall")
    
    print("\n2. **Update Chrome:**")
    print("   Make sure Chrome is up to date")
    
    print("\n3. **Clear cache:**")
    print("   rm -rf __pycache__ .cache")
    
    print("\n4. **Try without headless mode:**")
    print("   Uncheck 'Run in background' to see what's happening")
    
    print("\n5. **Check ChatGPT login:**")
    print("   - Make sure email/password are correct")
    print("   - Ensure 2FA is not enabled")
    print("   - Try logging in manually first")


def main():
    """Run all troubleshooting checks"""
    print("=" * 60)
    print("🔍 ChatGPT Analyzer - Troubleshooting")
    print("=" * 60)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("=" * 60)
    
    checks = [
        ("Port Availability", check_port_availability),
        ("Chrome Installation", check_chrome_installation),
        ("ChromeDriver", check_chromedriver),
        ("File Permissions", check_permissions),
        ("Virtual Environment", check_virtual_environment),
        ("Dependency Versions", check_dependencies_version),
        ("ChatGPT Connectivity", test_chatgpt_connectivity)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Check '{name}' failed with error: {e}")
            results.append((name, False))
            
    # Summary
    print("\n" + "=" * 60)
    print("📊 TROUBLESHOOTING SUMMARY")
    print("=" * 60)
    
    issues = [(name, result) for name, result in results if not result]
    
    if not issues:
        print("\n✅ No issues found! Everything should work.")
        print("\nIf you're still having problems:")
        print("1. Try running: python test_setup.py")
        print("2. Check the error messages in the browser console (F12)")
        print("3. Run without headless mode to see what's happening")
    else:
        print("\n❌ Issues found:")
        for name, _ in issues:
            print(f"   - {name}")
            
        suggest_fixes()
        
    print("\n📧 Still having issues?")
    print("1. Check if ChatGPT's interface has changed")
    print("2. Try the manual JSON export method as fallback")
    print("3. Review the README.md for more solutions")


if __name__ == "__main__":
    main()