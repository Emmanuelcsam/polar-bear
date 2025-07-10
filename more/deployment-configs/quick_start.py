#!/usr/bin/env python3
"""
Quick Start Script for Guitar Tab Reader
Automatically sets up and launches the system
"""

import os
import sys
import subprocess
import platform
import webbrowser
import time

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════╗
    ║       🎸 Guitar Tab Reader v1.0 🎸        ║
    ║   OCR-Powered Tab Visualization System    ║
    ╚═══════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ is required!")
        print(f"   Your version: {sys.version}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_tesseract():
    """Check if Tesseract is installed"""
    print("\n🔍 Checking Tesseract OCR...")
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Tesseract OCR is not installed!")
    print("\n📥 Installation instructions:")
    
    system = platform.system()
    if system == "Linux":
        print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("   Fedora: sudo dnf install tesseract")
        print("   Arch: sudo pacman -S tesseract")
    elif system == "Darwin":  # macOS
        print("   macOS: brew install tesseract")
        print("   (Install Homebrew first if needed: https://brew.sh)")
    elif system == "Windows":
        print("   Windows: Download installer from:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Make sure to add to PATH during installation!")
    
    return False

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if os.path.exists('requirements.txt'):
        print("   Using requirements.txt...")
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
    else:
        print("   Installing individual packages...")
        packages = [
            'opencv-python',
            'numpy',
            'pytesseract',
            'pdf2image',
            'Pillow',
            'Flask',
            'flask-cors',
            'pytest'
        ]
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ All Python dependencies installed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("   Try running with administrator privileges")
        return False

def create_test_files():
    """Create test files if they don't exist"""
    print("\n📄 Checking test files...")
    
    if not os.path.exists('create_test_tab.py'):
        print("   Test file creator not found")
        return
    
    if not os.path.exists('test_guitar_tab.png'):
        print("   Creating test tablature files...")
        try:
            subprocess.run([sys.executable, 'create_test_tab.py'], check=True)
            print("✅ Test files created!")
        except:
            print("⚠️  Could not create test files")

def run_tests():
    """Run unit tests"""
    print("\n🧪 Running unit tests...")
    try:
        result = subprocess.run([sys.executable, 'guitar_tab_reader.py', '--test'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed (non-critical)")
            return True
    except:
        print("⚠️  Could not run tests (non-critical)")
        return True

def start_server():
    """Start the Flask server"""
    print("\n🚀 Starting Guitar Tab Reader server...")
    print("   Server will start at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server\n")
    
    # Wait a moment then open browser
    time.sleep(2)
    webbrowser.open('http://localhost:5000')
    
    # Start the server
    try:
        subprocess.run([sys.executable, 'guitar_tab_reader.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Thanks for using Guitar Tab Reader!")

def main():
    """Main setup flow"""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    tesseract_ok = check_tesseract()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create test files
    create_test_files()
    
    # Run tests
    run_tests()
    
    if not tesseract_ok:
        print("\n⚠️  WARNING: Tesseract is not installed!")
        print("   The server will start, but OCR features won't work.")
        print("   Install Tesseract and restart for full functionality.\n")
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            sys.exit(0)
    
    # Start server
    print("\n✨ Setup complete! Starting server...")
    time.sleep(1)
    start_server()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the error and try again.")
        sys.exit(1)
