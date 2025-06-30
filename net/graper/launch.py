#!/usr/bin/env python3
"""
ChatGPT Analyzer Launcher
Simple script to launch the web application with automatic setup
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)


def check_and_install_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    # Check if flask is installed
    try:
        import flask
        import flask_cors
        import pandas
        print("✅ All core dependencies are installed")
        return True
    except ImportError:
        print("📦 Installing required dependencies...")
        
        # Install core dependencies
        core_deps = ['flask', 'flask-cors', 'pandas']
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', 'pip'
            ])
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + core_deps)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("   Please run manually: pip install flask flask-cors pandas")
            return False


def check_files():
    """Check if required files exist"""
    required_files = ['app.py', 'index.html']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("   Please ensure all files are in the same directory")
        return False
    
    return True


def find_free_port(start_port=5000):
    """Find a free port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    
    return start_port


def launch_server():
    """Launch the Flask server"""
    port = find_free_port()
    url = f"http://localhost:{port}"
    
    print(f"\n🚀 Starting ChatGPT Analyzer...")
    print(f"   Server URL: {url}")
    print("\n📝 Instructions:")
    print("   1. The browser will open automatically")
    print("   2. Upload your conversations.json file")
    print("   3. View and export your analysis")
    print("\n⚠️  To stop the server: Press Ctrl+C\n")
    
    # Set environment variables
    env = os.environ.copy()
    env['PORT'] = str(port)
    env['FLASK_APP'] = 'app.py'
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open(url)
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the server
    try:
        subprocess.run([sys.executable, 'app.py'], env=env)
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")
        print("👋 Thank you for using ChatGPT Analyzer!")


def main():
    """Main launcher function"""
    print("="*50)
    print("🤖 ChatGPT Conversation Analyzer")
    print("="*50)
    
    # Check Python version
    check_python_version()
    
    # Check required files
    if not check_files():
        sys.exit(1)
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        sys.exit(1)
    
    # Launch server
    try:
        launch_server()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure no other application is using port 5000")
        print("2. Try running directly: python app.py")
        print("3. Check that all files are in the same directory")
        sys.exit(1)


if __name__ == "__main__":
    main()