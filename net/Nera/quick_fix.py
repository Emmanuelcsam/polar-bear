#!/usr/bin/env python3
"""
Quick fix script to ensure Neural Nexus IDE runs properly
"""

import os
import sys
from pathlib import Path
import subprocess

def create_file_structure():
    """Ensure proper file structure"""
    # Create static directory
    Path("static").mkdir(exist_ok=True)
    
    # Check if HTML file exists in current directory
    html_files = list(Path(".").glob("*.html"))
    
    if html_files:
        # Move HTML to static/index.html
        html_file = html_files[0]
        target = Path("static/index.html")
        
        if not target.exists():
            print(f"Moving {html_file} to static/index.html")
            html_file.rename(target)
        else:
            print("static/index.html already exists")
    else:
        print("⚠️  No HTML file found. Please save the neural_nexus_ide.html file")
        print("   Then run this script again.")
        return False
    
    return True

def check_server_file():
    """Check if server file exists"""
    server_files = ["neural_nexus_server.py", "server.py", "*server*.py"]
    
    for pattern in server_files:
        files = list(Path(".").glob(pattern))
        if files:
            return files[0]
    
    print("⚠️  Server file not found. Please save neural_nexus_server.py")
    return None

def install_deps():
    """Quick dependency installation"""
    deps = ["fastapi", "uvicorn", "websockets", "aiofiles"]
    
    print("Installing dependencies...")
    cmd = [sys.executable, "-m", "pip", "install"] + deps
    subprocess.run(cmd)

def main():
    print("🔧 Neural Nexus IDE Quick Fix\n")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Fix file structure
    if not create_file_structure():
        sys.exit(1)
    
    # Check server
    server_file = check_server_file()
    if not server_file:
        sys.exit(1)
    
    # Install dependencies
    try:
        import fastapi
        import uvicorn
        print("✅ Dependencies already installed")
    except ImportError:
        install_deps()
    
    # Launch server
    print("\n🚀 Launching Neural Nexus IDE...")
    print("   Server: http://localhost:8765")
    print("   Press Ctrl+C to stop\n")
    
    subprocess.run([sys.executable, str(server_file)])

if __name__ == "__main__":
    main()
