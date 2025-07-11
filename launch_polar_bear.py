#!/usr/bin/env python3
"""
Polar Bear System Launcher
Quick launch script for the Polar Bear System
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_python_version():
    """Ensure Python 3.7+ is being used"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)

def install_requirements():
    """Install system requirements if needed"""
    print("Checking system requirements...")
    
    req_files = ['polar_bear_requirements.txt', 'requirements.txt', 'requirements_web.txt']
    
    for req_file in req_files:
        if os.path.exists(req_file):
            print(f"\nInstalling requirements from {req_file}...")
            try:
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-r', req_file],
                    check=True
                )
                print(f"✓ {req_file} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {req_file}: {e}")
                answer = input("Continue anyway? (y/n): ")
                if answer.lower() != 'y':
                    sys.exit(1)

def create_directories():
    """Ensure all required directories exist"""
    directories = ['logs', 'config', 'data']
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"Created directory: {directory}")

def launch_system():
    """Launch the main Polar Bear System"""
    print("\n" + "="*60)
    print("LAUNCHING POLAR BEAR SYSTEM")
    print("="*60)
    
    # Check if main script exists
    if not os.path.exists('polar_bear_system.py'):
        print("Error: polar_bear_system.py not found!")
        print("Please ensure you're running from the project root directory.")
        sys.exit(1)
    
    # Launch the system
    try:
        subprocess.run([sys.executable, 'polar_bear_system.py'])
    except KeyboardInterrupt:
        print("\n\nSystem stopped by user")
    except Exception as e:
        print(f"\nError launching system: {e}")
        sys.exit(1)

def quick_start():
    """Quick start mode - skip checks"""
    print("Quick start mode - launching system directly...")
    try:
        subprocess.run([sys.executable, 'polar_bear_system.py'])
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main launcher function"""
    print("\n" + "="*60)
    print("POLAR BEAR SYSTEM LAUNCHER")
    print("="*60)
    
    # Check Python version
    check_python_version()
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            quick_start()
            return
        elif sys.argv[1] == '--help':
            print("\nUsage:")
            print("  python launch_polar_bear.py         # Normal launch with checks")
            print("  python launch_polar_bear.py --quick # Quick launch, skip checks")
            print("  python launch_polar_bear.py --help  # Show this help")
            return
    
    # Normal launch process
    print("\nPreparing to launch Polar Bear System...")
    
    # Create directories
    create_directories()
    
    # Install requirements
    answer = input("\nCheck and install requirements? (y/n) [y]: ").strip()
    if answer.lower() != 'n':
        install_requirements()
    
    # Launch system
    print("\nReady to launch!")
    time.sleep(1)
    launch_system()

if __name__ == "__main__":
    main()