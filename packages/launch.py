#!/usr/bin/env python3
"""
Simple launcher for the Enhanced Multi-Script Runner
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import pylint
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pylint"])
    print("Installation complete!")

def main():
    # Check if running with uv
    if "uv" in sys.executable or os.environ.get("UV_PROJECT_ROOT"):
        print("Detected uv environment")
        
    # Check requirements
    if not check_requirements():
        print("Required packages not found.")
        response = input("Install requirements? (y/n): ")
        if response.lower() == 'y':
            install_requirements()
        else:
            print("Cannot run without requirements. Exiting.")
            sys.exit(1)
    
    # Import and run the main application
    try:
        # Save the main script if it doesn't exist
        script_path = Path(__file__).parent / "ide.py"
        if not script_path.exists():
            print("Main script not found. Please ensure ide.py is in the same directory.")
            sys.exit(1)
            
        # Import and run
        sys.path.insert(0, str(script_path.parent))
        from ide import main as run_app
        run_app()
        
    except Exception as e:
        print(f"Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
