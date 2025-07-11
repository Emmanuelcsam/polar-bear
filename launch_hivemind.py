#!/usr/bin/env python3
"""
Launch script for Polar Bear Hivemind System
Handles requirements checking and easy startup
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_python_version():
    """Ensure Python 3.6+ is being used"""
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)

def install_requirements():
    """Install required packages"""
    print("Checking and installing requirements...")
    
    required = ['psutil', 'colorama', 'requests']
    optional = ['watchdog']
    
    missing_required = []
    
    # Check required packages
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_required.append(package)
            print(f"✗ {package} is NOT installed (required)")
    
    # Check optional packages
    for package in optional:
        try:
            __import__(package)
            print(f"✓ {package} is installed (optional)")
        except ImportError:
            print(f"- {package} is not installed (optional - some features may be limited)")
    
    if missing_required:
        print(f"\nMissing required packages: {', '.join(missing_required)}")
        print("\nThis appears to be an externally managed environment.")
        print("Please install the missing packages using one of these methods:")
        print(f"  1. System package manager: sudo apt install python3-{missing_required[0]}")
        print(f"  2. User install: python3 -m pip install --user {' '.join(missing_required)}")
        print(f"  3. Virtual environment: python3 -m venv venv && source venv/bin/activate && pip install {' '.join(missing_required)}")
        return False
    
    return True

def check_existing_connectors():
    """Check if there are existing connector scripts"""
    root = Path.cwd()
    connector_count = 0
    
    for path in root.rglob('hivemind_connector.py'):
        connector_count += 1
        
    if connector_count > 0:
        print(f"\nFound {connector_count} existing connector scripts.")
        response = input("Do you want to remove them before starting? (y/n): ").strip().lower()
        if response == 'y':
            remove_existing_connectors()
            
def remove_existing_connectors():
    """Remove existing connector scripts"""
    root = Path.cwd()
    removed = 0
    
    for path in root.rglob('hivemind_connector.py'):
        try:
            path.unlink()
            removed += 1
        except Exception as e:
            print(f"Failed to remove {path}: {e}")
            
    print(f"Removed {removed} connector scripts")
    
def create_directories():
    """Create necessary directories"""
    dirs = ['hivemind_logs']
    
    for dir_name in dirs:
        path = Path(dir_name)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"Created directory: {dir_name}")

def show_banner():
    """Display startup banner"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              POLAR BEAR HIVEMIND SYSTEM v2.0                  ║
║                                                               ║
║  Deep Recursive Connector Network for Complete Project        ║
║  Coverage - Connectors in ALL Subdirectories                  ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def estimate_deployment_time():
    """Estimate deployment time based on directory count"""
    root = Path.cwd()
    dir_count = sum(1 for _ in root.rglob('*') if _.is_dir())
    
    # Skip count for common virtual env directories
    skip_patterns = ['venv', 'env', '__pycache__', '.git', 'node_modules']
    skip_count = sum(1 for d in root.rglob('*') if d.is_dir() and any(p in d.name.lower() for p in skip_patterns))
    
    actual_dirs = dir_count - skip_count
    estimated_time = actual_dirs * 0.1  # Roughly 0.1 seconds per directory
    
    print(f"\nDirectory Analysis:")
    print(f"  Total directories: {dir_count}")
    print(f"  Directories to skip: {skip_count}")
    print(f"  Connectors to deploy: {actual_dirs}")
    print(f"  Estimated deployment time: {estimated_time:.1f} seconds\n")
    
    return actual_dirs

def main():
    """Main launch function"""
    # Check Python version
    check_python_version()
    
    # Show banner
    show_banner()
    
    # Check for quick mode
    quick_mode = '--quick' in sys.argv or '-q' in sys.argv
    
    if not quick_mode:
        # Check existing connectors
        check_existing_connectors()
        
        # Estimate deployment
        dir_count = estimate_deployment_time()
        
        if dir_count > 1000:
            print("WARNING: Large number of directories detected!")
            print("This deployment may take several minutes.")
            response = input("Continue? (y/n): ").strip().lower()
            if response != 'y':
                print("Deployment cancelled")
                return
    
    # Install requirements
    if not install_requirements():
        print("\nFailed to install all requirements")
        print("Please install them manually and try again")
        return
        
    # Create directories
    create_directories()
    
    # Launch the hivemind
    print("\nStarting Polar Bear Hivemind...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Import and run the hivemind
        from polar_bear_hivemind import PolarBearHivemind
        
        hivemind = PolarBearHivemind()
        hivemind.run()
        
    except ImportError:
        print("Error: polar_bear_hivemind.py not found")
        print("Make sure the file exists in the current directory")
    except KeyboardInterrupt:
        print("\n\nHivemind stopped by user")
    except Exception as e:
        print(f"\nError running hivemind: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()