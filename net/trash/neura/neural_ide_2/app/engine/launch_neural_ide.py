#!/usr/bin/env python3
"""
Neural Script IDE Launcher
Handles dependency installation and launches the main application
"""

import sys
import subprocess
import os
from pathlib import Path
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_requirements():
    """Check if required packages are installed"""
    required = {
        'tkinter': 'tkinter',
        'pylint': 'pylint',
        'networkx': 'networkx',
        'matplotlib': 'matplotlib',
        'psutil': 'psutil',
        'numpy': 'numpy',
        'yaml': 'pyyaml',
        'websockets': 'websockets'
    }
    
    optional = {
        'flake8': 'flake8',
        'mypy': 'mypy',
        'bandit': 'bandit',
        'radon': 'radon',
        'vulture': 'vulture',
        'openai': 'openai'
    }
    
    missing = []
    optional_missing = []
    
    # Check required packages
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    # Check optional packages
    for module, package in optional.items():
        try:
            __import__(module)
        except ImportError:
            optional_missing.append(package)
            
    return missing, optional_missing

def install_requirements(packages):
    """Install required packages"""
    if not packages:
        return True
        
    print(f"\nInstalling required packages: {', '.join(packages)}")
    print("This may take a few minutes...\n")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install packages
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        
        print("\n✓ Installation complete!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if platform.system() != "Windows":
        return
        
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "Neural Script IDE.lnk")
        target = sys.executable
        wDir = os.path.dirname(os.path.abspath(__file__))
        args = os.path.abspath(__file__)
        icon = target
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.Arguments = args
        shortcut.WorkingDirectory = wDir
        shortcut.IconLocation = icon
        shortcut.save()
        
        print(f"Desktop shortcut created: {path}")
        
    except:
        pass  # Silently fail if can't create shortcut

def main():
    """Main launcher function"""
    print("=" * 60)
    print("Neural Script IDE Launcher")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Check requirements
    missing, optional_missing = check_requirements()
    
    if missing:
        print(f"\nRequired packages not found: {', '.join(missing)}")
        response = input("Install required packages? (y/n): ")
        
        if response.lower() == 'y':
            if not install_requirements(missing):
                input("\nPress Enter to exit...")
                sys.exit(1)
        else:
            print("\nCannot run without required packages.")
            input("Press Enter to exit...")
            sys.exit(1)
    
    if optional_missing:
        print(f"\nOptional packages not found: {', '.join(optional_missing)}")
        print("These enhance functionality but are not required.")
        response = input("Install optional packages? (y/n): ")
        
        if response.lower() == 'y':
            install_requirements(optional_missing)
    
    # Check if main script exists
    script_dir = Path(__file__).parent
    main_script = script_dir / "neural_script_ide.py"
    
    if not main_script.exists():
        print(f"\nError: Main script not found at {main_script}")
        print("Please ensure neural_script_ide.py is in the same directory as the launcher.")
        
        # Offer to create it
        response = input("\nCreate neural_script_ide.py with the full IDE code? (y/n): ")
        if response.lower() == 'y':
            print("\nPlease save the Neural Script IDE code as 'neural_script_ide.py' in this directory.")
            print("Then run the launcher again.")
        
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Create desktop shortcut on first run
    config_file = Path.home() / ".neural_script_ide_first_run"
    if not config_file.exists():
        create_desktop_shortcut()
        config_file.touch()
    
    # Launch the IDE
    print("\nLaunching Neural Script IDE...")
    print("-" * 60)
    
    try:
        # Import and run
        sys.path.insert(0, str(script_dir))
        from neural_script_ide import main as run_ide
        run_ide()
        
    except ImportError as e:
        print(f"\nError importing main script: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError launching IDE: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Neural Script IDE Launcher")
            print("\nUsage: python launch_neural_ide.py [options]")
            print("\nOptions:")
            print("  --help          Show this help message")
            print("  --no-install    Skip package installation")
            print("  --force-install Force reinstall all packages")
            sys.exit(0)
            
        elif sys.argv[1] == "--no-install":
            # Skip directly to running
            script_dir = Path(__file__).parent
            sys.path.insert(0, str(script_dir))
            from neural_script_ide import main as run_ide
            run_ide()
            sys.exit(0)
            
        elif sys.argv[1] == "--force-install":
            # Force reinstall all packages
            packages = ['pylint', 'networkx', 'matplotlib', 'psutil', 'numpy', 'pyyaml', 'websockets']
            install_requirements(packages)
            
    main()
