#!/usr/bin/env python3
"""
Synapse IDE Launcher
Handles dependency checks, installation, and launches the main application.
This is a robust launcher designed for a smooth startup experience.
"""

import sys
import subprocess
import os
from pathlib import Path

# --- Configuration: All required and optional packages ---
# This dictionary ensures all dependencies are checked before launch.
PACKAGES = {
    "required": {
        "tkinter": "tkinter",       # Standard GUI library
        "ttkthemes": "ttkthemes",   # For modern UI styling
        "Pillow": "Pillow",         # For image handling in the UI
        "pyyaml": "PyYAML",         # For human-readable project configurations
        "networkx": "networkx",     # For dependency visualization
        "matplotlib": "matplotlib", # For plotting graphs
        "psutil": "psutil"          # For performance monitoring
    },
    "optional": {
        "pylint": "pylint",         # For advanced code analysis
        "openai": "openai",         # For AI-powered suggestions
        "requests": "requests",     # For the Auto-Healer to search for solutions
        "beautifulsoup4": "beautifulsoup4" # For parsing web content in Auto-Healer
    }
}

def check_python_version():
    """Ensures the Python version is 3.7 or higher for compatibility."""
    print("Step 1: Checking Python version...")
    if sys.version_info < (3, 7):
        print(f"❌ Error: Python 3.7 or higher is required.")
        print(f"   Your version is: {sys.version}")
        return False
    print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible.")
    return True

def get_missing_packages():
    """
    Checks for all required and optional packages.
    Returns two lists: missing required and missing optional packages.
    """
    print("\nStep 2: Checking for required packages...")
    missing_required = []
    for module, package in PACKAGES["required"].items():
        try:
            # Tkinter is a special case
            if module == 'tkinter' and sys.version_info.major == 3:
                import tkinter
            else:
                __import__(module)
        except ImportError:
            # On Linux, tkinter might need a separate install
            if module == 'tkinter':
                missing_required.append('python3-tk') 
            else:
                missing_required.append(package)

    if not missing_required:
        print("✅ All required packages are installed.")
    else:
        print(f"⚠️ Missing required packages: {', '.join(missing_required)}")

    print("\nStep 3: Checking for optional AI & Analysis packages...")
    missing_optional = []
    for module, package in PACKAGES["optional"].items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(package)

    if not missing_optional:
        print("✅ All optional packages are installed.")
    else:
        print(f"⚠️ Missing optional packages: {', '.join(missing_optional)}")

    return missing_required, missing_optional

def install_packages(packages_to_install: list):
    """Installs a list of Python packages using pip."""
    if not packages_to_install:
        return True

    print(f"\nInstalling: {', '.join(packages_to_install)}")
    try:
        # We use sys.executable to ensure we're using the pip for the correct Python interpreter
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
        print("\n✅ Installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pip installation failed: {e}")
        print("   Please try installing the packages manually and run the launcher again.")
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during installation: {e}")
        return False

def launch_app():
    """Finds and launches the main Synapse IDE application."""
    print("\nStep 4: Launching Synapse IDE...")
    print("-" * 50)
    
    script_dir = Path(__file__).parent.resolve()
    main_app_file = script_dir / "synapse_ide_main.py"

    if not main_app_file.exists():
        print(f"❌ Critical Error: Main application file not found!")
        print(f"   Expected at: {main_app_file}")
        print("   Please ensure 'synapse_ide_main.py' is in the same directory.")
        return

    # Add the script's directory to the Python path to ensure local imports work
    sys.path.insert(0, str(script_dir))
    
    try:
        # Dynamically import and run the main application
        from synapse_ide_main import main as run_app
        run_app()
    except ImportError as e:
        print(f"\n❌ Error: Could not import the main application.")
        print(f"   Details: {e}")
        print("   This might be due to a missing dependency or an error in the main script.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred while launching the application:")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run the launcher sequence."""
    print("=" * 50)
    print("      Welcome to the Synapse IDE Launcher")
    print("=" * 50)

    if not check_python_version():
        input("\nPress Enter to exit.")
        sys.exit(1)

    missing_required, missing_optional = get_missing_packages()

    if missing_required:
        prompt = f"\nSome required packages are missing. Install them now? (y/n): "
        if input(prompt).lower() == 'y':
            if not install_packages(missing_required):
                input("\nPress Enter to exit.")
                sys.exit(1)
        else:
            print("\nCannot start without required packages. Exiting.")
            sys.exit(1)

    if missing_optional:
        prompt = f"\nOptional packages for AI features are missing. Install for the best experience? (y/n): "
        if input(prompt).lower() == 'y':
            install_packages(missing_optional)

    launch_app()
    print("-" * 50)
    print("Synapse IDE has closed. Thank you for using it!")

if __name__ == "__main__":
    main()
