#!/usr/bin/env python3
"""
Neural Weaver Launcher
Handles dependency checks and launches the main application.
This new version is designed to be more robust and user-friendly.
"""

import sys
import subprocess
import os
from pathlib import Path

# --- Configuration ---
# A dictionary of required and optional packages.
# Key: The name used for the import check.
# Value: The package name for installation via pip.
PACKAGES = {
    "required": {
        "tkinter": "tkinter", # Usually built-in
        "networkx": "networkx",
        "matplotlib": "matplotlib",
        "psutil": "psutil",
        "numpy": "numpy",
        "pyyaml": "pyyaml",
        "ttkthemes": "ttkthemes",
        "Pillow": "Pillow", # For image handling in Tkinter
    },
    "optional": {
        "openai": "openai",
        "requests": "requests",
        "beautifulsoup4": "beautifulsoup4",
        "pylint": "pylint"
    }
}

# --- Helper Functions ---

def check_python_version():
    """
    Checks if the current Python version is 3.7 or higher.
    This is crucial for modern language features and library compatibility.
    """
    print("Step 1: Checking Python version...")
    if sys.version_info < (3, 7):
        print(f"❌ Error: Python 3.7 or higher is required.")
        print(f"   Your version is: {sys.version}")
        return False
    print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible.")
    return True

def get_missing_packages():
    """
    Checks for missing required and optional packages.
    Returns two lists: one for missing required packages, one for missing optional.
    """
    print("\nStep 2: Checking for required packages...")
    missing_required = []
    missing_optional = []

    # Check required packages
    for module, package in PACKAGES["required"].items():
        try:
            # Skip checking for tkinter as it's handled differently
            if module != 'tkinter':
                __import__(module)
        except ImportError:
            missing_required.append(package)

    # Special check for tkinter
    try:
        __import__('tkinter')
    except ImportError:
        missing_required.append('python3-tk') # Suggests the common package name on Linux

    if not missing_required:
        print("✅ All required packages are installed.")
    else:
        print(f"⚠️ Missing required packages: {', '.join(missing_required)}")

    # Check optional packages for enhanced features
    print("\nStep 3: Checking for optional packages (for AI and advanced features)...")
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
    """
    Installs a list of packages using pip.
    """
    if not packages_to_install:
        return True

    print(f"\nInstalling packages: {', '.join(packages_to_install)}")
    print("This may take a moment...")
    try:
        # Ensure pip is up-to-date
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        # Install the packages
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
        print("\n✅ Installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during installation: {e}")
        print("Please try installing the packages manually and run the launcher again.")
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return False

def launch_app():
    """
    Finds and launches the main Neural Weaver application.
    """
    print("\nStep 4: Launching Neural Weaver...")
    print("-" * 50)
    script_dir = Path(__file__).parent.resolve()
    main_app_file = script_dir / "neural_weaver_main.py"

    if not main_app_file.exists():
        print(f"❌ Critical Error: Main application file not found!")
        print(f"   Expected at: {main_app_file}")
        print("   Please ensure 'neural_weaver_main.py' is in the same directory as this launcher.")
        return

    # Add the script's directory to the Python path to ensure imports work correctly
    sys.path.insert(0, str(script_dir))
    
    try:
        # Import the main function from the application file and run it
        from neural_weaver_main import main as run_app
        run_app()
    except ImportError as e:
        print(f"\n❌ Error: Could not import the main application.")
        print(f"   Details: {e}")
        print("   This might be due to a missing dependency or an issue in the main script.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred while launching the application:")
        import traceback
        traceback.print_exc()

# --- Main Execution ---

def main():
    """
    Main function to run the launcher sequence.
    """
    print("=" * 50)
    print("      Welcome to the Neural Weaver Launcher")
    print("=" * 50)

    if not check_python_version():
        input("\nPress Enter to exit.")
        sys.exit(1)

    missing_required, missing_optional = get_missing_packages()

    if missing_required:
        prompt = f"\nSome required packages are missing. Do you want to install them now? (y/n): "
        if input(prompt).lower() == 'y':
            if not install_packages(missing_required):
                input("\nPress Enter to exit.")
                sys.exit(1)
        else:
            print("\nCannot start without required packages. Exiting.")
            sys.exit(1)

    if missing_optional:
        prompt = f"\nOptional packages for AI features are missing. Install them for the best experience? (y/n): "
        if input(prompt).lower() == 'y':
            install_packages(missing_optional)

    launch_app()
    print("-" * 50)
    print("Neural Weaver has closed. Thank you for using it!")


if __name__ == "__main__":
    main()
