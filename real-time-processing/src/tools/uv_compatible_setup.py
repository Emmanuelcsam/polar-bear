#!/usr/bin/env python3
"""
UV-Compatible Setup Script for Geometry Detection System
========================================================

This script works with uv package manager for dependency installation.
"""

import subprocess
import sys
import os
import platform
import importlib.util

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text: str):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text: str):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def check_uv():
    """Check if uv is available"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_success(f"uv is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print_error("uv not found!")
    print_info("Install uv from: https://github.com/astral-sh/uv")
    return False

def install_with_uv():
    """Install packages using uv"""
    print_header("Installing Packages with UV")
    
    # Check if pyproject.toml exists
    if not os.path.exists('pyproject.toml'):
        print_warning("pyproject.toml not found. Creating one...")
        
        pyproject_content = '''[project]
name = "geometry-detection"
version = "1.0.0"
requires-python = ">=3.8"
dependencies = [
    "opencv-python>=4.8.0",
    "opencv-contrib-python>=4.8.0",
    "numpy>=1.24.0",
    "psutil>=5.9.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "pandas>=2.0.0",
    "pillow>=10.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
]
'''
        
        with open('pyproject.toml', 'w') as f:
            f.write(pyproject_content)
        
        print_success("Created pyproject.toml")
    
    print_info("Installing dependencies with uv...")
    
    try:
        # Install dependencies
        result = subprocess.run(['uv', 'pip', 'install', '-r', 'pyproject.toml'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("Dependencies installed successfully!")
            return True
        else:
            # Try alternative command
            print_warning("Trying alternative uv command...")
            result = subprocess.run(['uv', 'pip', 'sync'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print_success("Dependencies installed successfully!")
                return True
            else:
                print_error(f"Installation failed: {result.stderr}")
                
                # Try installing packages individually
                print_info("Trying individual package installation...")
                packages = [
                    'opencv-python',
                    'opencv-contrib-python', 
                    'numpy',
                    'psutil',
                    'scipy',
                    'matplotlib',
                    'pandas',
                    'pillow'
                ]
                
                for package in packages:
                    print(f"Installing {package}...", end=' ')
                    result = subprocess.run(['uv', 'pip', 'install', package],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print_success("OK")
                    else:
                        print_error("FAILED")
                
                return True
                
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def check_packages():
    """Check installed packages"""
    print_header("Checking Installed Packages")
    
    packages = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'psutil': 'PSUtil',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'PIL': 'Pillow'
    }
    
    all_installed = True
    
    for module, name in packages.items():
        try:
            spec = importlib.util.find_spec(module)
            if spec is not None:
                # Try to import and get version
                try:
                    imported = importlib.import_module(module)
                    version = getattr(imported, '__version__', 'unknown')
                    print_success(f"{name} installed (version: {version})")
                except:
                    print_success(f"{name} installed")
            else:
                print_error(f"{name} not installed")
                all_installed = False
        except:
            print_error(f"{name} not installed")
            all_installed = False
    
    return all_installed

def test_opencv():
    """Test OpenCV installation"""
    print_header("Testing OpenCV")
    
    try:
        import cv2
        import numpy as np
        
        print_success(f"OpenCV version: {cv2.__version__}")
        
        # Test basic operations
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        print_success("OpenCV basic operations work")
        
        # Check for CUDA
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count > 0:
                print_success(f"CUDA available: {cuda_count} device(s)")
            else:
                print_info("CUDA not available (CPU mode will be used)")
        except:
            print_info("CUDA module not available (CPU mode will be used)")
        
        return True
        
    except Exception as e:
        print_error(f"OpenCV test failed: {e}")
        return False

def create_test_script():
    """Create a simple test script"""
    test_content = '''#!/usr/bin/env python3
"""Simple test to verify installation"""

import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# Create and display test image
img = np.ones((200, 400, 3), dtype=np.uint8) * 255
cv2.putText(img, "Installation Successful!", (50, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.rectangle(img, (30, 30), (370, 170), (0, 0, 255), 3)

cv2.imshow("Test", img)
print("\\nPress any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✓ All tests passed!")
'''
    
    with open('test_installation.py', 'w') as f:
        f.write(test_content)
    
    print_success("Created test_installation.py")

def main():
    print_header("UV-Compatible Geometry Detection Setup")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or python_version.minor < 8:
        print_error("Python 3.8+ required!")
        return
    
    # Check if uv is available
    if not check_uv():
        print("\nAlternative: Install pip in this environment:")
        print("  uv pip install pip")
        print("  Then run the original setup_installer.py")
        return
    
    # Install packages
    if install_with_uv():
        # Check installation
        if check_packages():
            # Test OpenCV
            test_opencv()
            
            # Create test script
            create_test_script()
            
            print_header("Setup Complete!")
            print("\nNext steps:")
            print("1. Test your installation:")
            print(f"   {Colors.BOLD}uv run python test_installation.py{Colors.ENDC}")
            print("\n2. Run the main program:")
            print(f"   {Colors.BOLD}uv run python integrated_geometry_system.py{Colors.ENDC}")
            print("\n3. Or with the example:")
            print(f"   {Colors.BOLD}uv run python shape_analysis_dashboard.py{Colors.ENDC}")
        else:
            print_warning("Some packages failed to install")
            print("Try running:")
            print("  uv pip install opencv-python opencv-contrib-python numpy")
    else:
        print_error("Installation failed!")
        print("\nTry manual installation:")
        print("  uv pip install opencv-python opencv-contrib-python numpy psutil")

if __name__ == "__main__":
    main()
