"""
Refactored Auto Installer Module
Separates functions from main execution for better testability
"""
import subprocess
import sys
import importlib


def check_library_installed(library_name):
    """Check if a library is already installed"""
    try:
        lib_import_name = library_name.replace('-', '_')
        importlib.import_module(lib_import_name)
        return True
    except ImportError:
        return False


def install_library(library_name):
    """Install a library using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', library_name])
        return True
    except subprocess.CalledProcessError:
        return False


def auto_install_dependencies(libraries=None):
    """Auto-install all required dependencies"""
    if libraries is None:
        libraries = ['torch', 'torchvision', 'numpy', 'opencv-python', 'pillow', 'scikit-learn']
    
    results = {}
    for lib in libraries:
        if check_library_installed(lib):
            print(f"✓ {lib} already installed")
            results[lib] = 'already_installed'
        else:
            print(f"Installing {lib}...")
            if install_library(lib):
                print(f"✓ {lib} installed successfully")
                results[lib] = 'installed'
            else:
                print(f"✗ Failed to install {lib}")
                results[lib] = 'failed'
    
    return results


if __name__ == "__main__":
    auto_install_dependencies()
