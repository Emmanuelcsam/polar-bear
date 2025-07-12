"""
Refactored Auto Installer Module
Separates functions from main execution for better testability
Now integrated with the hivemind connector system
"""
import subprocess
import sys
import importlib
from connector_interface import setup_connector, get_hivemind_parameter, send_hivemind_status


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


def auto_install_dependencies(libraries=None, connector=None):
    """Auto-install all required dependencies"""
    # Get libraries from hivemind or use defaults
    if libraries is None:
        if connector:
            libraries = connector.get_parameter('libraries', 
                ['torch', 'torchvision', 'numpy', 'opencv-python', 'pillow', 'scikit-learn'])
        else:
            libraries = get_hivemind_parameter('libraries', 
                ['torch', 'torchvision', 'numpy', 'opencv-python', 'pillow', 'scikit-learn'])

    results = {}
    total = len(libraries)
    installed_count = 0
    
    for i, lib in enumerate(libraries):
        # Send progress status to hivemind
        if connector:
            connector.send_status({
                'action': 'checking',
                'library': lib,
                'progress': i / total * 100
            })
        
        if check_library_installed(lib):
            print(f"✓ {lib} already installed")
            results[lib] = 'already_installed'
            installed_count += 1
        else:
            print(f"Installing {lib}...")
            if install_library(lib):
                print(f"✓ {lib} installed successfully")
                results[lib] = 'installed'
                installed_count += 1
            else:
                print(f"✗ Failed to install {lib}")
                results[lib] = 'failed'

    # Send final status to hivemind
    final_status = {
        'action': 'complete',
        'total_libraries': total,
        'installed': installed_count,
        'failed': total - installed_count,
        'results': results
    }
    
    if connector:
        connector.send_status(final_status)
    else:
        send_hivemind_status(final_status)

    return results


def main():
    """Main function with hivemind integration"""
    # Setup connector
    connector = setup_connector("auto_installer_refactored.py")
    
    if connector.is_connected:
        print("Connected to hivemind system")
        # Register parameters
        connector.register_parameter("libraries", 
            ['torch', 'torchvision', 'numpy', 'opencv-python', 'pillow', 'scikit-learn'],
            "List of libraries to install")
        connector.register_parameter("upgrade", True, "Whether to upgrade existing libraries")
        
        # Register callbacks
        connector.register_callback("install", lambda libs=None: auto_install_dependencies(libs, connector))
        connector.register_callback("check", check_library_installed)
        connector.register_callback("get_installed", lambda: {lib: check_library_installed(lib) 
            for lib in connector.get_parameter('libraries', [])})
    else:
        print("Running in standalone mode")
    
    # Run the installation
    auto_install_dependencies(connector=connector)
    
    # If connected, listen for commands
    if connector.is_connected and connector.get_parameter('listen_mode', False):
        print("Entering listen mode for hivemind commands...")
        connector.listen_for_commands()


if __name__ == "__main__":
    main()
