#!/usr/bin/env python3
"""
Neural Nexus IDE Setup Script
Handles dependency installation and initial configuration
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil
import json

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Your version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def check_uv():
    """Check if uv is installed"""
    if shutil.which("uv"):
        print("✅ uv is installed (fast package manager)")
        return True
    else:
        print("ℹ️  uv not found (will use pip instead)")
        return False

def install_package(package, use_uv=False):
    """Install a single package"""
    try:
        if use_uv:
            subprocess.check_call(["uv", "pip", "install", package])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def install_dependencies():
    """Install all required dependencies"""
    print("\n📦 Installing dependencies...")
    
    use_uv = check_uv()
    
    # Core dependencies
    required_packages = [
        "psutil",           # Process management
        "networkx",         # Graph visualization
        "matplotlib",       # Plotting
        "openai",          # AI integration
        "requests",        # HTTP requests
        "beautifulsoup4",  # Web scraping for error lookup
        "websocket-client", # For potential Copilot integration
        "Pillow",          # Image handling
        "pywin32"          # Windows-specific features (if on Windows)
    ]
    
    # Remove Windows-specific packages on other platforms
    if sys.platform != "win32":
        required_packages = [p for p in required_packages if p != "pywin32"]
    
    failed = []
    
    for i, package in enumerate(required_packages, 1):
        print(f"[{i}/{len(required_packages)}] Installing {package}...", end=" ")
        if install_package(package, use_uv):
            print("✅")
        else:
            print("❌")
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  Failed to install: {', '.join(failed)}")
        print("You may need to install these manually")
    else:
        print("\n✅ All dependencies installed successfully!")
    
    return len(failed) == 0

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    dirs = [
        Path.home() / ".neural_nexus",
        Path.home() / ".neural_nexus" / "scripts",
        Path.home() / ".neural_nexus" / "logs",
        Path.home() / ".neural_nexus" / "templates"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if sys.platform != "win32":
        return
    
    try:
        import win32com.client
        from pathlib import Path
        
        desktop = Path.home() / "Desktop"
        shortcut_path = desktop / "Neural Nexus IDE.lnk"
        
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(str(shortcut_path))
        
        # Get the main script path
        main_script = Path(__file__).parent / "neural_nexus_ide.py"
        
        shortcut.TargetPath = sys.executable
        shortcut.Arguments = f'"{main_script}"'
        shortcut.WorkingDirectory = str(main_script.parent)
        shortcut.IconLocation = sys.executable
        shortcut.Description = "Neural Nexus IDE - Advanced Neural Network Development"
        shortcut.save()
        
        print(f"✅ Created desktop shortcut: {shortcut_path}")
    except Exception as e:
        print(f"ℹ️  Could not create desktop shortcut: {e}")

def check_optional_tools():
    """Check for optional tools"""
    print("\n🔧 Checking optional tools...")
    
    tools = {
        "git": "Git (for version control)",
        "code": "VS Code (for external editing)",
        "gh": "GitHub CLI (for Copilot integration)"
    }
    
    for cmd, name in tools.items():
        if shutil.which(cmd):
            print(f"✅ {name} is installed")
        else:
            print(f"ℹ️  {name} not found (optional)")
    
    # Check GitHub Copilot
    if shutil.which("gh"):
        try:
            result = subprocess.run(["gh", "copilot", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ GitHub Copilot CLI is installed")
            else:
                print("ℹ️  GitHub Copilot CLI not installed")
                print("   Install with: gh extension install github/gh-copilot")
        except:
            pass

def create_launcher_script():
    """Create a simple launcher script"""
    launcher_content = '''#!/usr/bin/env python3
"""Neural Nexus IDE Launcher"""
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
main_script = script_dir / "neural_nexus_ide.py"

if main_script.exists():
    subprocess.run([sys.executable, str(main_script)])
else:
    print(f"Error: Main script not found at {main_script}")
    input("Press Enter to exit...")
'''
    
    launcher_path = Path("launch_neural_nexus.py")
    launcher_path.write_text(launcher_content)
    
    if sys.platform != "win32":
        # Make executable on Unix-like systems
        os.chmod(launcher_path, 0o755)
    
    print(f"✅ Created launcher: {launcher_path}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("Neural Nexus IDE Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Install dependencies
    success = install_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Check optional tools
    check_optional_tools()
    
    # Create shortcuts and launcher
    create_desktop_shortcut()
    create_launcher_script()
    
    print("\n" + "=" * 60)
    
    if success:
        print("✅ Setup completed successfully!")
        print("\nTo start Neural Nexus IDE:")
        print("  - Run: python neural_nexus_ide.py")
        print("  - Or use: python launch_neural_nexus.py")
        if sys.platform == "win32":
            print("  - Or use the desktop shortcut")
    else:
        print("⚠️  Setup completed with warnings")
        print("Some dependencies failed to install")
        print("The IDE may still work, but some features might be unavailable")
    
    print("\n📝 First time setup:")
    print("  1. Go to Settings tab")
    print("  2. Add your OpenAI API key (optional)")
    print("  3. Install GitHub Copilot CLI for enhanced AI features")
    
    print("\n" + "=" * 60)
    
    # Ask to launch
    response = input("\nLaunch Neural Nexus IDE now? (y/n): ")
    if response.lower() == 'y':
        try:
            subprocess.run([sys.executable, "neural_nexus_ide.py"])
        except FileNotFoundError:
            print("Error: neural_nexus_ide.py not found in current directory")
            print("Please ensure both files are in the same directory")

if __name__ == "__main__":
    main()
