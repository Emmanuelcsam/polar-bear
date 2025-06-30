#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Nexus IDE Server v6.0 - Enhanced Setup Script
Automated setup and installation for the modular Neural Nexus IDE.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_system_dependencies():
    """Check for system-level dependencies."""
    print("\n🔍 Checking system dependencies...")

    dependencies = {
        'git': 'Git version control',
        'curl': 'HTTP client for downloads'
    }

    missing = []
    for cmd, desc in dependencies.items():
        try:
            subprocess.run([cmd, '--version'],
                         capture_output=True, check=True)
            print(f"✅ {desc} available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠️  {desc} not found (optional)")
            missing.append(cmd)

    if missing:
        print(f"\n💡 Optional dependencies missing: {', '.join(missing)}")
        print("   These are not required but may enhance functionality")


def install_pip_packages():
    """Install Python packages from requirements."""
    print("\n📦 Installing Python packages...")

    requirements_file = Path("requirements_modular.txt")
    if not requirements_file.exists():
        print("❌ requirements_modular.txt not found")
        return False

    try:
        # Upgrade pip first
        print("🔄 Upgrading pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])

        # Install requirements
        print("📥 Installing packages from requirements_modular.txt...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_modular.txt"
        ])

        print("✅ All packages installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Package installation failed: {e}")
        return False


def install_optional_tools():
    """Install optional system tools for enhanced functionality."""
    print("\n🛠️  Installing optional system tools...")

    system = platform.system().lower()

    if system == "linux":
        install_linux_tools()
    elif system == "darwin":  # macOS
        install_macos_tools()
    elif system == "windows":
        install_windows_tools()
    else:
        print(f"⚠️  Unsupported system: {system}")


def install_linux_tools():
    """Install tools on Linux systems."""
    tools = [
        ("semgrep", "curl -L https://github.com/returntocorp/semgrep/releases/latest/download/semgrep-linux -o /usr/local/bin/semgrep && chmod +x /usr/local/bin/semgrep"),
    ]

    print("🐧 Detected Linux system")
    for tool, install_cmd in tools:
        try:
            # Check if tool exists
            subprocess.run([tool, "--version"],
                         capture_output=True, check=True)
            print(f"✅ {tool} already installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"📥 Installing {tool}...")
            try:
                # This would need sudo privileges
                print(f"💡 To install {tool}, run manually:")
                print(f"   {install_cmd}")
            except Exception as e:
                print(f"⚠️  Could not install {tool}: {e}")


def install_macos_tools():
    """Install tools on macOS systems."""
    print("🍎 Detected macOS system")

    # Check for Homebrew
    try:
        subprocess.run(["brew", "--version"],
                     capture_output=True, check=True)
        print("✅ Homebrew detected")

        # Install semgrep via Homebrew
        try:
            subprocess.run(["semgrep", "--version"],
                         capture_output=True, check=True)
            print("✅ semgrep already installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("📥 Installing semgrep via Homebrew...")
            try:
                subprocess.check_call(["brew", "install", "semgrep"])
                print("✅ semgrep installed successfully")
            except subprocess.CalledProcessError:
                print("⚠️  Could not install semgrep via Homebrew")

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Homebrew not found. Install from: https://brew.sh/")


def install_windows_tools():
    """Install tools on Windows systems."""
    print("🪟 Detected Windows system")
    print("💡 For enhanced functionality on Windows:")
    print("   1. Install Windows Subsystem for Linux (WSL)")
    print("   2. Or use package managers like Chocolatey or Scoop")
    print("   3. Some tools may have limited functionality on Windows")


def setup_environment():
    """Set up the Neural Nexus environment."""
    print("\n🏗️  Setting up Neural Nexus environment...")

    # Create necessary directories
    home_dir = Path.home() / ".neural_nexus_server"
    directories = [
        home_dir,
        home_dir / "scripts",
        home_dir / "logs",
        home_dir / "temp",
        home_dir / "projects",
        home_dir / "cache",
        Path("static")
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Create basic configuration file
    config_file = home_dir / "config.json"
    if not config_file.exists():
        config_data = {
            "host": "127.0.0.1",
            "port": 8765,
            "debug": False,
            "security_enabled": True,
            "auto_heal_enabled": True,
            "max_heal_attempts": 10
        }

        import json
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"✅ Created config file: {config_file}")


def verify_installation():
    """Verify that the installation was successful."""
    print("\n🔍 Verifying installation...")

    # Test core imports
    test_imports = [
        ('fastapi', 'FastAPI framework'),
        ('uvicorn', 'ASGI server'),
        ('websockets', 'WebSocket support'),
        ('aiofiles', 'Async file operations'),
        ('psutil', 'System monitoring'),
    ]

    optional_imports = [
        ('uvloop', 'Enhanced event loop'),
        ('orjson', 'Fast JSON processing'),
        ('loguru', 'Enhanced logging'),
        ('mypy', 'Type checking'),
        ('ruff', 'Fast linting'),
        ('bandit', 'Security analysis'),
    ]

    print("📋 Core dependencies:")
    all_core_ok = True
    for module, desc in test_imports:
        try:
            __import__(module)
            print(f"✅ {desc}")
        except ImportError:
            print(f"❌ {desc} - MISSING")
            all_core_ok = False

    print("\n📋 Optional enhancements:")
    for module, desc in optional_imports:
        try:
            __import__(module)
            print(f"✅ {desc}")
        except ImportError:
            print(f"⚠️  {desc} - Not available")

    if all_core_ok:
        print("\n🎉 Core installation verified successfully!")
        return True
    else:
        print("\n❌ Core installation has issues")
        return False


def create_startup_scripts():
    """Create convenient startup scripts."""
    print("\n📝 Creating startup scripts...")

    # Create shell script for Unix systems
    if platform.system() != "Windows":
        startup_script = Path("start_neural_nexus.sh")
        script_content = f"""#!/bin/bash
# Neural Nexus IDE Server Startup Script

echo "🧠 Starting Neural Nexus IDE Server v6.0..."
echo "   Enhanced modular edition with AI-powered features"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the server
python3 neural_nexus_modular.py "$@"
"""
        with open(startup_script, 'w') as f:
            f.write(script_content)

        # Make executable
        startup_script.chmod(0o755)
        print(f"✅ Created startup script: {startup_script}")

    # Create batch script for Windows
    if platform.system() == "Windows":
        startup_script = Path("start_neural_nexus.bat")
        script_content = f"""@echo off
REM Neural Nexus IDE Server Startup Script

echo 🧠 Starting Neural Nexus IDE Server v6.0...
echo    Enhanced modular edition with AI-powered features
echo.

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Start the server
python neural_nexus_modular.py %*
"""
        with open(startup_script, 'w') as f:
            f.write(script_content)

        print(f"✅ Created startup script: {startup_script}")


def print_usage_instructions():
    """Print usage instructions after successful setup."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              🎉 Neural Nexus IDE Setup Complete! 🎉          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🚀 Quick Start:                                             ║
║                                                              ║
║  1. Start the server:                                        ║
║     python neural_nexus_modular.py                          ║
║                                                              ║
║  2. Open your browser to:                                    ║
║     http://localhost:8765                                    ║
║                                                              ║
║  3. Start coding with AI-powered features!                   ║
║                                                              ║
║  📋 Available Commands:                                      ║
║                                                              ║
║  python neural_nexus_modular.py --help        Show help     ║
║  python neural_nexus_modular.py --setup       Setup env     ║
║  python neural_nexus_modular.py --install     Install deps  ║
║  python neural_nexus_modular.py --port 9000   Custom port   ║
║  python neural_nexus_modular.py --debug       Debug mode    ║
║                                                              ║
║  🛠️  Startup Scripts:                                        ║""")

    if platform.system() != "Windows":
        print("║  ./start_neural_nexus.sh                  Unix/Linux/macOS ║")
    else:
        print("║  start_neural_nexus.bat                   Windows          ║")

    print(f"""║                                                              ║
║  📚 Documentation & Features:                                ║
║                                                              ║
║  • Real-time code analysis and auto-healing                 ║
║  • Security vulnerability scanning                          ║
║  • Advanced linting and formatting                          ║
║  • Type checking and error detection                        ║
║  • Project management system                                ║
║  • Performance monitoring                                   ║
║                                                              ║
║  🔧 Configuration:                                           ║
║  ~/.neural_nexus_server/config.json                         ║
║                                                              ║
║  Happy coding! 🚀                                           ║
╚══════════════════════════════════════════════════════════════╝""")


def main():
    """Main setup function."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║             Neural Nexus IDE Server v6.0 Setup              ║
║              🚀 Enhanced Modular Edition 🚀                  ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check system dependencies
    check_system_dependencies()

    # Install Python packages
    if not install_pip_packages():
        print("\n❌ Package installation failed. Please check the errors above.")
        sys.exit(1)

    # Install optional tools
    install_optional_tools()

    # Setup environment
    setup_environment()

    # Create startup scripts
    create_startup_scripts()

    # Verify installation
    if verify_installation():
        print_usage_instructions()
    else:
        print("\n⚠️  Installation completed with some issues.")
        print("   Core functionality should work, but some features may be limited.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n💥 Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
