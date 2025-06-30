#!/usr/bin/env python3
"""
ChatGPT Analyzer - One-Click Setup
Automatically sets up and launches the ChatGPT Analyzer
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


class SetupManager:
    """Manages the setup and installation process"""
    
    def __init__(self):
        self.system = platform.system()
        self.python_cmd = sys.executable
        self.venv_path = Path("venv")
        self.is_windows = self.system == "Windows"
        
    def print_banner(self):
        """Print welcome banner"""
        print("="*60)
        print("🤖 ChatGPT Analyzer - Automatic Setup")
        print("="*60)
        print(f"System: {self.system}")
        print(f"Python: {sys.version.split()[0]}")
        print("="*60)
        print()
        
    def check_python_version(self):
        """Ensure Python 3.8+"""
        if sys.version_info < (3, 8):
            print("❌ Error: Python 3.8 or higher is required")
            print(f"   Your version: {sys.version}")
            print("\n📥 Download Python from: https://www.python.org/downloads/")
            return False
        print("✅ Python version OK")
        return True
        
    def create_virtual_environment(self):
        """Create a virtual environment"""
        if self.venv_path.exists():
            print("📦 Virtual environment already exists")
            return True
            
        print("📦 Creating virtual environment...")
        try:
            subprocess.run([self.python_cmd, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
            
    def get_pip_command(self):
        """Get the pip command for the virtual environment"""
        if self.is_windows:
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = self.venv_path / "bin" / "pip"
        return str(pip_path)
        
    def get_python_command(self):
        """Get the python command for the virtual environment"""
        if self.is_windows:
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            python_path = self.venv_path / "bin" / "python"
        return str(python_path)
        
    def install_dependencies(self):
        """Install required packages"""
        print("\n📥 Installing dependencies...")
        
        pip_cmd = self.get_pip_command()
        
        # Upgrade pip first
        try:
            subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        except:
            pass
            
        # Core dependencies for live analysis
        packages = [
            "flask>=3.0.0",
            "flask-cors>=4.0.0",
            "flask-sock>=0.7.0",  # WebSocket support
            "selenium>=4.16.0",    # Browser automation
            "webdriver-manager>=4.0.1",  # Auto Chrome driver
            "python-dotenv>=1.0.0"
        ]
        
        # Optional but recommended for CAPTCHA handling
        optional_packages = [
            "opencv-python>=4.9.0.80",
            "pillow>=10.2.0"
        ]
        
        try:
            # Install core packages
            print("📦 Installing core packages...")
            subprocess.run([pip_cmd, "install"] + packages, check=True)
            
            # Try to install optional packages
            print("\n📦 Installing CAPTCHA handler dependencies...")
            for pkg in optional_packages:
                try:
                    subprocess.run([pip_cmd, "install", pkg], check=True)
                except:
                    print(f"   ⚠️  Optional package {pkg} failed (CAPTCHA handling may be limited)")
                    
            print("\n✅ All dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("   Try running manually:")
            print(f"   {pip_cmd} install flask flask-cors flask-sock selenium webdriver-manager")
            return False
            
    def create_launcher(self):
        """Create platform-specific launcher"""
        python_cmd = self.get_python_command()
        
        if self.is_windows:
            # Create batch file for Windows
            launcher_content = f"""@echo off
echo Starting ChatGPT Analyzer...
"{python_cmd}" launch.py
pause
"""
            launcher_path = Path("start-analyzer.bat")
            launcher_path.write_text(launcher_content)
            print(f"✅ Created launcher: {launcher_path}")
            print(f"   Double-click '{launcher_path}' to start the analyzer")
            
        else:
            # Create shell script for Unix
            launcher_content = f"""#!/bin/bash
echo "Starting ChatGPT Analyzer..."
{python_cmd} launch.py
"""
            launcher_path = Path("start-analyzer.sh")
            launcher_path.write_text(launcher_content)
            launcher_path.chmod(0o755)
            print(f"✅ Created launcher: {launcher_path}")
            print(f"   Run './start-analyzer.sh' to start the analyzer")
            
    def check_files(self):
        """Verify all required files exist"""
        required_files = {
            "app.py": "Server application",
            "index.html": "Web interface",
            "launch.py": "Launcher script"
        }
        
        optional_files = {
            "captcha_handler.py": "CAPTCHA bypass handler"
        }
        
        missing = []
        for file, desc in required_files.items():
            if not Path(file).exists():
                missing.append(f"{file} ({desc})")
                
        if missing:
            print("\n❌ Missing required files:")
            for file in missing:
                print(f"   - {file}")
            print("\nPlease ensure all files are in the current directory")
            return False
            
        print("✅ All required files present")
        
        # Check optional files
        for file, desc in optional_files.items():
            if not Path(file).exists():
                print(f"⚠️  Optional file missing: {file} ({desc})")
                print("   CAPTCHA handling may be limited without this file")
                
        return True
        
    def create_example_env(self):
        """Create example .env file"""
        env_content = """# ChatGPT Analyzer Configuration
# Uncomment and modify as needed

# Server settings
# PORT=5000
# DEBUG=False

# Upload limits (in bytes)
# MAX_CONTENT_LENGTH=104857600  # 100MB

# Optional features
# ENABLE_EXPORT_EXCEL=True
# ENABLE_EXPORT_PDF=False
"""
        env_path = Path(".env.example")
        env_path.write_text(env_content)
        print("📄 Created .env.example (optional configuration)")
        
    def run_setup(self):
        """Run the complete setup process"""
        self.print_banner()
        
        # Check Python version
        if not self.check_python_version():
            return False
            
        # Check required files
        if not self.check_files():
            return False
            
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
            
        # Install dependencies
        if not self.install_dependencies():
            return False
            
        # Create launcher
        self.create_launcher()
        
        # Create example config
        self.create_example_env()
        
        # Success message
        print("\n" + "="*60)
        print("✅ Setup completed successfully!")
        print("="*60)
        print("\n📚 Quick Start Guide:\n")
        
        if self.is_windows:
            print("1. Double-click 'start-analyzer.bat' to launch")
        else:
            print("1. Run './start-analyzer.sh' to launch")
            
        print("2. Your browser will open automatically")
        print("3. Upload your ChatGPT export (conversations.json)")
        print("4. View and export your analysis results")
        
        print("\n💡 Getting your ChatGPT export:")
        print("   1. Go to chat.openai.com")
        print("   2. Settings → Data controls → Export data")
        print("   3. Download and extract the ZIP file")
        print("   4. Use the conversations.json file")
        
        print("\n📖 For more information, see README.md")
        print("\n" + "="*60)
        
        # Ask if user wants to start now
        if input("\n🚀 Start the analyzer now? (y/n): ").lower().strip() == 'y':
            python_cmd = self.get_python_command()
            subprocess.run([python_cmd, "launch.py"])
            
        return True


def main():
    """Main setup entry point"""
    setup = SetupManager()
    
    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("\nFor help, please check:")
        print("- README.md for manual setup instructions")
        print("- Ensure all files are in the same directory")
        print("- Try running 'python app.py' directly")
        sys.exit(1)


if __name__ == "__main__":
    main()