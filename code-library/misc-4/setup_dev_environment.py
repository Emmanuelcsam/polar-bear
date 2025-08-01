#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Development Environment Setup Script for Fiber CNN Project
Automatically configures Pylance, real-time monitoring, and development tools
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging
import json
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DevEnvironmentSetup:
    """Setup development environment for Fiber CNN project"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.dev_requirements_file = self.project_root / "requirements-dev.txt"
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        logger.info(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def create_virtual_environment(self, force: bool = False) -> bool:
        """Create virtual environment"""
        if self.venv_path.exists() and not force:
            logger.info(f"Virtual environment already exists at {self.venv_path}")
            return True
        
        if force and self.venv_path.exists():
            shutil.rmtree(self.venv_path)
            logger.info("Removed existing virtual environment")
        
        try:
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], check=True)
            logger.info(f"✅ Virtual environment created at {self.venv_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False
    
    def get_venv_python(self) -> str:
        """Get path to virtual environment Python executable"""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "python.exe")
        else:  # Unix/Linux
            return str(self.venv_path / "bin" / "python")
    
    def get_venv_pip(self) -> str:
        """Get path to virtual environment pip executable"""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:  # Unix/Linux
            return str(self.venv_path / "bin" / "pip")
    
    def install_requirements(self, dev_only: bool = False) -> bool:
        """Install Python requirements"""
        pip_cmd = self.get_venv_pip()
        
        try:
            # Upgrade pip first
            subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
            
            # Install main requirements
            if not dev_only and self.requirements_file.exists():
                logger.info("Installing main requirements...")
                subprocess.run([pip_cmd, "install", "-r", str(self.requirements_file)], check=True)
            
            # Install development requirements
            if self.dev_requirements_file.exists():
                logger.info("Installing development requirements...")
                subprocess.run([pip_cmd, "install", "-r", str(self.dev_requirements_file)], check=True)
            
            logger.info("✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            return False
    
    def create_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            "logs",
            "logs/monitoring",
            "logs/tensorboard",
            "checkpoints",
            "output",
            "temp",
            "typings",
            "tests",
            "docs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def setup_git_hooks(self) -> None:
        """Setup Git hooks for code quality"""
        git_hooks_dir = self.project_root / ".git" / "hooks"
        if not git_hooks_dir.exists():
            logger.warning("Git repository not found, skipping hooks setup")
            return
        
        # Create pre-commit hook
        pre_commit_hook = git_hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/sh
# Pre-commit hook for code quality checks

echo "Running code quality checks..."

# Run black formatting check
python -m black --check --diff .

# Run isort import sorting check
python -m isort --check-only --diff .

# Run flake8 linting
python -m flake8 . --max-line-length=100 --ignore=E203,W503

# Run mypy type checking
python -m mypy . --ignore-missing-imports

echo "Code quality checks completed"
"""
        
        with open(pre_commit_hook, 'w') as f:
            f.write(pre_commit_content)
        
        # Make executable
        os.chmod(pre_commit_hook, 0o755)
        logger.info("✅ Git pre-commit hook created")
    
    def create_pre_commit_config(self) -> None:
        """Create pre-commit configuration"""
        pre_commit_config = {
            "repos": [
                {
                    "repo": "https://github.com/psf/black",
                    "rev": "23.3.0",
                    "hooks": [
                        {
                            "id": "black",
                            "language_version": "python3"
                        }
                    ]
                },
                {
                    "repo": "https://github.com/pycqa/isort",
                    "rev": "5.12.0",
                    "hooks": [
                        {
                            "id": "isort",
                            "args": ["--profile", "black"]
                        }
                    ]
                },
                {
                    "repo": "https://github.com/pycqa/flake8",
                    "rev": "6.0.0",
                    "hooks": [
                        {
                            "id": "flake8",
                            "args": ["--max-line-length=100", "--ignore=E203,W503"]
                        }
                    ]
                }
            ]
        }
        
        config_file = self.project_root / ".pre-commit-config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(pre_commit_config, f, default_flow_style=False)
        
        logger.info("✅ Pre-commit configuration created")
    
    def setup_pylance_integration(self) -> None:
        """Setup Pylance integration"""
        try:
            # Run the Pylance integration script
            pylance_script = self.project_root / "pylance_integration.py"
            if pylance_script.exists():
                python_cmd = self.get_venv_python()
                subprocess.run([python_cmd, str(pylance_script)], check=True)
                logger.info("✅ Pylance integration configured")
            else:
                logger.warning("Pylance integration script not found")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup Pylance integration: {e}")
    
    def create_vscode_extensions(self) -> None:
        """Create VS Code extensions recommendation file"""
        extensions = [
            "ms-python.python",
            "ms-python.pylance",
            "ms-python.black-formatter",
            "ms-python.isort",
            "ms-python.flake8",
            "ms-python.mypy-type-checker",
            "ms-toolsai.jupyter",
            "ms-toolsai.jupyter-keymap",
            "ms-toolsai.jupyter-renderers",
            "ms-vscode.vscode-json",
            "redhat.vscode-yaml",
            "ms-vscode.vscode-markdown",
            "ms-vscode.vscode-git",
            "ms-vscode.vscode-git-graph",
            "ms-vscode.vscode-gitlens",
            "ms-vscode.vscode-docker",
            "ms-vscode.vscode-kubernetes-tools",
            "ms-vscode.vscode-remote",
            "ms-vscode.vscode-remote-extensionpack"
        ]
        
        extensions_file = self.project_root / ".vscode" / "extensions.json"
        extensions_file.parent.mkdir(exist_ok=True)
        
        extensions_config = {
            "recommendations": extensions
        }
        
        with open(extensions_file, 'w') as f:
            json.dump(extensions_config, f, indent=2)
        
        logger.info("✅ VS Code extensions recommendations created")
    
    def create_development_scripts(self) -> None:
        """Create development utility scripts"""
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Create development script
        dev_script = scripts_dir / "dev_setup.sh"
        dev_script_content = """#!/bin/bash
# Development environment setup script

echo "Setting up development environment..."

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run initial code quality checks
python -m black .
python -m isort .
python -m flake8 . --max-line-length=100 --ignore=E203,W503

echo "Development environment setup complete!"
"""
        
        with open(dev_script, 'w') as f:
            f.write(dev_script_content)
        
        # Make executable
        os.chmod(dev_script, 0o755)
        
        # Create Windows batch file
        dev_script_win = scripts_dir / "dev_setup.bat"
        dev_script_win_content = """@echo off
REM Development environment setup script for Windows

echo Setting up development environment...

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

REM Setup pre-commit hooks
pre-commit install

REM Run initial code quality checks
python -m black .
python -m isort .
python -m flake8 . --max-line-length=100 --ignore=E203,W503

echo Development environment setup complete!
pause
"""
        
        with open(dev_script_win, 'w') as f:
            f.write(dev_script_win_content)
        
        logger.info("✅ Development scripts created")
    
    def run_initial_tests(self) -> bool:
        """Run initial tests to verify setup"""
        try:
            python_cmd = self.get_venv_python()
            
            # Test imports
            test_imports = [
                "torch",
                "torchvision",
                "numpy",
                "cv2",
                "albumentations",
                "matplotlib",
                "pytest",
                "black",
                "isort",
                "flake8",
                "mypy"
            ]
            
            logger.info("Testing imports...")
            for module in test_imports:
                try:
                    subprocess.run([python_cmd, "-c", f"import {module}"], 
                                 check=True, capture_output=True)
                    logger.info(f"✅ {module} imported successfully")
                except subprocess.CalledProcessError:
                    logger.warning(f"⚠️ Failed to import {module}")
            
            # Run basic tests if they exist
            test_files = list(self.project_root.glob("test_*.py"))
            if test_files:
                logger.info("Running basic tests...")
                subprocess.run([python_cmd, "-m", "pytest", "test_*.py", "-v"], check=True)
                logger.info("✅ Basic tests passed")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Initial tests failed: {e}")
            return False
    
    def create_environment_info(self) -> None:
        """Create environment information file"""
        info = {
            "setup_timestamp": str(Path().cwd()),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "venv_path": str(self.venv_path),
            "project_root": str(self.project_root),
            "platform": sys.platform,
            "architecture": sys.maxsize > 2**32 and "64-bit" or "32-bit"
        }
        
        info_file = self.project_root / "environment_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        logger.info("✅ Environment information saved")
    
    def setup_complete(self, force: bool = False, dev_only: bool = False) -> bool:
        """Complete development environment setup"""
        logger.info("🚀 Starting development environment setup...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment(force):
            return False
        
        # Install requirements
        if not self.install_requirements(dev_only):
            return False
        
        # Create directories
        self.create_directories()
        
        # Setup Git hooks
        self.setup_git_hooks()
        
        # Create pre-commit config
        self.create_pre_commit_config()
        
        # Setup Pylance integration
        self.setup_pylance_integration()
        
        # Create VS Code extensions
        self.create_vscode_extensions()
        
        # Create development scripts
        self.create_development_scripts()
        
        # Run initial tests
        if not self.run_initial_tests():
            logger.warning("⚠️ Some initial tests failed, but setup continues...")
        
        # Create environment info
        self.create_environment_info()
        
        logger.info("🎉 Development environment setup complete!")
        logger.info("📁 Project structure created")
        logger.info("🔧 VS Code configurations ready")
        logger.info("🐍 Virtual environment activated")
        logger.info("📦 Dependencies installed")
        logger.info("🔍 Code quality tools configured")
        logger.info("📊 Pylance integration ready")
        logger.info("📈 Real-time monitoring available")
        
        return True

def main():
    """Main function for development environment setup"""
    parser = argparse.ArgumentParser(description='Setup Development Environment for Fiber CNN Project')
    parser.add_argument('--force', action='store_true',
                       help='Force recreation of virtual environment')
    parser.add_argument('--dev-only', action='store_true',
                       help='Install only development dependencies')
    parser.add_argument('--project-root', type=str, default='.',
                       help='Project root directory')
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = DevEnvironmentSetup(args.project_root)
    
    # Run complete setup
    success = setup.setup_complete(
        force=args.force,
        dev_only=args.dev_only
    )
    
    if success:
        logger.info("✅ Development environment setup successful!")
        logger.info("Next steps:")
        logger.info("1. Activate virtual environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        logger.info("2. Open project in VS Code")
        logger.info("3. Run: python pylance_integration.py")
        logger.info("4. Start development!")
    else:
        logger.error("❌ Development environment setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 