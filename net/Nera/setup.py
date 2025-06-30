#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Nexus IDE Setup Script
Automated setup and launcher for the Neural Nexus IDE
"""

import os
import sys
import subprocess
import shutil
import platform
import webbrowser
import time
from pathlib import Path
import json

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    """Print welcome banner"""
    banner = f"""
{Colors.OKCYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  {Colors.BOLD}⚡ NEURAL NEXUS IDE{Colors.ENDC}{Colors.OKCYAN}  - AI-Powered Development Environment  ║
║                                                              ║
║  The Ultimate Python Script Development & Testing Platform   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)

def check_python_version():
    """Check if Python version is adequate"""
    if sys.version_info < (3, 8):
        print(f"{Colors.FAIL}Error: Python 3.8 or higher is required{Colors.ENDC}")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"{Colors.OKGREEN}✓ Python {sys.version.split()[0]} detected{Colors.ENDC}")

def check_dependencies():
    """Check and install dependencies"""
    print(f"\n{Colors.OKBLUE}Checking dependencies...{Colors.ENDC}")
    
    # Core requirements
    core_deps = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'websockets': 'WebSocket support',
        'aiofiles': 'Async file operations',
        'psutil': 'Process management'
    }
    
    # Optional dependencies
    optional_deps = {
        'openai': 'OpenAI GPT integration',
        'requests': 'HTTP requests',
        'matplotlib': 'Visualization',
        'networkx': 'Network graphs'
    }
    
    missing_core = []
    missing_optional = []
    
    # Check core dependencies
    for package, description in core_deps.items():
        try:
            __import__(package)
            print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {package} - {description}")
        except ImportError:
            missing_core.append(package)
            print(f"  {Colors.FAIL}✗{Colors.ENDC} {package} - {description}")
    
    # Check optional dependencies
    print(f"\n{Colors.OKBLUE}Optional features:{Colors.ENDC}")
    for package, description in optional_deps.items():
        try:
            __import__(package)
            print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {package} - {description}")
        except ImportError:
            missing_optional.append(package)
            print(f"  {Colors.WARNING}○{Colors.ENDC} {package} - {description}")
    
    # Install missing core dependencies
    if missing_core:
        print(f"\n{Colors.WARNING}Installing required dependencies...{Colors.ENDC}")
        
        # Try UV first, then pip
        if shutil.which('uv'):
            cmd = ['uv', 'pip', 'install'] + missing_core
        else:
            cmd = [sys.executable, '-m', 'pip', 'install'] + missing_core
        
        try:
            subprocess.check_call(cmd)
            print(f"{Colors.OKGREEN}✓ Core dependencies installed{Colors.ENDC}")
        except subprocess.CalledProcessError:
            print(f"{Colors.FAIL}Failed to install dependencies{Colors.ENDC}")
            sys.exit(1)
    
    # Ask about optional dependencies
    if missing_optional:
        print(f"\n{Colors.WARNING}Optional dependencies can enhance functionality:{Colors.ENDC}")
        for package in missing_optional:
            response = input(f"Install {package}? (y/n): ").lower()
            if response == 'y':
                try:
                    if shutil.which('uv'):
                        subprocess.check_call(['uv', 'pip', 'install', package])
                    else:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"{Colors.OKGREEN}✓ {package} installed{Colors.ENDC}")
                except:
                    print(f"{Colors.WARNING}Failed to install {package}{Colors.ENDC}")

def setup_directories():
    """Create necessary directories"""
    print(f"\n{Colors.OKBLUE}Setting up directories...{Colors.ENDC}")
    
    dirs = [
        Path.home() / ".neural_nexus_server",
        Path.home() / ".neural_nexus_server" / "scripts",
        Path.home() / ".neural_nexus_server" / "logs",
        Path.home() / ".neural_nexus_server" / "temp",
        Path("static")
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  {Colors.OKGREEN}✓{Colors.ENDC} Created {dir_path}")

def save_files():
    """Save the HTML and server files"""
    print(f"\n{Colors.OKBLUE}Creating application files...{Colors.ENDC}")
    
    # The HTML content would be saved here
    # For this demo, we'll create a placeholder
    
    html_path = Path("static") / "index.html"
    if not html_path.exists():
        print(f"  {Colors.WARNING}! Please save the HTML file as: {html_path}{Colors.ENDC}")
    
    server_path = Path("neural_nexus_server.py")
    if not server_path.exists():
        print(f"  {Colors.WARNING}! Please save the server file as: {server_path}{Colors.ENDC}")
    
    # Create a config file
    config_path = Path("config.json")
    if not config_path.exists():
        config = {
            "port": 8765,
            "host": "127.0.0.1",
            "auto_open_browser": True,
            "theme": "dark",
            "terminal_type": "auto"
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  {Colors.OKGREEN}✓{Colors.ENDC} Created config.json")

def check_tools():
    """Check for optional tools"""
    print(f"\n{Colors.OKBLUE}Checking optional tools...{Colors.ENDC}")
    
    tools = {
        'git': 'Git version control',
        'code': 'Visual Studio Code',
        'gh': 'GitHub CLI',
        'uv': 'UV package manager',
        'node': 'Node.js runtime'
    }
    
    for tool, description in tools.items():
        if shutil.which(tool):
            print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {tool} - {description}")
        else:
            print(f"  {Colors.WARNING}○{Colors.ENDC} {tool} - {description} (optional)")

def start_server():
    """Start the Neural Nexus server"""
    print(f"\n{Colors.OKGREEN}Starting Neural Nexus IDE...{Colors.ENDC}")
    
    # Check if server file exists
    server_file = Path("neural_nexus_server.py")
    if not server_file.exists():
        print(f"{Colors.FAIL}Error: neural_nexus_server.py not found!{Colors.ENDC}")
        print("Please ensure all files are in the same directory.")
        sys.exit(1)
    
    # Load config
    config = {"port": 8765, "host": "127.0.0.1", "auto_open_browser": True}
    if Path("config.json").exists():
        with open("config.json") as f:
            config.update(json.load(f))
    
    # Start server
    cmd = [sys.executable, "neural_nexus_server.py", "--port", str(config["port"])]
    
    try:
        if platform.system() == "Windows":
            # On Windows, use CREATE_NEW_CONSOLE to open in new window
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # On Unix-like systems
            process = subprocess.Popen(cmd)
        
        print(f"\n{Colors.OKGREEN}✓ Server started on http://{config['host']}:{config['port']}{Colors.ENDC}")
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Open browser
        if config.get("auto_open_browser", True):
            url = f"http://{config['host']}:{config['port']}"
            print(f"{Colors.OKBLUE}Opening browser...{Colors.ENDC}")
            webbrowser.open(url)
        
        print(f"\n{Colors.OKCYAN}Neural Nexus IDE is running!{Colors.ENDC}")
        print(f"\nPress Ctrl+C to stop the server")
        
        # Keep script running
        try:
            process.wait()
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Shutting down...{Colors.ENDC}")
            process.terminate()
            
    except Exception as e:
        print(f"{Colors.FAIL}Error starting server: {e}{Colors.ENDC}")
        sys.exit(1)

def main():
    """Main setup function"""
    print_banner()
    
    # Run checks
    check_python_version()
    check_dependencies()
    setup_directories()
    save_files()
    check_tools()
    
    # Ask to start server
    print(f"\n{Colors.BOLD}Setup complete!{Colors.ENDC}")
    response = input("\nStart Neural Nexus IDE now? (y/n): ").lower()
    
    if response == 'y':
        start_server()
    else:
        print(f"\n{Colors.OKBLUE}To start the server later, run:{Colors.ENDC}")
        print(f"  python neural_nexus_server.py")
        print(f"\n{Colors.OKBLUE}Or use this setup script:{Colors.ENDC}")
        print(f"  python setup.py --start")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Nexus IDE Setup")
    parser.add_argument("--start", action="store_true", help="Start server immediately")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")
    args = parser.parse_args()
    
    if args.start:
        start_server()
    elif args.check:
        check_python_version()
        check_dependencies()
        check_tools()
    else:
        main()
