#!/usr/bin/env python3
"""
Integrated Geometry Detection System - Setup and Installation Helper
===================================================================

This script automatically sets up your environment for the geometry detection system.
It checks dependencies, installs packages, tests cameras, and provides troubleshooting.

Usage: python setup_installer.py
"""

import subprocess
import sys
import os
import platform
import time
import json
import urllib.request
import zipfile
import tarfile
from typing import List, Dict, Tuple, Optional
import importlib.util
import shared_config # Import the shared configuration module

# ANSI color codes for pretty output
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

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

class SystemChecker:
    """Check system requirements and capabilities"""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.python_version = sys.version_info
        self.issues = []
        self.warnings = []
    
    def check_all(self) -> bool:
        """Run all system checks"""
        print_header("System Requirements Check")
        
        all_good = True
        
        # Check Python version
        if not self.check_python():
            all_good = False
        
        # Check system
        self.check_system()
        
        # Check hardware
        self.check_hardware()
        
        # Check existing packages
        self.check_existing_packages()
        
        # Print summary
        print("\n" + "-"*60)
        if self.issues:
            print_error(f"Found {len(self.issues)} critical issues:")
            for issue in self.issues:
                print(f"  • {issue}")
        
        if self.warnings:
            print_warning(f"Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if all_good and not self.issues:
            print_success("System check passed!")
        
        return all_good and not self.issues
    
    def check_python(self) -> bool:
        """Check Python version"""
        print(f"Python version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
        if self.python_version.major < 3:
            self.issues.append("Python 3.x required")
            return False
        
        if self.python_version.minor < 7:
            self.issues.append("Python 3.7 or higher required")
            return False
        
        if self.python_version.minor > 11:
            self.warnings.append("Python 3.12+ may have compatibility issues with some packages")
        
        print_success("Python version OK")
        return True
    
    def check_system(self):
        """Check operating system"""
        print(f"\nOperating System: {self.system} {platform.release()}")
        print(f"Architecture: {self.machine}")
        
        if self.system == "Windows":
            # Check Windows version
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                    r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
                version = winreg.QueryValueEx(key, "CurrentVersion")[0]
                build = winreg.QueryValueEx(key, "CurrentBuild")[0]
                print(f"Windows version: {version} (Build {build})")
                winreg.CloseKey(key)
            except:
                pass
        
        elif self.system == "Linux":
            # Check distribution
            try:
                with open("/etc/os-release") as f:
                    for line in f:
                        if line.startswith("PRETTY_NAME"):
                            distro = line.split("=")[1].strip().strip('"')
                            print(f"Distribution: {distro}")
                            break
            except:
                pass
        
        elif self.system == "Darwin":
            # Check macOS version
            mac_ver = platform.mac_ver()[0]
            print(f"macOS version: {mac_ver}")
            
            # Check for M1/M2
            if self.machine == "arm64":
                print_info("Apple Silicon detected (M1/M2)")
                self.warnings.append("Some packages may need Rosetta 2 or arm64 versions")
    
    def check_hardware(self):
        """Check hardware capabilities"""
        print("\nHardware Check:")
        
        # CPU cores
        try:
            import multiprocessing
            cores = multiprocessing.cpu_count()
            print(f"CPU cores: {cores}")
            if cores < 2:
                self.warnings.append("Low CPU core count may affect performance")
        except:
            pass
        
        # RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            print(f"RAM: {ram_gb:.1f} GB")
            if ram_gb < 4:
                self.warnings.append("Less than 4GB RAM may cause performance issues")
        except:
            pass
        
        # GPU check
        self.check_gpu()
    
    def check_gpu(self):
        """Check for GPU availability"""
        print("\nGPU Check:")
        
        # Check NVIDIA GPU
        nvidia_found = False
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                nvidia_found = True
                print_success("NVIDIA GPU detected")
                
                # Parse GPU info
                for line in result.stdout.split('\n'):
                    if 'NVIDIA' in line and 'CUDA' not in line:
                        gpu_info = line.strip()
                        if gpu_info:
                            print(f"  GPU: {gpu_info}")
                            break
        except:
            print_info("No NVIDIA GPU found")
        
        # Check CUDA
        if nvidia_found:
            try:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'release' in line:
                            cuda_version = line.split('release')[1].split(',')[0].strip()
                            print_success(f"CUDA installed: version {cuda_version}")
                            break
                else:
                    print_warning("CUDA toolkit not found - GPU acceleration will be limited")
                    self.warnings.append("Install CUDA for full GPU acceleration")
            except:
                print_warning("CUDA toolkit not found")
    
    def check_existing_packages(self):
        """Check for existing package installations"""
        print("\nChecking existing packages:")
        
        packages = {
            'cv2': ('opencv-python', 'OpenCV'),
            'numpy': ('numpy', 'NumPy'),
            'PIL': ('pillow', 'Pillow'),
            'pypylon': ('pypylon', 'Pylon/Basler SDK'),
            'scipy': ('scipy', 'SciPy'),
            'matplotlib': ('matplotlib', 'Matplotlib'),
            'pandas': ('pandas', 'Pandas'),
            'psutil': ('psutil', 'PSUtil'),
            'GPUtil': ('gputil', 'GPUtil')
        }
        
        for module, (package, name) in packages.items():
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
                    print_info(f"{name} not installed")
            except:
                print_info(f"{name} not installed")

class PackageInstaller:
    """Handle package installation"""
    
    def __init__(self):
        self.system = platform.system()
        self.required_packages = [
            'opencv-python',
            'opencv-contrib-python',
            'numpy',
            'psutil'
        ]
        self.optional_packages = [
            'scipy',
            'matplotlib',
            'pandas',
            'pillow',
            'gputil'
        ]
        self.system_packages = {
            'Linux': {
                'apt': [
                    'python3-dev',
                    'python3-pip',
                    'libgl1-mesa-glx',
                    'libglib2.0-0',
                    'libsm6',
                    'libxext6',
                    'libxrender-dev',
                    'libgomp1',
                    'v4l-utils'
                ],
                'yum': [
                    'python3-devel',
                    'mesa-libGL',
                    'glib2',
                    'libSM',
                    'libXext',
                    'libXrender',
                    'libgomp'
                ]
            }
        }
    
    def install_all(self) -> bool:
        """Install all required packages"""
        print_header("Package Installation")
        
        # Update pip first
        if not self.update_pip():
            print_warning("Failed to update pip, continuing anyway...")
        
        # Install system dependencies
        if self.system == "Linux":
            self.install_system_dependencies()
        
        # Install Python packages
        success = self.install_python_packages()
        
        # Check for Pylon SDK
        self.check_pylon_sdk()
        
        return success
    
    def update_pip(self) -> bool:
        """Update pip to latest version"""
        print("Updating pip...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
            print_success("pip updated")
            return True
        except subprocess.CalledProcessError:
            return False
    
    def install_system_dependencies(self):
        """Install system-level dependencies on Linux"""
        print("\nInstalling system dependencies...")
        
        # Detect package manager
        if os.path.exists('/usr/bin/apt-get'):
            pkg_manager = 'apt'
            install_cmd = ['sudo', 'apt-get', 'install', '-y']
        elif os.path.exists('/usr/bin/yum'):
            pkg_manager = 'yum'
            install_cmd = ['sudo', 'yum', 'install', '-y']
        else:
            print_warning("Could not detect package manager")
            return
        
        packages = self.system_packages['Linux'][pkg_manager]
        
        print(f"Using {pkg_manager} to install system packages...")
        print("This may require your sudo password.")
        
        try:
            # Update package list first
            if pkg_manager == 'apt':
                subprocess.run(['sudo', 'apt-get', 'update'], check=False)
            
            # Install packages
            subprocess.run(install_cmd + packages, check=False)
            print_success("System dependencies installed")
        except Exception as e:
            print_warning(f"Failed to install some system dependencies: {e}")
    
    def install_python_packages(self) -> bool:
        """Install Python packages"""
        print("\nInstalling Python packages...")
        
        all_success = True
        
        # Install required packages
        print("\nRequired packages:")
        for package in self.required_packages:
            if not self.install_package(package):
                all_success = False
        
        # Ask about optional packages
        print("\nOptional packages enhance functionality:")
        for package in self.optional_packages:
            if self.ask_install_optional(package):
                self.install_package(package)
        
        return all_success
    
    def install_package(self, package: str) -> bool:
        """Install a single package"""
        print(f"Installing {package}...", end=' ')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print_success("OK")
            return True
        except subprocess.CalledProcessError:
            print_error("FAILED")
            return False
    
    def ask_install_optional(self, package: str) -> bool:
        """Ask user about optional package"""
        descriptions = {
            'scipy': 'Scientific computing (recommended)',
            'matplotlib': 'Plotting and visualization',
            'pandas': 'Data analysis and export',
            'pillow': 'Additional image format support',
            'gputil': 'GPU monitoring'
        }
        
        desc = descriptions.get(package, '')
        response = input(f"Install {package}? {desc} (y/n): ").lower()
        return response in ['y', 'yes']
    
    def check_pylon_sdk(self):
        """Check for Pylon SDK installation"""
        print("\nChecking for Basler Pylon SDK...")
        
        try:
            import pypylon
            print_success("pypylon is installed")
        except ImportError:
            print_info("pypylon not installed (required only for Basler cameras)")
            print("To install pypylon:")
            print("  1. Download Pylon SDK from https://www.baslerweb.com/en/software/pylon/")
            print("  2. Install the SDK for your platform")
            print("  3. Run: pip install pypylon")

class CameraDetector:
    """Detect and test available cameras"""
    
    def __init__(self):
        self.cameras = []
        self.working_cameras = []
    
    def detect_all(self):
        """Detect all available cameras"""
        print_header("Camera Detection")
        
        # Try to import OpenCV
        try:
            import cv2
        except ImportError:
            print_error("OpenCV not installed - cannot detect cameras")
            return
        
        print("Scanning for cameras...")
        
        # Check OpenCV cameras
        self.detect_opencv_cameras()
        
        # Check for Pylon cameras
        self.detect_pylon_cameras()
        
        # Test cameras
        if self.cameras:
            self.test_cameras()
        else:
            print_warning("No cameras detected!")
            self.suggest_camera_fixes()
    
    def detect_opencv_cameras(self):
        """Detect OpenCV-compatible cameras"""
        import cv2
        
        for i in range(5):  # Check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    self.cameras.append({
                        'type': 'opencv',
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'name': f'Camera {i}'
                    })
                    
                    print_success(f"Found camera at index {i}: {width}x{height} @ {fps}fps")
                cap.release()
    
    def detect_pylon_cameras(self):
        """Detect Pylon/Basler cameras"""
        try:
            from pypylon import pylon
            
            factory = pylon.TlFactory.GetInstance()
            devices = factory.EnumerateDevices()
            
            if devices:
                print(f"\nFound {len(devices)} Pylon camera(s):")
                for i, device in enumerate(devices):
                    name = device.GetFriendlyName()
                    self.cameras.append({
                        'type': 'pylon',
                        'index': i,
                        'name': name
                    })
                    print_success(f"Pylon camera {i}: {name}")
        except ImportError:
            pass
        except Exception as e:
            print_warning(f"Error detecting Pylon cameras: {e}")
    
    def test_cameras(self):
        """Test detected cameras"""
        print("\nTesting cameras...")
        
        for camera in self.cameras:
            if camera['type'] == 'opencv':
                if self.test_opencv_camera(camera['index']):
                    self.working_cameras.append(camera)
    
    def test_opencv_camera(self, index: int) -> bool:
        """Test OpenCV camera"""
        import cv2
        
        print(f"Testing camera {index}...", end=' ')
        cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            print_error("Failed to open")
            return False
        
        # Try to read multiple frames
        success_count = 0
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
        
        cap.release()
        
        if success_count >= 8:
            print_success(f"OK ({success_count}/10 frames)")
            return True
        else:
            print_error(f"Unstable ({success_count}/10 frames)")
            return False
    
    def suggest_camera_fixes(self):
        """Suggest fixes for camera issues"""
        system = platform.system()
        
        print("\nTroubleshooting suggestions:")
        
        if system == "Windows":
            print("  • Check Device Manager for camera drivers")
            print("  • Try Windows Camera app to verify camera works")
            print("  • Close other applications using the camera")
            print("  • Run as Administrator")
            
        elif system == "Linux":
            print("  • Check camera permissions: ls -la /dev/video*")
            print("  • Add user to video group: sudo usermod -a -G video $USER")
            print("  • Install v4l-utils: sudo apt-get install v4l-utils")
            print("  • List cameras: v4l2-ctl --list-devices")
            
        elif system == "Darwin":
            print("  • Check System Preferences > Security & Privacy > Camera")
            print("  • Allow Terminal/Python camera access")
            print("  • Test with Photo Booth app first")

class ConfigurationWizard:
    """Interactive configuration wizard"""
    
    def __init__(self):
        self.config = {
            'camera_backend': 'opencv',
            'camera_index': 0,
            'use_gpu': True,
            'enable_tube_detection': True,
            'enable_benchmarking': True,
            'log_level': 'INFO'
        }
    
    def run(self, working_cameras: List[Dict]):
        """Run configuration wizard"""
        print_header("Configuration Wizard")
        
        # Camera selection
        if working_cameras:
            self.select_camera(working_cameras)
        
        # GPU settings
        self.configure_gpu()
        
        # Feature selection
        self.select_features()
        
        # Save configuration
        self.save_config()
    
    def select_camera(self, cameras: List[Dict]):
        """Select default camera"""
        print("Available cameras:")
        for i, cam in enumerate(cameras):
            print(f"  {i}: {cam['name']} ({cam.get('width', '?')}x{cam.get('height', '?')})")
        
        while True:
            try:
                choice = input(f"\nSelect default camera (0-{len(cameras)-1}): ")
                idx = int(choice)
                if 0 <= idx < len(cameras):
                    selected = cameras[idx]
                    self.config['camera_backend'] = selected['type']
                    self.config['camera_index'] = selected['index']
                    print_success(f"Selected: {selected['name']}")
                    break
            except:
                print_error("Invalid selection")
    
    def configure_gpu(self):
        """Configure GPU settings"""
        print("\nGPU Configuration:")
        
        # Check if CUDA is available
        cuda_available = False
        try:
            import cv2
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            pass
        
        if cuda_available:
            print_success("CUDA GPU acceleration is available")
            response = input("Enable GPU acceleration? (Y/n): ").lower()
            self.config['use_gpu'] = response != 'n'
        else:
            print_info("GPU acceleration not available")
            self.config['use_gpu'] = False
    
    def select_features(self):
        """Select which features to enable"""
        print("\nFeature Selection:")
        
        # Tube detection
        response = input("Enable tube angle detection? (Y/n): ").lower()
        self.config['enable_tube_detection'] = response != 'n'
        
        # Benchmarking
        response = input("Enable performance benchmarking? (Y/n): ").lower()
        self.config['enable_benchmarking'] = response != 'n'
        
        # Log level
        print("\nLog level (DEBUG/INFO/WARNING/ERROR):")
        level = input("Select log level [INFO]: ").upper() or 'INFO'
        if level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            self.config['log_level'] = level
    
    def save_config(self):
        """Save configuration to file and update shared_config"""
        with open('geometry_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Update shared_config as well
        shared_config.update_config(self.config)

        print_success("Configuration saved to geometry_config.json")

class SetupInstaller:
    """Main class to orchestrate the setup and installation process"""
    def __init__(self):
        self.status = "initialized"
        self.system_checker = SystemChecker()
        self.package_installer = PackageInstaller()
        self.camera_detector = CameraDetector()
        self.config_wizard = ConfigurationWizard()
        self.test_runner = TestRunner()
        self.working_cameras = []
        self.setup_config = {}

    def get_script_info(self):
        """Returns information about the installer's status and collected configuration."""
        return {
            "name": "Setup and Installation Helper",
            "status": self.status,
            "setup_config": self.setup_config,
            "system_check_issues": self.system_checker.issues,
            "system_check_warnings": self.system_checker.warnings,
            "detected_cameras": self.camera_detector.cameras,
            "working_cameras": self.camera_detector.working_cameras
        }

    def set_script_parameter(self, key, value):
        """Allows triggering specific setup phases or setting configuration values."""
        if key == "run_system_check":
            if value:
                self.status = "running_system_check"
                self.system_checker.check_all()
                self.status = "system_check_complete"
                return True
        elif key == "install_packages":
            if value:
                self.status = "installing_packages"
                self.package_installer.install_all()
                self.status = "package_installation_complete"
                return True
        elif key == "detect_cameras":
            if value:
                self.status = "detecting_cameras"
                self.camera_detector.detect_all()
                self.working_cameras = self.camera_detector.working_cameras
                self.status = "camera_detection_complete"
                return True
        elif key == "run_config_wizard":
            if value and self.working_cameras:
                self.status = "running_config_wizard"
                self.config_wizard.run(self.working_cameras)
                self.setup_config = self.config_wizard.config # Update internal config
                self.status = "config_wizard_complete"
                return True
        elif key == "run_tests":
            if value:
                self.status = "running_tests"
                self.test_runner.run_all_tests()
                self.status = "tests_complete"
                return True
        elif key == "create_launcher_scripts":
            if value:
                self.status = "creating_launchers"
                create_launcher_scripts()
                self.status = "launchers_created"
                return True
        elif key == "start_main_program":
            if value:
                self.status = "starting_main_program"
                print("\nStarting system...")
                time.sleep(1)
                try:
                    if os.path.exists('integrated_geometry_system.py'):
                        subprocess.run([sys.executable, 'integrated_geometry_system.py'])
                        self.status = "main_program_running"
                    else:
                        print_error("integrated_geometry_system.py not found!")
                        print("Make sure both files are in the same directory.")
                        self.status = "main_program_start_failed"
                except KeyboardInterrupt:
                    print("\nProgram interrupted.")
                    self.status = "main_program_interrupted"
                except Exception as e:
                    print_error(f"Failed to start: {e}")
                    self.status = "main_program_start_failed"
                return True
        return False

    def run_interactive_setup(self):
        """Runs the full interactive setup process."""
        print_header("Integrated Geometry Detection System Setup")
        print("This wizard will help you set up everything needed for geometry detection.\n")
        
        # System check
        if not self.system_checker.check_all():
            print("\n⚠️  System check found issues. Continue anyway? (y/n): ", end='')
            if input().lower() != 'y':
                print("Setup cancelled.")
                self.status = "cancelled"
                return
        
        # Package installation
        print("\n" + "="*60)
        print("Ready to install packages? (y/n): ", end='')
        if input().lower() == 'y':
            if not self.package_installer.install_all():
                print_warning("Some packages failed to install")
        
        # Camera detection
        print("\n" + "="*60)
        print("Detect cameras? (y/n): ", end='')
        if input().lower() == 'y':
            self.camera_detector.detect_all()
            self.working_cameras = self.camera_detector.working_cameras
            
            # Configuration wizard
            if self.working_cameras:
                print("\n" + "="*60)
                print("Run configuration wizard? (y/n): ", end='')
                if input().lower() == 'y':
                    self.config_wizard.run(self.working_cameras)
                    self.setup_config = self.config_wizard.config # Update internal config
        
        # Run tests
        print("\n" + "="*60)
        print("Run system tests? (y/n): ", end='')
        if input().lower() == 'y':
            self.test_runner.run_all_tests()
        
        # Create launcher scripts
        create_launcher_scripts()
        
        # Show next steps
        show_next_steps()
        
        # Offer to run main program
        print("\n" + "="*60)
        print("Start the geometry detection system now? (y/n): ", end='')
        if input().lower() == 'y':
            self.set_script_parameter("start_main_program", True)
        
        self.status = "completed"

installer_instance = None

def get_script_info():
    """Global function to get installer information."""
    if installer_instance:
        return installer_instance.get_script_info()
    return {"name": "Setup and Installation Helper", "status": "not_initialized"}

def set_script_parameter(key, value):
    """Global function to set installer parameters."""
    if installer_instance:
        return installer_instance.set_script_parameter(key, value)
    return False

def main():
    """Main entry point for the setup installer."""
    global installer_instance
    installer_instance = SetupInstaller()
    try:
        installer_instance.run_interactive_setup()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        installer_instance.status = "interrupted"
    except Exception as e:
        print(f"\n{Colors.FAIL}Setup error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        installer_instance.status = "error"

if __name__ == "__main__":
    main()
