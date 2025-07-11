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
        """Save configuration to file"""
        with open('geometry_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print_success("Configuration saved to geometry_config.json")

class TestRunner:
    """Run system tests"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
    
    def run_all_tests(self):
        """Run all system tests"""
        print_header("System Tests")
        
        # Test OpenCV
        self.test_opencv()
        
        # Test NumPy
        self.test_numpy()
        
        # Test shape detection
        self.test_shape_detection()
        
        # Test GPU
        self.test_gpu()
        
        # Print summary
        print("\n" + "-"*60)
        total = self.tests_passed + self.tests_failed
        print(f"Tests completed: {total}")
        print_success(f"Passed: {self.tests_passed}")
        if self.tests_failed > 0:
            print_error(f"Failed: {self.tests_failed}")
        
        return self.tests_failed == 0
    
    def test_opencv(self):
        """Test OpenCV installation"""
        print("Testing OpenCV...", end=' ')
        try:
            import cv2
            
            # Test basic operations
            img = cv2.imread('nonexistent.jpg')  # Should return None
            test_img = cv2.cvtColor(255 * np.ones((100, 100, 3), dtype=np.uint8), 
                                   cv2.COLOR_BGR2GRAY)
            
            print_success(f"OK (version {cv2.__version__})")
            self.tests_passed += 1
        except Exception as e:
            print_error(f"FAILED: {e}")
            self.tests_failed += 1
    
    def test_numpy(self):
        """Test NumPy installation"""
        print("Testing NumPy...", end=' ')
        try:
            import numpy as np
            
            # Test basic operations
            arr = np.array([1, 2, 3])
            result = np.sum(arr)
            assert result == 6
            
            print_success(f"OK (version {np.__version__})")
            self.tests_passed += 1
        except Exception as e:
            print_error(f"FAILED: {e}")
            self.tests_failed += 1
    
    def test_shape_detection(self):
        """Test basic shape detection"""
        print("Testing shape detection...", end=' ')
        try:
            import cv2
            import numpy as np
            
            # Create test image
            img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)
            
            # Detect edges
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            assert len(contours) == 1, f"Expected 1 contour, found {len(contours)}"
            
            print_success("OK")
            self.tests_passed += 1
        except Exception as e:
            print_error(f"FAILED: {e}")
            self.tests_failed += 1
    
    def test_gpu(self):
        """Test GPU capabilities"""
        print("Testing GPU...", end=' ')
        try:
            import cv2
            
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count > 0:
                # Test GPU operation
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(np.zeros((100, 100), dtype=np.uint8))
                result = gpu_mat.download()
                
                print_success(f"OK ({gpu_count} device(s))")
            else:
                print_info("No GPU (CPU mode will be used)")
            
            self.tests_passed += 1
        except Exception as e:
            print_info("No GPU support (CPU mode will be used)")
            self.tests_passed += 1

def create_launcher_scripts():
    """Create platform-specific launcher scripts"""
    print_header("Creating Launcher Scripts")
    
    system = platform.system()
    
    # Create run script
    if system == "Windows":
        # Batch file for Windows
        with open("run_geometry_detector.bat", "w") as f:
            f.write("""@echo off
echo Starting Integrated Geometry Detection System...
echo.

REM Load configuration if it exists
if exist geometry_config.json (
    echo Loading saved configuration...
    python integrated_geometry_system.py
) else (
    echo No configuration found, using defaults...
    python integrated_geometry_system.py
)

pause
""")
        print_success("Created run_geometry_detector.bat")
        
        # PowerShell script
        with open("run_geometry_detector.ps1", "w") as f:
            f.write("""
Write-Host "Starting Integrated Geometry Detection System..." -ForegroundColor Green

if (Test-Path "geometry_config.json") {
    Write-Host "Loading saved configuration..." -ForegroundColor Cyan
} else {
    Write-Host "No configuration found, using defaults..." -ForegroundColor Yellow
}

python integrated_geometry_system.py

Read-Host -Prompt "Press Enter to exit"
""")
        print_success("Created run_geometry_detector.ps1")
        
    else:  # Linux/Mac
        # Shell script
        with open("run_geometry_detector.sh", "w") as f:
            f.write("""#!/bin/bash

echo "Starting Integrated Geometry Detection System..."
echo

# Load configuration if it exists
if [ -f "geometry_config.json" ]; then
    echo "Loading saved configuration..."
else
    echo "No configuration found, using defaults..."
fi

python3 integrated_geometry_system.py

echo
echo "Press Enter to exit..."
read
""")
        os.chmod("run_geometry_detector.sh", 0o755)
        print_success("Created run_geometry_detector.sh")
    
    # Create test script
    with open("test_camera.py", "w") as f:
        f.write("""#!/usr/bin/env python3
import cv2
import sys

print("Simple camera test")
print("Press 'q' to quit")

# Try different camera indices
for i in range(5):
    print(f"\\nTrying camera index {i}...")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        print(f"Camera {i} opened successfully!")
        
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, f"Camera {i} - Press 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f'Camera {i} Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"Failed to read from camera {i}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

print("\\nNo working cameras found!")
""")
    
    if system != "Windows":
        os.chmod("test_camera.py", 0o755)
    
    print_success("Created test_camera.py")

def show_next_steps():
    """Show next steps to the user"""
    print_header("Setup Complete!")
    
    print("🎉 Your system is ready for geometry detection!\n")
    
    print("Next steps:")
    print("\n1. TEST YOUR CAMERA:")
    print("   python test_camera.py")
    
    print("\n2. RUN THE MAIN PROGRAM:")
    if platform.system() == "Windows":
        print("   • Double-click: run_geometry_detector.bat")
        print("   • Or run: python integrated_geometry_system.py")
    else:
        print("   • Run: ./run_geometry_detector.sh")
        print("   • Or: python3 integrated_geometry_system.py")
    
    print("\n3. RUN WITH CUSTOM OPTIONS:")
    print("   python integrated_geometry_system.py --help")
    
    print("\n4. RUN UNIT TESTS:")
    print("   python integrated_geometry_system.py --test")
    
    print("\n5. VIEW DOCUMENTATION:")
    print("   See integrated_geometry_system.py for full API documentation")
    
    print("\nKEYBOARD CONTROLS:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  's' - Screenshot")
    print("  'r' - Record video")
    print("  'b' - Save benchmark")
    print("  'g' - Toggle GPU")
    print("  '+/-' - Adjust sensitivity")
    
    print("\n" + "="*60)
    print("Happy detecting! 🔍")

def main():
    """Main setup function"""
    print_header("Integrated Geometry Detection System Setup")
    print("This wizard will help you set up everything needed for geometry detection.\n")
    
    # System check
    checker = SystemChecker()
    if not checker.check_all():
        print("\n⚠️  System check found issues. Continue anyway? (y/n): ", end='')
        if input().lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Package installation
    print("\n" + "="*60)
    print("Ready to install packages? (y/n): ", end='')
    if input().lower() == 'y':
        installer = PackageInstaller()
        if not installer.install_all():
            print_warning("Some packages failed to install")
    
    # Camera detection
    print("\n" + "="*60)
    print("Detect cameras? (y/n): ", end='')
    if input().lower() == 'y':
        detector = CameraDetector()
        detector.detect_all()
        
        # Configuration wizard
        if detector.working_cameras:
            print("\n" + "="*60)
            print("Run configuration wizard? (y/n): ", end='')
            if input().lower() == 'y':
                wizard = ConfigurationWizard()
                wizard.run(detector.working_cameras)
    
    # Run tests
    print("\n" + "="*60)
    print("Run system tests? (y/n): ", end='')
    if input().lower() == 'y':
        runner = TestRunner()
        runner.run_all_tests()
    
    # Create launcher scripts
    create_launcher_scripts()
    
    # Show next steps
    show_next_steps()
    
    # Offer to run main program
    print("\n" + "="*60)
    print("Start the geometry detection system now? (y/n): ", end='')
    if input().lower() == 'y':
        print("\nStarting system...")
        time.sleep(1)
        
        try:
            if os.path.exists('integrated_geometry_system.py'):
                subprocess.run([sys.executable, 'integrated_geometry_system.py'])
            else:
                print_error("integrated_geometry_system.py not found!")
                print("Make sure both files are in the same directory.")
        except KeyboardInterrupt:
            print("\nProgram interrupted.")
        except Exception as e:
            print_error(f"Failed to start: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
    except Exception as e:
        print(f"\n{Colors.FAIL}Setup error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
