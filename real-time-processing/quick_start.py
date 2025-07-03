#!/usr/bin/env python3
"""
Quick Start Script for Geometry Detection System
===============================================

This script automatically checks your setup and helps you get started
with the geometry detection system.

Just run: python quick_start.py
"""

import subprocess
import sys
import os
import platform

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60)

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher required!")
        print("Please upgrade Python: https://www.python.org/downloads/")
        return False
    
    print("✅ Python version OK")
    return True

def check_and_install_packages():
    """Check and install required packages"""
    print_header("Checking Required Packages")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
    }
    
    optional = {
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas'
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    print("\nRequired packages:")
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_required.append(package)
    
    # Check optional packages
    print("\nOptional packages:")
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} - Missing (optional)")
            missing_optional.append(package)
    
    # Install missing required packages
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        response = input("\nInstall missing packages? (y/n): ").lower()
        
        if response == 'y':
            print("\nInstalling packages...")
            packages = missing_required + ['opencv-contrib-python']
            
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
                print("✅ Packages installed successfully!")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install packages")
                print("\nTry manually:")
                print(f"pip install {' '.join(packages)}")
                return False
        else:
            print("\n❌ Cannot continue without required packages")
            return False
    
    # Offer to install optional packages
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        response = input("Install optional packages? (y/n): ").lower()
        
        if response == 'y':
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_optional)
                print("✅ Optional packages installed")
            except:
                print("⚠️  Some optional packages failed to install")
    
    return True

def test_opencv_import():
    """Test OpenCV import and basic functionality"""
    print_header("Testing OpenCV")
    
    try:
        import cv2
        import numpy as np
        
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test basic operations
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("✅ Basic operations work")
        
        # Check for CUDA
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count > 0:
                print(f"✅ CUDA available: {cuda_count} device(s)")
            else:
                print("ℹ️  CUDA not available (CPU mode will be used)")
        except:
            print("ℹ️  CUDA module not available (CPU mode will be used)")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def find_working_camera():
    """Find a working camera"""
    print_header("Finding Camera")
    
    try:
        import cv2
        
        # Try to find a working camera
        working_index = None
        
        for i in range(5):
            print(f"Checking camera index {i}...", end=' ')
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print("✅ Works!")
                    working_index = i
                    cap.release()
                    break
                else:
                    print("❌ No frames")
            else:
                print("❌ Cannot open")
            
            cap.release()
        
        if working_index is not None:
            print(f"\n✅ Found working camera at index {working_index}")
            return working_index
        else:
            print("\n❌ No working camera found!")
            print("\nTroubleshooting:")
            
            system = platform.system()
            if system == "Windows":
                print("- Check Device Manager for camera")
                print("- Try Windows Camera app")
                print("- Close other apps using camera")
            elif system == "Linux":
                print("- Run: ls /dev/video*")
                print("- Install v4l-utils: sudo apt-get install v4l-utils")
                print("- Check permissions: sudo chmod 666 /dev/video0")
            elif system == "Darwin":
                print("- Check System Preferences > Security & Privacy > Camera")
                print("- Allow Terminal/Python camera access")
            
            return None
            
    except ImportError:
        print("❌ OpenCV not installed!")
        return None

def test_simple_detection():
    """Test simple shape detection"""
    print_header("Testing Shape Detection")
    
    try:
        import cv2
        import numpy as np
        
        # Create test image with shapes
        test_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Draw test shapes
        cv2.rectangle(test_img, (50, 50), (150, 150), (0, 0, 0), -1)
        cv2.circle(test_img, (250, 100), 50, (0, 0, 0), -1)
        pts = np.array([[350, 50], [450, 50], [400, 150]], np.int32)
        cv2.fillPoly(test_img, [pts], (0, 0, 0))
        
        # Convert to grayscale
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"✅ Detected {len(contours)} shapes in test image")
        
        if len(contours) == 3:
            print("✅ Shape detection working correctly!")
            return True
        else:
            print("⚠️  Unexpected number of shapes detected")
            return True  # Still OK, just a warning
            
    except Exception as e:
        print(f"❌ Shape detection test failed: {e}")
        return False

def create_test_files():
    """Create test files for users"""
    print_header("Creating Test Files")
    
    files_created = []
    
    # Create run_geometry_detector.bat for Windows
    if platform.system() == "Windows":
        bat_content = """@echo off
echo Starting Geometry Detector...
python advanced_geometry_detector.py
pause
"""
        with open("run_geometry_detector.bat", "w") as f:
            f.write(bat_content)
        files_created.append("run_geometry_detector.bat")
    
    # Create run_geometry_detector.sh for Linux/Mac
    else:
        sh_content = """#!/bin/bash
echo "Starting Geometry Detector..."
python3 advanced_geometry_detector.py
"""
        with open("run_geometry_detector.sh", "w") as f:
            f.write(sh_content)
        os.chmod("run_geometry_detector.sh", 0o755)
        files_created.append("run_geometry_detector.sh")
    
    # Create test_camera.py if it doesn't exist
    if not os.path.exists("test_camera.py"):
        test_content = '''import cv2
print("Testing camera...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera works! Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Camera not found!")
'''
        with open("test_camera.py", "w") as f:
            f.write(test_content)
        files_created.append("test_camera.py")
    
    if files_created:
        print(f"✅ Created helper files: {', '.join(files_created)}")
    
    return True

def show_next_steps(camera_index):
    """Show next steps to user"""
    print_header("Setup Complete! Next Steps")
    
    print("\n✅ Your system is ready for geometry detection!")
    
    print("\n1. TEST YOUR CAMERA:")
    print("   python camera_test.py")
    
    print("\n2. RUN SIMPLE VERSION:")
    print("   python simple_geometry_detector.py")
    
    print("\n3. RUN FULL VERSION:")
    if camera_index is not None and camera_index != 0:
        print(f"   python advanced_geometry_detector.py -s {camera_index}")
    else:
        print("   python advanced_geometry_detector.py")
    
    if platform.system() == "Windows":
        print("\n   Or double-click: run_geometry_detector.bat")
    else:
        print("\n   Or run: ./run_geometry_detector.sh")
    
    print("\n4. USE VIDEO FILE:")
    print("   python advanced_geometry_detector.py -f video.mp4")
    
    print("\n5. GET HELP:")
    print("   python advanced_geometry_detector.py --help")
    
    print("\nCONTROLS:")
    print("  'q' - Quit")
    print("  'p' - Pause")
    print("  's' - Screenshot")
    print("  'g' - Toggle GPU")
    print("  '+/-' - Adjust sensitivity")

def main():
    """Main setup function"""
    print_header("Geometry Detection System - Quick Start")
    print("\nThis script will check your system and help you get started.")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check and install packages
    if not check_and_install_packages():
        return
    
    # Test OpenCV
    if not test_opencv_import():
        print("\n❌ OpenCV not working properly")
        print("Try reinstalling:")
        print("pip uninstall opencv-python opencv-contrib-python")
        print("pip install opencv-python opencv-contrib-python")
        return
    
    # Find camera
    camera_index = find_working_camera()
    
    # Test shape detection
    test_simple_detection()
    
    # Create helper files
    create_test_files()
    
    # Show next steps
    show_next_steps(camera_index)
    
    # Offer to run simple test
    print("\n" + "="*60)
    response = input("\nRun simple camera test now? (y/n): ").lower()
    
    if response == 'y':
        print("\nStarting simple camera test...")
        print("Press 'q' to quit\n")
        
        try:
            if camera_index is not None:
                subprocess.run([sys.executable, "simple_geometry_detector.py"])
            else:
                print("No camera found. Please check your camera and try again.")
        except FileNotFoundError:
            print("simple_geometry_detector.py not found!")
            print("Make sure all scripts are in the same directory.")
        except KeyboardInterrupt:
            print("\nTest interrupted")
    
    print("\n✅ Setup complete! Happy shape detecting! 🎉")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nPlease report this issue with the full error message.")
