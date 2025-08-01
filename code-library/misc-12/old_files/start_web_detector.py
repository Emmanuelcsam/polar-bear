#!/usr/bin/env python3
"""
Web Core Detector Startup Script
Launches the web-based visual core detection system
"""

import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required dependencies are available"""
    print("=== CHECKING DEPENDENCIES ===")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    
    # Check NumPy
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    # Check Flask
    try:
        import flask
        print(f"✓ Flask {flask.__version__}")
    except ImportError:
        print("❌ Flask not installed")
        return False
    
    # Check Pylon (optional)
    try:
        from pypylon import pylon
        print("✓ Pylon SDK available")
    except ImportError:
        print("⚠ Pylon SDK not available - will use OpenCV fallback")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("=== INSTALLING DEPENDENCIES ===")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "opencv-python", "numpy", "flask"])
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main startup function"""
    print("=========================================")
    print("   Web Core Detection System")
    print("=========================================")
    
    # Check dependencies
    if not check_dependencies():
        print("\nAttempting to install dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            print("Please install manually: pip install opencv-python numpy flask")
            return 1
    
    # Import and run web core detector
    try:
        from web_core_detector import main as run_web_detector
        print("\n=== STARTING WEB CORE DETECTOR ===")
        print("🌐 Opening web browser in 3 seconds...")
        
        # Start the web detector in a separate thread
        import threading
        detector_thread = threading.Thread(target=run_web_detector)
        detector_thread.daemon = True
        detector_thread.start()
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open web browser
        try:
            webbrowser.open('http://localhost:5000')
            print("✅ Web browser opened!")
        except Exception as e:
            print(f"⚠ Could not open browser automatically: {e}")
            print("🌐 Please open your browser and go to: http://localhost:5000")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1) 