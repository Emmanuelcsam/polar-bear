#!/usr/bin/env python3
"""
Interactive Troubleshooting Helper
==================================

This script helps you troubleshoot camera and OpenCV issues step by step.
Just run: python troubleshoot.py
"""

import sys
import os
import time
import subprocess
import platform

class TroubleshootHelper:
    def __init__(self):
        self.solutions = []
        self.system = platform.system()
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if self.system == 'Windows' else 'clear')
    
    def print_header(self, text):
        """Print a formatted header"""
        self.clear_screen()
        print("="*60)
        print(text.center(60))
        print("="*60)
        print()
    
    def ask_yes_no(self, question):
        """Ask a yes/no question"""
        while True:
            answer = input(f"{question} (y/n): ").lower().strip()
            if answer in ['y', 'yes']:
                return True
            elif answer in ['n', 'no']:
                return False
            else:
                print("Please answer 'y' for yes or 'n' for no.")
    
    def pause(self):
        """Pause and wait for user"""
        input("\nPress Enter to continue...")
    
    def add_solution(self, solution):
        """Add a solution to the list"""
        self.solutions.append(solution)
        
    def check_python(self):
        """Check Python installation"""
        self.print_header("Step 1: Checking Python")
        
        version = sys.version_info
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            print("\n❌ Python 3.7 or higher is required!")
            self.add_solution("Update Python to version 3.7 or higher from python.org")
            return False
        
        print("\n✅ Python version is OK!")
        return True
    
    def check_opencv(self):
        """Check OpenCV installation"""
        self.print_header("Step 2: Checking OpenCV")
        
        try:
            import cv2
            print(f"✅ OpenCV is installed: version {cv2.__version__}")
            
            # Test basic functionality
            test_img = cv2.imread('nonexistent.jpg')
            print("✅ OpenCV basic functions work")
            
            return True
            
        except ImportError:
            print("❌ OpenCV is not installed!")
            self.add_solution("Install OpenCV: pip install opencv-python opencv-contrib-python")
            
            if self.ask_yes_no("\nWould you like me to install it now?"):
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                                         'opencv-python', 'opencv-contrib-python'])
                    print("\n✅ OpenCV installed successfully!")
                    return True
                except:
                    print("\n❌ Failed to install automatically")
                    self.add_solution("Try manual install: pip install opencv-python")
                    
        except Exception as e:
            print(f"❌ OpenCV error: {e}")
            self.add_solution("Reinstall OpenCV: pip uninstall opencv-python && pip install opencv-python")
            
        return False
    
    def check_camera_access(self):
        """Check camera access"""
        self.print_header("Step 3: Checking Camera Access")
        
        try:
            import cv2
            
            print("Looking for cameras...")
            found_camera = False
            working_index = None
            
            for i in range(5):
                print(f"\nTrying camera index {i}...", end=' ')
                cap = cv2.VideoCapture(i)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print("✅ WORKS!")
                        found_camera = True
                        working_index = i
                        
                        # Show camera info
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        print(f"   Resolution: {width}x{height}")
                        print(f"   FPS: {fps}")
                        
                        cap.release()
                        
                        if self.ask_yes_no("\nIs this your camera?"):
                            break
                    else:
                        print("❌ Opens but no frames")
                        cap.release()
                else:
                    print("❌ Cannot open")
            
            if found_camera:
                print(f"\n✅ Found working camera at index {working_index}")
                self.add_solution(f"Use camera index {working_index} in your scripts")
                return True, working_index
            else:
                print("\n❌ No working camera found!")
                return False, None
                
        except Exception as e:
            print(f"❌ Error checking cameras: {e}")
            return False, None
    
    def diagnose_camera_issues(self):
        """Diagnose why camera isn't working"""
        self.print_header("Camera Diagnostics")
        
        print("Let's figure out why the camera isn't working...\n")
        
        # Check if camera is being used
        if self.ask_yes_no("Are any of these apps open: Zoom, Teams, Skype, Discord, OBS?"):
            self.add_solution("Close all applications that might be using the camera")
            print("\n⚠️  Close those apps and try again")
            
        # Check physical connection
        if self.ask_yes_no("Are you using an external USB camera?"):
            self.add_solution("Try different USB ports")
            self.add_solution("Check USB cable connection")
            self.add_solution("Try the camera on another computer")
            print("\n⚠️  USB connection might be the issue")
        
        # System-specific checks
        if self.system == "Windows":
            print("\nWindows-specific checks:")
            print("1. Open Device Manager")
            print("2. Look for 'Cameras' or 'Imaging devices'")
            print("3. Check for yellow warning triangles")
            
            if self.ask_yes_no("\nDo you see any warnings or missing drivers?"):
                self.add_solution("Update camera drivers in Device Manager")
                self.add_solution("Download drivers from manufacturer website")
                
            print("\n4. Try Windows Camera app")
            if self.ask_yes_no("Does the Windows Camera app work?"):
                self.add_solution("The camera works! Issue is with Python permissions")
                self.add_solution("Try running Python as Administrator")
            else:
                self.add_solution("Camera has system-wide issues - check drivers")
                
        elif self.system == "Linux":
            print("\nLinux-specific checks:")
            
            # Check video devices
            print("\nChecking for video devices...")
            try:
                result = subprocess.run(['ls', '-la', '/dev/video*'], 
                                      capture_output=True, text=True, shell=True)
                print(result.stdout)
                
                if '/dev/video' not in result.stdout:
                    self.add_solution("No video devices found - check if camera is recognized")
                    self.add_solution("Run: sudo dmesg | grep -i camera")
                else:
                    self.add_solution("Camera device exists - likely a permission issue")
                    self.add_solution("Run: sudo chmod 666 /dev/video0")
                    self.add_solution("Or add user to video group: sudo usermod -a -G video $USER")
            except:
                pass
                
            # Check with v4l2
            if self.ask_yes_no("\nDo you have v4l-utils installed?"):
                print("Run: v4l2-ctl --list-devices")
            else:
                self.add_solution("Install v4l-utils: sudo apt-get install v4l-utils")
                
        elif self.system == "Darwin":  # macOS
            print("\nmacOS-specific checks:")
            print("\n1. Go to System Preferences")
            print("2. Click Security & Privacy")
            print("3. Click Camera")
            print("4. Make sure Terminal (or your Python app) is checked")
            
            if not self.ask_yes_no("\nIs Terminal/Python allowed camera access?"):
                self.add_solution("Enable camera access for Terminal in System Preferences")
                print("\n⚠️  This is likely the issue!")
    
    def test_solutions(self, camera_index=None):
        """Test recommended solutions"""
        self.print_header("Testing Solutions")
        
        if camera_index is not None:
            print(f"Let's test camera index {camera_index}...\n")
            
            try:
                import cv2
                
                # Create simple test script
                test_code = f"""import cv2
import numpy as np

print("Opening camera at index {camera_index}...")
cap = cv2.VideoCapture({camera_index})

if not cap.isOpened():
    print("Failed to open camera!")
else:
    print("Camera opened! Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if ret:
            # Add text
            cv2.putText(frame, "Camera Works! Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw test shapes
            cv2.rectangle(frame, (50, 50), (200, 200), (255, 0, 0), 2)
            cv2.circle(frame, (300, 150), 50, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No frames!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete!")
"""
                
                # Save test script
                with open('camera_test_simple.py', 'w') as f:
                    f.write(test_code)
                
                print("Created camera_test_simple.py")
                print("\nRun it with: python camera_test_simple.py")
                
                if self.ask_yes_no("\nRun the test now?"):
                    subprocess.run([sys.executable, 'camera_test_simple.py'])
                    
            except Exception as e:
                print(f"Error creating test: {e}")
    
    def show_summary(self):
        """Show summary of solutions"""
        self.print_header("Troubleshooting Summary")
        
        if not self.solutions:
            print("✅ Everything seems to be working!")
            print("\nYou should be able to run:")
            print("  python advanced_geometry_detector.py")
        else:
            print("Here are the issues found and solutions:\n")
            
            for i, solution in enumerate(self.solutions, 1):
                print(f"{i}. {solution}")
            
            print("\n" + "-"*60)
            print("\nTry these solutions in order, then run:")
            print("  python quick_start.py")
            print("\nOr test directly with:")
            print("  python simple_geometry_detector.py")
    
    def run(self):
        """Run the troubleshooting process"""
        self.print_header("Camera & OpenCV Troubleshooting Helper")
        
        print("This will help you fix common issues step by step.")
        print("Answer the questions and I'll find solutions!\n")
        
        self.pause()
        
        # Check Python
        if not self.check_python():
            self.pause()
            self.show_summary()
            return
        
        self.pause()
        
        # Check OpenCV
        if not self.check_opencv():
            self.pause()
            self.show_summary()
            return
            
        self.pause()
        
        # Check camera
        camera_works, camera_index = self.check_camera_access()
        
        if not camera_works:
            self.pause()
            self.diagnose_camera_issues()
            self.pause()
        else:
            # Test with working camera
            if self.ask_yes_no("\nWould you like to create a test script?"):
                self.test_solutions(camera_index)
        
        self.pause()
        self.show_summary()
        
        # Offer to run geometry detector
        if camera_works:
            print("\n" + "="*60)
            if self.ask_yes_no("\nEverything looks good! Run geometry detector now?"):
                if os.path.exists('simple_geometry_detector.py'):
                    print("\nStarting Simple Geometry Detector...")
                    print("Press 'q' to quit\n")
                    time.sleep(2)
                    subprocess.run([sys.executable, 'simple_geometry_detector.py'])
                else:
                    print("\nsimple_geometry_detector.py not found!")
                    print("Make sure all scripts are in the same directory")

def main():
    """Main function"""
    try:
        helper = TroubleshootHelper()
        helper.run()
    except KeyboardInterrupt:
        print("\n\nTroubleshooting cancelled by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("Please report this error for help")

if __name__ == "__main__":
    main()
