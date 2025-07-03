#!/usr/bin/env python3
"""
Camera Test and Debug Script
============================

This script helps you test your camera setup and debug any issues
before running the main geometry detection programs.

Usage:
    python camera_test.py              # Test default camera
    python camera_test.py --index 1    # Test specific camera
    python camera_test.py --scan       # Scan for all cameras
    python camera_test.py --file video.mp4  # Test video file
"""

import cv2
import numpy as np
import sys
import argparse
import platform
import time

def print_system_info():
    """Print system information"""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    
    # Check CUDA
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"CUDA Devices: {cuda_count}")
        if cuda_count > 0:
            print("CUDA is available!")
    except:
        print("CUDA: Not available")
    
    print("="*60)

def test_backends():
    """Test different camera backends"""
    print("\nTesting camera backends...")
    
    backends = [
        (cv2.CAP_ANY, "ANY (Auto)"),
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_V4L2, "V4L2 (Linux)"),
        (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS)"),
    ]
    
    working_backends = []
    
    for backend, name in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ {name}: WORKS")
                    working_backends.append((backend, name))
                else:
                    print(f"✗ {name}: Opens but no frames")
                cap.release()
            else:
                print(f"✗ {name}: Cannot open")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")
    
    return working_backends

def scan_cameras(max_index=10):
    """Scan for available cameras"""
    print(f"\nScanning for cameras (testing indices 0-{max_index-1})...")
    cameras = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✓ Camera {i}: {width}x{height} @ {fps}fps")
                cameras.append(i)
                
                # Show preview
                cv2.putText(frame, f"Camera {i}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f'Camera {i} Preview', frame)
                cv2.waitKey(500)
                cv2.destroyWindow(f'Camera {i} Preview')
            else:
                print(f"? Camera {i}: Opens but no frames")
            cap.release()
    
    if not cameras:
        print("\nNo cameras found!")
        print("Troubleshooting tips:")
        print("- Check if camera is connected")
        print("- Check camera permissions")
        print("- Try running as administrator/sudo")
        print("- Close other applications using the camera")
    else:
        print(f"\nFound {len(cameras)} camera(s) at indices: {cameras}")
    
    return cameras

def test_camera(index=0, backend=cv2.CAP_ANY):
    """Test a specific camera"""
    print(f"\nTesting camera at index {index}...")
    
    cap = cv2.VideoCapture(index, backend)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {index}")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ Camera opened successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    # Try different resolutions
    print("\nTesting different resolutions...")
    resolutions = [
        (640, 480),
        (800, 600),
        (1280, 720),
        (1920, 1080)
    ]
    
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == w and actual_h == h:
            print(f"  ✓ {w}x{h}: Supported")
        else:
            print(f"  ✗ {w}x{h}: Not supported (got {actual_w}x{actual_h})")
    
    # Reset to original
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Test frame capture
    print("\nTesting frame capture...")
    success_count = 0
    test_frames = 30
    
    start_time = time.time()
    for i in range(test_frames):
        ret, frame = cap.read()
        if ret and frame is not None:
            success_count += 1
    elapsed = time.time() - start_time
    
    print(f"  Captured {success_count}/{test_frames} frames")
    print(f"  Actual FPS: {test_frames/elapsed:.2f}")
    
    if success_count == 0:
        print("✗ No frames captured!")
        cap.release()
        return False
    
    # Show live preview
    print("\nShowing live preview...")
    print("Press 'q' to quit, 's' to save a test image")
    
    frame_count = 0
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_time > 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Add info to frame
        info = [
            f"Camera {index}",
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
            f"FPS: {current_fps}",
            f"Frame: {frame_count}",
            "",
            "Press 'q' to quit",
            "Press 's' to save image"
        ]
        
        y = 30
        for text in info:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            y += 30
        
        # Draw test shapes
        cv2.rectangle(frame, (50, 200), (200, 350), (255, 0, 0), 2)
        cv2.circle(frame, (300, 275), 50, (0, 255, 0), 2)
        pts = np.array([[400, 200], [450, 300], [350, 350]], np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        
        cv2.imshow('Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'camera_test_{frame_count}.png'
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nTest completed. Captured {frame_count} frames")
    return True

def test_video_file(filepath):
    """Test video file"""
    print(f"\nTesting video file: {filepath}")
    
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print(f"✗ Failed to open video file")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Video opened successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {frame_count}")
    print(f"  Duration: {frame_count/fps:.2f} seconds")
    
    # Test reading frames
    print("\nTesting frame reading...")
    test_positions = [0, frame_count//4, frame_count//2, 3*frame_count//4]
    
    for pos in test_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Frame {pos}: OK")
        else:
            print(f"  ✗ Frame {pos}: Failed")
    
    # Play video
    print("\nPlaying video preview...")
    print("Press 'q' to quit, SPACE to pause")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    paused = False
    current_frame = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Add info
        progress = current_frame / frame_count * 100
        cv2.putText(frame, f"Frame {current_frame}/{frame_count} ({progress:.1f}%)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Video Test', frame)
        
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    
    return True

def diagnose_issues():
    """Diagnose common camera issues"""
    print("\n" + "="*60)
    print("DIAGNOSTIC INFORMATION")
    print("="*60)
    
    # Platform-specific checks
    system = platform.system()
    
    if system == "Windows":
        print("\nWindows-specific checks:")
        print("- Check Device Manager for camera drivers")
        print("- Try Windows Camera app to verify camera works")
        print("- Disable antivirus temporarily")
        print("- Run as Administrator")
        
    elif system == "Linux":
        print("\nLinux-specific checks:")
        import subprocess
        
        # Check video devices
        try:
            result = subprocess.run(['ls', '/dev/video*'], 
                                  capture_output=True, text=True, shell=True)
            print(f"Video devices: {result.stdout.strip()}")
        except:
            pass
        
        # Check permissions
        print("\nTo fix permission issues:")
        print("  sudo usermod -a -G video $USER")
        print("  sudo chmod 666 /dev/video0")
        
        # Check v4l2
        print("\nInstall v4l2 tools:")
        print("  sudo apt-get install v4l-utils")
        print("  v4l2-ctl --list-devices")
        
    elif system == "Darwin":  # macOS
        print("\nmacOS-specific checks:")
        print("- Check System Preferences > Security & Privacy > Camera")
        print("- Allow Terminal/Python access to camera")
        print("- Test with Photo Booth app")
        print("- Restart the system")
    
    print("\nGeneral troubleshooting:")
    print("1. Close all other applications using the camera")
    print("2. Restart the computer")
    print("3. Try different USB ports (for external cameras)")
    print("4. Update camera drivers")
    print("5. Check camera cable connections")
    print("6. Try a different camera")

def main():
    parser = argparse.ArgumentParser(description='Camera Test and Debug Tool')
    parser.add_argument('--index', type=int, default=0, 
                       help='Camera index to test (default: 0)')
    parser.add_argument('--scan', action='store_true', 
                       help='Scan for all available cameras')
    parser.add_argument('--file', type=str, 
                       help='Test video file instead of camera')
    parser.add_argument('--backends', action='store_true', 
                       help='Test different camera backends')
    parser.add_argument('--diagnose', action='store_true', 
                       help='Show diagnostic information')
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    if args.file:
        # Test video file
        test_video_file(args.file)
    elif args.scan:
        # Scan for cameras
        scan_cameras()
    elif args.backends:
        # Test backends
        working = test_backends()
        if working:
            print(f"\nWorking backends: {len(working)}")
            print("Testing with first working backend...")
            test_camera(args.index, working[0][0])
    elif args.diagnose:
        # Show diagnostic info
        diagnose_issues()
    else:
        # Test specific camera
        if not test_camera(args.index):
            print("\nCamera test failed!")
            diagnose_issues()

if __name__ == "__main__":
    main()
