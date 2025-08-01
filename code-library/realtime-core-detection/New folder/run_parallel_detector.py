#!/usr/bin/env python3
"""
Parallel Live Core Detector with Circle Overlay
Launches both the live core detector and circle overlay as separate processes.
"""

import subprocess
import sys
import time
import signal
from typing import List


class ParallelDetectorLauncher:
    """Launcher for parallel core detector and circle overlay"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.is_running = False
        
    def start_live_core_detector(self, camera_index: int = 0,
                                use_pylon: bool = True) -> subprocess.Popen:
        """Start live core detector as a separate process"""
        cmd = [
            sys.executable, "live_core_detector.py",
            "--camera", str(camera_index)
        ]
        
        if not use_pylon:
            cmd.append("--no-pylon")
        
        print(f"Starting Live Core Detector: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes.append(process)
        return process
    
    def start_circle_overlay(self, window_name: str = "Circle Overlay",
                            overlay_on_window: str = "Live Core Detector",
                            width: int = 800, height: int = 600) -> subprocess.Popen:
        """Start circle overlay as a separate process"""
        cmd = [
            sys.executable, "circle_overlay.py",
            "--window-name", window_name,
            "--overlay-on", overlay_on_window,
            "--width", str(width),
            "--height", str(height)
        ]
        
        print(f"Starting Circle Overlay: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes.append(process)
        return process
    
    def monitor_processes(self):
        """Monitor running processes and handle output"""
        print("Monitoring parallel processes...")
        print("Press Ctrl+C to stop all processes")
        
        try:
            while self.is_running and any(p.poll() is None for p in self.processes):
                # Check process status
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        # Process has ended
                        stdout, stderr = process.communicate()
                        if stdout:
                            print(f"Process {i} stdout: {stdout}")
                        if stderr:
                            print(f"Process {i} stderr: {stderr}")
                        print(f"Process {i} ended with code: {process.returncode}")
                
                time.sleep(0.1)  # Check every 100ms
                
        except KeyboardInterrupt:
            print("\nReceived interrupt signal, stopping all processes...")
            self.stop_all_processes()
    
    def stop_all_processes(self):
        """Stop all running processes"""
        self.is_running = False
        
        for i, process in enumerate(self.processes):
            if process.poll() is None:  # Process is still running
                print(f"Stopping process {i}...")
                try:
                    process.terminate()
                    process.wait(timeout=5)  # Wait up to 5 seconds
                except subprocess.TimeoutExpired:
                    print(f"Force killing process {i}...")
                    process.kill()
                    process.wait()
                except Exception as e:
                    print(f"Error stopping process {i}: {e}")
        
        self.processes.clear()
        print("All processes stopped")
    
    def run(self, camera_index: int = 0, use_pylon: bool = True,
            overlay_width: int = 800, overlay_height: int = 600):
        """Run both processes in parallel"""
        print("Starting Parallel Live Core Detector with Circle Overlay")
        print("=" * 60)
        
        self.is_running = True
        
        try:
            # Start live core detector first
            self.start_live_core_detector(camera_index, use_pylon)
            
            # Wait a moment for the detector to initialize
            time.sleep(2)
            
            # Start circle overlay
            self.start_circle_overlay(
                width=overlay_width,
                height=overlay_height
            )
            
            print("\nBoth processes started successfully!")
            print("Circle overlay controls:")
            print("  WASD: Move circle")
            print("  Q/E: Resize circle")
            print("  L: Lock/Unlock position")
            print("  R: Reset to center")
            print("  ESC: Exit overlay")
            print("\nPress Ctrl+C to stop all processes")
            
            # Monitor processes
            self.monitor_processes()
            
        except Exception as e:
            print(f"Error running parallel processes: {e}")
            self.stop_all_processes()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parallel Live Core Detector with Circle Overlay"
    )
    parser.add_argument(
        "--camera", type=int, default=0, 
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--no-pylon", action="store_true", 
        help="Disable Pylon SDK and use webcam only"
    )
    parser.add_argument(
        "--overlay-width", type=int, default=800,
        help="Circle overlay window width (default: 800)"
    )
    parser.add_argument(
        "--overlay-height", type=int, default=600,
        help="Circle overlay window height (default: 600)"
    )
    
    args = parser.parse_args()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, stopping processes...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run launcher
    launcher = ParallelDetectorLauncher()
    
    try:
        launcher.run(
            camera_index=args.camera,
            use_pylon=not args.no_pylon,
            overlay_width=args.overlay_width,
            overlay_height=args.overlay_height
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        launcher.stop_all_processes()


if __name__ == "__main__":
    main() 