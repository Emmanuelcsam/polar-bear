#!/usr/bin/env python3
"""
Enhanced Pylon Viewer Integration Module
Automatically opens Pylon Viewer when the program starts and manages the integration.
Provides seamless integration with the core detection system.
"""

import os
import subprocess
import time
import threading
import platform
from typing import Optional, List, Dict
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PylonViewerManager:
    """Manages Pylon Viewer integration and automatic startup"""
    
    def __init__(self, auto_start: bool = True, 
                 viewer_path: Optional[str] = None):
        """
        Initialize Pylon Viewer Manager
        
        Args:
            auto_start: Whether to automatically start Pylon Viewer
            viewer_path: Custom path to Pylon Viewer executable
        """
        self.auto_start = auto_start
        self.viewer_path = viewer_path
        self.viewer_process = None
        self.is_running = False
        self.startup_timeout = 10  # seconds to wait for startup
        
        # Common Pylon Viewer paths for different platforms
        self.default_paths = {
            'Windows': [
                (r'C:\Program Files\Basler\pylon 7\Runtime\x64'
                 r'\PylonViewerApp.exe'),
                (r'C:\Program Files\Basler\pylon 6\Runtime\x64'
                 r'\PylonViewerApp.exe'),
                (r'C:\Program Files\Basler\pylon 5\Runtime\x64'
                 r'\PylonViewerApp.exe'),
                (r'C:\Program Files (x86)\Basler\pylon 7\Runtime\x64'
                 r'\PylonViewerApp.exe'),
                (r'C:\Program Files (x86)\Basler\pylon 6\Runtime\x64'
                 r'\PylonViewerApp.exe'),
                (r'C:\Program Files (x86)\Basler\pylon 5\Runtime\x64'
                 r'\PylonViewerApp.exe'),
            ],
            'Linux': [
                '/opt/pylon/bin/PylonViewerApp',
                '/usr/local/pylon/bin/PylonViewerApp',
            ],
            'Darwin': [
                '/Applications/PylonViewerApp.app/Contents/MacOS/PylonViewerApp',
                '/opt/pylon/bin/PylonViewerApp',
            ]
        }
    
    def find_pylon_viewer(self) -> Optional[str]:
        """Find Pylon Viewer executable on the system"""
        if self.viewer_path and os.path.exists(self.viewer_path):
            logger.info(f"Using custom Pylon Viewer path: {self.viewer_path}")
            return self.viewer_path
        
        system = platform.system()
        if system not in self.default_paths:
            logger.warning(f"Unsupported platform: {system}")
            return None
        
        for path in self.default_paths[system]:
            if os.path.exists(path):
                logger.info(f"Found Pylon Viewer at: {path}")
                return path
        
        logger.warning("Pylon Viewer not found in default locations")
        logger.info("Available search paths:")
        for path in self.default_paths[system]:
            logger.info(f"  - {path}")
        return None
    
    def start_pylon_viewer(self) -> bool:
        """Start Pylon Viewer application with enhanced error handling"""
        try:
            viewer_path = self.find_pylon_viewer()
            if not viewer_path:
                logger.error("Pylon Viewer not found. Please install Pylon SDK or specify custom path.")
                logger.info("You can continue without Pylon Viewer - the system will use webcam fallback.")
                return False
            
            logger.info(f"Starting Pylon Viewer from: {viewer_path}")
            
            # Start Pylon Viewer in background with proper error handling
            if platform.system() == 'Windows':
                self.viewer_process = subprocess.Popen(
                    [viewer_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                self.viewer_process = subprocess.Popen(
                    [viewer_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Wait a moment to see if process starts successfully
            time.sleep(1)
            
            if self.viewer_process.poll() is not None:
                # Process exited immediately
                stdout, stderr = self.viewer_process.communicate()
                logger.error(f"Pylon Viewer failed to start. Exit code: {self.viewer_process.returncode}")
                if stdout:
                    logger.error(f"stdout: {stdout.decode()}")
                if stderr:
                    logger.error(f"stderr: {stderr.decode()}")
                return False
            
            self.is_running = True
            logger.info("Pylon Viewer started successfully")
            logger.info(f"Process ID: {self.viewer_process.pid}")
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_viewer, daemon=True)
            monitor_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Pylon Viewer: {e}")
            return False
    
    def _monitor_viewer(self):
        """Monitor Pylon Viewer process with enhanced logging"""
        try:
            logger.info("Starting Pylon Viewer monitoring thread")
            while self.is_running and self.viewer_process:
                if self.viewer_process.poll() is not None:
                    logger.info("Pylon Viewer process terminated")
                    self.is_running = False
                    break
                time.sleep(2)  # Check every 2 seconds
        except Exception as e:
            logger.error(f"Error monitoring Pylon Viewer: {e}")
    
    def stop_pylon_viewer(self):
        """Stop Pylon Viewer application gracefully"""
        if self.viewer_process:
            try:
                logger.info("Stopping Pylon Viewer...")
                self.viewer_process.terminate()
                
                # Wait for graceful termination
                try:
                    self.viewer_process.wait(timeout=5)
                    logger.info("Pylon Viewer stopped successfully")
                except subprocess.TimeoutExpired:
                    logger.warning("Pylon Viewer did not terminate gracefully, forcing kill")
                    self.viewer_process.kill()
                    self.viewer_process.wait()
                    logger.info("Pylon Viewer force-killed")
                    
            except Exception as e:
                logger.error(f"Error stopping Pylon Viewer: {e}")
            finally:
                self.is_running = False
                self.viewer_process = None
    
    def get_camera_list(self) -> List[Dict]:
        """Get list of available Pylon cameras"""
        try:
            # This would require Pylon SDK to be installed
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            logger.error(f"Error getting camera list: {e}")
            return []
    
    def is_pylon_available(self) -> bool:
        """Check if Pylon SDK is available"""
        try:
            import pypylon
            return True
        except ImportError:
            return False
    
    def get_status(self) -> Dict:
        """Get current status of Pylon Viewer integration"""
        return {
            'is_running': self.is_running,
            'process_id': self.viewer_process.pid if self.viewer_process else None,
            'viewer_path': self.find_pylon_viewer(),
            'pylon_sdk_available': self.is_pylon_available(),
            'platform': platform.system()
        }


def integrate_with_orchestrator():
    """Integrate Pylon Viewer with the main orchestrator"""
    try:
        logger.info("Starting Pylon Viewer integration...")
        
        # Create Pylon Viewer manager
        viewer_manager = PylonViewerManager(auto_start=True)
        
        # Check if Pylon SDK is available
        if not viewer_manager.is_pylon_available():
            logger.warning("Pylon SDK not available. System will use webcam fallback.")
            logger.info("To enable Pylon support, install pypylon: pip install pypylon")
            return None
        
        # Start Pylon Viewer if auto_start is enabled
        if viewer_manager.auto_start:
            logger.info("Attempting to start Pylon Viewer...")
            if viewer_manager.start_pylon_viewer():
                logger.info("Pylon Viewer integration successful")
                logger.info("Pylon Viewer should now be open and ready for camera interaction")
                return viewer_manager
            else:
                logger.warning("Pylon Viewer integration failed, continuing without it")
                logger.info("The system will use webcam fallback for camera input")
                return None
        
        return viewer_manager
        
    except Exception as e:
        logger.error(f"Error in Pylon Viewer integration: {e}")
        return None


def main():
    """Test Pylon Viewer integration"""
    print("Testing Pylon Viewer Integration...")
    print("=" * 50)
    
    # Check if Pylon SDK is available
    viewer_manager = PylonViewerManager()
    if not viewer_manager.is_pylon_available():
        print("Pylon SDK not available. Please install pypylon package.")
        print("You can continue without Pylon - the system will use webcam fallback.")
        return
    
    # Test integration
    viewer_manager = integrate_with_orchestrator()
    
    if viewer_manager:
        print("Pylon Viewer integration test successful")
        print("Status:", viewer_manager.get_status())
        print("Press Enter to stop Pylon Viewer...")
        input()
        viewer_manager.stop_pylon_viewer()
    else:
        print("Pylon Viewer integration test failed")
        print("The system will continue with webcam fallback")


if __name__ == "__main__":
    main() 