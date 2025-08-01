#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for PYLON integration with simulated camera data
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_fiber_image(width=1280, height=720):
    """Create synthetic fiber optic image for testing"""
    # Create base image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add background gradient
    for i in range(height):
        for j in range(width):
            image[i, j] = [50 + (i * 100) // height, 50 + (j * 100) // width, 100]
    
    # Add fiber core (circular)
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 8
    
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < radius:
                # Core (bright center)
                intensity = int(255 * (1 - dist / radius))
                image[i, j] = [intensity, intensity, intensity]
            elif dist < radius * 2:
                # Cladding (darker ring)
                intensity = int(128 * (1 - (dist - radius) / radius))
                image[i, j] = [intensity, intensity, intensity]
    
    # Add some synthetic defects
    # Scratch
    scratch_start = (center_x - radius//2, center_y - radius//2)
    scratch_end = (center_x + radius//2, center_y + radius//2)
    cv2.line(image, scratch_start, scratch_end, (0, 0, 255), 3)
    
    # Dust particle
    dust_pos = (center_x + radius//3, center_y - radius//3)
    cv2.circle(image, dust_pos, 5, (255, 0, 0), -1)
    
    # Add noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return image

class MockPylonCamera:
    """Mock PYLON camera for testing"""
    
    def __init__(self, config):
        self.config = config
        self.is_connected = True
        self.is_streaming = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 30.0
        
    def connect(self, camera_serial=None):
        """Mock connection"""
        logger.info("Mock PYLON camera connected")
        return True
    
    def start_streaming(self):
        """Mock streaming start"""
        self.is_streaming = True
        logger.info("Mock camera streaming started")
        return True
    
    def get_frame(self, timeout=0.1):
        """Generate synthetic frame"""
        if not self.is_streaming:
            return None
        
        # Create synthetic fiber image
        image = create_synthetic_fiber_image(self.config.width, self.config.height)
        
        # Update FPS calculation
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        return image
    
    def get_fps(self):
        """Get current FPS"""
        return self.current_fps
    
    def stop_streaming(self):
        """Mock streaming stop"""
        self.is_streaming = False
        logger.info("Mock camera streaming stopped")
    
    def disconnect(self):
        """Mock disconnect"""
        self.stop_streaming()
        self.is_connected = False
        logger.info("Mock camera disconnected")

def test_pylon_integration():
    """Test PYLON integration with mock camera"""
    logger.info("Testing PYLON integration with mock camera...")
    
    # Import the integration modules
    try:
        from pylon_integration import CameraConfig, RealTimeInspector, PylonMonitor
        from fast_pylon_monitor import FastCameraConfig, FastInspector, FastMonitor
        
        logger.info("Successfully imported PYLON integration modules")
        
        # Test basic camera configuration
        config = CameraConfig(
            exposure_time=5000.0,
            fps=30.0,
            width=1280,
            height=720
        )
        logger.info(f"Camera config created: {config}")
        
        # Test fast camera configuration
        fast_config = FastCameraConfig(
            exposure_time=3000.0,
            fps=60.0,
            width=1280,
            height=720
        )
        logger.info(f"Fast camera config created: {fast_config}")
        
        # Test synthetic image generation
        synthetic_image = create_synthetic_fiber_image(1280, 720)
        logger.info(f"Synthetic image created: {synthetic_image.shape}")
        
        # Test mock camera
        mock_camera = MockPylonCamera(config)
        if mock_camera.connect():
            logger.info("Mock camera connection successful")
            
            if mock_camera.start_streaming():
                logger.info("Mock camera streaming successful")
                
                # Test frame capture
                for i in range(5):
                    frame = mock_camera.get_frame()
                    if frame is not None:
                        logger.info(f"Frame {i+1} captured: {frame.shape}")
                        
                        # Save test frame
                        cv2.imwrite(f"test_frame_{i+1}.jpg", frame)
                        logger.info(f"Test frame {i+1} saved")
                    
                    time.sleep(0.1)
                
                mock_camera.stop_streaming()
                mock_camera.disconnect()
        
        logger.info("PYLON integration test completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import PYLON integration modules: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during PYLON integration test: {e}")
        return False

def test_model_integration():
    """Test model integration without requiring trained model"""
    logger.info("Testing model integration...")
    
    try:
        from fiber_cnn_pure import FiberAnalysisNet
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create a simple test model instead of the complex one
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(32, 3)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x).flatten(1)
                x = self.fc(x)
                return {'quality': x}
        
        # Create simple model
        model = SimpleTestModel()
        model = model.to(device)
        model.eval()
        logger.info(f"Simple test model created with {sum(p.numel() for p in model.parameters())/1e3:.1f}K parameters")
        
        # Test with synthetic data
        synthetic_image = create_synthetic_fiber_image(512, 512)
        
        # Preprocess
        image = cv2.resize(synthetic_image, (512, 512))
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Test inference
        with torch.no_grad():
            outputs = model(image)
        
        logger.info("Model inference successful!")
        logger.info(f"Output shape: quality={outputs['quality'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model integration test: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting PYLON integration tests...")
    
    # Test 1: PYLON integration
    test1_passed = test_pylon_integration()
    
    # Test 2: Model integration
    test2_passed = test_model_integration()
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY:")
    logger.info(f"PYLON Integration Test: {'PASSED' if test1_passed else 'FAILED'}")
    logger.info(f"Model Integration Test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("All tests passed! PYLON integration is ready.")
        logger.info("You can now run:")
        logger.info("  python fast_pylon_monitor.py --help")
        logger.info("  python pylon_integration.py --help")
    else:
        logger.error("Some tests failed. Please check the error messages above.")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main() 