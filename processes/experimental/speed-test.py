#!/usr/bin/env python3
"""
Optimized Fiber Optic Analysis Pipeline
Combines preprocessing, zone separation, and defect detection with GPU acceleration and deep learning.
Total runtime target: ~20 seconds per image.

This script is integrated with the Hivemind Connector system for parameter sharing and remote control.
"""

import os
import json
import cv2
import numpy as np
import tempfile
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import multiprocessing as mp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cupy as cp  # For GPU array operations
from cupyx.scipy import ndimage as cp_ndimage  # GPU-accelerated ndimage

# Configure logging
logger = logging.getLogger('FiberAnalysis')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# Connector-exposed parameters
PROCESSING_CONFIG = {
    'force_cpu': False,
    'max_processing_time': 30,
    'confidence_threshold': 0.8,
    'defect_sensitivity': 0.95,
    'enable_deep_learning': True,
    'enable_gpu_acceleration': True,
    'method_weights': {
        'adaptive_intensity': 0.85,
        'bright_core': 0.80,
        'computational': 0.90,
        'geometric': 0.88,
        'gradient': 0.87,
        'guess_approach': 0.75,
        'hough_separation': 0.82,
        'segmentation': 0.86,
        'threshold_separation': 0.78,
        'unified_detector': 0.91
    }
}

# Shared state for connector integration
SHARED_STATE = {
    'last_processed_image': None,
    'last_processing_time': 0,
    'total_images_processed': 0,
    'last_result': None,
    'pipeline_instance': None
}

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Pre-trained models for deep learning
class DeepFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove final layer for features

    def forward(self, x):
        return self.resnet(x)

deep_extractor = DeepFeatureExtractor().to(device).eval()

# Autoencoder for anomaly detection
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

autoencoder = Autoencoder().to(device).eval()  # Assume pre-trained; in practice, train on normal data

@dataclass
class MethodResult:
    name: str
    center: Optional[Tuple[float, float]]
    core_radius: Optional[float]
    cladding_radius: Optional[float]
    confidence: float
    execution_time: float
    parameters: Dict[str, Any]
    error: Optional[str] = None

@dataclass
class SeparationResult:
    masks: Dict[str, np.ndarray]
    regions: Dict[str, np.ndarray]
    consensus: Dict[str, Any]
    defect_mask: np.ndarray
    metadata: Dict[str, Any]

@dataclass
class DefectInfo:
    location: Tuple[int, int]
    size: float
    severity: str
    confidence: float
    region: str
    features: Dict[str, float]
    anomaly_score: float

@dataclass
class DetectionResult:
    core_result: Any
    cladding_result: Any
    ferrule_result: Any
    overall_quality: float
    total_defects: int
    critical_defects: int
    total_processing_time: float
    metadata: Dict[str, Any]
    combined_defect_map: np.ndarray

class GPUManager:
    def __init__(self, force_cpu=False):
        self.use_gpu = torch.cuda.is_available() and not force_cpu
        self.xp = cp if self.use_gpu else np

    def array_to_gpu(self, arr):
        return cp.asarray(arr) if self.use_gpu else arr

    def array_to_cpu(self, arr):
        return cp.asnumpy(arr) if self.use_gpu else arr

def gpu_accelerated(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class UnifiedPipeline:
    def __init__(self, config: Optional[Dict] = None, force_cpu: bool = False):
        self.config = config or {}
        self.gpu_manager = GPUManager(force_cpu)
        self.methods = self._initialize_methods()
        self.consensus_system = self._init_consensus()
        self.vulnerable_methods = {'adaptive_intensity', 'bright_core', 'guess_approach', 'hough_separation', 'threshold_separation'}
        self.deep_extractor = deep_extractor
        self.autoencoder = autoencoder

    def _initialize_methods(self):
        return {
            'adaptive_intensity': {'score': 0.85},
            'bright_core': {'score': 0.80},
            'computational': {'score': 0.90},
            'geometric': {'score': 0.88},
            'gradient': {'score': 0.87},
            'guess_approach': {'score': 0.75},
            'hough_separation': {'score': 0.82},
            'segmentation': {'score': 0.86},
            'threshold_separation': {'score': 0.78},
            'unified_detector': {'score': 0.91}
        }

    def _init_consensus(self):
        class Consensus:
            def generate_consensus(self, results, method_weights, image_shape):
                valid = [r for r in results if not r.error]
                if len(valid) < 3:
                    return None
                centers = np.array([r.center for r in valid])
                core_r = np.array([r.core_radius for r in valid])
                clad_r = np.array([r.cladding_radius for r in valid])
                confs = np.array([r.confidence for r in valid])
                weights = np.array([method_weights[r.name] for r in valid])
                total_w = np.sum(weights * confs)
                cons_center = np.sum(centers * (weights * confs)[:, np.newaxis], axis=0) / total_w
                cons_core_r = np.sum(core_r * weights * confs) / total_w
                cons_clad_r = np.sum(clad_r * weights * confs) / total_w
                masks = self._generate_masks_gpu(cons_center, cons_core_r, cons_clad_r, image_shape)
                return {'masks': masks, 'center': tuple(cons_center), 'core_radius': float(cons_core_r), 'cladding_radius': float(cons_clad_r), 'confidence': float(np.mean(confs)), 'contributing_methods': [r.name for r in valid], 'all_results': results}

            @gpu_accelerated
            def _generate_masks_gpu(self, center, core_r, clad_r, shape):
                xp = self.gpu_manager.xp
                h, w = shape
                y, x = xp.ogrid[:h, :w]
                dist_sq = (x - center[0])**2 + (y - center[1])**2
                core = (dist_sq <= core_r**2).astype(xp.uint8)
                clad = ((dist_sq > core_r**2) & (dist_sq <= clad_r**2)).astype(xp.uint8)
                ferr = (dist_sq > clad_r**2).astype(xp.uint8)
                return {'core': core, 'cladding': clad, 'ferrule': ferr}

        return Consensus()

    def process_image(self, image_path: str):
        start = time.time()
        original_img = cv2.imread(image_path)
        if original_img is None:
            return None
        image_shape = original_img.shape[:2]

        # Preprocessing with GPU
        processed_arrays = self._preprocess_gpu(original_img)

        # Separation with GPU and parallel methods
        with mp.Pool(processes=4) as pool:
            results = pool.starmap(self._run_method_parallel, [(name, image_path, image_shape) for name in self.methods])

        consensus = self.consensus_system.generate_consensus(results, {name: info['score'] for name, info in self.methods.items()}, image_shape)
        if not consensus:
            return None

        # Extract regions
        masks = consensus['masks']
        regions = {k: cv2.bitwise_and(original_img, original_img, mask=m) for k, m in masks.items()}

        # Detection on each region separately with deep learning
        detection_results = {}
        for region_name, region_img in regions.items():
            detection_results[region_name] = self._detect_defects_gpu(region_img, region_name)

        # Combine defect maps
        combined_defect_map = np.zeros(image_shape, dtype=np.uint8)
        for res in detection_results.values():
            # Assume each detection_result has a 'defect_mask'
            combined_defect_map = np.maximum(combined_defect_map, res['defect_mask'])

        total_time = time.time() - start
        logger.info(f"Total processing time: {total_time:.2f}s")

        return {
            'separation': consensus,
            'detection': detection_results,
            'combined_defect_map': combined_defect_map
        }

    def _preprocess_gpu(self, img):
        # Simplified GPU preprocessing for speed
        img_gpu = self.gpu_manager.array_to_gpu(img)
        gray = cp.dot(img_gpu[..., :3], cp.array([0.299, 0.587, 0.114])).astype(cp.uint8)
        blurred = cp_ndimage.gaussian_filter(gray, sigma=1.5)
        # Add more if needed, but keep minimal for speed
        return {'gray': self.gpu_manager.array_to_cpu(gray), 'blurred': self.gpu_manager.array_to_cpu(blurred)}

    def _run_method_parallel(self, method_name, image_path, image_shape):
        # Simulate method run; in practice, implement or call actual methods
        # For speed, use deep learning approximation if possible
        # Placeholder for now
        return MethodResult(name=method_name, center=(image_shape[1]/2, image_shape[0]/2), core_radius=50, cladding_radius=100, confidence=0.8, execution_time=1.0, parameters={})

    def _detect_defects_gpu(self, region_img, region_name):
        # Use PyTorch for deep learning detection
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])
        img_tensor = transform(cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)).unsqueeze(0).to(device)
        
        # Autoencoder anomaly detection
        with torch.no_grad():
            recon = self.autoencoder(img_tensor)
        anomaly_map = torch.abs(img_tensor - recon).squeeze().cpu().numpy()
        
        # Threshold and find defects
        threshold = np.percentile(anomaly_map, 95)
        binary = (anomaly_map > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 5000:
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                defects.append(DefectInfo((cx, cy), area, 'MEDIUM', 0.7, region_name, {}, area/100))
        
        return {'defects': defects, 'defect_mask': binary * 255}

# Connector-exposed functions
def initialize_pipeline(force_cpu=None):
    """Initialize the pipeline with optional configuration"""
    global SHARED_STATE
    
    if force_cpu is not None:
        PROCESSING_CONFIG['force_cpu'] = force_cpu
    
    pipeline = UnifiedPipeline(config=PROCESSING_CONFIG, force_cpu=PROCESSING_CONFIG['force_cpu'])
    SHARED_STATE['pipeline_instance'] = pipeline
    logger.info("Pipeline initialized with connector configuration")
    return True

def process_single_image(image_path):
    """Process a single image through the pipeline"""
    global SHARED_STATE
    
    if SHARED_STATE['pipeline_instance'] is None:
        initialize_pipeline()
    
    pipeline = SHARED_STATE['pipeline_instance']
    
    start_time = time.time()
    result = pipeline.process_image(image_path)
    processing_time = time.time() - start_time
    
    # Update shared state
    SHARED_STATE['last_processed_image'] = image_path
    SHARED_STATE['last_processing_time'] = processing_time
    SHARED_STATE['total_images_processed'] += 1
    SHARED_STATE['last_result'] = result
    
    logger.info(f"Processed {image_path} in {processing_time:.2f}s")
    
    return {
        'success': result is not None,
        'processing_time': processing_time,
        'defect_count': len(result['detection']) if result else 0,
        'image_path': image_path
    }

def batch_process_images(image_directory, max_images=None):
    """Process multiple images from a directory"""
    image_dir = Path(image_directory)
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    if max_images:
        image_files = image_files[:max_images]
    
    results = []
    for img_path in image_files:
        result = process_single_image(str(img_path))
        results.append(result)
    
    return {
        'total_processed': len(results),
        'successful': sum(1 for r in results if r['success']),
        'average_time': np.mean([r['processing_time'] for r in results]),
        'results': results
    }

def get_pipeline_status():
    """Get current pipeline status and statistics"""
    global SHARED_STATE
    
    return {
        'initialized': SHARED_STATE['pipeline_instance'] is not None,
        'last_processed_image': SHARED_STATE['last_processed_image'],
        'last_processing_time': SHARED_STATE['last_processing_time'],
        'total_images_processed': SHARED_STATE['total_images_processed'],
        'gpu_available': torch.cuda.is_available(),
        'using_gpu': not PROCESSING_CONFIG['force_cpu'] and torch.cuda.is_available(),
        'configuration': PROCESSING_CONFIG
    }

def update_configuration(config_updates):
    """Update processing configuration"""
    global PROCESSING_CONFIG
    
    for key, value in config_updates.items():
        if key in PROCESSING_CONFIG:
            PROCESSING_CONFIG[key] = value
            logger.info(f"Updated configuration: {key} = {value}")
    
    # Reinitialize pipeline if needed
    if SHARED_STATE['pipeline_instance'] is not None:
        initialize_pipeline()
    
    return PROCESSING_CONFIG

def get_last_result():
    """Get the last processing result"""
    return SHARED_STATE['last_result']

def clear_cache():
    """Clear any cached data and reset state"""
    global SHARED_STATE
    
    SHARED_STATE = {
        'last_processed_image': None,
        'last_processing_time': 0,
        'total_images_processed': 0,
        'last_result': None,
        'pipeline_instance': None
    }
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Cache cleared and state reset")
    return True

# Main execution for standalone mode
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == "--test":
            # Test mode
            initialize_pipeline()
            print("Pipeline initialized successfully")
            print(f"Status: {get_pipeline_status()}")
        elif sys.argv[1] == "--process":
            if len(sys.argv) > 2:
                result = process_single_image(sys.argv[2])
                print(f"Processing result: {result}")
            else:
                print("Usage: python speed-test.py --process <image_path>")
        else:
            # Process single image
            result = process_single_image(sys.argv[1])
            print(f"Processing result: {result}")
    else:
        # Interactive mode
        print("Speed Test Pipeline - Connector Integrated")
        print("Available functions:")
        print("  - initialize_pipeline()")
        print("  - process_single_image(image_path)")
        print("  - batch_process_images(directory)")
        print("  - get_pipeline_status()")
        print("  - update_configuration(config)")
        print("  - clear_cache()")
        
        # Initialize by default
        initialize_pipeline()
        print("\nPipeline initialized and ready for connector commands")
