#!/usr/bin/env python3
"""
Fast Background Removal Processor - CLI Version
Uses rembg CLI to avoid segmentation faults
"""

import os
import sys
import subprocess
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
import time
from scipy import stats
import gc
import logging
from pathlib import Path
import json
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_background_removal_cli.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODELS_LIST = ['u2net', 'u2netp', 'u2net_human_seg']
NUM_METHODS = len(MODELS_LIST)
MAX_SIZE = 2048  # Max image size for processing

class FastBackgroundRemoverCLI:
    def __init__(self, model_path='crop_method_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Check rembg CLI availability
        try:
            subprocess.run(['rembg', '--help'], capture_output=True, check=True)
            logger.info("rembg CLI is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("rembg CLI not found. Please install: pip install rembg[cli]")
            raise RuntimeError("rembg CLI not available")
        
        # Initialize preprocessing
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.CenterCrop(112),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load feature extractor
        self.feature_extractor = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor.fc = torch.nn.Identity()
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Load classifier
        input_dim = 512 + 512 + 17
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, NUM_METHODS)
        ).to(self.device)
        
        # Load saved model
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.classifier.load_state_dict(checkpoint['state_dict'])
                logger.info("Loaded existing classifier model.")
            except Exception as e:
                logger.error(f"Error loading saved model: {e}")
                logger.info("Using untrained classifier.")
        else:
            logger.warning(f"Model file {model_path} not found. Using untrained classifier.")
        
        self.classifier.eval()
        
        # Statistics tracking
        self.stats = {
            'processed': 0,
            'failed': 0,
            'start_time': time.time(),
            'method_usage': {i: 0 for i in range(NUM_METHODS)}
        }
    
    def extract_features(self, img_path: str) -> torch.Tensor:
        """Extract features from an image"""
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                resnet_features = self.feature_extractor(input_tensor).squeeze()
            
            # Extract additional statistics
            img_cv = cv2.imread(img_path)
            height, width, _ = img_cv.shape
            aspect_ratio = width / float(height)
            
            # Color statistics
            img_flat = img_cv.reshape(-1, 3).astype(np.float64)
            mean = np.mean(img_flat, axis=0) / 255.0
            std = np.std(img_flat, axis=0) / 255.0
            skew_rgb = stats.skew(img_flat, axis=0)
            kurt_rgb = stats.kurtosis(img_flat, axis=0)
            
            # Grayscale statistics
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            gray_flat = gray.flatten().astype(np.float64)
            skew_gray = stats.skew(gray_flat)
            kurt_gray = stats.kurtosis(gray_flat)
            
            # Histogram
            hist = cv2.calcHist([img_cv], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten() / (height * width)
            
            # Entropy
            hist_gray, _ = np.histogram(gray, bins=256, range=(0,255))
            hist_gray = hist_gray / (height * width)
            entropy = -np.sum(hist_gray * np.log2(hist_gray + 1e-7))
            
            # Edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (height * width * 255.0)
            
            # Combine features
            additional = np.concatenate(([aspect_ratio, entropy, edge_density], mean, std, skew_rgb, kurt_rgb, [skew_gray, kurt_gray], hist))
            additional_t = torch.from_numpy(additional).float().to(self.device)
            full_features = torch.cat((resnet_features, additional_t))
            
            return full_features
            
        except Exception as e:
            logger.error(f"Error extracting features from {img_path}: {e}")
            return None
    
    def predict_best_method(self, features: torch.Tensor) -> int:
        """Predict the best background removal method for given features"""
        with torch.no_grad():
            logits = self.classifier(features.unsqueeze(0))[0]
            best_method = torch.argmax(logits).item()
        return best_method
    
    def remove_background_cli(self, img_path: str, output_path: str, method_idx: int) -> bool:
        """Remove background using rembg CLI"""
        try:
            model_name = MODELS_LIST[method_idx]
            
            # Prepare image if too large
            img = Image.open(img_path)
            temp_input = None
            
            if img.width > MAX_SIZE or img.height > MAX_SIZE:
                # Create temporary resized image
                temp_input = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
                img.save(temp_input.name, 'PNG')
                input_file = temp_input.name
            else:
                input_file = img_path
            
            # Run rembg CLI
            cmd = ['rembg', 'i', '-m', model_name, input_file, output_path]
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up temporary file
            if temp_input:
                os.unlink(temp_input.name)
            
            if result.returncode == 0:
                logger.info(f"Successfully removed background with {model_name}")
                return True
            else:
                logger.error(f"rembg failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing background from {img_path} with method {model_name}: {e}")
            return False
    
    def process_image(self, img_path: str, output_path: str) -> bool:
        """Process a single image"""
        try:
            # Extract features
            features = self.extract_features(img_path)
            if features is None:
                logger.error(f"Failed to extract features from {img_path}")
                return False
            
            # Predict best method
            best_method = self.predict_best_method(features)
            self.stats['method_usage'][best_method] += 1
            
            # Remove background
            success = self.remove_background_cli(img_path, output_path, best_method)
            
            if not success:
                # Try fallback methods
                logger.warning(f"Primary method {MODELS_LIST[best_method]} failed, trying alternatives...")
                with torch.no_grad():
                    logits = self.classifier(features.unsqueeze(0))[0]
                    sorted_indices = torch.argsort(logits, descending=True)
                
                for idx in sorted_indices[1:]:  # Skip the first one we already tried
                    alt_method = idx.item()
                    success = self.remove_background_cli(img_path, output_path, alt_method)
                    if success:
                        logger.info(f"Fallback method {MODELS_LIST[alt_method]} succeeded")
                        self.stats['method_usage'][alt_method] += 1
                        break
            
            if not success:
                logger.error(f"All methods failed for {img_path}")
                return False
            
            logger.info(f"Processed {img_path} -> {output_path} using {MODELS_LIST[best_method]}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str, extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        """Process all images in a directory"""
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(input_dir).glob(f"*{ext}"))
            image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process images one by one
        for idx, img_path in enumerate(image_paths):
            output_path = os.path.join(output_dir, img_path.name)
            
            logger.info(f"[{idx+1}/{len(image_paths)}] Processing {img_path.name}...")
            
            if self.process_image(str(img_path), output_path):
                self.stats['processed'] += 1
            else:
                self.stats['failed'] += 1
            
            # Periodic garbage collection
            if (idx + 1) % 10 == 0:
                gc.collect()
        
        # Print statistics
        elapsed = time.time() - self.stats['start_time']
        logger.info(f"\nProcessing complete!")
        logger.info(f"Total images: {len(image_paths)}")
        logger.info(f"Successfully processed: {self.stats['processed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Time elapsed: {elapsed:.2f}s")
        logger.info(f"Average time per image: {elapsed/max(1, self.stats['processed']):.2f}s")
        logger.info(f"\nMethod usage:")
        for i, count in self.stats['method_usage'].items():
            logger.info(f"  {MODELS_LIST[i]}: {count} times")
    
    def save_stats(self, output_file: str = 'processing_stats.json'):
        """Save processing statistics to a JSON file"""
        stats_data = {
            'processed': self.stats['processed'],
            'failed': self.stats['failed'],
            'elapsed_time': time.time() - self.stats['start_time'],
            'method_usage': {MODELS_LIST[i]: count for i, count in self.stats['method_usage'].items()}
        }
        with open(output_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        logger.info(f"Saved statistics to {output_file}")


def main():
    print("Fast Background Removal Processor (CLI Version)")
    print("==============================================")
    print("This tool uses the learned data from auto-background-removal.py")
    print("to quickly process images using the rembg CLI.\n")
    
    # Get input directory
    input_dir = input("Enter the full path to the directory containing images to process: ").strip()
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Get output directory
    output_dir = input("Enter the full path to the output directory for processed images: ").strip()
    
    # Ask about model path
    use_custom_model = input("\nDo you want to use a custom model path? (y/n) [default: n]: ").strip().lower()
    
    if use_custom_model == 'y':
        model_path = input("Path to trained classifier model [default: crop_method_classifier.pth]: ").strip()
        if not model_path:
            model_path = 'crop_method_classifier.pth'
    else:
        model_path = 'crop_method_classifier.pth'
    
    print(f"\nConfiguration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Model path: {model_path}")
    
    proceed = input("\nProceed with processing? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Processing cancelled.")
        sys.exit(0)
    
    print("\nStarting processing...")
    
    try:
        # Create processor and run
        processor = FastBackgroundRemoverCLI(model_path)
        processor.process_directory(input_dir, output_dir)
        processor.save_stats('processing_stats_cli.json')
        
        print(f"\nProcessing complete! Check processing_stats_cli.json for detailed statistics.")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()