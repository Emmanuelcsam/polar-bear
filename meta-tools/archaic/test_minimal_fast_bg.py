#!/usr/bin/env python3
"""
Minimal test to isolate the segfault issue
"""

import os
import sys
import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
from scipy import stats
import rembg
import gc

print("Creating FastBackgroundRemover class...")

class MinimalFastBackgroundRemover:
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
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
        print("Feature extractor loaded")
        
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
            torch.nn.Linear(256, 3)
        ).to(self.device)
        print("Classifier created")
        
        # Load saved model
        model_path = 'crop_method_classifier.pth'
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.classifier.load_state_dict(checkpoint['state_dict'])
                print("Loaded existing classifier model.")
            except Exception as e:
                print(f"Error loading saved model: {e}")
        
        self.classifier.eval()
        
        # Don't initialize rembg sessions yet
        self.sessions = {}
        self.current_session = None
        print("Initialization complete")
    
    def extract_features(self, img_path):
        """Extract features from an image"""
        print(f"Extracting features from {img_path}...")
        
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
        
        print(f"Features extracted, shape: {full_features.shape}")
        return full_features
    
    def predict_best_method(self, features):
        """Predict the best background removal method"""
        with torch.no_grad():
            logits = self.classifier(features.unsqueeze(0))[0]
            best_method = torch.argmax(logits).item()
        print(f"Best method predicted: {best_method}")
        return best_method
    
    def load_session(self, model_name):
        """Load a rembg session"""
        print(f"Loading session for {model_name}...")
        
        if model_name not in self.sessions:
            try:
                # First, let's test if rembg works at all
                print("Testing rembg.new_session()...")
                session = rembg.new_session(model_name)
                self.sessions[model_name] = session
                self.current_session = model_name
                print(f"Session loaded for {model_name}")
            except Exception as e:
                print(f"Error loading session: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return self.sessions.get(model_name)
    
    def remove_background(self, img_path, method_idx):
        """Remove background using specified method"""
        models_list = ['u2net', 'u2netp', 'u2net_human_seg']
        model_name = models_list[method_idx]
        
        print(f"Removing background with {model_name}...")
        
        session = self.load_session(model_name)
        if session is None:
            return None
        
        # Read image
        img = Image.open(img_path)
        print(f"Image size: {img.size}")
        
        # Convert to bytes
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        input_data = buffer.getvalue()
        print(f"Input data size: {len(input_data)} bytes")
        
        # Remove background
        print("Calling rembg.remove()...")
        output_data = rembg.remove(input_data, session=session)
        print(f"Output data size: {len(output_data)} bytes")
        
        # Convert back to PIL Image
        output_image = Image.open(BytesIO(output_data))
        
        # Ensure RGBA format
        if output_image.mode != 'RGBA':
            output_image = output_image.convert('RGBA')
        
        print(f"Successfully removed background")
        return output_image

# Test the minimal version
print("\nTesting minimal version...")
test_image = "/media/jarvis/6E7A-FA6E/polar-bear/meta-tools/frontend/icon.png"

try:
    # Create instance
    processor = MinimalFastBackgroundRemover()
    
    # Extract features
    features = processor.extract_features(test_image)
    
    # Predict best method
    best_method = processor.predict_best_method(features)
    
    # Remove background
    result = processor.remove_background(test_image, best_method)
    
    if result:
        output_path = "/tmp/minimal_test_output.png"
        result.save(output_path)
        print(f"\n✓ Success! Output saved to {output_path}")
    else:
        print("\n✗ Failed to remove background")
        
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()