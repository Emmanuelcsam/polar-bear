#!/usr/bin/env python3
"""
Debug segmentation fault in fast-background-removal
"""

import os
import sys
import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
from scipy import stats

print("Step 1: Testing basic imports...")
print(f"  torch: {torch.__version__}")
print(f"  torchvision: {torchvision.__version__}")
print(f"  numpy: {np.__version__}")
print(f"  OpenCV: {cv2.__version__}")

print("\nStep 2: Testing device...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

print("\nStep 3: Testing feature extractor...")
try:
    feature_extractor = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    print("  ✓ Feature extractor created")
except Exception as e:
    print(f"  ✗ Feature extractor error: {e}")
    sys.exit(1)

print("\nStep 4: Testing classifier...")
try:
    input_dim = 512 + 512 + 17
    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 3)
    ).to(device)
    print("  ✓ Classifier created")
except Exception as e:
    print(f"  ✗ Classifier error: {e}")
    sys.exit(1)

print("\nStep 5: Testing model loading...")
model_path = 'crop_method_classifier.pth'
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        classifier.load_state_dict(checkpoint['state_dict'])
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Model loading error: {e}")
else:
    print("  ⚠ Model file not found")

print("\nStep 6: Testing image preprocessing...")
test_image_path = "/media/jarvis/6E7A-FA6E/polar-bear/meta-tools/frontend/icon.png"
if os.path.exists(test_image_path):
    try:
        # Load with PIL
        img_pil = Image.open(test_image_path).convert('RGB')
        print(f"  PIL image size: {img_pil.size}")
        
        # Preprocess
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.CenterCrop(112),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        print(f"  Input tensor shape: {input_tensor.shape}")
        
        # Extract features
        with torch.no_grad():
            resnet_features = feature_extractor(input_tensor).squeeze()
        print(f"  ResNet features shape: {resnet_features.shape}")
        
        # Load with OpenCV
        img_cv = cv2.imread(test_image_path)
        print(f"  OpenCV image shape: {img_cv.shape}")
        
        # Extract statistics
        height, width, _ = img_cv.shape
        aspect_ratio = width / float(height)
        
        # Color statistics
        img_flat = img_cv.reshape(-1, 3).astype(np.float64)
        mean = np.mean(img_flat, axis=0) / 255.0
        std = np.std(img_flat, axis=0) / 255.0
        print(f"  Color mean: {mean}")
        print(f"  Color std: {std}")
        
        # Skewness and kurtosis
        skew_rgb = stats.skew(img_flat, axis=0)
        kurt_rgb = stats.kurtosis(img_flat, axis=0)
        print(f"  Skewness RGB: {skew_rgb}")
        print(f"  Kurtosis RGB: {kurt_rgb}")
        
        # Grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray_flat = gray.flatten().astype(np.float64)
        skew_gray = stats.skew(gray_flat)
        kurt_gray = stats.kurtosis(gray_flat)
        print(f"  Skewness gray: {skew_gray}")
        print(f"  Kurtosis gray: {kurt_gray}")
        
        # Histogram
        hist = cv2.calcHist([img_cv], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten() / (height * width)
        print(f"  Histogram shape: {hist.shape}")
        
        # Entropy
        hist_gray, _ = np.histogram(gray, bins=256, range=(0,255))
        hist_gray = hist_gray / (height * width)
        entropy = -np.sum(hist_gray * np.log2(hist_gray + 1e-7))
        print(f"  Entropy: {entropy}")
        
        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (height * width * 255.0)
        print(f"  Edge density: {edge_density}")
        
        # Combine features
        additional = np.concatenate(([aspect_ratio, entropy, edge_density], mean, std, skew_rgb, kurt_rgb, [skew_gray, kurt_gray], hist))
        print(f"  Additional features shape: {additional.shape}")
        
        additional_t = torch.from_numpy(additional).float().to(device)
        full_features = torch.cat((resnet_features, additional_t))
        print(f"  Full features shape: {full_features.shape}")
        
        # Test prediction
        with torch.no_grad():
            logits = classifier(full_features.unsqueeze(0))[0]
            best_method = torch.argmax(logits).item()
        print(f"  Best method predicted: {best_method}")
        
        print("\n✓ All steps completed successfully!")
        
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ✗ Test image not found: {test_image_path}")

print("\nStep 7: Testing rembg separately...")
try:
    import rembg
    print("  ✓ rembg imported")
    
    # Don't actually create a session here to avoid the segfault
    print("  (Skipping session creation to avoid potential segfault)")
    
except ImportError as e:
    print(f"  ✗ rembg import error: {e}")