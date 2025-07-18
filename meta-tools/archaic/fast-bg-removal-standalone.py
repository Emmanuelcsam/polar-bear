#!/usr/bin/env python3
"""
Standalone Fast Background Removal - Simplified version
Uses the trained model to process images without complex imports
"""

import os
import sys
import time
from pathlib import Path

def main():
    print("Fast Background Removal Processor (Standalone)")
    print("============================================")
    print()
    
    # Check for required libraries
    try:
        import numpy as np
        import torch
        import torchvision
        import cv2
        from PIL import Image
        import rembg
        from scipy import stats
        print("✓ All required libraries are available")
    except ImportError as e:
        print(f"✗ Missing required library: {e}")
        print("Please install: pip install torch torchvision opencv-python pillow rembg scipy")
        sys.exit(1)
    
    # Configuration
    MODELS_LIST = ['u2net', 'u2netp', 'u2net_human_seg']
    NUM_METHODS = len(MODELS_LIST)
    MAX_SIZE = 2048
    
    # Get directories
    input_dir = input("Enter the full path to the directory containing images: ").strip()
    if not os.path.exists(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir = input("Enter the full path to the output directory: ").strip()
    os.makedirs(output_dir, exist_ok=True)
    
    # Find images
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(input_dir).glob(f"*{ext}"))
        image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_paths:
        print("No images found in the input directory.")
        sys.exit(0)
    
    print(f"\nFound {len(image_paths)} images to process")
    
    # Initialize components
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Preprocessing
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(128),
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Feature extractor
    print("Loading feature extractor...")
    feature_extractor = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Classifier
    print("Loading classifier...")
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
        torch.nn.Linear(256, NUM_METHODS)
    ).to(device)
    
    # Load saved model
    model_path = 'crop_method_classifier.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            classifier.load_state_dict(checkpoint['state_dict'])
            print("✓ Loaded trained classifier model")
        except Exception as e:
            print(f"⚠ Could not load saved model: {e}")
            print("Using untrained classifier")
    else:
        print("⚠ No saved model found, using untrained classifier")
    
    classifier.eval()
    
    # Process images
    print("\nProcessing images...")
    start_time = time.time()
    processed = 0
    failed = 0
    method_usage = {i: 0 for i in range(NUM_METHODS)}
    
    # Load rembg session once
    current_session = None
    current_model = None
    
    for idx, img_path in enumerate(image_paths):
        print(f"\n[{idx+1}/{len(image_paths)}] Processing {img_path.name}...")
        
        try:
            # Extract features
            img = Image.open(str(img_path)).convert('RGB')
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                resnet_features = feature_extractor(input_tensor).squeeze()
            
            # Additional statistics
            img_cv = cv2.imread(str(img_path))
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
            additional_t = torch.from_numpy(additional).float().to(device)
            full_features = torch.cat((resnet_features, additional_t))
            
            # Predict best method
            with torch.no_grad():
                logits = classifier(full_features.unsqueeze(0))[0]
                best_method = torch.argmax(logits).item()
            
            model_name = MODELS_LIST[best_method]
            print(f"  Using method: {model_name}")
            method_usage[best_method] += 1
            
            # Load session if needed
            if current_model != model_name:
                print(f"  Loading {model_name} session...")
                current_session = rembg.new_session(model_name)
                current_model = model_name
            
            # Remove background
            img_pil = Image.open(str(img_path))
            
            # Resize if too large
            if img_pil.width > MAX_SIZE or img_pil.height > MAX_SIZE:
                img_pil.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
            
            # Convert to bytes
            from io import BytesIO
            buffer = BytesIO()
            img_pil.save(buffer, format='PNG')
            input_data = buffer.getvalue()
            
            # Process
            output_data = rembg.remove(input_data, session=current_session)
            
            # Save result
            output_image = Image.open(BytesIO(output_data))
            if output_image.mode != 'RGBA':
                output_image = output_image.convert('RGBA')
            
            output_path = os.path.join(output_dir, img_path.name)
            output_image.save(output_path, 'PNG')
            print(f"  ✓ Saved to {output_path}")
            processed += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total images: {len(image_paths)}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Average time per image: {elapsed/max(1, processed):.2f}s")
    print(f"\nMethod usage:")
    for i, count in method_usage.items():
        print(f"  {MODELS_LIST[i]}: {count} times")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()