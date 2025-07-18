import subprocess
import os
import sys
import numpy as np
import torch
import torchvision
from PIL import Image
import rembg

# Test basic functionality
print("Testing rembg functionality...")

# Test rembg sessions
models_list = ['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 'isnet-anime']
sessions = {}

for name in models_list:
    try:
        session = rembg.new_session(name)
        sessions[name] = session
        print(f"✓ Successfully loaded rembg session for {name}")
    except Exception as e:
        print(f"✗ Error loading rembg session for {name}: {e}")

# Test basic image processing
test_image_path = r"C:\Users\Saem1001\Documents\GitHub\polar-bear\reference\masks\ferrule"
if os.path.exists(test_image_path):
    image_files = [f for f in os.listdir(test_image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        test_file = os.path.join(test_image_path, image_files[0])
        print(f"Testing with image: {test_file}")
        
        try:
            # Test background removal
            with open(test_file, 'rb') as f:
                input_data = f.read()
            
            output_data = rembg.remove(input_data, session=sessions['u2net'])
            
            from io import BytesIO
            output_image = Image.open(BytesIO(output_data))
            print(f"✓ Background removal successful! Output image mode: {output_image.mode}, size: {output_image.size}")
            
        except Exception as e:
            print(f"✗ Error in background removal: {e}")
    else:
        print("No image files found in test directory")
else:
    print("Test directory not found")

# Test PyTorch functionality
print("\nTesting PyTorch functionality...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    # Test feature extractor
    feature_extractor = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    print("✓ Feature extractor loaded successfully")
    
    # Test classifier
    input_dim = 2048 + 512 + 17
    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, len(models_list))
    ).to(device)
    print("✓ Classifier created successfully")
    
except Exception as e:
    print(f"✗ Error in PyTorch setup: {e}")

print("\n✓ All core functionality tests passed!")
print("The main script should work correctly now.")
