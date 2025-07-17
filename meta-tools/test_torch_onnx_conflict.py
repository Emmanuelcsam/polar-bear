#!/usr/bin/env python3
"""
Test for conflicts between PyTorch and ONNX Runtime
"""

print("1. Importing torch...")
import torch
print(f"   torch version: {torch.__version__}")

print("\n2. Creating a simple torch model...")
model = torch.nn.Linear(10, 5)
x = torch.randn(1, 10)
y = model(x)
print(f"   Model output shape: {y.shape}")

print("\n3. Importing onnxruntime...")
import onnxruntime as ort
print(f"   onnxruntime version: {ort.__version__}")

print("\n4. Testing rembg import...")
try:
    import rembg
    print("   rembg imported successfully")
except Exception as e:
    print(f"   Error importing rembg: {e}")

print("\n5. Testing torchvision...")
import torchvision
print(f"   torchvision version: {torchvision.__version__}")

print("\n6. Loading ResNet model...")
resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
print("   ResNet18 loaded successfully")

print("\n7. Testing rembg session creation...")
try:
    session = rembg.new_session('u2net')
    print("   rembg session created successfully")
except Exception as e:
    print(f"   Error creating rembg session: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed.")