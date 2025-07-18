#!/usr/bin/env python3
"""
Test rembg functionality in isolation
"""

import rembg
import os
from PIL import Image
from io import BytesIO

print("Testing rembg functionality...")

# Test loading a session
try:
    print("Loading u2net session...")
    session = rembg.new_session('u2net')
    print("✓ u2net session loaded successfully")
except Exception as e:
    print(f"✗ Error loading u2net session: {e}")
    exit(1)

# Test with a simple image
test_image_path = "/media/jarvis/6E7A-FA6E/polar-bear/meta-tools/frontend/icon.png"

if os.path.exists(test_image_path):
    try:
        print(f"\nProcessing test image: {test_image_path}")
        
        # Load image
        img = Image.open(test_image_path)
        print(f"  Image size: {img.size}")
        print(f"  Image mode: {img.mode}")
        
        # Convert to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        input_data = buffer.getvalue()
        print(f"  Input data size: {len(input_data)} bytes")
        
        # Remove background
        print("  Removing background...")
        output_data = rembg.remove(input_data, session=session)
        print(f"  Output data size: {len(output_data)} bytes")
        
        # Convert back to PIL Image
        output_image = Image.open(BytesIO(output_data))
        print(f"  Output image size: {output_image.size}")
        print(f"  Output image mode: {output_image.mode}")
        
        # Save result
        output_path = "/tmp/test_rembg_output.png"
        output_image.save(output_path)
        print(f"✓ Successfully saved result to {output_path}")
        
    except Exception as e:
        print(f"✗ Error processing image: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ Test image not found: {test_image_path}")

print("\nTest completed.")