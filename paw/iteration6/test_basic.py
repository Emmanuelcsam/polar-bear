#!/usr/bin/env python3
import os
import sys

print("Testing basic imports and functionality...")

try:
    # Test core imports
    print("\n1. Testing core modules...")
    from core.config import BASE, DATA, DEVICE, CORES
    print(f"   ✓ Config loaded: BASE={BASE.name}, DEVICE={DEVICE}, CORES={CORES}")
    
    from core.logger import log
    log("test", "Logger working")
    print("   ✓ Logger working")
    
    from core.datastore import put, get, scan
    put("test_key", "test_value")
    assert get("test_key") == "test_value"
    print("   ✓ Datastore working")
    
    # Test module imports
    print("\n2. Testing module imports...")
    modules = [
        "cv_module", "torch_module", "random_pixel", "intensity_reader",
        "pattern_recognizer", "anomaly_detector", "batch_processor",
        "realtime_processor", "hpc"
    ]
    
    for mod in modules:
        exec(f"from modules import {mod}")
        print(f"   ✓ {mod} imported successfully")
    
    # Test basic functionality
    print("\n3. Testing basic functionality...")
    
    # Generate a random image
    from modules.random_pixel import gen
    img = gen()
    print(f"   ✓ Generated random image: shape={img.shape}")
    
    # Test histogram computation
    from modules.cv_module import hist
    h = hist(img)
    print(f"   ✓ Computed histogram: len={len(h)}, sum={h.sum()}")
    
    # Test distribution learning
    from modules.cv_module import save_hist
    from modules.intensity_reader import learn
    import tempfile
    import cv2
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        cv2.imwrite(f.name, img)
        save_hist(f.name)
        os.unlink(f.name)
    
    learn()
    dist = get("dist")
    if dist is not None:
        print(f"   ✓ Learned distribution: shape={dist.shape}")
    
    print("\n✅ All basic tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)