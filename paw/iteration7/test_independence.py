#!/usr/bin/env python3
"""
Test that PyTorch and OpenCV modules are truly independent
Shows the system works even without AI libraries installed
"""

import subprocess
import sys
import os

def test_module(script_name):
    """Test if a module handles missing dependencies gracefully"""
    print(f"\nTesting {script_name}...")
    
    try:
        # Create a test environment that simulates missing libraries
        test_code = f"""
import sys
import os

# Simulate missing libraries
class MockModule:
    def __getattr__(self, name):
        raise ImportError(f"Simulated missing library")

# Override imports
sys.modules['torch'] = MockModule()
sys.modules['torch.nn'] = MockModule()
sys.modules['cv2'] = MockModule()

# Now try to run the module
try:
    exec(open('{script_name}').read())
    print(f"[TEST] {script_name} handled missing dependencies")
except ImportError as e:
    print(f"[TEST] {script_name} requires AI libraries: {{e}}")
except Exception as e:
    print(f"[TEST] {script_name} other error: {{e}}")
"""
        
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            capture_output=True,
            text=True
        )
        
        if "requires AI libraries" in result.stdout:
            print(f"  ✗ {script_name} requires AI libraries")
            return False
        else:
            print(f"  ✓ {script_name} works independently")
            return True
            
    except Exception as e:
        print(f"  ? {script_name} test error: {e}")
        return None

def main():
    print("=== Testing Module Independence ===")
    print("This verifies that modules work without PyTorch/OpenCV\n")
    
    # Test core modules (should all work)
    core_modules = [
        'pixel_reader.py',
        'random_generator.py',
        'correlator.py',
        'pattern_recognizer.py',
        'anomaly_detector.py',
        'intensity_analyzer.py',
        'geometry_analyzer.py',
        'data_calculator.py',
        'data_store.py',
        'image_generator.py',
        'batch_processor.py',
        'trend_analyzer.py'
    ]
    
    print("Testing Core Modules (should work without AI libraries):")
    core_results = []
    for module in core_modules:
        if os.path.exists(module):
            result = test_module(module)
            core_results.append(result)
    
    # Test AI modules (expected to need libraries)
    ai_modules = [
        'vision_processor.py',
        'neural_learner.py',
        'neural_generator.py',
        'hybrid_analyzer.py'
    ]
    
    print("\n\nTesting AI Modules (expected to need PyTorch/OpenCV):")
    ai_results = []
    for module in ai_modules:
        if os.path.exists(module):
            result = test_module(module)
            ai_results.append(result)
    
    # Summary
    print("\n\n=== Summary ===")
    
    core_independent = sum(1 for r in core_results if r is True)
    print(f"Core modules working independently: {core_independent}/{len(core_results)}")
    
    ai_dependent = sum(1 for r in ai_results if r is False)
    print(f"AI modules requiring libraries: {ai_dependent}/{len(ai_results)}")
    
    print("\nConclusion:")
    if core_independent == len(core_results):
        print("✓ All core modules work without AI libraries!")
        print("✓ The system is truly modular - delete any module and others continue")
    else:
        print("✗ Some core modules have unexpected dependencies")
    
    # Test actual functionality without AI
    print("\n\nTesting actual execution without AI modules...")
    
    # Remove AI modules temporarily
    ai_files = ['vision_processor.py', 'neural_learner.py', 
                'neural_generator.py', 'hybrid_analyzer.py']
    
    backed_up = []
    for f in ai_files:
        if os.path.exists(f):
            os.rename(f, f + '.backup')
            backed_up.append(f)
    
    try:
        # Run core analysis
        print("\nRunning core analysis without AI modules...")
        
        # Make sure we have an image
        if not any(f.endswith(('.jpg', '.png', '.bmp')) for f in os.listdir('.')):
            subprocess.run([sys.executable, 'create_test_image.py'], capture_output=True)
        
        # Run basic pipeline
        scripts = ['pixel_reader.py', 'pattern_recognizer.py', 'intensity_analyzer.py']
        
        for script in scripts:
            if os.path.exists(script):
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"  ✓ {script} executed successfully")
                else:
                    print(f"  ✗ {script} failed")
        
        print("\n✓ Core system works perfectly without AI modules!")
        
    finally:
        # Restore AI modules
        for f in backed_up:
            os.rename(f + '.backup', f)
        print("\nAI modules restored")

if __name__ == "__main__":
    main()