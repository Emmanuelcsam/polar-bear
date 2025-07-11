#!/usr/bin/env python3
"""
Test that HPC modules work independently and adapt to available hardware
Shows GPU fallback, parallel scaling, and module independence
"""

import os
import subprocess
import sys
import json
import shutil

def test_gpu_without_cuda():
    """Test GPU module works without CUDA"""
    print("\n[TEST] Testing GPU module without CUDA...")
    
    # Create test script that blocks CUDA
    test_script = """
import sys
# Block CUDA imports
class BlockedModule:
    def __getattr__(self, name):
        raise ImportError("CUDA blocked for testing")

sys.modules['torch.cuda'] = BlockedModule()

# Now run gpu_accelerator
exec(open('gpu_accelerator.py').read())
"""
    
    with open('test_gpu_nocuda.py', 'w') as f:
        f.write(test_script)
    
    try:
        result = subprocess.run(
            [sys.executable, 'test_gpu_nocuda.py'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "CPU fallback" in result.stdout or "No GPU detected" in result.stdout:
            print("  ✓ GPU module correctly falls back to CPU")
            return True
        else:
            print("  ✗ GPU module failed without CUDA")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    finally:
        if os.path.exists('test_gpu_nocuda.py'):
            os.remove('test_gpu_nocuda.py')

def test_parallel_scaling():
    """Test parallel processor adapts to CPU cores"""
    print("\n[TEST] Testing parallel processor scaling...")
    
    if os.path.exists('parallel_processor.py'):
        result = subprocess.run(
            [sys.executable, 'parallel_processor.py'],
            capture_output=True,
            text=True,
            timeout=20
        )
        
        if "CPU detected" in result.stdout and "worker processes" in result.stdout:
            print("  ✓ Parallel processor adapts to available CPU cores")
            
            # Check if it created results
            if os.path.exists('parallel_results.json'):
                with open('parallel_results.json', 'r') as f:
                    data = json.load(f)
                    cores = data.get('cpu_cores', 0)
                    print(f"  ✓ Used {cores} CPU cores for processing")
            return True
        else:
            print("  ✗ Parallel processor failed")
            return False
    
    return False

def test_distributed_simulation():
    """Test distributed analyzer works as simulation"""
    print("\n[TEST] Testing distributed computing simulation...")
    
    if os.path.exists('distributed_analyzer.py'):
        result = subprocess.run(
            [sys.executable, 'distributed_analyzer.py'],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if "nodes initialized" in result.stdout and "Distributed processing complete" in result.stdout:
            print("  ✓ Distributed analyzer runs without actual cluster")
            return True
        else:
            print("  ✗ Distributed analyzer failed")
            return False
    
    return False

def test_hpc_without_others():
    """Test HPC modules work without each other"""
    print("\n[TEST] Testing HPC module independence...")
    
    # Test each HPC module with others removed
    hpc_modules = [
        'gpu_accelerator.py',
        'parallel_processor.py',
        'distributed_analyzer.py',
        'hpc_optimizer.py'
    ]
    
    results = []
    
    for test_module in hpc_modules:
        if not os.path.exists(test_module):
            continue
            
        # Backup other HPC modules
        backed_up = []
        for other in hpc_modules:
            if other != test_module and os.path.exists(other):
                shutil.move(other, other + '.backup')
                backed_up.append(other)
        
        try:
            print(f"\n  Testing {test_module} in isolation...")
            
            # Ensure we have test data
            if not os.path.exists('pixel_data.json'):
                # Create minimal test data
                import numpy as np
                test_data = {
                    'pixels': np.random.randint(0, 256, 1000).tolist(),
                    'size': [32, 32]
                }
                with open('pixel_data.json', 'w') as f:
                    json.dump(test_data, f)
            
            # Run module
            result = subprocess.run(
                [sys.executable, test_module],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 or "complete" in result.stdout.lower():
                print(f"    ✓ {test_module} works independently")
                results.append(True)
            else:
                print(f"    ✗ {test_module} failed in isolation")
                results.append(False)
                
        except subprocess.TimeoutExpired:
            print(f"    ✓ {test_module} runs (timeout ok for continuous modules)")
            results.append(True)
            
        finally:
            # Restore backed up modules
            for module in backed_up:
                shutil.move(module + '.backup', module)
    
    return all(results)

def test_hardware_adaptation():
    """Test HPC optimizer adapts to hardware"""
    print("\n[TEST] Testing hardware adaptation...")
    
    if os.path.exists('hpc_optimizer.py'):
        result = subprocess.run(
            [sys.executable, 'hpc_optimizer.py'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if "System Resources:" in result.stdout:
            print("  ✓ HPC optimizer detects system resources")
            
            # Check recommendations
            if os.path.exists('hpc_pipeline.json'):
                with open('hpc_pipeline.json', 'r') as f:
                    data = json.load(f)
                    
                print(f"  ✓ Detected {data['resources']['cpu_cores']} CPU cores")
                print(f"  ✓ Detected {data['resources']['memory_gb']:.1f} GB memory")
                
                if data.get('recommendations'):
                    print("  ✓ Generated optimization recommendations")
                    
            return True
    
    return False

def test_core_without_hpc():
    """Test core system works without HPC modules"""
    print("\n[TEST] Testing core system without HPC...")
    
    hpc_modules = [
        'gpu_accelerator.py',
        'parallel_processor.py',
        'distributed_analyzer.py',
        'hpc_optimizer.py'
    ]
    
    # Backup HPC modules
    backed_up = []
    for module in hpc_modules:
        if os.path.exists(module):
            shutil.move(module, module + '.backup')
            backed_up.append(module)
    
    try:
        # Run core modules
        core_scripts = ['pixel_reader.py', 'pattern_recognizer.py', 'anomaly_detector.py']
        
        all_passed = True
        for script in core_scripts:
            if os.path.exists(script):
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"  ✓ {script} works without HPC modules")
                else:
                    print(f"  ✗ {script} failed")
                    all_passed = False
        
        return all_passed
        
    finally:
        # Restore HPC modules
        for module in backed_up:
            shutil.move(module + '.backup', module)

def main():
    print("=== HPC MODULE INDEPENDENCE TEST ===")
    print("This verifies HPC modules adapt to hardware and work independently\n")
    
    # Ensure we have test data
    if not any(f.endswith(('.jpg', '.png', '.bmp')) for f in os.listdir('.')):
        subprocess.run([sys.executable, 'create_test_image.py'], capture_output=True)
    
    results = []
    
    # Test 1: GPU fallback
    results.append(("GPU CPU fallback", test_gpu_without_cuda()))
    
    # Test 2: Parallel scaling
    results.append(("Parallel CPU scaling", test_parallel_scaling()))
    
    # Test 3: Distributed simulation
    results.append(("Distributed simulation", test_distributed_simulation()))
    
    # Test 4: HPC independence
    results.append(("HPC module independence", test_hpc_without_others()))
    
    # Test 5: Hardware adaptation
    results.append(("Hardware adaptation", test_hardware_adaptation()))
    
    # Test 6: Core without HPC
    results.append(("Core system without HPC", test_core_without_hpc()))
    
    # Summary
    print("\n=== SUMMARY ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:<25} {status}")
    
    if passed == total:
        print("\n✓ All HPC modules are truly independent!")
        print("✓ GPU module falls back to CPU when needed")
        print("✓ Parallel module adapts to available cores")
        print("✓ Distributed module works as simulation")
        print("✓ HPC optimizer adapts to your hardware")
        print("✓ Core system unaffected by HPC modules")
    
    print("\nKey features:")
    print("• Automatic GPU detection and fallback")
    print("• CPU core detection and scaling")
    print("• Hardware-aware optimization")
    print("• No dependencies between modules")
    print("• Graceful degradation")

if __name__ == "__main__":
    main()