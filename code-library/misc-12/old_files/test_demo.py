#!/usr/bin/env python3
"""
Simple demo test to verify the integrated system works without cameras.
"""

import sys
import time

def test_demo_mode():
    """Test that demo mode works correctly"""
    print("Testing demo mode integration...")
    
    try:
        from live_feed import LiveFeed
        
        # Test live feed in demo mode
        live_feed = LiveFeed(demo_mode=True)
        print("✓ Live feed demo mode initialized successfully")
        
        # Test that it can read frames
        frame = live_feed.read_frame()
        if frame is not None:
            print(f"✓ Demo frame generated: {frame.shape}")
        else:
            print("✗ Demo frame generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Demo mode test failed: {e}")
        return False

def test_main_demo():
    """Test main.py in demo mode"""
    print("\nTesting main.py demo mode...")
    
    try:
        # Import main components
        from main import UltraFastCoreDetector
        
        # Create detector in demo mode
        detector = UltraFastCoreDetector(demo_mode=True)
        print("✓ Main detector demo mode initialized successfully")
        
        # Test system info
        info = detector.get_system_info()
        print(f"✓ System info retrieved: {len(info)} sections")
        
        return True
        
    except Exception as e:
        print(f"✗ Main demo test failed: {e}")
        return False

def main():
    """Run demo tests"""
    print("=" * 50)
    print("Demo Mode Integration Test")
    print("=" * 50)
    
    tests = [
        ("Live Feed Demo", test_demo_mode),
        ("Main Demo", test_main_demo),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"Demo Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Demo mode works correctly!")
        print("The system can run without physical cameras.")
    else:
        print("⚠ Some demo tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 