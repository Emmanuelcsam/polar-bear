#!/usr/bin/env python3
"""
Integration Test Script
Tests the complete integrated system with Pylon Viewer and circle overlay.
"""

import sys


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        from circle_overlay import UltraFastCircleOverlay
        print("✓ circle_overlay imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import circle_overlay: {e}")
        return False
    
    try:
        from live_feed import LiveFeed
        print("✓ live_feed imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import live_feed: {e}")
        return False
    
    try:
        from config_manager import ConfigManager
        print("✓ config_manager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import config_manager: {e}")
        return False
    
    try:
        from pylon_viewer_integration import PylonViewerManager
        print("✓ pylon_viewer_integration imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pylon_viewer_integration: {e}")
        return False
    
    return True


def test_config_manager():
    """Test configuration manager"""
    print("\nTesting configuration manager...")
    
    try:
        from config_manager import ConfigManager
        config_manager = ConfigManager("config.json")
        config = config_manager.get_circle_overlay_config()
        
        if config:
            print("✓ Configuration loaded successfully")
            print(f"  - Circle overlay config: {len(config)} sections")
            return True
        else:
            print("✗ Configuration is empty")
            return False
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_circle_overlay():
    """Test circle overlay functionality"""
    print("\nTesting circle overlay...")
    
    try:
        from circle_overlay import UltraFastCircleOverlay
        circle_overlay = UltraFastCircleOverlay("config.json")
        
        # Test basic functionality
        circle_info = circle_overlay.get_circle_info()
        print("✓ Circle overlay initialized successfully")
        print(f"  - Center: {circle_info['center']}")
        print(f"  - Radius: {circle_info['radius']}")
        
        # Test performance mode
        circle_overlay.set_performance_mode(True)
        print("✓ Performance mode set successfully")
        
        return True
    except Exception as e:
        print(f"✗ Circle overlay error: {e}")
        return False


def test_pylon_integration():
    """Test Pylon Viewer integration"""
    print("\nTesting Pylon Viewer integration...")
    
    try:
        from pylon_viewer_integration import PylonViewerManager
        pylon_manager = PylonViewerManager(auto_start=False)
        
        # Check if Pylon SDK is available
        pylon_available = pylon_manager.is_pylon_available()
        print(f"✓ Pylon SDK available: {pylon_available}")
        
        # Check if Pylon Viewer executable exists
        viewer_path = pylon_manager.find_pylon_viewer()
        if viewer_path:
            print(f"✓ Pylon Viewer found at: {viewer_path}")
        else:
            print("⚠ Pylon Viewer not found (will use webcam fallback)")
        
        # Get status
        status = pylon_manager.get_status()
        print(f"✓ Status: {status}")
        
        return True
    except Exception as e:
        print(f"✗ Pylon integration error: {e}")
        return False


def test_live_feed():
    """Test live feed functionality"""
    print("\nTesting live feed...")
    
    try:
        from live_feed import LiveFeed
        # Test in demo mode to avoid camera issues
        live_feed = LiveFeed(demo_mode=True)
        print("✓ Live feed initialized successfully (demo mode)")
        
        # Test performance mode
        live_feed.set_performance_mode(True)
        print("✓ Performance mode set successfully")
        
        return True
    except Exception as e:
        print(f"✗ Live feed error: {e}")
        return False


def main():
    """Run all integration tests"""
    print("=" * 50)
    print("Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration Manager", test_config_manager),
        ("Circle Overlay", test_circle_overlay),
        ("Pylon Integration", test_pylon_integration),
        ("Live Feed", test_live_feed),
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
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The integrated system is ready.")
        print("\nTo start the system:")
        print("  - Run: ./start_orchestrator.sh")
        print("  - Or: node monitor.js")
        print("  - Web interface: http://localhost:3000")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("The system may still work with reduced functionality.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 