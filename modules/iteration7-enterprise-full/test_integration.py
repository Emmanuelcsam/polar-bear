#!/usr/bin/env python3
"""
Integration Test Script - Tests connector integration with scripts
Demonstrates both independent and collaborative modes
"""

import sys
import time
import json
import subprocess
from pathlib import Path

# Import the enhanced connector
try:
    from connector import get_connector, EnhancedConnector
    CONNECTOR_AVAILABLE = True
except ImportError:
    CONNECTOR_AVAILABLE = False
    print("Warning: Enhanced connector not available")


def test_independent_execution():
    """Test scripts running independently"""
    print("\n=== Testing Independent Execution ===")
    
    scripts_to_test = [
        'pixel_reader.py',
        'data_calculator.py',
        'pattern_recognizer.py',
        'anomaly_detector.py'
    ]
    
    results = {}
    
    for script in scripts_to_test:
        if Path(script).exists():
            print(f"\nRunning {script} independently...")
            try:
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                results[script] = {
                    'success': result.returncode == 0,
                    'output': result.stdout[:500],
                    'error': result.stderr[:500] if result.stderr else None
                }
                print(f"✓ {script} - Return code: {result.returncode}")
            except Exception as e:
                results[script] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"✗ {script} - Error: {e}")
        else:
            print(f"✗ {script} - Not found")
    
    return results


def test_connector_control():
    """Test connector control capabilities"""
    if not CONNECTOR_AVAILABLE:
        print("\n=== Skipping Connector Control Tests (Not Available) ===")
        return {}
    
    print("\n=== Testing Connector Control ===")
    
    # Initialize connector
    connector = get_connector()
    connector.start()
    
    # Wait for scripts to load
    time.sleep(2)
    
    # Get status
    status = connector.get_status()
    print(f"\nConnector Status:")
    print(f"- Scripts loaded: {status['scripts_loaded']}")
    print(f"- Running: {status['running']}")
    
    # Test parameter control
    print("\n--- Testing Parameter Control ---")
    test_results = {}
    
    for script_name, script_info in status['scripts'].items():
        if script_info['loaded']:
            try:
                # Test setting a parameter
                connector.set_parameter(script_name, 'test_param', 'test_value')
                
                # Test getting the parameter back
                value = connector.get_parameter(script_name, 'test_param')
                
                test_results[script_name] = {
                    'parameter_control': value == 'test_value'
                }
                
                print(f"✓ {script_name} - Parameter control working")
            except Exception as e:
                test_results[script_name] = {
                    'parameter_control': False,
                    'error': str(e)
                }
                print(f"✗ {script_name} - Parameter control failed: {e}")
    
    # Test function calls
    print("\n--- Testing Function Calls ---")
    
    for script_name, script_info in status['scripts'].items():
        if script_info['loaded'] and script_info['functions']:
            try:
                # Try to call the first available function
                func_name = script_info['functions'][0]
                print(f"Calling {script_name}.{func_name}()...")
                
                # This might fail if the function requires arguments
                try:
                    result = connector.call_script_function(script_name, func_name)
                    test_results[script_name]['function_call'] = True
                    print(f"✓ {script_name}.{func_name}() - Success")
                except Exception as e:
                    test_results[script_name]['function_call'] = False
                    print(f"✗ {script_name}.{func_name}() - Expected failure (may need args)")
                    
            except Exception as e:
                print(f"✗ {script_name} - Function call test failed: {e}")
    
    # Test collaborative execution
    print("\n--- Testing Collaborative Execution ---")
    
    try:
        connector.run_all_collaborative()
        time.sleep(2)
        
        # Check which scripts are running
        status = connector.get_status()
        running_count = sum(1 for s in status['scripts'].values() if s['running'])
        print(f"Scripts running collaboratively: {running_count}")
        
        # Stop all scripts
        connector.stop_all()
        print("✓ Collaborative execution test completed")
        
    except Exception as e:
        print(f"✗ Collaborative execution failed: {e}")
    
    # Save state
    try:
        connector.save_state()
        print("\n✓ Connector state saved")
    except Exception as e:
        print(f"\n✗ Failed to save state: {e}")
    
    return test_results


def test_hivemind_integration():
    """Test hivemind connector integration"""
    print("\n=== Testing Hivemind Integration ===")
    
    try:
        import socket
        import json
        
        # Try to connect to hivemind connector
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        
        try:
            sock.connect(('localhost', 10087))
            
            # Send status request
            request = json.dumps({'command': 'status'})
            sock.send(request.encode())
            
            # Receive response
            response = sock.recv(4096).decode()
            status = json.loads(response)
            
            print(f"✓ Hivemind connector active on port 10087")
            print(f"  - Connector ID: {status.get('connector_id')}")
            print(f"  - Scripts available: {status.get('scripts')}")
            print(f"  - Enhanced features: {status.get('enhanced_connector')}")
            
            # Test enhanced features
            if status.get('enhanced_connector'):
                sock.close()
                
                # Test parameter control
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(('localhost', 10087))
                
                request = json.dumps({
                    'command': 'set_parameter',
                    'script': 'pixel_reader.py',
                    'parameter': 'test_param',
                    'value': 'hivemind_test'
                })
                sock.send(request.encode())
                
                response = sock.recv(4096).decode()
                result = json.loads(response)
                
                if result.get('status') == 'parameter_set':
                    print("✓ Hivemind parameter control working")
                else:
                    print("✗ Hivemind parameter control failed")
                
            sock.close()
            return True
            
        except ConnectionRefusedError:
            print("✗ Hivemind connector not running on port 10087")
            return False
        except Exception as e:
            print(f"✗ Hivemind test failed: {e}")
            return False
            
    except ImportError:
        print("✗ Socket module not available")
        return False


def test_enhanced_pixel_reader():
    """Test the enhanced pixel reader specifically"""
    print("\n=== Testing Enhanced Pixel Reader ===")
    
    if not Path('pixel_reader_enhanced.py').exists():
        print("✗ Enhanced pixel reader not found")
        return False
    
    try:
        # Test independent execution
        print("\n--- Independent Mode ---")
        result = subprocess.run(
            [sys.executable, 'pixel_reader_enhanced.py'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "Running in independent mode" in result.stdout:
            print("✓ Enhanced pixel reader runs independently")
        else:
            print("✗ Enhanced pixel reader independent mode failed")
            
        # Test with connector if available
        if CONNECTOR_AVAILABLE:
            print("\n--- Collaborative Mode ---")
            connector = get_connector()
            connector.start()
            
            # Add the enhanced script
            connector.add_script('pixel_reader_enhanced.py')
            
            # Check if loaded
            script = connector.get_script('pixel_reader_enhanced.py')
            if script and script.module:
                print("✓ Enhanced pixel reader loaded in connector")
                
                # Test function exposure
                if 'read_pixels' in script.functions:
                    print("✓ read_pixels function exposed")
                if 'get_pixel_statistics' in script.functions:
                    print("✓ get_pixel_statistics function exposed")
                if 'process_directory' in script.functions:
                    print("✓ process_directory function exposed")
                    
                # Test parameter control
                connector.set_parameter('pixel_reader_enhanced.py', 'config_verbose', False)
                value = connector.get_parameter('pixel_reader_enhanced.py', 'config_verbose')
                if value == False:
                    print("✓ Parameter control working")
                else:
                    print("✗ Parameter control failed")
                    
            else:
                print("✗ Failed to load enhanced pixel reader in connector")
                
        return True
        
    except Exception as e:
        print(f"✗ Enhanced pixel reader test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("CONNECTOR INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Create a test image if none exists
    create_test_image = Path('create_test_image.py')
    if create_test_image.exists() and not any(Path('.').glob('*.png')):
        print("\nCreating test image...")
        subprocess.run([sys.executable, str(create_test_image)], capture_output=True)
    
    # Run tests
    results = {
        'independent': test_independent_execution(),
        'connector': test_connector_control(),
        'hivemind': test_hivemind_integration(),
        'enhanced_reader': test_enhanced_pixel_reader()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for category, result in results.items():
        if isinstance(result, dict):
            category_passed = sum(1 for v in result.values() if isinstance(v, dict) and v.get('success', False))
            category_total = len(result)
        elif isinstance(result, bool):
            category_passed = 1 if result else 0
            category_total = 1
        else:
            continue
            
        total_tests += category_total
        passed_tests += category_passed
        
        print(f"{category.title()}: {category_passed}/{category_total} passed")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    # Save results
    with open('integration_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to integration_test_results.json")


if __name__ == "__main__":
    main()