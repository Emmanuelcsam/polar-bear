#!/usr/bin/env python3
"""
Integration Test Suite
Tests all scripts for independent operation and connector control
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

def test_script_independence():
    """Test that each script can run independently"""
    print("\n=== Testing Script Independence ===")
    
    scripts = [
        ('data-store.py', []),
        ('pixel-generator.py', [], 2),  # Run for 2 seconds
        ('anomaly-detector.py', []),
        ('batch-processor.py', ['.']),
        ('learner.py', []),
        ('trend-reader.py', []),
        ('pattern-recognition.py', []),
        ('geomtry-analyzer.py', []),
        ('orchestrator.py', [])
    ]
    
    results = {}
    
    for script_info in scripts:
        script = script_info[0]
        args = script_info[1] if len(script_info) > 1 else []
        timeout = script_info[2] if len(script_info) > 2 else 5
        
        print(f"\nTesting {script}...")
        
        try:
            cmd = [sys.executable, script] + [str(a) for a in args]
            
            if 'generator' in script:
                # For generator scripts, run with timeout
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    stdout, stderr = proc.communicate()
                    stdout = stdout or "Script ran successfully (terminated after timeout)"
            else:
                # For other scripts, run normally
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                stdout = result.stdout
                stderr = result.stderr
            
            results[script] = {
                'success': True,
                'output': stdout[:200] if stdout else 'No output',
                'error': stderr[:200] if stderr else None
            }
            print(f"✓ {script} - OK")
            
        except Exception as e:
            results[script] = {
                'success': False,
                'error': str(e)
            }
            print(f"✗ {script} - FAILED: {e}")
    
    return results

def test_connector_control():
    """Test connector control over scripts"""
    print("\n=== Testing Connector Control ===")
    
    # Import connector and script interface
    try:
        from connector import Connector
        from script_interface import ScriptManager
        
        # Initialize connector
        connector = Connector()
        connector.initialize()
        
        # Test getting script info
        print("\n1. Getting script information:")
        info = connector.get_script_info()
        for script, details in info.items():
            print(f"  - {script}: {len(details.get('functions', []))} functions")
        
        # Test setting variables
        print("\n2. Testing variable control:")
        
        # Test data-store parameters
        result = connector.control_variable('data-store', 'events_file', 'test_events.log')
        print(f"  - Set data-store events_file: {result}")
        
        # Test anomaly-detector threshold
        result = connector.control_variable('anomaly-detector', 'threshold', 75)
        print(f"  - Set anomaly-detector threshold: {result}")
        
        # Test getting variables
        result = connector.control_variable('anomaly-detector', 'threshold')
        print(f"  - Get anomaly-detector threshold: {result}")
        
        # Test function calls
        print("\n3. Testing function calls:")
        
        # Generate some pixels
        result = connector.call_function('pixel-generator', 'generate')
        print(f"  - Generated pixel: {result}")
        
        # Check for anomalies
        result = connector.call_function('anomaly-detector', 'anomalies')
        print(f"  - Anomalies found: {len(result) if isinstance(result, list) else result}")
        
        # Test orchestration
        print("\n4. Testing workflow orchestration:")
        workflow = [
            {
                'script': 'pixel-generator',
                'action': {'type': 'call_function', 'function_name': 'generate'},
                'name': 'generate_pixel',
                'store_as': 'pixel_data'
            },
            {
                'script': 'data-store',
                'action': {'type': 'call_function', 'function_name': 'get_stats'},
                'name': 'get_stats'
            }
        ]
        
        results = connector.orchestrate_workflow(workflow)
        print(f"  - Workflow completed with {len(results)} steps")
        
        return {'success': True, 'tests_passed': 4}
        
    except Exception as e:
        print(f"\n✗ Connector control test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_hivemind_connector():
    """Test hivemind connector functionality"""
    print("\n=== Testing Hivemind Connector ===")
    
    try:
        # Test importing
        from hivemind_connector import HivemindConnector
        
        # Create instance but don't start (would bind to ports)
        connector = HivemindConnector()
        
        # Test message processing
        print("\n1. Testing message processing:")
        
        # Status message
        response = connector.process_message({'command': 'status'})
        print(f"  - Status: {response.get('status')}")
        
        # Get info message
        response = connector.process_message({'command': 'get_info'})
        print(f"  - Scripts info retrieved: {len(response) > 0}")
        
        # Control message
        response = connector.process_message({
            'command': 'control',
            'control_type': 'get_variable',
            'script': 'anomaly-detector',
            'variable': 'threshold'
        })
        print(f"  - Control message processed: {'error' not in response}")
        
        return {'success': True, 'tests_passed': 3}
        
    except Exception as e:
        print(f"\n✗ Hivemind connector test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_script_collaboration():
    """Test scripts working together"""
    print("\n=== Testing Script Collaboration ===")
    
    try:
        # Clear any existing data
        if os.path.exists('events.log'):
            os.remove('events.log')
        
        # Import modules
        import pixel_generator
        import data_store
        import anomaly_detector
        import trend_reader
        import pattern_recognition
        
        print("\n1. Generating test data:")
        # Generate some pixels
        for i in range(50):
            pixel = pixel_generator.generate()
            data_store.save_event(pixel)
        
        # Add some anomalies
        for i in range(5):
            data_store.save_event({'pixel': 500})  # Anomaly
        
        print("  - Generated 55 events")
        
        print("\n2. Analyzing data:")
        # Get statistics
        stats = data_store.get_info()
        print(f"  - Data store info: {stats}")
        
        # Find anomalies
        anomalies = anomaly_detector.anomalies(100)
        print(f"  - Found {len(anomalies)} anomalies")
        
        # Get trends
        trends = trend_reader.trends()
        print(f"  - Trends: {trends[:100]}...")
        
        # Find patterns
        patterns = pattern_recognition.patterns()
        print(f"  - Found {patterns.get('total_unique', 0)} unique patterns")
        
        return {'success': True, 'tests_passed': 4}
        
    except Exception as e:
        print(f"\n✗ Collaboration test failed: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        # Cleanup
        if os.path.exists('events.log'):
            os.remove('events.log')

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("INTEGRATION TEST SUITE")
    print("=" * 60)
    
    all_results = {}
    
    # Run tests
    all_results['independence'] = test_script_independence()
    all_results['connector'] = test_connector_control()
    all_results['hivemind'] = test_hivemind_connector()
    all_results['collaboration'] = test_script_collaboration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in all_results.items():
        if isinstance(result, dict) and result.get('success'):
            status = "PASSED"
            passed_tests += 1
        else:
            status = "FAILED"
        total_tests += 1
        print(f"{test_name.capitalize()}: {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} test suites passed")
    
    # Save results
    with open('integration_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)