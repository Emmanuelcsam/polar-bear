#!/usr/bin/env python3
"""
Comprehensive Troubleshooting and Testing Suite for ML Models Connector System
Tests all scripts for connector integration and collaboration capabilities
"""

import sys
import socket
import json
import time
import subprocess
import os
from pathlib import Path
from datetime import datetime
import threading
import importlib.util

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60)

def print_result(test_name, success, details=""):
    """Print test result"""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{test_name:<40} [{status}]")
    if details:
        print(f"  └─ {details}")

def test_port_availability(port):
    """Test if a port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0  # True if port is available (connection failed)
    except:
        return False

def test_connector_communication(port):
    """Test communication with a connector"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', port))
        
        # Send status request
        message = json.dumps({"command": "status"})
        sock.send(message.encode())
        
        # Receive response
        response = sock.recv(4096).decode()
        data = json.loads(response)
        sock.close()
        
        return True, data
    except Exception as e:
        return False, str(e)

def test_script_interface():
    """Test if script interface module is available and working"""
    try:
        spec = importlib.util.spec_from_file_location("script_interface", "script_interface.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check for required classes
        has_interface = hasattr(module, 'ScriptInterface')
        has_client = hasattr(module, 'ConnectorClient')
        has_wrapper = hasattr(module, 'create_standalone_wrapper')
        
        return all([has_interface, has_client, has_wrapper]), {
            "ScriptInterface": has_interface,
            "ConnectorClient": has_client,
            "create_standalone_wrapper": has_wrapper
        }
    except Exception as e:
        return False, str(e)

def test_script_wrapper():
    """Test script wrapper functionality"""
    try:
        spec = importlib.util.spec_from_file_location("script_wrappers", "script_wrappers.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check for wrapper classes
        has_generic = hasattr(module, 'GenericScriptWrapper')
        has_anomaly = hasattr(module, 'AnomalyDetectionWrapper')
        has_cnn = hasattr(module, 'CNNScriptWrapper')
        has_collab = hasattr(module, 'CollaborativeWrapper')
        
        return all([has_generic, has_anomaly, has_cnn, has_collab]), {
            "GenericScriptWrapper": has_generic,
            "AnomalyDetectionWrapper": has_anomaly,
            "CNNScriptWrapper": has_cnn,
            "CollaborativeWrapper": has_collab
        }
    except Exception as e:
        return False, str(e)

def test_script_execution(script_name, with_connector=False):
    """Test if a script can be executed"""
    script_path = Path(script_name)
    
    if not script_path.exists():
        return False, "Script not found"
    
    try:
        cmd = [sys.executable, str(script_path)]
        if with_connector:
            cmd.append("--with-connector")
        
        # Run with timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit to see if it starts
        time.sleep(2)
        
        if process.poll() is None:
            # Still running, that's good
            process.terminate()
            process.wait()
            return True, "Script started successfully"
        else:
            # Check if it exited with error
            if process.returncode != 0:
                stderr = process.stderr.read()
                return False, f"Exit code {process.returncode}: {stderr[:100]}"
            else:
                return True, "Script completed successfully"
                
    except Exception as e:
        return False, str(e)

def test_enhanced_anomaly_script():
    """Test the enhanced anomaly detection script"""
    try:
        # First check if it can be imported
        spec = importlib.util.spec_from_file_location("anomaly_detection_script", "anomaly_detection_script.py")
        module = importlib.util.module_from_spec(spec)
        
        # Check for required class
        with open("anomaly_detection_script.py", 'r') as f:
            content = f.read()
            has_class = "class AnomalyDetectionSystem" in content
            has_interface = "ScriptInterface" in content
            has_connector_check = "CONNECTOR_AVAILABLE" in content
            
        return all([has_class, has_interface, has_connector_check]), {
            "AnomalyDetectionSystem class": has_class,
            "ScriptInterface integration": has_interface,
            "Connector availability check": has_connector_check
        }
    except Exception as e:
        return False, str(e)

def run_integration_test():
    """Run a full integration test"""
    print_header("STARTING INTEGRATION TEST")
    
    results = []
    
    # 1. Start hivemind connector
    print("\n1. Starting Hivemind Connector...")
    hivemind_proc = subprocess.Popen(
        [sys.executable, "hivemind_connector.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(3)  # Wait for startup
    
    # 2. Test connection
    success, data = test_connector_communication(10117)
    results.append(("Hivemind connector connection", success, str(data) if success else data))
    
    # 3. Start enhanced connector
    print("\n2. Starting Enhanced Connector...")
    connector_proc = subprocess.Popen(
        [sys.executable, "connector.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True
    )
    time.sleep(2)
    
    # 4. Run anomaly detection with connector
    print("\n3. Running Anomaly Detection with Connector...")
    anomaly_proc = subprocess.Popen(
        [sys.executable, "anomaly_detection_script.py", "--with-connector"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for execution
    time.sleep(10)
    
    # 5. Check if script registered
    success, data = test_connector_communication(10117)
    if success:
        registered = data.get('registered_scripts', 0) > 0
        results.append(("Script registration", registered, f"Registered scripts: {data.get('registered_scripts', 0)}"))
    
    # Clean up
    print("\n4. Cleaning up processes...")
    for proc in [anomaly_proc, connector_proc, hivemind_proc]:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    
    return results

def generate_report(all_results):
    """Generate a troubleshooting report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"troubleshooting_report_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("ML MODELS CONNECTOR SYSTEM - TROUBLESHOOTING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        # Summary
        total_tests = sum(len(results) for _, results in all_results)
        passed_tests = sum(1 for _, results in all_results for _, success, _ in results if success)
        
        f.write(f"SUMMARY: {passed_tests}/{total_tests} tests passed\n\n")
        
        # Detailed results
        for category, results in all_results:
            f.write(f"\n{category}\n")
            f.write("-" * len(category) + "\n")
            for test_name, success, details in results:
                status = "PASS" if success else "FAIL"
                f.write(f"{test_name:<40} [{status}]\n")
                if details:
                    f.write(f"  Details: {details}\n")
            f.write("\n")
        
        # Recommendations
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        
        # Check for common issues
        port_issues = any("port" in test.lower() and not success 
                         for _, results in all_results 
                         for test, success, _ in results)
        
        if port_issues:
            f.write("- Port conflicts detected. Check if ports 10117-10118 are in use.\n")
            f.write("  Run: lsof -i :10117 and lsof -i :10118\n")
        
        interface_issues = any("interface" in test.lower() and not success 
                             for _, results in all_results 
                             for test, success, _ in results)
        
        if interface_issues:
            f.write("- Script interface issues detected. Ensure script_interface.py is present.\n")
        
        f.write("\nFor full integration:\n")
        f.write("1. Ensure all connectors are running (hivemind_connector.py)\n")
        f.write("2. Update scripts to use ScriptInterface base class\n")
        f.write("3. Run scripts with --with-connector flag\n")
        f.write("4. Use connector.py menu to control and monitor scripts\n")
    
    return filename

def main():
    """Main troubleshooting function"""
    print_header("ML MODELS CONNECTOR TROUBLESHOOTING")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # Test 1: Basic Requirements
    print_header("TESTING BASIC REQUIREMENTS")
    results = []
    
    # Check Python version
    py_version = sys.version_info
    py_ok = py_version.major >= 3 and py_version.minor >= 6
    results.append(("Python version >= 3.6", py_ok, f"{py_version.major}.{py_version.minor}.{py_version.micro}"))
    
    # Check required files
    required_files = [
        "connector.py",
        "hivemind_connector.py",
        "script_interface.py",
        "script_wrappers.py",
        "anomaly_detection_script.py"
    ]
    
    for file in required_files:
        exists = Path(file).exists()
        results.append((f"File: {file}", exists, ""))
    
    all_results.append(("Basic Requirements", results))
    
    # Test 2: Port Availability
    print_header("TESTING PORT AVAILABILITY")
    results = []
    
    ports = {
        10117: "Hivemind Connector",
        10118: "Script Control Server",
        10004: "Parent Connector"
    }
    
    for port, description in ports.items():
        available = test_port_availability(port)
        results.append((f"Port {port} ({description})", available, 
                       "Available" if available else "In use or blocked"))
    
    all_results.append(("Port Availability", results))
    
    # Test 3: Module Functionality
    print_header("TESTING MODULE FUNCTIONALITY")
    results = []
    
    # Test script interface
    success, details = test_script_interface()
    results.append(("Script Interface Module", success, str(details)))
    
    # Test script wrappers
    success, details = test_script_wrapper()
    results.append(("Script Wrappers Module", success, str(details)))
    
    # Test enhanced anomaly script
    success, details = test_enhanced_anomaly_script()
    results.append(("Enhanced Anomaly Script", success, str(details)))
    
    all_results.append(("Module Functionality", results))
    
    # Test 4: Script Execution
    print_header("TESTING SCRIPT EXECUTION")
    results = []
    
    # Test anomaly script standalone
    success, details = test_script_execution("anomaly_detection_script.py", False)
    results.append(("Anomaly Script (standalone)", success, details))
    
    all_results.append(("Script Execution", results))
    
    # Test 5: Connector Communication
    print_header("TESTING CONNECTOR COMMUNICATION")
    results = []
    
    # Try to communicate with running connectors
    success, data = test_connector_communication(10117)
    results.append(("Hivemind Connector (10117)", success, 
                   str(data) if success else "Not running or not responding"))
    
    success, data = test_connector_communication(10118)
    results.append(("Script Control Server (10118)", success,
                   str(data) if success else "Not running or not responding"))
    
    all_results.append(("Connector Communication", results))
    
    # Test 6: Integration Test (optional)
    user_input = input("\nRun full integration test? This will start connectors and scripts. (y/n): ")
    if user_input.lower() == 'y':
        integration_results = run_integration_test()
        all_results.append(("Integration Test", integration_results))
    
    # Print summary
    print_header("TEST SUMMARY")
    total_tests = sum(len(results) for _, results in all_results)
    passed_tests = sum(1 for _, results in all_results for _, success, _ in results if success)
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Display results
    for category, results in all_results:
        print(f"\n{category}:")
        for test_name, success, details in results:
            print_result(test_name, success, details)
    
    # Generate report
    report_file = generate_report(all_results)
    print(f"\n✓ Detailed report saved to: {report_file}")
    
    # Provide recommendations
    print_header("RECOMMENDATIONS")
    
    if passed_tests < total_tests:
        print("\nTo fix issues:")
        print("1. Ensure all required files are present")
        print("2. Check that ports 10117-10118 are not in use")
        print("3. Install missing Python packages")
        print("4. Run: python hivemind_connector.py")
        print("5. Run: python connector.py")
        print("6. Then run scripts with: python <script_name> --with-connector")
    else:
        print("\n✓ All tests passed! The connector system is ready to use.")
        print("\nTo use the system:")
        print("1. Start hivemind connector: python hivemind_connector.py")
        print("2. Start enhanced connector: python connector.py")
        print("3. Run scripts with: python <script_name> --with-connector")
        print("4. Use the connector menu to control scripts")

if __name__ == "__main__":
    main()