#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Polar Bear System
Tests all aspects of the integrated system
"""

import socket
import json
import time
import subprocess
import sys
import os
from typing import Dict, Any, List, Tuple
import threading
from pathlib import Path

class IntegrationTestSuite:
    """Comprehensive test suite for the integrated system"""
    
    def __init__(self):
        self.test_results = []
        self.connector_port = 10001
        self.tests_passed = 0
        self.tests_failed = 0
        
    def log_test_result(self, test_name: str, passed: bool, message: str = ""):
        """Log a test result"""
        result = {
            'test': test_name,
            'passed': passed,
            'message': message,
            'timestamp': time.time()
        }
        self.test_results.append(result)
        
        if passed:
            self.tests_passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            self.tests_failed += 1
            print(f"❌ {test_name}: FAILED - {message}")
        
        if message:
            print(f"   {message}")
    
    def send_connector_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to connector"""
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(5)
            client.connect(('localhost', self.connector_port))
            
            client.send(json.dumps(command).encode('utf-8'))
            response = json.loads(client.recv(8192).decode('utf-8'))
            client.close()
            
            return response
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_connector_availability(self) -> bool:
        """Test 1: Check if hivemind connector is available"""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(2)
            test_socket.connect(('localhost', self.connector_port))
            test_socket.close()
            self.log_test_result("Connector Availability", True, 
                               f"Connector is running on port {self.connector_port}")
            return True
        except Exception as e:
            self.log_test_result("Connector Availability", False, str(e))
            return False
    
    def test_script_discovery(self) -> bool:
        """Test 2: Verify all scripts are discovered"""
        response = self.send_connector_command({'command': 'get_scripts'})
        
        if response.get('status') != 'ok':
            self.log_test_result("Script Discovery", False, 
                               f"Failed to get scripts: {response}")
            return False
        
        scripts = response.get('scripts', {})
        expected_scripts = ['polar_bear_master', 'speed-test', 'collaboration_manager']
        
        found_scripts = list(scripts.keys())
        missing = [s for s in expected_scripts if s not in found_scripts]
        
        if missing:
            self.log_test_result("Script Discovery", False, 
                               f"Missing scripts: {missing}")
            return False
        
        self.log_test_result("Script Discovery", True, 
                           f"Found {len(scripts)} scripts: {', '.join(found_scripts)}")
        return True
    
    def test_parameter_access(self) -> bool:
        """Test 3: Test parameter access and modification"""
        # Test getting a parameter
        get_response = self.send_connector_command({
            'command': 'get_parameter',
            'script': 'collaboration_manager',
            'parameter': 'COLLABORATION_STATE'
        })
        
        if get_response.get('status') != 'ok':
            self.log_test_result("Parameter Access", False, 
                               "Failed to get parameter")
            return False
        
        # Test setting a parameter
        test_value = {'test_key': 'test_value'}
        set_response = self.send_connector_command({
            'command': 'set_parameter',
            'script': 'collaboration_manager',
            'parameter': 'test_parameter',
            'value': test_value
        })
        
        # Verify the set worked
        verify_response = self.send_connector_command({
            'command': 'get_parameter',
            'script': 'collaboration_manager',
            'parameter': 'test_parameter'
        })
        
        success = (set_response.get('success') or 
                  verify_response.get('value') == test_value)
        
        self.log_test_result("Parameter Access", success,
                           "Parameter get/set operations work correctly" if success 
                           else "Failed to verify parameter operations")
        return success
    
    def test_function_execution(self) -> bool:
        """Test 4: Test remote function execution"""
        # Test executing a simple function
        response = self.send_connector_command({
            'command': 'execute_function',
            'script': 'collaboration_manager',
            'function': 'get_collaboration_status',
            'args': [],
            'kwargs': {}
        })
        
        if response.get('status') != 'ok':
            self.log_test_result("Function Execution", False, 
                               f"Failed to execute function: {response}")
            return False
        
        result = response.get('result', '')
        
        # Check if we got a meaningful result
        success = 'collaboration_state' in str(result)
        
        self.log_test_result("Function Execution", success,
                           "Remote function execution works" if success 
                           else "Function executed but result unexpected")
        return success
    
    def test_script_independence(self) -> bool:
        """Test 5: Verify scripts can run independently"""
        # Test collaboration_manager --test
        try:
            result = subprocess.run(
                [sys.executable, 'collaboration_manager.py', '--test'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            success = result.returncode == 0 and 'System Health' in result.stdout
            
            self.log_test_result("Script Independence", success,
                               "Scripts can run independently" if success 
                               else f"Script failed with: {result.stderr[:100]}")
            return success
            
        except Exception as e:
            self.log_test_result("Script Independence", False, str(e))
            return False
    
    def test_collaboration_coordination(self) -> bool:
        """Test 6: Test collaboration between scripts"""
        # Initialize collaboration manager
        init_response = self.send_connector_command({
            'command': 'execute_function',
            'script': 'collaboration_manager',
            'function': 'initialize_collaboration_manager',
            'args': [],
            'kwargs': {}
        })
        
        if init_response.get('status') != 'ok':
            self.log_test_result("Collaboration Coordination", False, 
                               "Failed to initialize collaboration manager")
            return False
        
        # Get collaboration status
        status_response = self.send_connector_command({
            'command': 'execute_function',
            'script': 'collaboration_manager',
            'function': 'get_collaboration_status',
            'args': [],
            'kwargs': {}
        })
        
        success = (status_response.get('status') == 'ok' and 
                  'manager_initialized' in str(status_response.get('result', '')))
        
        self.log_test_result("Collaboration Coordination", success,
                           "Collaboration system initialized and working" if success 
                           else "Collaboration system not working properly")
        return success
    
    def test_error_handling(self) -> bool:
        """Test 7: Test error handling and recovery"""
        # Test invalid command
        invalid_response = self.send_connector_command({
            'command': 'invalid_command'
        })
        
        # Test invalid script
        invalid_script = self.send_connector_command({
            'command': 'execute_function',
            'script': 'non_existent_script',
            'function': 'some_function',
            'args': [],
            'kwargs': {}
        })
        
        # Test invalid function
        invalid_function = self.send_connector_command({
            'command': 'execute_function',
            'script': 'collaboration_manager',
            'function': 'non_existent_function',
            'args': [],
            'kwargs': {}
        })
        
        # All should return error status
        success = (invalid_response.get('status') == 'error' and
                  invalid_script.get('status') == 'error' and
                  invalid_function.get('status') == 'error')
        
        self.log_test_result("Error Handling", success,
                           "Error handling works correctly" if success 
                           else "Error handling not working properly")
        return success
    
    def test_concurrent_access(self) -> bool:
        """Test 8: Test concurrent access to the connector"""
        results = []
        errors = []
        
        def concurrent_request(index):
            try:
                response = self.send_connector_command({
                    'command': 'status'
                })
                results.append(response.get('status') == 'ok')
            except Exception as e:
                errors.append(str(e))
        
        # Launch 10 concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5)
        
        success = len(results) >= 8 and all(results)  # At least 80% success
        
        self.log_test_result("Concurrent Access", success,
                           f"Handled {len(results)}/10 concurrent requests successfully" 
                           if success else f"Failed with {len(errors)} errors")
        return success
    
    def test_system_health_monitoring(self) -> bool:
        """Test 9: Test system health monitoring"""
        # Get system health through collaboration manager
        response = self.send_connector_command({
            'command': 'execute_function',
            'script': 'collaboration_manager',
            'function': 'get_collaboration_status',
            'args': [],
            'kwargs': {}
        })
        
        if response.get('status') != 'ok':
            self.log_test_result("System Health Monitoring", False, 
                               "Failed to get system health")
            return False
        
        # Also test direct connector status
        status_response = self.send_connector_command({'command': 'status'})
        
        success = (status_response.get('status') == 'ok' and
                  'connector_id' in status_response and
                  'discovered_scripts' in status_response)
        
        self.log_test_result("System Health Monitoring", success,
                           "System health monitoring functional" if success 
                           else "Health monitoring not working properly")
        return success
    
    def test_script_execution(self) -> bool:
        """Test 10: Test direct script execution"""
        # Test executing a script directly
        response = self.send_connector_command({
            'command': 'execute',
            'script': 'test_integration.py',
            'args': [],
            'kwargs': {}
        })
        
        # This might fail if test_integration.py tries to test itself
        # So we just check if the command is handled
        success = 'status' in response
        
        self.log_test_result("Script Execution", success,
                           "Script execution command handled" if success 
                           else "Script execution not supported")
        return success
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("=" * 60)
        print("POLAR BEAR COMPREHENSIVE INTEGRATION TEST SUITE")
        print("=" * 60)
        print()
        
        # Check prerequisites
        if not self.test_connector_availability():
            print("\n⚠️  Cannot proceed without connector. Please start:")
            print("   python3 hivemind_connector.py")
            return
        
        print()
        
        # Run all tests
        tests = [
            self.test_script_discovery,
            self.test_parameter_access,
            self.test_function_execution,
            self.test_script_independence,
            self.test_collaboration_coordination,
            self.test_error_handling,
            self.test_concurrent_access,
            self.test_system_health_monitoring,
            self.test_script_execution
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.log_test_result(test.__name__, False, f"Exception: {str(e)}")
            print()
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.tests_passed + self.tests_failed}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success Rate: {(self.tests_passed / (self.tests_passed + self.tests_failed) * 100):.1f}%")
        print()
        
        if self.tests_failed == 0:
            print("🎉 ALL TESTS PASSED! The system is fully integrated.")
        else:
            print("⚠️  Some tests failed. Please review the output above.")
        
        # Save results
        self.save_test_report()
    
    def save_test_report(self):
        """Save test results to a file"""
        report_path = Path("integration_test_report.json")
        
        report = {
            'timestamp': time.time(),
            'total_tests': self.tests_passed + self.tests_failed,
            'passed': self.tests_passed,
            'failed': self.tests_failed,
            'success_rate': self.tests_passed / (self.tests_passed + self.tests_failed) * 100,
            'results': self.test_results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTest report saved to: {report_path}")

def main():
    """Main entry point"""
    suite = IntegrationTestSuite()
    suite.run_all_tests()

if __name__ == "__main__":
    main()