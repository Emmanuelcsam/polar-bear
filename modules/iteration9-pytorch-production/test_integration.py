#!/usr/bin/env python3
"""
Test Integration Script for PyTorch Production Module
Tests both independent script execution and connector integration
"""

import os
import sys
import json
import socket
import subprocess
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegrationTester:
    """Test all script integrations"""
    
    def __init__(self):
        self.results = []
        self.connector_port = 10051
        self.hivemind_port = 10050
        
    def test_independent_scripts(self):
        """Test scripts running independently"""
        logger.info("=== Testing Independent Script Execution ===")
        
        # Test preprocess.py
        logger.info("Testing preprocess.py...")
        try:
            result = subprocess.run([sys.executable, 'preprocess.py'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.results.append(('preprocess.py (independent)', 'PASS', 'Script executed successfully'))
            else:
                self.results.append(('preprocess.py (independent)', 'FAIL', result.stderr))
        except Exception as e:
            self.results.append(('preprocess.py (independent)', 'ERROR', str(e)))
            
        # Test script interface
        logger.info("Testing script_interface.py...")
        try:
            from script_interface import interface
            scripts = interface.list_scripts()
            if len(scripts) == 4:  # We expect 4 scripts
                self.results.append(('script_interface.py', 'PASS', f'Found {len(scripts)} scripts'))
            else:
                self.results.append(('script_interface.py', 'FAIL', f'Expected 4 scripts, found {len(scripts)}'))
        except Exception as e:
            self.results.append(('script_interface.py', 'ERROR', str(e)))
            
    def test_connector_integration(self):
        """Test connector integration with scripts"""
        logger.info("\n=== Testing Connector Integration ===")
        
        # Test basic connector
        logger.info("Testing connector.py...")
        try:
            result = subprocess.run([sys.executable, 'connector.py'], 
                                  capture_output=True, text=True, timeout=5)
            if 'Enhanced Connector Script Initialized' in result.stdout:
                self.results.append(('connector.py', 'PASS', 'Connector initialized successfully'))
            else:
                self.results.append(('connector.py', 'FAIL', 'Unexpected output'))
        except Exception as e:
            self.results.append(('connector.py', 'ERROR', str(e)))
            
    def test_script_control_via_interface(self):
        """Test controlling scripts through the interface"""
        logger.info("\n=== Testing Script Control via Interface ===")
        
        try:
            from script_interface import interface, handle_connector_command
            from script_wrappers import wrappers
            
            # Test listing scripts
            cmd = {'command': 'list_scripts'}
            response = handle_connector_command(cmd)
            if 'scripts' in response:
                self.results.append(('Script listing', 'PASS', f'Listed {len(response["scripts"])} scripts'))
            else:
                self.results.append(('Script listing', 'FAIL', 'No scripts returned'))
                
            # Test setting parameter
            cmd = {
                'command': 'set_parameter',
                'script': 'preprocess.py',
                'parameter': 'img_size',
                'value': 256
            }
            response = handle_connector_command(cmd)
            if response.get('status') == 'success':
                self.results.append(('Parameter setting', 'PASS', 'Set img_size to 256'))
            else:
                self.results.append(('Parameter setting', 'FAIL', response.get('error', 'Unknown error')))
                
            # Test wrapper function
            logger.info("Testing wrapper function...")
            result = wrappers.initialize_generator(img_size=64, force=True)
            if result.get('status') == 'success':
                self.results.append(('Wrapper function', 'PASS', 'Generator initialized via wrapper'))
            else:
                self.results.append(('Wrapper function', 'FAIL', result.get('error', 'Unknown error'))
                
        except Exception as e:
            self.results.append(('Script control', 'ERROR', str(e)))
            
    def test_connector_server(self):
        """Test connector server functionality"""
        logger.info("\n=== Testing Connector Server ===")
        
        # Start connector server in background
        server_process = subprocess.Popen([sys.executable, 'connector.py', '--server'],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Give server time to start
        
        try:
            # Test connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(('localhost', self.connector_port))
            
            # Send status command
            cmd = {'type': 'status'}
            sock.send(json.dumps(cmd).encode())
            response = json.loads(sock.recv(4096).decode())
            
            if response.get('status') == 'active':
                self.results.append(('Connector server', 'PASS', 'Server responded to status request'))
            else:
                self.results.append(('Connector server', 'FAIL', 'Invalid status response'))
                
            sock.close()
            
        except Exception as e:
            self.results.append(('Connector server', 'ERROR', str(e)))
        finally:
            server_process.terminate()
            server_process.wait()
            
    def generate_report(self):
        """Generate test report"""
        logger.info("\n=== TEST REPORT ===")
        passed = sum(1 for _, status, _ in self.results if status == 'PASS')
        failed = sum(1 for _, status, _ in self.results if status == 'FAIL')
        errors = sum(1 for _, status, _ in self.results if status == 'ERROR')
        
        logger.info(f"Total tests: {len(self.results)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Errors: {errors}")
        
        logger.info("\nDetailed Results:")
        for test, status, message in self.results:
            symbol = '✓' if status == 'PASS' else '✗' if status == 'FAIL' else '!'
            logger.info(f"{symbol} {test}: {status} - {message}")
            
        return passed == len(self.results)
        
    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("Starting PyTorch Production Module Integration Tests")
        
        self.test_independent_scripts()
        self.test_connector_integration()
        self.test_script_control_via_interface()
        self.test_connector_server()
        
        success = self.generate_report()
        
        # Cleanup test files
        test_files = ['generator_model.pth', 'target_data.pt', 'final_generated_image.png']
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Cleaned up: {file}")
                
        return success


if __name__ == "__main__":
    tester = IntegrationTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)