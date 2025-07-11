#!/usr/bin/env python3
"""
Test script for Polar Bear Hivemind System
Validates the deployment and communication between connectors
"""

import os
import sys
import socket
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

class HivemindTester:
    def __init__(self):
        self.root_path = Path.cwd()
        self.test_results = []
        self.connector_paths = []
        
    def run_all_tests(self):
        """Run all diagnostic tests"""
        print("=" * 60)
        print("POLAR BEAR HIVEMIND DIAGNOSTIC TEST")
        print("=" * 60)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Root directory: {self.root_path}\n")
        
        # Run tests
        self.test_python_version()
        self.test_requirements()
        self.test_connector_deployment()
        self.test_port_availability()
        self.test_directory_structure()
        self.test_skip_directories()
        self.test_connector_scripts()
        self.test_root_server()
        self.test_connector_communication()
        
        # Show results
        self.show_results()
        
    def test_python_version(self):
        """Test Python version"""
        test_name = "Python Version Check"
        try:
            version = sys.version_info
            if version >= (3, 6):
                self.test_results.append((test_name, "PASS", f"Python {version.major}.{version.minor}.{version.micro}"))
            else:
                self.test_results.append((test_name, "FAIL", f"Python {version.major}.{version.minor} (3.6+ required)"))
        except Exception as e:
            self.test_results.append((test_name, "ERROR", str(e)))
            
    def test_requirements(self):
        """Test required packages"""
        packages = ['psutil', 'colorama', 'watchdog', 'requests']
        
        for package in packages:
            test_name = f"Package: {package}"
            try:
                __import__(package)
                self.test_results.append((test_name, "PASS", "Installed"))
            except ImportError:
                self.test_results.append((test_name, "FAIL", "Not installed"))
                
    def test_connector_deployment(self):
        """Test connector deployment"""
        test_name = "Connector Deployment"
        
        try:
            # Find all hivemind_connector.py files
            connectors = list(self.root_path.rglob('hivemind_connector.py'))
            self.connector_paths = connectors
            
            if connectors:
                # Count by depth
                depth_counts = {}
                for conn in connectors:
                    depth = len(conn.relative_to(self.root_path).parts) - 1
                    depth_counts[depth] = depth_counts.get(depth, 0) + 1
                    
                details = f"Total: {len(connectors)}, Depths: {depth_counts}"
                self.test_results.append((test_name, "PASS", details))
            else:
                self.test_results.append((test_name, "FAIL", "No connectors found"))
        except Exception as e:
            self.test_results.append((test_name, "ERROR", str(e)))
            
    def test_port_availability(self):
        """Test port range availability"""
        test_name = "Port Availability (10000-10100)"
        
        try:
            blocked_ports = []
            for port in range(10000, 10101):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                if result == 0:
                    blocked_ports.append(port)
                    
            if len(blocked_ports) == 0:
                self.test_results.append((test_name, "PASS", "All ports available"))
            elif len(blocked_ports) < 10:
                self.test_results.append((test_name, "WARN", f"{len(blocked_ports)} ports in use: {blocked_ports[:5]}..."))
            else:
                self.test_results.append((test_name, "FAIL", f"{len(blocked_ports)} ports blocked"))
        except Exception as e:
            self.test_results.append((test_name, "ERROR", str(e)))
            
    def test_directory_structure(self):
        """Test directory structure"""
        test_name = "Directory Structure"
        
        try:
            expected_dirs = ['modules', 'ruleset', 'training', 'static', 'templates']
            missing = []
            
            for dir_name in expected_dirs:
                if not (self.root_path / dir_name).exists():
                    missing.append(dir_name)
                    
            if not missing:
                self.test_results.append((test_name, "PASS", "All expected directories exist"))
            else:
                self.test_results.append((test_name, "WARN", f"Missing: {', '.join(missing)}"))
        except Exception as e:
            self.test_results.append((test_name, "ERROR", str(e)))
            
    def test_skip_directories(self):
        """Test that virtual environments are skipped"""
        test_name = "Skip Virtual Environments"
        
        try:
            skip_dirs = ['venv', 'env', '.env', '__pycache__', 'node_modules', '.git']
            found_skipped = []
            
            for skip_dir in skip_dirs:
                for path in self.root_path.rglob(skip_dir):
                    if path.is_dir():
                        # Check if connector exists in this directory
                        if (path / 'hivemind_connector.py').exists():
                            found_skipped.append(str(path.relative_to(self.root_path)))
                            
            if not found_skipped:
                self.test_results.append((test_name, "PASS", "No connectors in skip directories"))
            else:
                self.test_results.append((test_name, "FAIL", f"Found in: {', '.join(found_skipped[:3])}..."))
        except Exception as e:
            self.test_results.append((test_name, "ERROR", str(e)))
            
    def test_connector_scripts(self):
        """Test connector script validity"""
        test_name = "Connector Script Validity"
        
        try:
            if not self.connector_paths:
                self.test_results.append((test_name, "SKIP", "No connectors to test"))
                return
                
            # Test first few connectors
            test_count = min(5, len(self.connector_paths))
            valid_count = 0
            
            for conn_path in self.connector_paths[:test_count]:
                try:
                    # Check if script is valid Python
                    result = subprocess.run(
                        [sys.executable, '-m', 'py_compile', str(conn_path)],
                        capture_output=True
                    )
                    if result.returncode == 0:
                        valid_count += 1
                except:
                    pass
                    
            if valid_count == test_count:
                self.test_results.append((test_name, "PASS", f"All {test_count} tested scripts are valid"))
            else:
                self.test_results.append((test_name, "FAIL", f"{valid_count}/{test_count} scripts valid"))
        except Exception as e:
            self.test_results.append((test_name, "ERROR", str(e)))
            
    def test_root_server(self):
        """Test root server connectivity"""
        test_name = "Root Server (port 10000)"
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 10000))
            sock.close()
            
            if result == 0:
                self.test_results.append((test_name, "PASS", "Root server is running"))
            else:
                self.test_results.append((test_name, "INFO", "Root server not running (normal if hivemind not started)"))
        except Exception as e:
            self.test_results.append((test_name, "ERROR", str(e)))
            
    def test_connector_communication(self):
        """Test connector communication if running"""
        test_name = "Connector Communication"
        
        try:
            # Try to connect to a few connector ports
            active_connectors = 0
            tested_ports = []
            
            for port in range(10001, 10011):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    if sock.connect_ex(('localhost', port)) == 0:
                        # Try to send status command
                        sock.send(json.dumps({'command': 'status'}).encode())
                        response = sock.recv(4096)
                        if response:
                            active_connectors += 1
                            tested_ports.append(port)
                    sock.close()
                except:
                    pass
                    
            if active_connectors > 0:
                self.test_results.append((test_name, "PASS", f"{active_connectors} active connectors found"))
            else:
                self.test_results.append((test_name, "INFO", "No active connectors (normal if hivemind not started)"))
        except Exception as e:
            self.test_results.append((test_name, "ERROR", str(e)))
            
    def show_results(self):
        """Display test results"""
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        # Count results
        counts = {'PASS': 0, 'FAIL': 0, 'WARN': 0, 'ERROR': 0, 'INFO': 0, 'SKIP': 0}
        
        # Display results
        for test_name, status, details in self.test_results:
            counts[status] = counts.get(status, 0) + 1
            
            # Color coding for terminal
            if status == 'PASS':
                symbol = '✓'
            elif status == 'FAIL':
                symbol = '✗'
            elif status == 'WARN':
                symbol = '⚠'
            elif status == 'ERROR':
                symbol = '⚠'
            elif status == 'INFO':
                symbol = 'ℹ'
            else:
                symbol = '-'
                
            print(f"{symbol} {status:5} | {test_name:30} | {details}")
            
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total tests: {len(self.test_results)}")
        print(f"Passed: {counts['PASS']}")
        print(f"Failed: {counts['FAIL']}")
        print(f"Warnings: {counts['WARN']}")
        print(f"Errors: {counts['ERROR']}")
        print(f"Info: {counts['INFO']}")
        
        # Overall status
        if counts['FAIL'] == 0 and counts['ERROR'] == 0:
            print("\nOVERALL: System is ready for deployment ✓")
        else:
            print("\nOVERALL: Issues detected, please review failures ✗")
            
        # Recommendations
        print("\nRECOMMENDATIONS:")
        if counts['FAIL'] > 0 or counts['ERROR'] > 0:
            print("- Fix failed tests before running hivemind")
        if 'Root server not running' in str(self.test_results):
            print("- Run 'python launch_hivemind.py' to start the system")
        if len(self.connector_paths) == 0:
            print("- No connectors deployed yet, run hivemind to deploy")
            
def main():
    """Main test function"""
    tester = HivemindTester()
    tester.run_all_tests()
    
if __name__ == "__main__":
    main()