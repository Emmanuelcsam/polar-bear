#!/usr/bin/env python3
"""
Comprehensive troubleshooting script for iteration2-basic-stats
Checks all scripts, connectors, and dependencies
"""

import os
import sys
import subprocess
import socket
import json
import importlib
import traceback
from pathlib import Path
from datetime import datetime

class Troubleshooter:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.successes = []
        self.directory = Path(__file__).parent
        
    def log_issue(self, category, message):
        self.issues.append(f"[{category}] {message}")
        
    def log_warning(self, category, message):
        self.warnings.append(f"[{category}] {message}")
        
    def log_success(self, category, message):
        self.successes.append(f"[{category}] {message}")
    
    def check_dependencies(self):
        """Check if all required packages are installed"""
        print("\n=== Checking Dependencies ===")
        
        required_packages = {
            'numpy': 'numpy',
            'scipy': 'scipy', 
            'PIL': 'Pillow',
            'queue': None,  # Built-in
            'socket': None,  # Built-in
            'threading': None,  # Built-in
            'ast': None,  # Built-in
        }
        
        for import_name, package_name in required_packages.items():
            try:
                importlib.import_module(import_name)
                self.log_success('DEPENDENCY', f"{import_name} is available")
            except ImportError:
                if package_name:
                    self.log_issue('DEPENDENCY', f"{import_name} not found. Install with: pip install {package_name}")
                else:
                    self.log_issue('DEPENDENCY', f"{import_name} not found (should be built-in)")
    
    def check_scripts(self):
        """Check if all scripts are present and have correct syntax"""
        print("\n=== Checking Scripts ===")
        
        expected_scripts = [
            'connector.py',
            'hivemind_connector.py',
            'correlation-finder.py',
            'image-anomaly-detector.py',
            'intensity-matcher.py'
        ]
        
        for script in expected_scripts:
            script_path = self.directory / script
            
            if not script_path.exists():
                self.log_issue('SCRIPT', f"{script} not found")
                continue
            
            # Check syntax
            try:
                with open(script_path, 'r') as f:
                    compile(f.read(), script, 'exec')
                self.log_success('SCRIPT', f"{script} has valid syntax")
            except SyntaxError as e:
                self.log_issue('SCRIPT', f"{script} has syntax error: {e}")
            
            # Check if executable
            if not os.access(script_path, os.R_OK):
                self.log_warning('SCRIPT', f"{script} is not readable")
    
    def check_connector_system(self):
        """Check the connector system functionality"""
        print("\n=== Checking Connector System ===")
        
        try:
            from connector import ConnectorSystem
            
            connector = ConnectorSystem()
            connector.scan_scripts()
            
            scripts_found = len(connector.scripts)
            if scripts_found > 0:
                self.log_success('CONNECTOR', f"Found {scripts_found} scripts")
                
                # Check each script controller
                for name, controller in connector.script_controllers.items():
                    if controller.is_loaded:
                        self.log_success('CONNECTOR', f"Script {name} loaded successfully")
                    else:
                        self.log_issue('CONNECTOR', f"Script {name} failed to load")
            else:
                self.log_issue('CONNECTOR', "No scripts found by connector")
                
            # Check collaboration
            connector.enable_collaboration()
            if hasattr(connector, 'shared_memory'):
                self.log_success('CONNECTOR', "Collaboration system enabled")
            else:
                self.log_issue('CONNECTOR', "Collaboration system failed to initialize")
                
        except Exception as e:
            self.log_issue('CONNECTOR', f"Failed to initialize connector system: {e}")
            traceback.print_exc()
    
    def check_hivemind_connectivity(self):
        """Check if hivemind connector can be reached"""
        print("\n=== Checking Hivemind Connectivity ===")
        
        # Check if hivemind_connector.py is running
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(('localhost', 10113))
            
            # Send status command
            status_msg = json.dumps({'command': 'status'})
            sock.send(status_msg.encode())
            response = json.loads(sock.recv(4096).decode())
            
            if response.get('status') == 'active':
                self.log_success('HIVEMIND', f"Hivemind connector is active on port 10113")
                self.log_success('HIVEMIND', f"Connector ID: {response.get('connector_id')}")
            else:
                self.log_warning('HIVEMIND', "Hivemind connector responded but status is not active")
                
            sock.close()
            
        except socket.error:
            self.log_warning('HIVEMIND', "Cannot connect to hivemind connector on port 10113")
            self.log_warning('HIVEMIND', "Run 'python hivemind_connector.py' in another terminal")
        except Exception as e:
            self.log_issue('HIVEMIND', f"Error checking hivemind connector: {e}")
    
    def check_script_execution(self):
        """Test basic execution of each script"""
        print("\n=== Testing Script Execution ===")
        
        # Test correlation-finder
        print("\n--- Testing correlation-finder ---")
        test_input = "1 10%\n2 20%\n3 30%"
        
        try:
            result = subprocess.run(
                [sys.executable, 'correlation-finder.py'],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.directory
            )
            
            if result.returncode == 0:
                self.log_success('EXECUTION', "correlation-finder.py runs successfully")
            else:
                self.log_issue('EXECUTION', f"correlation-finder.py failed with code {result.returncode}")
                if result.stderr:
                    self.log_issue('EXECUTION', f"Error: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            self.log_issue('EXECUTION', "correlation-finder.py timed out")
        except Exception as e:
            self.log_issue('EXECUTION', f"Failed to run correlation-finder.py: {e}")
        
        # Test other scripts similarly (with appropriate test cases)
        print("\n--- Testing intensity-matcher ---")
        
        # Create a small test image
        try:
            from PIL import Image
            import numpy as np
            
            test_img_path = self.directory / 'test_troubleshoot.jpg'
            if not test_img_path.exists():
                img_array = np.full((10, 10), 128, dtype=np.uint8)
                Image.fromarray(img_array, 'L').save(test_img_path)
            
            result = subprocess.run(
                [sys.executable, 'intensity-matcher.py', str(test_img_path), '128'],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.directory
            )
            
            if result.returncode == 0:
                self.log_success('EXECUTION', "intensity-matcher.py runs successfully")
            else:
                self.log_issue('EXECUTION', f"intensity-matcher.py failed with code {result.returncode}")
                
            # Cleanup
            if test_img_path.exists():
                test_img_path.unlink()
                
        except Exception as e:
            self.log_issue('EXECUTION', f"Failed to test intensity-matcher.py: {e}")
    
    def check_permissions(self):
        """Check file and directory permissions"""
        print("\n=== Checking Permissions ===")
        
        # Check directory permissions
        if os.access(self.directory, os.R_OK | os.W_OK | os.X_OK):
            self.log_success('PERMISSION', f"Directory {self.directory} has read/write/execute permissions")
        else:
            self.log_issue('PERMISSION', f"Directory {self.directory} has permission issues")
        
        # Check log file permissions
        log_file = self.directory / 'connector.log'
        if log_file.exists():
            if os.access(log_file, os.R_OK | os.W_OK):
                self.log_success('PERMISSION', "connector.log is readable and writable")
            else:
                self.log_warning('PERMISSION', "connector.log has permission issues")
    
    def generate_report(self):
        """Generate a comprehensive troubleshooting report"""
        print("\n" + "=" * 60)
        print("TROUBLESHOOTING REPORT")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Directory: {self.directory}")
        print("\n")
        
        if self.successes:
            print(f"✓ SUCCESSES ({len(self.successes)}):")
            for success in self.successes:
                print(f"  {success}")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.issues:
            print(f"\n✗ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
        
        print("\n" + "=" * 60)
        print(f"Summary: {len(self.successes)} successes, {len(self.warnings)} warnings, {len(self.issues)} issues")
        
        if not self.issues:
            print("\n✓ All systems operational!")
        else:
            print("\n✗ Issues detected. Please address the problems listed above.")
        
        # Save report to file
        report_file = self.directory / f"troubleshooting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write("TROUBLESHOOTING REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory: {self.directory}\n\n")
            
            if self.successes:
                f.write(f"SUCCESSES ({len(self.successes)}):\n")
                for success in self.successes:
                    f.write(f"  {success}\n")
            
            if self.warnings:
                f.write(f"\nWARNINGS ({len(self.warnings)}):\n")
                for warning in self.warnings:
                    f.write(f"  {warning}\n")
            
            if self.issues:
                f.write(f"\nISSUES ({len(self.issues)}):\n")
                for issue in self.issues:
                    f.write(f"  {issue}\n")
        
        print(f"\nReport saved to: {report_file}")

def main():
    """Run complete troubleshooting"""
    troubleshooter = Troubleshooter()
    
    print("Starting comprehensive troubleshooting for iteration2-basic-stats")
    print("=" * 60)
    
    # Run all checks
    troubleshooter.check_dependencies()
    troubleshooter.check_scripts()
    troubleshooter.check_permissions()
    troubleshooter.check_connector_system()
    troubleshooter.check_script_execution()
    troubleshooter.check_hivemind_connectivity()
    
    # Generate report
    troubleshooter.generate_report()

if __name__ == "__main__":
    main()