#!/usr/bin/env python3
"""
Comprehensive Troubleshooting Script for PyTorch Production Module
Checks all scripts, dependencies, and integration points
"""

import os
import sys
import subprocess
import importlib.util
import socket
import json
import time
from pathlib import Path
from datetime import datetime

class TroubleshootingTool:
    """Comprehensive troubleshooting for all scripts"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        self.script_dir = Path(__file__).parent
        
    def check_python_version(self):
        """Check Python version compatibility"""
        print("Checking Python version...")
        version = sys.version_info
        if version.major == 3 and version.minor >= 7:
            self.info.append(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
        else:
            self.issues.append(f"Python version {version.major}.{version.minor} may not be compatible")
            
    def check_dependencies(self):
        """Check required dependencies"""
        print("\nChecking dependencies...")
        required_packages = {
            'torch': 'PyTorch',
            'numpy': 'NumPy',
            'cv2': 'OpenCV',
        }
        
        for module, name in required_packages.items():
            try:
                importlib.import_module(module)
                self.info.append(f"{name} ({module}) installed ✓")
            except ImportError:
                self.issues.append(f"{name} ({module}) not installed - run: pip install {module}")
                
    def check_script_files(self):
        """Check all script files exist and are readable"""
        print("\nChecking script files...")
        expected_scripts = [
            'connector.py',
            'hivemind_connector.py',
            'preprocess.py',
            'load.py',
            'train.py',
            'final.py',
            'script_interface.py',
            'script_wrappers.py',
            'test_integration.py',
            'troubleshoot_all.py'
        ]
        
        for script in expected_scripts:
            script_path = self.script_dir / script
            if script_path.exists():
                if os.access(script_path, os.R_OK):
                    self.info.append(f"{script} exists and readable ✓")
                else:
                    self.issues.append(f"{script} exists but not readable")
            else:
                self.issues.append(f"{script} missing")
                
    def test_script_syntax(self):
        """Test Python syntax of all scripts"""
        print("\nChecking script syntax...")
        scripts = list(self.script_dir.glob("*.py"))
        
        for script in scripts:
            try:
                result = subprocess.run([sys.executable, '-m', 'py_compile', str(script)],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.info.append(f"{script.name} syntax valid ✓")
                else:
                    self.issues.append(f"{script.name} syntax error: {result.stderr}")
            except Exception as e:
                self.issues.append(f"{script.name} syntax check failed: {str(e)}")
                
    def test_script_imports(self):
        """Test if scripts can import their dependencies"""
        print("\nTesting script imports...")
        
        # Test script interface
        try:
            from script_interface import interface, handle_connector_command
            self.info.append("script_interface imports successfully ✓")
        except Exception as e:
            self.issues.append(f"script_interface import failed: {str(e)}")
            
        # Test script wrappers
        try:
            from script_wrappers import wrappers
            self.info.append("script_wrappers imports successfully ✓")
        except Exception as e:
            self.issues.append(f"script_wrappers import failed: {str(e)}")
            
    def test_connector_ports(self):
        """Test if connector ports are available"""
        print("\nChecking connector ports...")
        ports = {
            10050: 'hivemind_connector',
            10051: 'connector'
        }
        
        for port, name in ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('localhost', port))
                sock.close()
                self.info.append(f"Port {port} ({name}) available ✓")
            except OSError:
                self.warnings.append(f"Port {port} ({name}) in use - may be running")
                
    def test_script_execution(self):
        """Test basic execution of each script"""
        print("\nTesting script execution...")
        
        # Test preprocess.py
        print("  Testing preprocess.py...")
        try:
            result = subprocess.run([sys.executable, 'preprocess.py'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.info.append("preprocess.py executes successfully ✓")
            else:
                self.warnings.append(f"preprocess.py execution warning: {result.stderr}")
        except Exception as e:
            self.issues.append(f"preprocess.py execution failed: {str(e)}")
            
    def test_integration_capabilities(self):
        """Test integration between scripts and connectors"""
        print("\nTesting integration capabilities...")
        
        try:
            from script_interface import interface
            scripts = interface.list_scripts()
            
            for script in scripts:
                deps = interface.check_dependencies(script['name'])
                if deps['ready']:
                    self.info.append(f"{script['name']} ready for integration ✓")
                else:
                    self.warnings.append(f"{script['name']} missing dependencies: {deps['missing']}")
                    
        except Exception as e:
            self.issues.append(f"Integration test failed: {str(e)}")
            
    def check_file_permissions(self):
        """Check file permissions"""
        print("\nChecking file permissions...")
        
        # Check if we can create files in directory
        test_file = self.script_dir / '.test_write'
        try:
            test_file.write_text('test')
            test_file.unlink()
            self.info.append("Directory writable ✓")
        except Exception as e:
            self.issues.append(f"Cannot write to directory: {str(e)}")
            
    def generate_report(self):
        """Generate troubleshooting report"""
        print("\n" + "="*60)
        print("TROUBLESHOOTING REPORT")
        print("="*60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Directory: {self.script_dir}")
        
        if self.issues:
            print(f"\n🔴 ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  - {issue}")
        else:
            print("\n✅ No critical issues found!")
            
        if self.warnings:
            print(f"\n🟡 WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
                
        print(f"\n🟢 OK ({len(self.info)}):")
        for info in self.info[:5]:  # Show first 5
            print(f"  - {info}")
        if len(self.info) > 5:
            print(f"  ... and {len(self.info) - 5} more")
            
        print("\n" + "="*60)
        
        # Save report
        report_file = self.script_dir / f"troubleshooting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write("TROUBLESHOOTING REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory: {self.script_dir}\n\n")
            
            if self.issues:
                f.write(f"ISSUES ({len(self.issues)}):\n")
                for issue in self.issues:
                    f.write(f"  - {issue}\n")
            
            if self.warnings:
                f.write(f"\nWARNINGS ({len(self.warnings)}):\n")
                for warning in self.warnings:
                    f.write(f"  - {warning}\n")
                    
            f.write(f"\nOK ({len(self.info)}):\n")
            for info in self.info:
                f.write(f"  - {info}\n")
                
        print(f"\nReport saved to: {report_file}")
        
        return len(self.issues) == 0
        
    def run_diagnostics(self):
        """Run all diagnostic checks"""
        print("Starting PyTorch Production Module Diagnostics...")
        print("="*60)
        
        self.check_python_version()
        self.check_dependencies()
        self.check_script_files()
        self.test_script_syntax()
        self.test_script_imports()
        self.test_connector_ports()
        self.test_script_execution()
        self.test_integration_capabilities()
        self.check_file_permissions()
        
        success = self.generate_report()
        
        if success:
            print("\n✅ System appears to be functioning correctly!")
            print("You can now:")
            print("  1. Run scripts independently: python <script_name>.py")
            print("  2. Start connector server: python connector.py --server")
            print("  3. Run integration tests: python test_integration.py")
        else:
            print("\n❌ Issues detected - please resolve them before proceeding")
            
        return success


if __name__ == "__main__":
    tool = TroubleshootingTool()
    success = tool.run_diagnostics()
    sys.exit(0 if success else 1)