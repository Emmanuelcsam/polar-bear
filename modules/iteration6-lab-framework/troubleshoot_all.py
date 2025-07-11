#!/usr/bin/env python3
"""
Comprehensive troubleshooting script for the lab framework
Diagnoses and fixes common issues with scripts and connectors
"""

import os
import sys
import subprocess
import socket
import json
import time
import importlib
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


class LabFrameworkTroubleshooter:
    """Troubleshooter for the lab framework"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.issues = []
        self.fixes_applied = []
        self.test_results = {}
        
    def log(self, message, level="INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def check_python_version(self):
        """Check Python version compatibility"""
        self.log("Checking Python version...")
        version = sys.version_info
        
        if version.major < 3 or (version.major == 3 and version.minor < 6):
            self.issues.append(f"Python version {version.major}.{version.minor} is too old. Need 3.6+")
            return False
            
        self.log(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        self.log("Checking dependencies...")
        
        required_packages = {
            'numpy': 'numpy',
            'cv2': 'opencv-python',
            'torch': 'torch',
            'sklearn': 'scikit-learn'
        }
        
        missing = []
        for module, package in required_packages.items():
            try:
                importlib.import_module(module)
                self.log(f"✓ {module} is installed")
            except ImportError:
                missing.append(package)
                self.issues.append(f"Missing dependency: {package}")
                
        if missing:
            self.log(f"✗ Missing packages: {', '.join(missing)}", "ERROR")
            self.log("Run: pip install " + " ".join(missing))
            return False
            
        return True
        
    def check_directory_structure(self):
        """Check if all required directories exist"""
        self.log("Checking directory structure...")
        
        required_dirs = ['core', 'modules', 'data', 'tests']
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.base_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                self.issues.append(f"Missing directory: {dir_name}")
            else:
                self.log(f"✓ Directory '{dir_name}' exists")
                
        if missing_dirs:
            # Create missing directories
            for dir_name in missing_dirs:
                dir_path = self.base_path / dir_name
                dir_path.mkdir(exist_ok=True)
                self.fixes_applied.append(f"Created directory: {dir_name}")
                self.log(f"✓ Created missing directory: {dir_name}")
                
        return True
        
    def check_core_modules(self):
        """Check if core modules are working"""
        self.log("Checking core modules...")
        
        core_modules = ['config', 'logger', 'datastore']
        
        for module_name in core_modules:
            try:
                module = importlib.import_module(f'core.{module_name}')
                self.log(f"✓ core.{module_name} imports successfully")
                
                # Test basic functionality
                if module_name == 'datastore':
                    from core.datastore import put, get
                    put("test_key", "test_value")
                    if get("test_key") == "test_value":
                        self.log(f"✓ core.datastore functionality verified")
                    else:
                        self.issues.append(f"core.datastore not functioning properly")
                        
            except Exception as e:
                self.issues.append(f"Error with core.{module_name}: {str(e)}")
                self.log(f"✗ core.{module_name} failed: {e}", "ERROR")
                return False
                
        return True
        
    def check_module_scripts(self):
        """Check if module scripts are valid"""
        self.log("Checking module scripts...")
        
        modules_dir = self.base_path / 'modules'
        if not modules_dir.exists():
            self.issues.append("Modules directory not found")
            return False
            
        module_files = list(modules_dir.glob("*.py"))
        self.log(f"Found {len(module_files)} module files")
        
        for module_file in module_files:
            if module_file.name == "__init__.py":
                continue
                
            module_name = module_file.stem
            try:
                # Try to import the module
                spec = importlib.util.spec_from_file_location(module_name, str(module_file))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.log(f"✓ Module '{module_name}' loads successfully")
                
            except Exception as e:
                self.issues.append(f"Module '{module_name}' failed to load: {str(e)}")
                self.log(f"✗ Module '{module_name}' failed: {e}", "ERROR")
                
        return len(self.issues) == 0
        
    def check_hivemind_connector(self):
        """Check if hivemind connector is running"""
        self.log("Checking hivemind connector...")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(('localhost', 10006))
            
            # Send status command
            status_msg = {'command': 'status'}
            sock.send(json.dumps(status_msg).encode())
            response = json.loads(sock.recv(4096).decode())
            sock.close()
            
            self.log(f"✓ Hivemind connector is running: {response}")
            return True
            
        except Exception as e:
            self.log(f"✗ Hivemind connector not accessible: {e}", "WARNING")
            
            # Try to start it
            self.log("Attempting to start hivemind connector...")
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(self.base_path / "hivemind_connector.py")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(3)  # Give it time to start
                
                # Check again
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    sock.connect(('localhost', 10006))
                    sock.close()
                    self.log("✓ Hivemind connector started successfully")
                    self.fixes_applied.append("Started hivemind connector")
                    return True
                except:
                    self.log("✗ Failed to start hivemind connector", "ERROR")
                    return False
                    
            except Exception as e:
                self.log(f"✗ Could not start hivemind connector: {e}", "ERROR")
                return False
                
    def test_basic_functionality(self):
        """Test basic script functionality"""
        self.log("Testing basic functionality...")
        
        # Test 1: Random pixel generation
        try:
            from modules import random_pixel
            img = random_pixel.gen()
            self.test_results['random_pixel_gen'] = True
            self.log("✓ Random pixel generation works")
        except Exception as e:
            self.test_results['random_pixel_gen'] = False
            self.issues.append(f"Random pixel generation failed: {str(e)}")
            self.log(f"✗ Random pixel generation failed: {e}", "ERROR")
            
        # Test 2: Basic script execution
        try:
            result = subprocess.run(
                [sys.executable, str(self.base_path / "test_basic.py")],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.test_results['test_basic'] = True
                self.log("✓ Basic test script passes")
            else:
                self.test_results['test_basic'] = False
                self.issues.append(f"Basic test script failed: {result.stderr}")
                self.log(f"✗ Basic test script failed", "ERROR")
        except Exception as e:
            self.test_results['test_basic'] = False
            self.issues.append(f"Could not run basic test: {str(e)}")
            
        return all(self.test_results.values())
        
    def test_connector_integration(self):
        """Test connector integration"""
        self.log("Testing connector integration...")
        
        try:
            from connector import LabFrameworkConnector
            
            connector = LabFrameworkConnector()
            connector.discover_modules()
            
            if len(connector.modules) > 0:
                self.test_results['connector_discovery'] = True
                self.log(f"✓ Connector discovered {len(connector.modules)} modules")
            else:
                self.test_results['connector_discovery'] = False
                self.issues.append("Connector failed to discover modules")
                
        except Exception as e:
            self.test_results['connector_discovery'] = False
            self.issues.append(f"Connector integration failed: {str(e)}")
            self.log(f"✗ Connector integration failed: {e}", "ERROR")
            
        return self.test_results.get('connector_discovery', False)
        
    def fix_common_issues(self):
        """Apply fixes for common issues"""
        self.log("Applying automatic fixes...")
        
        # Fix 1: Ensure __init__.py files exist
        for dir_name in ['core', 'modules', 'tests']:
            init_file = self.base_path / dir_name / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                self.fixes_applied.append(f"Created {dir_name}/__init__.py")
                self.log(f"✓ Created missing {dir_name}/__init__.py")
                
        # Fix 2: Create data directory if missing
        data_dir = self.base_path / "data"
        if not data_dir.exists():
            data_dir.mkdir()
            self.fixes_applied.append("Created data directory")
            self.log("✓ Created data directory")
            
        # Fix 3: Ensure proper permissions
        try:
            for py_file in self.base_path.rglob("*.py"):
                if not os.access(py_file, os.R_OK):
                    py_file.chmod(0o644)
                    self.fixes_applied.append(f"Fixed permissions for {py_file.name}")
        except Exception as e:
            self.log(f"Could not fix permissions: {e}", "WARNING")
            
    def generate_report(self):
        """Generate a troubleshooting report"""
        report_lines = [
            "LAB FRAMEWORK TROUBLESHOOTING REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Path: {self.base_path}",
            "",
            "SYSTEM INFORMATION:",
            f"Python Version: {sys.version}",
            f"Platform: {sys.platform}",
            "",
            "TEST RESULTS:",
        ]
        
        for test_name, result in self.test_results.items():
            status = "PASSED" if result else "FAILED"
            report_lines.append(f"  {test_name}: {status}")
            
        if self.issues:
            report_lines.extend([
                "",
                "ISSUES FOUND:",
            ])
            for issue in self.issues:
                report_lines.append(f"  - {issue}")
                
        if self.fixes_applied:
            report_lines.extend([
                "",
                "FIXES APPLIED:",
            ])
            for fix in self.fixes_applied:
                report_lines.append(f"  - {fix}")
                
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
        ])
        
        if self.issues:
            if any("dependency" in issue.lower() for issue in self.issues):
                report_lines.append("  - Install missing dependencies with pip")
            if any("connector" in issue.lower() for issue in self.issues):
                report_lines.append("  - Ensure hivemind_connector.py is running")
            if any("module" in issue.lower() for issue in self.issues):
                report_lines.append("  - Check module syntax and imports")
        else:
            report_lines.append("  - No issues found. System is healthy!")
            
        report = "\n".join(report_lines)
        
        # Save report
        report_file = self.base_path / f"troubleshooting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
            
        self.log(f"Report saved to: {report_file}")
        return report
        
    def run_full_diagnostic(self):
        """Run complete diagnostic and troubleshooting"""
        self.log("Starting full system diagnostic...")
        self.log("=" * 60)
        
        # Run all checks
        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Directory Structure", self.check_directory_structure),
            ("Core Modules", self.check_core_modules),
            ("Module Scripts", self.check_module_scripts),
            ("Hivemind Connector", self.check_hivemind_connector),
            ("Basic Functionality", self.test_basic_functionality),
            ("Connector Integration", self.test_connector_integration),
        ]
        
        for check_name, check_func in checks:
            self.log(f"\nRunning: {check_name}")
            try:
                check_func()
            except Exception as e:
                self.log(f"Check failed with exception: {e}", "ERROR")
                self.issues.append(f"{check_name} check failed: {str(e)}")
                
        # Apply fixes
        if self.issues:
            self.log("\n" + "=" * 60)
            self.fix_common_issues()
            
        # Generate report
        self.log("\n" + "=" * 60)
        report = self.generate_report()
        
        # Print summary
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        if not self.issues:
            print("✅ All systems operational! No issues found.")
        else:
            print(f"⚠️ Found {len(self.issues)} issues")
            print(f"✓ Applied {len(self.fixes_applied)} automatic fixes")
            print("\nRemaining issues require manual intervention:")
            for issue in self.issues:
                if not any(fix in issue for fix in self.fixes_applied):
                    print(f"  - {issue}")
                    
        print(f"\nFull report saved to troubleshooting_report_*.txt")


def main():
    """Main entry point"""
    troubleshooter = LabFrameworkTroubleshooter()
    troubleshooter.run_full_diagnostic()


if __name__ == "__main__":
    main()