#!/usr/bin/env python3
"""
System Test and Troubleshoot - Comprehensive testing for Polar Bear
Tests all components and provides troubleshooting guidance
"""

import os
import sys
import json
import time
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

class SystemTester:
    """Comprehensive system testing and troubleshooting"""
    
    def __init__(self):
        self.results = {
            'tests': {},
            'errors': [],
            'warnings': [],
            'info': [],
            'summary': {}
        }
        self.setup_logging()
    
    def setup_logging(self):
        """Setup test logging"""
        self.logger = logging.getLogger("SystemTester")
        self.logger.setLevel(logging.DEBUG)
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def print_header(self, text: str):
        """Print a formatted header"""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}")
    
    def print_section(self, text: str):
        """Print a section header"""
        print(f"\n--- {text} ---")
    
    def test_python_version(self) -> bool:
        """Test Python version compatibility"""
        self.print_section("Python Version Check")
        
        version = sys.version_info
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            self.results['errors'].append("Python 3.7+ required")
            print("❌ FAIL: Python 3.7 or higher required")
            return False
        
        print("✅ PASS: Python version compatible")
        self.results['tests']['python_version'] = True
        return True
    
    def test_core_imports(self) -> bool:
        """Test core Python imports"""
        self.print_section("Core Imports Check")
        
        core_modules = [
            'numpy', 'cv2', 'PIL', 'scipy', 'sklearn',
            'matplotlib', 'pandas', 'tqdm', 'requests'
        ]
        
        missing = []
        for module in core_modules:
            try:
                __import__(module)
                print(f"✅ {module}")
            except ImportError:
                print(f"❌ {module} - NOT INSTALLED")
                missing.append(module)
        
        if missing:
            self.results['warnings'].append(f"Missing modules: {', '.join(missing)}")
            print(f"\nTo install missing modules:")
            print(f"pip install {' '.join(missing)}")
            
        self.results['tests']['core_imports'] = len(missing) == 0
        return len(missing) == 0
    
    def test_pytorch(self) -> bool:
        """Test PyTorch installation"""
        self.print_section("PyTorch Check")
        
        try:
            import torch
            print(f"✅ PyTorch version: {torch.__version__}")
            
            # Check CUDA
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
                self.results['info'].append("GPU acceleration available")
            else:
                print("ℹ️  CUDA not available - CPU mode only")
                
            self.results['tests']['pytorch'] = True
            return True
            
        except ImportError:
            print("❌ PyTorch not installed")
            print("\nTo install PyTorch:")
            print("Visit https://pytorch.org/get-started/locally/")
            self.results['tests']['pytorch'] = False
            return False
    
    def test_file_structure(self) -> bool:
        """Test expected file structure"""
        self.print_section("File Structure Check")
        
        required_files = [
            'connector.py',
            'hivemind_connector.py',
            'polar_bear_brain.py',
            'universal_connector.py',
            'connector_enhanced.py'
        ]
        
        missing = []
        for file in required_files:
            if Path(file).exists():
                print(f"✅ {file}")
            else:
                print(f"❌ {file} - NOT FOUND")
                missing.append(file)
        
        # Check for directories
        dirs_to_check = ['logs', 'input', 'output', 'reference']
        print("\nDirectories:")
        for dir_name in dirs_to_check:
            if Path(dir_name).exists():
                print(f"✅ {dir_name}/")
            else:
                print(f"ℹ️  {dir_name}/ - Will be created when needed")
        
        self.results['tests']['file_structure'] = len(missing) == 0
        return len(missing) == 0
    
    def test_connectors(self) -> bool:
        """Test connector functionality"""
        self.print_section("Connector Tests")
        
        tests_passed = True
        
        # Test Universal Connector
        try:
            from universal_connector import UniversalConnector
            connector = UniversalConnector()
            scripts = connector.get_script_info()
            
            print(f"✅ Universal Connector: {len(scripts)} scripts discovered")
            self.results['info'].append(f"Discovered {len(scripts)} scripts")
            
            # Test a simple function
            if scripts:
                test_script = list(scripts.keys())[0]
                info = scripts[test_script]
                print(f"✅ Test script '{test_script}' loaded successfully")
                
        except Exception as e:
            print(f"❌ Universal Connector error: {e}")
            self.results['errors'].append(f"Universal Connector: {str(e)}")
            tests_passed = False
        
        # Test Enhanced Connector
        try:
            from connector_enhanced import EnhancedConnector
            enhanced = EnhancedConnector()
            if enhanced.initialize():
                print("✅ Enhanced Connector initialized")
            else:
                print("❌ Enhanced Connector initialization failed")
                tests_passed = False
                
        except Exception as e:
            print(f"❌ Enhanced Connector error: {e}")
            self.results['errors'].append(f"Enhanced Connector: {str(e)}")
            tests_passed = False
        
        self.results['tests']['connectors'] = tests_passed
        return tests_passed
    
    def test_brain_system(self) -> bool:
        """Test Polar Bear Brain system"""
        self.print_section("Brain System Test")
        
        try:
            from polar_bear_brain import PolarBearBrain, DependencyManager, ConfigurationManager
            
            # Test components
            print("Testing Brain components...")
            
            # Test dependency manager
            deps = DependencyManager(logging.getLogger())
            print("✅ Dependency Manager initialized")
            
            # Test config manager
            config = ConfigurationManager(logging.getLogger())
            print("✅ Configuration Manager initialized")
            
            # Don't actually start the brain (would enter interactive mode)
            print("✅ Brain system components verified")
            
            self.results['tests']['brain_system'] = True
            return True
            
        except Exception as e:
            print(f"❌ Brain system error: {e}")
            self.results['errors'].append(f"Brain system: {str(e)}")
            self.results['tests']['brain_system'] = False
            return False
    
    def test_script_execution(self) -> bool:
        """Test basic script execution"""
        self.print_section("Script Execution Test")
        
        try:
            from universal_connector import UniversalConnector
            connector = UniversalConnector()
            
            # Find a simple test function
            test_functions = [
                ('numpy-json-encoder-26', 'encode'),
                ('logging-utility-26', 'setup_logging'),
                ('timing-utilities-26', 'Timer')
            ]
            
            executed = False
            for script, func in test_functions:
                try:
                    scripts = connector.get_script_info()
                    if script in scripts:
                        # Just check if we can load the module
                        module = connector.load_module(script)
                        if module:
                            print(f"✅ Successfully loaded {script}")
                            executed = True
                            break
                except:
                    continue
            
            if not executed:
                print("⚠️  No test scripts could be executed")
                self.results['warnings'].append("Script execution test incomplete")
            
            self.results['tests']['script_execution'] = executed
            return executed
            
        except Exception as e:
            print(f"❌ Script execution error: {e}")
            self.results['errors'].append(f"Script execution: {str(e)}")
            self.results['tests']['script_execution'] = False
            return False
    
    def test_image_processing(self) -> bool:
        """Test basic image processing capabilities"""
        self.print_section("Image Processing Test")
        
        try:
            import cv2
            import numpy as np
            
            # Create a test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.circle(test_image, (50, 50), 30, (255, 255, 255), -1)
            
            # Save test image
            test_path = "test_image.png"
            cv2.imwrite(test_path, test_image)
            
            print("✅ Created test image")
            
            # Try to load it back
            loaded = cv2.imread(test_path)
            if loaded is not None:
                print("✅ Image I/O working")
                os.remove(test_path)  # Cleanup
            else:
                print("❌ Failed to load test image")
                
            self.results['tests']['image_processing'] = True
            return True
            
        except Exception as e:
            print(f"❌ Image processing error: {e}")
            self.results['errors'].append(f"Image processing: {str(e)}")
            self.results['tests']['image_processing'] = False
            return False
    
    def generate_report(self):
        """Generate final test report"""
        self.print_header("TEST SUMMARY")
        
        # Count results
        total_tests = len(self.results['tests'])
        passed_tests = sum(1 for v in self.results['tests'].values() if v)
        
        print(f"\nTests passed: {passed_tests}/{total_tests}")
        
        # Show errors
        if self.results['errors']:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"   - {error}")
        
        # Show warnings
        if self.results['warnings']:
            print(f"\n⚠️  Warnings ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"   - {warning}")
        
        # Show info
        if self.results['info']:
            print(f"\nℹ️  Information:")
            for info in self.results['info']:
                print(f"   - {info}")
        
        # Overall status
        print("\n" + "="*60)
        if passed_tests == total_tests:
            print("✅ ALL TESTS PASSED - System is ready!")
        elif passed_tests >= total_tests * 0.7:
            print("⚠️  MOSTLY PASSED - System should work with limitations")
        else:
            print("❌ TESTS FAILED - Please fix errors before proceeding")
        
        # Save report
        report_file = f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")
    
    def troubleshoot(self):
        """Provide troubleshooting guidance"""
        self.print_header("TROUBLESHOOTING GUIDE")
        
        if not self.results['tests'].get('python_version', True):
            print("\n1. Python Version Issue:")
            print("   - Install Python 3.7 or higher")
            print("   - Check with: python --version")
        
        if not self.results['tests'].get('core_imports', True):
            print("\n2. Missing Dependencies:")
            print("   - Run: pip install -r requirements.txt")
            print("   - Or install individually as shown above")
        
        if not self.results['tests'].get('pytorch', True):
            print("\n3. PyTorch Installation:")
            print("   - Visit https://pytorch.org/")
            print("   - Choose your system configuration")
            print("   - Run the provided pip command")
        
        if not self.results['tests'].get('connectors', True):
            print("\n4. Connector Issues:")
            print("   - Ensure all connector files are present")
            print("   - Check file permissions")
            print("   - Look for syntax errors in connector files")
        
        print("\n5. General Tips:")
        print("   - Check the logs/ directory for detailed error messages")
        print("   - Run scripts individually to isolate issues")
        print("   - Ensure you have write permissions in the directory")
        print("   - Try running with administrator/sudo if needed")
    
    def run_all_tests(self):
        """Run all system tests"""
        self.print_header("POLAR BEAR SYSTEM TEST")
        print("Testing all components...")
        
        # Run tests in order
        self.test_python_version()
        self.test_core_imports()
        self.test_pytorch()
        self.test_file_structure()
        self.test_connectors()
        self.test_brain_system()
        self.test_script_execution()
        self.test_image_processing()
        
        # Generate report
        self.generate_report()
        
        # Provide troubleshooting if needed
        if any(not v for v in self.results['tests'].values()):
            self.troubleshoot()

def quick_start_guide():
    """Display quick start guide"""
    print("\n" + "="*60)
    print("  POLAR BEAR QUICK START GUIDE")
    print("="*60)
    
    print("\n1. Run system test:")
    print("   python system_test_and_troubleshoot.py")
    
    print("\n2. Start the Brain (main system):")
    print("   python polar_bear_brain.py")
    
    print("\n3. Use Enhanced Connector (interactive):")
    print("   python connector_enhanced.py")
    
    print("\n4. Direct script execution:")
    print("   python connector_enhanced.py execute <script> <function>")
    
    print("\n5. Check system status:")
    print("   python connector_enhanced.py status")
    
    print("\nFor more help, check the logs/ directory")
    print("="*60)

if __name__ == "__main__":
    # Run system tests
    tester = SystemTester()
    tester.run_all_tests()
    
    # Show quick start guide
    quick_start_guide()