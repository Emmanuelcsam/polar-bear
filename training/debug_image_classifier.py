#!/usr/bin/env python3
"""
Debug script for image-classifier.py
Identifies and fixes common issues
"""

import os
import sys
import traceback
import subprocess
import importlib
import json
import numpy as np
from pathlib import Path
import logging

class ImageClassifierDebugger:
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("\n🔍 Checking dependencies...")
        
        required_packages = {
            "PIL": "Pillow",
            "numpy": "numpy",
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "imagehash": "imagehash",
            "tqdm": "tqdm",
            "scipy": "scipy",
            "matplotlib": "matplotlib"
        }
        
        missing_packages = []
        
        for import_name, package_name in required_packages.items():
            try:
                importlib.import_module(import_name)
                print(f"✅ {package_name} is installed")
            except ImportError:
                print(f"❌ {package_name} is NOT installed")
                missing_packages.append(package_name)
                self.issues_found.append(f"Missing package: {package_name}")
                
        if missing_packages:
            print(f"\n⚠️  Installing missing packages: {', '.join(missing_packages)}")
            for package in missing_packages:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "--upgrade", "--no-cache-dir", package
                    ])
                    self.fixes_applied.append(f"Installed {package}")
                except Exception as e:
                    print(f"Failed to install {package}: {e}")
                    
        return len(missing_packages) == 0
        
    def check_file_structure(self):
        """Check if the main script exists and is valid"""
        print("\n🔍 Checking file structure...")
        
        if not os.path.exists("image-classifier.py"):
            print("❌ image-classifier.py not found!")
            self.issues_found.append("Main script not found")
            return False
            
        # Check if file is readable and valid Python
        try:
            with open("image-classifier.py", 'r') as f:
                content = f.read()
                
            # Try to compile it
            compile(content, "image-classifier.py", "exec")
            print("✅ image-classifier.py is valid Python code")
            
            # Check for required classes
            required_classes = ["KnowledgeBank", "ImageClassifierGUI", "UltimateImageClassifier"]
            for class_name in required_classes:
                if f"class {class_name}" in content:
                    print(f"✅ Found class: {class_name}")
                else:
                    print(f"❌ Missing class: {class_name}")
                    self.issues_found.append(f"Missing class: {class_name}")
                    
            return True
            
        except SyntaxError as e:
            print(f"❌ Syntax error in image-classifier.py: {e}")
            self.issues_found.append(f"Syntax error: {e}")
            return False
            
    def test_imports(self):
        """Test if the script can be imported without errors"""
        print("\n🔍 Testing imports...")
        
        try:
            # Add current directory to path
            sys.path.insert(0, os.getcwd())
            
            # Try to import
            import image_classifier
            
            print("✅ Successfully imported image_classifier module")
            
            # Check for required functions/classes
            required_attrs = [
                "setup_logging",
                "KnowledgeBank", 
                "ImageClassifierGUI",
                "UltimateImageClassifier",
                "main"
            ]
            
            for attr in required_attrs:
                if hasattr(image_classifier, attr):
                    print(f"✅ Found: {attr}")
                else:
                    print(f"❌ Missing: {attr}")
                    self.issues_found.append(f"Missing attribute: {attr}")
                    
            return True
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            self.issues_found.append(f"Import error: {e}")
            traceback.print_exc()
            return False
            
    def test_basic_functionality(self):
        """Test basic functionality of the classifier"""
        print("\n🔍 Testing basic functionality...")
        
        try:
            import image_classifier
            
            # Test logging setup
            print("Testing logging setup...")
            logger = image_classifier.setup_logging()
            if logger:
                print("✅ Logging setup successful")
            else:
                print("❌ Logging setup failed")
                self.issues_found.append("Logging setup failed")
                
            # Test KnowledgeBank
            print("\nTesting KnowledgeBank...")
            kb = image_classifier.KnowledgeBank("test_kb.pkl")
            kb.add_custom_keyword("test")
            kb.add_image("test_hash", np.array([1, 2, 3]), ["test_class"], {})
            stats = kb.get_statistics()
            
            if stats['total_images'] == 1:
                print("✅ KnowledgeBank working correctly")
            else:
                print("❌ KnowledgeBank not storing data correctly")
                self.issues_found.append("KnowledgeBank storage issue")
                
            # Test classifier initialization
            print("\nTesting UltimateImageClassifier...")
            classifier = image_classifier.UltimateImageClassifier()
            
            # Test parsing
            result = classifier.parse_classification("50-fc-core-clean")
            expected_keys = ["core_diameter", "connector_type", "region", "condition"]
            
            if all(key in result for key in expected_keys):
                print("✅ Classification parsing working")
            else:
                print("❌ Classification parsing not working correctly")
                self.issues_found.append("Classification parsing issue")
                
            # Clean up
            if os.path.exists("test_kb.pkl"):
                os.remove("test_kb.pkl")
            if os.path.exists("classifier_config.json"):
                os.remove("classifier_config.json")
                
            return True
            
        except Exception as e:
            print(f"❌ Functionality test failed: {e}")
            self.issues_found.append(f"Functionality error: {e}")
            traceback.print_exc()
            return False
            
    def check_gui_availability(self):
        """Check if GUI (tkinter) is available"""
        print("\n🔍 Checking GUI availability...")
        
        try:
            import tkinter
            print("✅ tkinter (GUI) is available")
            
            # Try to create a test window
            root = tkinter.Tk()
            root.withdraw()  # Hide it immediately
            root.destroy()
            print("✅ GUI can be created")
            
            return True
            
        except Exception as e:
            print(f"⚠️  GUI not available: {e}")
            print("   Manual mode will fall back to console interface")
            self.issues_found.append("GUI not available (non-critical)")
            return False
            
    def check_permissions(self):
        """Check file system permissions"""
        print("\n🔍 Checking permissions...")
        
        # Check if we can create directories
        test_dir = "test_permissions_dir"
        try:
            os.makedirs(test_dir, exist_ok=True)
            print("✅ Can create directories")
            
            # Check if we can create files
            test_file = os.path.join(test_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            print("✅ Can create files")
            
            # Check if we can rename files
            new_name = os.path.join(test_dir, "renamed.txt")
            os.rename(test_file, new_name)
            print("✅ Can rename files")
            
            # Clean up
            os.remove(new_name)
            os.rmdir(test_dir)
            
            return True
            
        except Exception as e:
            print(f"❌ Permission error: {e}")
            self.issues_found.append(f"Permission error: {e}")
            return False
            
    def suggest_fixes(self):
        """Suggest fixes for found issues"""
        print("\n💡 Suggested fixes:")
        
        if not self.issues_found:
            print("✅ No issues found! The classifier should work correctly.")
            return
            
        for issue in self.issues_found:
            print(f"\n⚠️  Issue: {issue}")
            
            if "Missing package" in issue:
                package = issue.split(": ")[1]
                print(f"   Fix: Run 'pip install {package}'")
                
            elif "GUI not available" in issue:
                print("   Fix: Install tkinter (python3-tk on Ubuntu/Debian)")
                print("        Or use console mode (will fall back automatically)")
                
            elif "Permission error" in issue:
                print("   Fix: Check file system permissions")
                print("        Ensure you have write access to the directory")
                
            elif "Syntax error" in issue:
                print("   Fix: Check the image-classifier.py file for syntax errors")
                print("        The error message above shows the location")
                
            elif "Import error" in issue:
                print("   Fix: Ensure all dependencies are installed")
                print("        Check for circular imports or missing modules")
                
    def run_diagnostic(self):
        """Run complete diagnostic"""
        print("="*60)
        print("IMAGE CLASSIFIER DIAGNOSTIC TOOL")
        print("="*60)
        
        # Run all checks
        checks = [
            ("Dependencies", self.check_dependencies),
            ("File Structure", self.check_file_structure),
            ("Imports", self.test_imports),
            ("Basic Functionality", self.test_basic_functionality),
            ("GUI Availability", self.check_gui_availability),
            ("Permissions", self.check_permissions)
        ]
        
        all_passed = True
        
        for name, check_func in checks:
            try:
                if not check_func():
                    all_passed = False
            except Exception as e:
                print(f"❌ {name} check failed with error: {e}")
                self.issues_found.append(f"{name} check error: {e}")
                all_passed = False
                
        # Summary
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        
        if all_passed and not self.issues_found:
            print("\n✅ All checks passed! The image classifier should work correctly.")
            print("\nYou can now run:")
            print("  python image-classifier.py --help")
            print("\nOr for a quick test:")
            print("  python image-classifier.py --mode auto")
        else:
            print(f"\n❌ Found {len(self.issues_found)} issue(s)")
            self.suggest_fixes()
            
        if self.fixes_applied:
            print(f"\n✅ Applied {len(self.fixes_applied)} fix(es):")
            for fix in self.fixes_applied:
                print(f"   - {fix}")
                
        return all_passed


def main():
    """Run the debugger"""
    debugger = ImageClassifierDebugger()
    
    # Check if we're in the right directory
    if not os.path.exists("image-classifier.py"):
        print("❌ Error: image-classifier.py not found in current directory!")
        print("   Make sure you're running this from the training directory")
        return
        
    success = debugger.run_diagnostic()
    
    if success:
        print("\n🎉 Ready to run the image classifier!")
    else:
        print("\n⚠️  Please fix the issues above before running the classifier")
        

if __name__ == "__main__":
    main()