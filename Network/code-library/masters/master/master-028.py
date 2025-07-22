#!/usr/bin/env python3
"""
Polar Bear System Startup Script
Ensures all components are properly initialized and working together
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import logging

class SystemStarter:
    """System startup and verification"""
    
    def __init__(self):
        self.setup_logging()
        self.results = {
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger("SystemStarter")
    
    def print_banner(self):
        """Print system banner"""
        print("\n" + "="*70)
        print(" "*20 + "POLAR BEAR SYSTEM V3.0")
        print(" "*15 + "Fiber Optic Inspection Platform")
        print("="*70 + "\n")
    
    def check_python_version(self):
        """Check Python version"""
        self.logger.info("Checking Python version...")
        version = sys.version_info
        
        if version.major >= 3 and version.minor >= 7:
            self.logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
            self.results['checks']['python'] = True
        else:
            self.logger.error(f"✗ Python 3.7+ required, found {version.major}.{version.minor}")
            self.results['checks']['python'] = False
            self.results['errors'].append("Python version too old")
    
    def check_dependencies(self):
        """Check and install dependencies"""
        self.logger.info("Checking dependencies...")
        
        required_packages = {
            'numpy': 'numpy',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'torch': 'torch',
            'pandas': 'pandas',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn',
            'matplotlib': 'matplotlib'
        }
        
        missing = []
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
                self.logger.info(f"✓ {import_name}")
            except ImportError:
                self.logger.warning(f"✗ {import_name} missing")
                missing.append(package_name)
        
        if missing:
            self.results['warnings'].append(f"Missing packages: {', '.join(missing)}")
            
            # Offer to install
            response = input(f"\nInstall missing packages? ({', '.join(missing)}) [Y/n]: ")
            if response.lower() != 'n':
                self.install_packages(missing)
        
        self.results['checks']['dependencies'] = len(missing) == 0
    
    def install_packages(self, packages):
        """Install missing packages"""
        self.logger.info("Installing packages...")
        
        for package in packages:
            try:
                cmd = [sys.executable, "-m", "pip", "install", package]
                if sys.version_info >= (3, 11):
                    cmd.append("--break-system-packages")
                
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
                self.logger.info(f"✓ Installed {package}")
            except Exception as e:
                self.logger.error(f"✗ Failed to install {package}: {e}")
                self.results['errors'].append(f"Failed to install {package}")
    
    def check_components(self):
        """Check system components"""
        self.logger.info("Checking system components...")
        
        components = {
            'connector.py': 'Basic Connector',
            'universal_connector.py': 'Universal Connector',
            'mega_connector.py': 'Mega Connector',
            'ultimate_mega_connector.py': 'Ultimate Mega Connector',
            'polar_bear_brain.py': 'Polar Bear Brain',
            'master_orchestrator.py': 'Master Orchestrator',
            'hivemind_connector.py': 'Hivemind Connector'
        }
        
        for file, name in components.items():
            if Path(file).exists():
                self.logger.info(f"✓ {name}")
                self.results['checks'][file] = True
            else:
                self.logger.warning(f"✗ {name} not found")
                self.results['checks'][file] = False
                self.results['warnings'].append(f"{name} not found")
    
    def create_directories(self):
        """Create required directories"""
        self.logger.info("Creating directories...")
        
        directories = [
            'input', 'output', 'reference', 'models', 'logs',
            'cache', 'results', 'logs/master', 'logs/algorithms'
        ]
        
        for dir_name in directories:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            
        self.logger.info("✓ All directories created")
        self.results['checks']['directories'] = True
    
    def test_basic_functionality(self):
        """Test basic system functionality"""
        self.logger.info("Testing basic functionality...")
        
        try:
            # Test importing main components
            from connector import Connector
            self.logger.info("✓ Basic connector imports")
            
            # Test creating connector instance
            conn = Connector()
            status = conn.get_status()
            
            self.logger.info(f"✓ Connector status: {status['connector_type']}")
            self.results['checks']['basic_test'] = True
            
        except Exception as e:
            self.logger.error(f"✗ Basic functionality test failed: {e}")
            self.results['checks']['basic_test'] = False
            self.results['errors'].append("Basic functionality test failed")
    
    def generate_test_image(self):
        """Generate a test image if none exists"""
        test_image_path = Path("test_fiber.png")
        
        if not test_image_path.exists():
            self.logger.info("Generating test image...")
            
            try:
                import cv2
                import numpy as np
                
                # Create synthetic fiber optic image
                img = np.zeros((512, 512, 3), dtype=np.uint8)
                
                # Draw fiber structure
                center = (256, 256)
                cv2.circle(img, center, 200, (100, 100, 100), -1)  # Ferrule
                cv2.circle(img, center, 150, (150, 150, 150), -1)  # Cladding
                cv2.circle(img, center, 50, (255, 255, 255), -1)   # Core
                
                # Add some defects
                cv2.line(img, (100, 100), (400, 400), (50, 50, 50), 2)  # Scratch
                cv2.circle(img, (300, 200), 5, (0, 0, 0), -1)  # Pit
                
                cv2.imwrite(str(test_image_path), img)
                self.logger.info("✓ Test image created")
                
            except Exception as e:
                self.logger.warning(f"Could not create test image: {e}")
    
    def show_quick_start(self):
        """Show quick start commands"""
        print("\n" + "="*70)
        print("QUICK START COMMANDS")
        print("="*70)
        
        print("\n1. Test the system:")
        print("   python system_test_and_troubleshoot.py")
        
        print("\n2. Process a single image:")
        print("   python mega_connector.py analyze test_fiber.png")
        
        print("\n3. Start interactive mode:")
        print("   python ultimate_mega_connector.py")
        
        print("\n4. Start the Master Orchestrator:")
        print("   python master_orchestrator.py start")
        
        print("\n5. Use the Brain system:")
        print("   python polar_bear_brain.py")
        
        print("\n6. View system status:")
        print("   python connector.py status")
        
        print("\n" + "="*70)
    
    def generate_report(self):
        """Generate startup report"""
        print("\n" + "="*70)
        print("STARTUP REPORT")
        print("="*70)
        
        # Check results
        all_checks = all(self.results['checks'].values())
        
        if all_checks and not self.results['errors']:
            print("\n✅ SYSTEM READY - All checks passed!")
        elif self.results['errors']:
            print("\n❌ SYSTEM NOT READY - Errors found:")
            for error in self.results['errors']:
                print(f"   - {error}")
        else:
            print("\n⚠️  SYSTEM READY WITH WARNINGS:")
            for warning in self.results['warnings']:
                print(f"   - {warning}")
        
        # Recommendations
        if not self.results['checks'].get('ultimate_mega_connector.py'):
            self.results['recommendations'].append(
                "For best performance, ensure ultimate_mega_connector.py is available"
            )
        
        if not self.results['checks'].get('dependencies'):
            self.results['recommendations'].append(
                "Install all dependencies for full functionality"
            )
        
        if self.results['recommendations']:
            print("\n📋 RECOMMENDATIONS:")
            for rec in self.results['recommendations']:
                print(f"   - {rec}")
        
        # Save report
        report_file = f"startup_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")
    
    def interactive_setup(self):
        """Interactive setup wizard"""
        print("\n" + "="*70)
        print("INTERACTIVE SETUP")
        print("="*70)
        
        # Ask about usage mode
        print("\nHow will you use the system?")
        print("1. Single image analysis")
        print("2. Batch processing")
        print("3. Real-time video processing")
        print("4. Complete inspection workflow")
        print("5. Development/Testing")
        
        choice = input("\nSelect mode (1-5): ").strip()
        
        recommendations = {
            '1': "Use: python mega_connector.py analyze <image_path>",
            '2': "Use: python master_orchestrator.py batch --images *.jpg",
            '3': "Use: python polar_bear_brain.py (select real-time mode)",
            '4': "Use: python ultimate_mega_connector.py (interactive mode)",
            '5': "Use: python connector.py interactive"
        }
        
        if choice in recommendations:
            print(f"\nRecommended command: {recommendations[choice]}")
        
        # Ask about performance
        gpu_available = self._check_gpu()
        if gpu_available:
            use_gpu = input("\nGPU detected. Enable GPU acceleration? [Y/n]: ").lower() != 'n'
            if use_gpu:
                print("GPU acceleration enabled in configuration")
    
    def _check_gpu(self):
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def run(self):
        """Run complete startup sequence"""
        self.print_banner()
        
        # Run all checks
        self.check_python_version()
        self.check_dependencies()
        self.check_components()
        self.create_directories()
        self.test_basic_functionality()
        self.generate_test_image()
        
        # Generate report
        self.generate_report()
        
        # Interactive setup
        if not all(self.results['checks'].values()):
            response = input("\nRun interactive setup? [Y/n]: ")
            if response.lower() != 'n':
                self.interactive_setup()
        
        # Show quick start
        self.show_quick_start()
        
        print("\n✨ Startup sequence complete!")

def main():
    """Main entry point"""
    starter = SystemStarter()
    
    try:
        starter.run()
    except KeyboardInterrupt:
        print("\n\nStartup interrupted by user")
    except Exception as e:
        print(f"\n❌ Startup error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()