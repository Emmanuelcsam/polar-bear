#!/usr/bin/env python3
"""
Deep Image Crawler and Organizer
Finds specific images across all subdirectories and organizes them
Includes unit testing, logging, and dependency management
Windows-compatible version with ASCII output

Features:
- Deep recursive directory crawling
- Automatic duplicate detection and renaming
- Interactive execution of organization scripts
- Comprehensive logging and reporting
- Automatic dependency installation
- Full unit testing suite
"""

import os
import sys
import shutil
import subprocess
import logging
import json
import hashlib
import unittest
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import importlib.util

# Configure logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"image_crawler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages automatic installation of dependencies"""
    
    REQUIRED_PACKAGES = {
        'PIL': 'pillow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'imagehash': 'imagehash'
    }
    
    @staticmethod
    def check_and_install_dependencies():
        """Check and install required dependencies"""
        logger.info("Checking dependencies...")
        print("[DEPS] Checking required packages...")
        missing_packages = []
        
        for module_name, package_name in DependencyManager.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(module_name)
                logger.info(f"[OK] {module_name} is already installed")
                print(f"[DEPS] [OK] {module_name} ({package_name}) - Already installed")
            except ImportError:
                missing_packages.append(package_name)
                logger.warning(f"[MISSING] {module_name} is not installed")
                print(f"[DEPS] [MISSING] {module_name} ({package_name}) - Not found")
        
        if missing_packages:
            print(f"\n[DEPS] Need to install {len(missing_packages)} packages: {', '.join(missing_packages)}")
            logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
            try:
                print("[DEPS] Running pip install...")
                print(f"[DEPS] Command: {sys.executable} -m pip install {' '.join(missing_packages)}")
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                logger.info("All dependencies installed successfully")
                print("[DEPS] [OK] All dependencies installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e}")
                print(f"[DEPS] [FAIL] Failed to install dependencies: {e}")
                sys.exit(1)
        else:
            logger.info("All dependencies are already satisfied")
            print("[DEPS] [OK] All dependencies are already satisfied")

class ImageCrawler:
    """Main class for crawling and organizing images"""
    
    TARGET_IMAGES = {
        'region_core.png': 'core',
        'region_cladding.png': 'cladding',
        'region_ferrule.png': 'ferrule'
    }
    
    def __init__(self):
        self.start_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_output_dir = r"C:\Users\Saem1001\Documents\GitHub\polar-bear\dataset"
        self.found_images = {category: [] for category in self.TARGET_IMAGES.values()}
        self.stats = {
            'total_files_scanned': 0,
            'total_images_found': 0,
            'duplicates_renamed': 0,
            'errors': 0
        }
        self.character_sort_path = None
        self.size_sort_path = None
        logger.info(f"ImageCrawler initialized. Starting directory: {self.start_dir}")
        print(f"\n[INIT] ImageCrawler initialized")
        print(f"[INIT] Starting directory: {self.start_dir}")
        print(f"[INIT] Target output base: {self.base_output_dir}")
    
    def get_organization_script_paths(self):
        """Get paths to organization scripts from user"""
        print("\n" + "="*60)
        print("ORGANIZATION SCRIPTS CONFIGURATION")
        print("="*60)
        print("\n[CONFIG] Note: character-sort.py and size-sort.py are interactive scripts.")
        print("[CONFIG] They will ask you questions during execution.")
        print("[CONFIG] When they ask for a directory, enter '.' (current directory)")
        print("[CONFIG] since we'll already be in the target directory.\n")
        
        # Get character-sort.py path
        print("[CONFIG] Please provide the path to character-sort.py")
        print("[CONFIG] (Press Enter to skip if you don't want to use it)")
        
        while True:
            char_path = input("Path to character-sort.py: ").strip()
            
            if char_path == "":
                print("[CONFIG] Skipping character-sort.py")
                self.character_sort_path = None
                break
            elif os.path.exists(char_path) and char_path.endswith('.py'):
                self.character_sort_path = os.path.abspath(char_path)
                print(f"[CONFIG] [OK] Found character-sort.py at: {self.character_sort_path}")
                break
            else:
                print(f"[ERROR] File not found or not a .py file: {char_path}")
                if not self.ask_user_confirmation("Try again?", "yes"):
                    self.character_sort_path = None
                    break
        
        # Get size-sort.py path
        print("\n[CONFIG] Please provide the path to size-sort.py")
        print("[CONFIG] (Press Enter to skip if you don't want to use it)")
        
        while True:
            size_path = input("Path to size-sort.py: ").strip()
            
            if size_path == "":
                print("[CONFIG] Skipping size-sort.py")
                self.size_sort_path = None
                break
            elif os.path.exists(size_path) and size_path.endswith('.py'):
                self.size_sort_path = os.path.abspath(size_path)
                print(f"[CONFIG] [OK] Found size-sort.py at: {self.size_sort_path}")
                break
            else:
                print(f"[ERROR] File not found or not a .py file: {size_path}")
                if not self.ask_user_confirmation("Try again?", "yes"):
                    self.size_sort_path = None
                    break
        
        print("\n[CONFIG] Script configuration complete:")
        print(f"[CONFIG] character-sort.py: {self.character_sort_path or 'Not configured'}")
        print(f"[CONFIG] size-sort.py: {self.size_sort_path or 'Not configured'}")
    
    def ask_user_confirmation(self, question: str, default: str = "yes") -> bool:
        """Ask user a yes/no question"""
        valid = {"yes": True, "y": True, "no": False, "n": False}
        prompt = " [Y/n] " if default == "yes" else " [y/N] "
        
        while True:
            choice = input(question + prompt).lower().strip()
            if choice == "":
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                print("Please respond with 'yes' or 'no' (or 'y' or 'n').")
    
    def get_file_hash(self, filepath: str) -> str:
        """Generate MD5 hash of a file for duplicate detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {filepath}: {e}")
            return ""
    
    def deep_crawl(self) -> Dict[str, List[str]]:
        """Recursively crawl all subdirectories for target images"""
        logger.info("Starting deep crawl for images...")
        print("\n" + "="*60)
        print("STARTING DEEP CRAWL")
        print("="*60)
        print(f"[CRAWL] Scanning directory tree from: {self.start_dir}")
        print(f"[CRAWL] Looking for: {', '.join(self.TARGET_IMAGES.keys())}")
        
        dir_count = 0
        
        for root, dirs, files in os.walk(self.start_dir):
            # Skip the logs directory
            if 'logs' in root:
                continue
            
            dir_count += 1
            relative_path = os.path.relpath(root, self.start_dir)
            print(f"\n[CRAWL] Scanning directory #{dir_count}: {relative_path}")
            print(f"[CRAWL] Contains {len(files)} files, {len(dirs)} subdirectories")
                
            self.stats['total_files_scanned'] += len(files)
            
            for file in files:
                if file.lower() in [img.lower() for img in self.TARGET_IMAGES.keys()]:
                    filepath = os.path.join(root, file)
                    category = self.TARGET_IMAGES.get(file.lower(), 
                                                   self.TARGET_IMAGES.get(file))
                    
                    if category:
                        self.found_images[category].append(filepath)
                        self.stats['total_images_found'] += 1
                        logger.info(f"Found {file} at: {filepath}")
                        print(f"[FOUND] [OK] {file} -> Category: {category}")
                        print(f"        Path: {filepath}")
        
        # Log summary
        print("\n" + "-"*60)
        print("[CRAWL] Crawl complete!")
        print(f"[STATS] Directories scanned: {dir_count}")
        print(f"[STATS] Total files scanned: {self.stats['total_files_scanned']}")
        print(f"[STATS] Total images found: {self.stats['total_images_found']}")
        print("\n[SUMMARY] Images found by category:")
        for category, paths in self.found_images.items():
            print(f"  - {category}: {len(paths)} images")
            if len(paths) > 0 and len(paths) <= 3:
                for path in paths:
                    print(f"    • {os.path.basename(path)}")
            elif len(paths) > 3:
                for path in paths[:2]:
                    print(f"    • {os.path.basename(path)}")
                print(f"    • ... and {len(paths) - 2} more")
        
        logger.info(f"Crawl complete. Found {self.stats['total_images_found']} images")
        
        return self.found_images
    
    def create_output_directories(self) -> Dict[str, str]:
        """Create output directories if they don't exist"""
        print("\n[DIRS] Creating output directories...")
        output_dirs = {}
        
        for category in self.TARGET_IMAGES.values():
            dir_path = os.path.join(self.base_output_dir, category)
            output_dirs[category] = dir_path
            
            try:
                if os.path.exists(dir_path):
                    print(f"[DIRS] [OK] Directory already exists: {dir_path}")
                else:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"[DIRS] [OK] Created directory: {dir_path}")
                logger.info(f"Output directory ready: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                print(f"[DIRS] [FAIL] Failed to create directory {dir_path}: {e}")
                raise
        
        print("[DIRS] All output directories ready")
        return output_dirs
    
    def generate_unique_filename(self, original_name: str, destination_dir: str, 
                               file_hash: str) -> str:
        """Generate a unique filename to avoid overwriting"""
        base_name = Path(original_name).stem
        extension = Path(original_name).suffix
        
        # First try with hash suffix
        new_name = f"{base_name}_{file_hash[:8]}{extension}"
        full_path = os.path.join(destination_dir, new_name)
        
        # If still exists, add counter
        counter = 1
        while os.path.exists(full_path):
            new_name = f"{base_name}_{file_hash[:8]}_{counter}{extension}"
            full_path = os.path.join(destination_dir, new_name)
            counter += 1
            
        return new_name
    
    def copy_and_organize_images(self) -> None:
        """Copy found images to their respective directories"""
        logger.info("Starting to copy and organize images...")
        print("\n" + "="*60)
        print("COPYING AND ORGANIZING IMAGES")
        print("="*60)
        
        output_dirs = self.create_output_directories()
        copied_hashes = {category: {} for category in self.TARGET_IMAGES.values()}
        
        for category, image_paths in self.found_images.items():
            if not image_paths:
                continue
                
            dest_dir = output_dirs[category]
            print(f"\n[COPY] Processing {category} images...")
            print(f"[COPY] Destination: {dest_dir}")
            print(f"[COPY] Total images to process: {len(image_paths)}")
            
            for idx, src_path in enumerate(image_paths, 1):
                try:
                    print(f"\n[COPY] Processing image {idx}/{len(image_paths)}")
                    print(f"[COPY] Source: {src_path}")
                    
                    # Get file hash for duplicate detection
                    print("[COPY] Calculating file hash...")
                    file_hash = self.get_file_hash(src_path)
                    print(f"[COPY] Hash: {file_hash[:16]}...")
                    
                    # Check if we've already copied this exact file
                    if file_hash in copied_hashes[category]:
                        logger.info(f"Skipping duplicate file: {src_path}")
                        print("[COPY] [WARNING] Duplicate detected, skipping...")
                        continue
                    
                    # Generate unique filename
                    original_name = os.path.basename(src_path)
                    new_name = self.generate_unique_filename(original_name, dest_dir, file_hash)
                    dest_path = os.path.join(dest_dir, new_name)
                    
                    # Copy the file
                    print(f"[COPY] Copying to: {dest_path}")
                    shutil.copy2(src_path, dest_path)
                    copied_hashes[category][file_hash] = dest_path
                    
                    if new_name != original_name:
                        self.stats['duplicates_renamed'] += 1
                        logger.info(f"Copied and renamed: {src_path} -> {dest_path}")
                        print(f"[COPY] [OK] Copied and renamed: {original_name} -> {new_name}")
                    else:
                        logger.info(f"Copied: {src_path} -> {dest_path}")
                        print(f"[COPY] [OK] Copied successfully")
                        
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"Error copying {src_path}: {e}")
                    print(f"[ERROR] Failed to copy: {e}")
        
        print("\n" + "-"*60)
        print("[COPY] Copy operation complete!")
        print(f"[STATS] Files copied: {self.stats['total_images_found'] - len([h for cat in copied_hashes.values() for h in cat])}")
        print(f"[STATS] Duplicates renamed: {self.stats['duplicates_renamed']}")
        print(f"[STATS] Errors: {self.stats['errors']}")
    
    def run_external_script(self, script_path: str, target_dir: str) -> bool:
        """Run an external Python script on a specific directory"""
        logger.info(f"Running {script_path} on {target_dir}")
        print(f"\n[EXEC] Preparing to run external script...")
        print(f"[EXEC] Script: {os.path.basename(script_path)}")
        print(f"[EXEC] Target directory: {target_dir}")
        
        try:
            # Save current directory
            original_dir = os.getcwd()
            print(f"[EXEC] Saving current directory: {original_dir}")
            
            # Change to target directory
            os.chdir(target_dir)
            print(f"[EXEC] Changed working directory to: {target_dir}")
            
            # Run the script interactively (don't capture output)
            print(f"[EXEC] Executing script...")
            print(f"[EXEC] Command: {sys.executable} {script_path}")
            print("[EXEC] " + "-"*50)
            print("[EXEC] Script is starting - please respond to any prompts...")
            print("[EXEC] " + "-"*50 + "\n")
            
            # Run without capturing output so the script can interact with the user
            result = subprocess.run([sys.executable, script_path])
            
            print("\n[EXEC] " + "-"*50)
            if result.returncode == 0:
                logger.info(f"Successfully ran {script_path}")
                print(f"[EXEC] [OK] Script completed successfully")
                return True
            else:
                logger.error(f"Script returned non-zero exit code: {result.returncode}")
                print(f"[EXEC] [FAIL] Script failed with return code: {result.returncode}")
                return False
                
        except KeyboardInterrupt:
            print("\n[EXEC] Script interrupted by user")
            logger.warning(f"Script {script_path} interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Exception running {script_path}: {e}")
            print(f"[EXEC] [FAIL] Exception occurred: {e}")
            return False
        finally:
            # Restore original directory
            os.chdir(original_dir)
            print(f"[EXEC] Restored working directory to: {original_dir}")
    
    def run_organization_scripts(self) -> None:
        """Run character-sort.py and size-sort.py on organized directories"""
        logger.info("Running organization scripts...")
        print("\n" + "="*60)
        print("RUNNING ORGANIZATION SCRIPTS")
        print("="*60)
        
        # Check if any scripts are configured
        if not self.character_sort_path and not self.size_sort_path:
            print("[INFO] No organization scripts configured, skipping...")
            return
        
        # Run scripts on each output directory
        output_dirs = self.create_output_directories()
        
        for category, dir_path in output_dirs.items():
            print(f"\n[SCRIPT] Processing {category} directory...")
            print(f"[SCRIPT] Directory path: {dir_path}")
            
            # Check if directory has images
            image_files = [f for f in os.listdir(dir_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_count = len(image_files)
            
            if image_count == 0:
                print(f"[SCRIPT] No images in {category} directory, skipping...")
                continue
            
            print(f"[SCRIPT] Found {image_count} images to organize")
            if image_count <= 5:
                for img in image_files:
                    print(f"  • {img}")
            else:
                for img in image_files[:3]:
                    print(f"  • {img}")
                print(f"  • ... and {image_count - 3} more")
            
            # Run character-sort.py
            if self.character_sort_path:
                if self.ask_user_confirmation(f"\nRun character-sort.py on {category} directory?", "yes"):
                    print(f"\n[SCRIPT] Running character-sort.py...")
                    print(f"[SCRIPT] Script path: {self.character_sort_path}")
                    print(f"[SCRIPT] NOTE: When the script asks for a directory, enter '.' for current directory")
                    print(f"[SCRIPT]       (We've already changed to: {dir_path})")
                    success = self.run_external_script(self.character_sort_path, dir_path)
                    if success:
                        print("[SCRIPT] [OK] character-sort.py completed successfully")
                    else:
                        print("[SCRIPT] [FAIL] character-sort.py encountered errors")
            
            # Run size-sort.py
            if self.size_sort_path:
                if self.ask_user_confirmation(f"\nRun size-sort.py on {category} directory?", "yes"):
                    print(f"\n[SCRIPT] Running size-sort.py...")
                    print(f"[SCRIPT] Script path: {self.size_sort_path}")
                    print(f"[SCRIPT] NOTE: When the script asks for a directory, enter '.' for current directory")
                    print(f"[SCRIPT]       (We've already changed to: {dir_path})")
                    success = self.run_external_script(self.size_sort_path, dir_path)
                    if success:
                        print("[SCRIPT] [OK] size-sort.py completed successfully")
                    else:
                        print("[SCRIPT] [FAIL] size-sort.py encountered errors")
    
    def generate_report(self) -> None:
        """Generate a detailed report of the operation"""
        report_path = os.path.join(LOG_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        print("[REPORT] Generating detailed report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'start_directory': self.start_dir,
            'output_directory': self.base_output_dir,
            'statistics': self.stats,
            'found_images': {
                category: len(paths) for category, paths in self.found_images.items()
            },
            'organization_scripts': {
                'character_sort': self.character_sort_path or 'Not configured',
                'size_sort': self.size_sort_path or 'Not configured'
            },
            'log_file': LOG_FILE
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {report_path}")
        print(f"[REPORT] [OK] Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("OPERATION SUMMARY")
        print("="*60)
        print(f"Start directory: {self.start_dir}")
        print(f"Output directory: {self.base_output_dir}")
        print(f"\nStatistics:")
        print(f"  • Total files scanned: {self.stats['total_files_scanned']:,}")
        print(f"  • Total images found: {self.stats['total_images_found']}")
        print(f"  • Duplicates renamed: {self.stats['duplicates_renamed']}")
        print(f"  • Errors encountered: {self.stats['errors']}")
        print(f"\nImages by category:")
        for category, count in report['found_images'].items():
            print(f"  • {category}: {count} images")
        print(f"\nOrganization scripts used:")
        print(f"  • character-sort.py: {self.character_sort_path or 'Not used'}")
        print(f"  • size-sort.py: {self.size_sort_path or 'Not used'}")
        print(f"\nLogs and reports:")
        print(f"  • Full log: {LOG_FILE}")
        print(f"  • JSON report: {report_path}")
        print("="*60)


class TestImageCrawler(unittest.TestCase):
    """Unit tests for ImageCrawler functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = "test_images"
        self.crawler = ImageCrawler()
        self.crawler.start_dir = self.test_dir
        self.crawler.base_output_dir = "test_output"
        
        # Create test directory structure
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "subdir1"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "subdir2", "subsubdir"), exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
    
    def test_get_file_hash(self):
        """Test file hashing function"""
        # Create a test file
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Test hashing
        hash1 = self.crawler.get_file_hash(test_file)
        hash2 = self.crawler.get_file_hash(test_file)
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 32)  # MD5 hash length
        
        logger.info(f"[OK] test_get_file_hash passed")
    
    def test_create_output_directories(self):
        """Test output directory creation"""
        output_dirs = self.crawler.create_output_directories()
        
        self.assertEqual(len(output_dirs), 3)
        for category, path in output_dirs.items():
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.isdir(path))
        
        logger.info(f"[OK] test_create_output_directories passed")
    
    def test_generate_unique_filename(self):
        """Test unique filename generation"""
        test_dir = os.path.join(self.test_dir, "unique_test")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create existing file
        existing_file = os.path.join(test_dir, "region_core_12345678.png")
        with open(existing_file, 'w') as f:
            f.write("test")
        
        # Test unique name generation
        new_name = self.crawler.generate_unique_filename(
            "region_core.png", test_dir, "12345678abcdef"
        )
        
        self.assertIn("12345678", new_name)
        self.assertTrue(new_name.endswith(".png"))
        
        logger.info(f"[OK] test_generate_unique_filename passed")
    
    def test_deep_crawl(self):
        """Test deep directory crawling"""
        # Create test images
        test_images = [
            os.path.join(self.test_dir, "region_core.png"),
            os.path.join(self.test_dir, "subdir1", "region_cladding.png"),
            os.path.join(self.test_dir, "subdir2", "subsubdir", "region_ferrule.png")
        ]
        
        for img_path in test_images:
            with open(img_path, 'w') as f:
                f.write("fake image data")
        
        # Test crawling
        found_images = self.crawler.deep_crawl()
        
        self.assertEqual(len(found_images['core']), 1)
        self.assertEqual(len(found_images['cladding']), 1)
        self.assertEqual(len(found_images['ferrule']), 1)
        self.assertEqual(self.crawler.stats['total_images_found'], 3)
        
        logger.info(f"[OK] test_deep_crawl passed")


def run_unit_tests():
    """Run all unit tests"""
    logger.info("Running unit tests...")
    print("[TEST] Preparing to run unit tests...")
    print("[TEST] This ensures all functions work correctly")
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImageCrawler)
    
    # Run tests
    print("[TEST] Running tests...\n")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n[TEST] Tests run: {result.testsRun}")
    print(f"[TEST] Failures: {len(result.failures)}")
    print(f"[TEST] Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        logger.info("All unit tests passed!")
        print("[TEST] [OK] All unit tests passed!")
        return True
    else:
        logger.error("Some unit tests failed!")
        print("[TEST] [FAIL] Some unit tests failed!")
        return False


def main():
    """Main execution function"""
    print("="*60)
    print("DEEP IMAGE CRAWLER AND ORGANIZER")
    print("="*60)
    print("[INFO] Starting application...")
    print(f"[INFO] Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Log file: {LOG_FILE}")
    
    try:
        # Check and install dependencies
        print("\n[DEPS] Checking dependencies...")
        DependencyManager.check_and_install_dependencies()
        
        # Run unit tests
        print("\n[TEST] Running unit tests...")
        if not run_unit_tests():
            if not ImageCrawler().ask_user_confirmation("\n[TEST] Unit tests failed. Continue anyway?", "no"):
                sys.exit(1)
        
        # Create crawler instance
        crawler = ImageCrawler()
        
        # Show configuration
        print(f"\n[CONFIG] Configuration:")
        print(f"[CONFIG]   Start directory: {crawler.start_dir}")
        print(f"[CONFIG]   Output directory: {crawler.base_output_dir}")
        print(f"[CONFIG]   Target images: {', '.join(crawler.TARGET_IMAGES.keys())}")
        
        if not crawler.ask_user_confirmation("\n[CONFIRM] Proceed with image crawling and organization?", "yes"):
            logger.info("Operation cancelled by user")
            print("[INFO] Operation cancelled by user")
            sys.exit(0)
        
        # Get organization script paths
        crawler.get_organization_script_paths()
        
        # Run the crawler
        crawler.deep_crawl()
        
        if crawler.stats['total_images_found'] == 0:
            logger.warning("No target images found!")
            print("\n[WARNING] No target images found!")
            if crawler.ask_user_confirmation("[CONFIRM] No images found. Exit?", "yes"):
                sys.exit(0)
        
        # Copy and organize images
        if crawler.ask_user_confirmation("\n[CONFIRM] Proceed with copying images?", "yes"):
            crawler.copy_and_organize_images()
        
        # Run organization scripts
        if (crawler.character_sort_path or crawler.size_sort_path) and \
           crawler.ask_user_confirmation("\n[CONFIRM] Run organization scripts on the copied images?", "yes"):
            crawler.run_organization_scripts()
        
        # Generate report
        print("\n[REPORT] Generating final report...")
        crawler.generate_report()
        
        logger.info("Operation completed successfully!")
        print("\n[SUCCESS] Operation completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        print("\n[INTERRUPT] Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n[ERROR] Unexpected error: {e}")
        print("[ERROR] Check log file for full traceback")
        sys.exit(1)
    finally:
        input("\n[DONE] Press Enter to exit...")


if __name__ == "__main__":
    main()