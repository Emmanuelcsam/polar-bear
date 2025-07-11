#!/usr/bin/env python3
"""
Automated Interactive Tester for Image Classifier
Handles all interactive prompts and tests every mode
"""

import subprocess
import sys
import os
import time
import tempfile
import shutil
from PIL import Image
import numpy as np
import threading
from queue import Queue
import json

class AutomatedTester:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.reference_dir = os.path.join(self.temp_dir, "reference")
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        self.test_results = []
        
        # Copy necessary files to temp directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copy(os.path.join(script_dir, "image-classifier.py"), self.temp_dir)
        if os.path.exists(os.path.join(script_dir, "test_image_classifier_comprehensive.py")):
            shutil.copy(os.path.join(script_dir, "test_image_classifier_comprehensive.py"), self.temp_dir)
        
    def setup_test_environment(self):
        """Create test directories and images"""
        print("Setting up test environment...")
        
        # Create directories
        os.makedirs(self.reference_dir)
        os.makedirs(self.dataset_dir)
        
        # Create reference images with different classifications
        reference_structure = {
            "fc/50/core/clean": [(255, 0, 0), (200, 50, 50)],  # Red variations
            "fc/50/core/dirty": [(100, 50, 50), (150, 100, 100)],
            "fc/91/cladding/clean": [(0, 255, 0), (50, 200, 50)],  # Green variations
            "sma/50/ferrule/scratched": [(0, 0, 255), (50, 50, 200)],  # Blue variations
            "sma/91/core/oil": [(255, 255, 0), (200, 200, 50)],  # Yellow variations
            "lc/50/core/wet": [(255, 0, 255), (200, 50, 200)],  # Magenta variations
            "st/91/cladding/blob": [(0, 255, 255), (50, 200, 200)],  # Cyan variations
            "sc/50/ferrule/anomaly": [(128, 128, 128), (100, 100, 100)]  # Gray variations
        }
        
        for path, colors in reference_structure.items():
            full_path = os.path.join(self.reference_dir, path)
            os.makedirs(full_path, exist_ok=True)
            
            # Create multiple reference images for each classification
            for i, color in enumerate(colors):
                img_name = f"ref_{path.replace('/', '_')}_{i}.jpg"
                img_path = os.path.join(full_path, img_name)
                self.create_test_image(img_path, color)
                
        # Create dataset images (unclassified)
        dataset_images = [
            ("unclassified_red.jpg", (250, 10, 10)),
            ("unclassified_green.jpg", (10, 250, 10)),
            ("unclassified_blue.jpg", (10, 10, 250)),
            ("unclassified_yellow.jpg", (250, 250, 10)),
            ("unclassified_mixed.jpg", (128, 64, 192)),
            ("test_image_1.png", (200, 100, 50)),
            ("test_image_2.bmp", (50, 100, 200)),
            ("sample_fiber.jpg", (150, 150, 150))
        ]
        
        for img_name, color in dataset_images:
            img_path = os.path.join(self.dataset_dir, img_name)
            self.create_test_image(img_path, color)
            
        print(f"Created {len(reference_structure)} reference classifications")
        print(f"Created {len(dataset_images)} dataset images")
        
    def create_test_image(self, path, color, size=(128, 128)):
        """Create a test image with specified color and some noise"""
        # Create base image with color
        img_array = np.full((*size, 3), color, dtype=np.uint8)
        
        # Add some noise for variety
        noise = np.random.randint(-20, 20, (*size, 3))
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add some patterns
        # Horizontal lines
        for y in range(0, size[1], 10):
            img_array[y:y+2, :] = np.clip(img_array[y:y+2, :] + 30, 0, 255)
            
        # Vertical lines
        for x in range(0, size[0], 10):
            img_array[:, x:x+2] = np.clip(img_array[:, x:x+2] - 30, 0, 255)
            
        img = Image.fromarray(img_array)
        img.save(path)
        
    def run_test(self, test_name, command, input_sequence=None, timeout=60):
        """Run a test with automated input"""
        print(f"\n{'='*60}")
        print(f"Running test: {test_name}")
        print(f"Command: {command}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if input_sequence:
                # Run with automated input
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.temp_dir
                )
                
                # Send input sequence
                for inp in input_sequence:
                    if inp == "WAIT":
                        time.sleep(1)
                    else:
                        process.stdin.write(inp + '\n')
                        process.stdin.flush()
                        time.sleep(0.5)
                
                # Wait for completion
                stdout, stderr = process.communicate(timeout=timeout)
                
            else:
                # Run without input
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.temp_dir
                )
                stdout = result.stdout
                stderr = result.stderr
                process = result
                
            duration = time.time() - start_time
            
            # Check results
            success = process.returncode == 0
            
            self.test_results.append({
                'test_name': test_name,
                'command': command,
                'success': success,
                'duration': duration,
                'stdout': stdout[-1000:] if len(stdout) > 1000 else stdout,  # Last 1000 chars
                'stderr': stderr[-1000:] if len(stderr) > 1000 else stderr
            })
            
            print(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")
            print(f"Duration: {duration:.2f}s")
            
            if not success and stderr:
                print(f"Error output:\n{stderr}")
                
            return success
            
        except subprocess.TimeoutExpired:
            print(f"❌ Test timed out after {timeout}s")
            self.test_results.append({
                'test_name': test_name,
                'command': command,
                'success': False,
                'duration': timeout,
                'error': 'Timeout'
            })
            return False
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            self.test_results.append({
                'test_name': test_name,
                'command': command,
                'success': False,
                'error': str(e)
            })
            return False
            
    def test_automatic_mode(self):
        """Test automatic classification mode"""
        cmd = f"python image-classifier.py --reference_folder {self.reference_dir} --dataset_folder {self.dataset_dir} --mode auto"
        return self.run_test("Automatic Mode", cmd)
        
    def test_manual_console_mode(self):
        """Test manual mode with console interface"""
        cmd = f"python image-classifier.py --reference_folder {self.reference_dir} --dataset_folder {self.dataset_dir} --mode manual"
        
        # Simulate user inputs for manual mode
        input_sequence = [
            "1",  # Accept suggestion for first image
            "2",  # Enter custom classification for second image
            "50-fc-core-clean",  # Custom classification
            "3",  # Skip third image
            "1",  # Accept suggestion for fourth image
            "4",  # Exit
        ]
        
        return self.run_test("Manual Console Mode", cmd, input_sequence)
        
    def test_exit_mode(self):
        """Test exit mode (just analyzes references)"""
        cmd = f"python image-classifier.py --reference_folder {self.reference_dir} --dataset_folder {self.dataset_dir} --mode exit"
        return self.run_test("Exit Mode", cmd)
        
    def test_custom_parameters(self):
        """Test with custom parameters"""
        cmd = (f"python image-classifier.py "
               f"--reference_folder {self.reference_dir} "
               f"--dataset_folder {self.dataset_dir} "
               f"--mode auto "
               f"--similarity_threshold 0.8 "
               f"--auto_create_folders False "
               f"--custom_keywords fiber optic connector")
        
        return self.run_test("Custom Parameters", cmd)
        
    def test_help_command(self):
        """Test help command"""
        cmd = "python image-classifier.py --help"
        return self.run_test("Help Command", cmd)
        
    def test_missing_directories(self):
        """Test behavior with missing directories"""
        missing_ref = os.path.join(self.temp_dir, "missing_ref")
        cmd = f"python image-classifier.py --reference_folder {missing_ref} --dataset_folder {self.dataset_dir} --mode auto"
        
        # Should prompt to create directory
        input_sequence = ["n"]  # Don't create directory
        
        return self.run_test("Missing Reference Directory", cmd, input_sequence)
        
    def test_empty_dataset(self):
        """Test with empty dataset folder"""
        empty_dataset = os.path.join(self.temp_dir, "empty_dataset")
        os.makedirs(empty_dataset)
        
        cmd = f"python image-classifier.py --reference_folder {self.reference_dir} --dataset_folder {empty_dataset} --mode auto"
        return self.run_test("Empty Dataset", cmd)
        
    def verify_results(self):
        """Verify the results of classification"""
        print(f"\n{'='*60}")
        print("Verifying classification results...")
        print(f"{'='*60}")
        
        # Check if files were moved/renamed
        classified_count = 0
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if '-' in file and not file.startswith('unclassified'):
                    classified_count += 1
                    print(f"✅ Classified: {os.path.join(root, file)}")
                    
        print(f"\nTotal classified images: {classified_count}")
        
        # Check if knowledge bank was created
        kb_files = ['knowledge_bank.pkl', 'classifier_config.json']
        for kb_file in kb_files:
            if os.path.exists(os.path.join(self.temp_dir, kb_file)):
                print(f"✅ Created: {kb_file}")
                
        # Check log files
        log_dir = os.path.join(self.temp_dir, "logs")
        if os.path.exists(log_dir):
            log_files = os.listdir(log_dir)
            print(f"✅ Created {len(log_files)} log file(s)")
            
    def run_unit_tests(self):
        """Run the comprehensive unit test suite"""
        cmd = "python test_image_classifier_comprehensive.py"
        return self.run_test("Comprehensive Unit Tests", cmd, timeout=120)
        
    def generate_report(self):
        """Generate test report"""
        print(f"\n{'='*80}")
        print("TEST REPORT")
        print(f"{'='*80}\n")
        
        passed = sum(1 for r in self.test_results if r.get('success', False))
        failed = len(self.test_results) - passed
        
        print(f"Total tests: {len(self.test_results)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success rate: {(passed/len(self.test_results)*100):.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 60)
        
        for result in self.test_results:
            status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
            print(f"\n{result['test_name']}: {status}")
            if 'duration' in result:
                print(f"Duration: {result['duration']:.2f}s")
            if not result.get('success', False):
                if 'error' in result:
                    print(f"Error: {result['error']}")
                if 'stderr' in result and result['stderr']:
                    print(f"Stderr: {result['stderr'][:200]}...")
                    
        # Save report to file
        report_path = os.path.join(self.temp_dir, "test_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        
    def cleanup(self):
        """Clean up test environment"""
        print("\nCleaning up test environment...")
        try:
            shutil.rmtree(self.temp_dir)
            print("✅ Cleanup complete")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
            
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "="*80)
        print("AUTOMATED INTERACTIVE TESTING SUITE")
        print("Testing all modes and features of image-classifier.py")
        print("="*80 + "\n")
        
        try:
            # Setup
            self.setup_test_environment()
            
            # Run tests
            tests = [
                self.test_help_command,
                self.test_automatic_mode,
                self.test_manual_console_mode,
                self.test_exit_mode,
                self.test_custom_parameters,
                self.test_missing_directories,
                self.test_empty_dataset,
                self.run_unit_tests
            ]
            
            for test in tests:
                test()
                time.sleep(1)  # Brief pause between tests
                
            # Verify results
            self.verify_results()
            
            # Generate report
            self.generate_report()
            
        finally:
            # Always cleanup
            self.cleanup()


def main():
    """Main entry point"""
    print("Starting automated interactive testing...")
    print("This will test all modes and handle all interactive prompts automatically")
    print("-" * 80)
    
    tester = AutomatedTester()
    tester.run_all_tests()
    
    print("\n✅ All automated tests completed!")
    

if __name__ == "__main__":
    main()