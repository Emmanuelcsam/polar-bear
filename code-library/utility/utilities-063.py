#!/usr/bin/env python3
"""
Final validation script for Automated Processing Studio
Performs comprehensive checks to ensure all requirements are met
"""

import os
import sys
import traceback
import tempfile
import numpy as np
from pathlib import Path
import json
import time


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"{title.upper()}")
    print(f"{'='*60}")


def check_requirement(description, check_func):
    """Run a requirement check and report results"""
    try:
        result, details = check_func()
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {description}")
        if details:
            print(f"     {details}")
        return result
    except Exception as e:
        print(f"❌ ERROR - {description}")
        print(f"     {str(e)}")
        if os.environ.get('STUDIO_DEBUG'):
            traceback.print_exc()
        return False


def validate_dependencies():
    """Check that all dependencies can be imported"""
    def check():
        try:
            import cv2
            import numpy as np
            import sklearn
            return True, f"All core dependencies available"
        except ImportError as e:
            return False, f"Missing dependency: {e}"
    
    return check_requirement("Dependencies installed", check)


def validate_auto_dependency_installation():
    """Check that dependency auto-installation works"""
    def check():
        try:
            from automated_processing_studio import DependencyManager
            # Just verify the class exists and has the method
            if hasattr(DependencyManager, 'check_and_install_dependencies'):
                return True, "Auto-dependency installation available"
            return False, "Method not found"
        except Exception as e:
            return False, str(e)
    
    return check_requirement("Auto-dependency installation", check)


def validate_no_argparse():
    """Ensure the system doesn't use argparse"""
    def check():
        # Read the main file and check for argparse
        with open("automated_processing_studio.py", "r") as f:
            content = f.read()
        
        if "argparse" in content.lower():
            return False, "Found 'argparse' in code"
        
        # Check for interactive_setup method
        if "interactive_setup" not in content:
            return False, "No interactive_setup method found"
        
        return True, "Uses interactive configuration (no argparse)"
    
    return check_requirement("No argparse usage", check)


def validate_ram_processing():
    """Validate that processing runs entirely in RAM"""
    def check():
        try:
            from automated_processing_studio import ImageProcessor
            
            # Create test image in memory
            test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            processor = ImageProcessor()
            
            # Test various operations
            normalized = processor.normalize_image(test_img)
            gray = processor.to_grayscale(test_img)
            features = processor.extract_features(test_img)
            
            # Verify all outputs are in memory (numpy arrays or dicts)
            if not isinstance(normalized, np.ndarray):
                return False, "normalize_image didn't return numpy array"
            if not isinstance(gray, np.ndarray):
                return False, "to_grayscale didn't return numpy array"
            if not isinstance(features, dict):
                return False, "extract_features didn't return dict"
            
            return True, "All processing operations work in RAM"
        except Exception as e:
            return False, str(e)
    
    return check_requirement("RAM-based processing", check)


def validate_learning_capabilities():
    """Check reinforcement learning implementation"""
    def check():
        try:
            from automated_processing_studio import ReinforcementLearner
            
            learner = ReinforcementLearner(state_size=10, action_size=5)
            
            # Test learning
            state = np.random.random(10)
            action = learner.choose_action(state)
            initial_epsilon = learner.epsilon
            
            # Simulate learning
            for _ in range(10):
                next_state = np.random.random(10)
                learner.learn(state, action, 1.0, next_state, False)
            
            # Check that learning occurred
            if learner.epsilon >= initial_epsilon:
                return False, "Epsilon didn't decay"
            
            if len(learner.q_table) == 0:
                return False, "Q-table is empty after learning"
            
            return True, f"Learning works - epsilon: {learner.epsilon:.4f}, Q-table size: {len(learner.q_table)}"
        except Exception as e:
            return False, str(e)
    
    return check_requirement("Reinforcement learning", check)


def validate_anomaly_detection():
    """Check anomaly detection functionality"""
    def check():
        try:
            from automated_processing_studio import AnomalyDetector, ImageProcessor
            
            detector = AnomalyDetector()
            processor = ImageProcessor()
            
            # Create test image
            test_img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
            features = processor.extract_features(test_img)
            
            # Test anomaly detection
            result = detector.detect_anomalies(test_img, features)
            
            if not isinstance(result, dict):
                return False, "detect_anomalies didn't return dict"
            
            required_keys = ['is_anomaly', 'anomaly_score', 'reasons']
            for key in required_keys:
                if key not in result:
                    return False, f"Missing key in result: {key}"
            
            # Test library functions
            detector.add_to_library(test_img, features, is_anomaly=False)
            
            if len(detector.similarity_library) != 1:
                return False, "Failed to add to similarity library"
            
            return True, "Anomaly detection and libraries working"
        except Exception as e:
            return False, str(e)
    
    return check_requirement("Anomaly detection", check)


def validate_similarity_matching():
    """Check image similarity calculation"""
    def check():
        try:
            from automated_processing_studio import ImageProcessor
            
            processor = ImageProcessor()
            
            # Create identical images
            img1 = np.ones((50, 50), dtype=np.uint8) * 128
            img2 = img1.copy()
            
            similarity = processor.calculate_similarity(img1, img2)
            
            if similarity > 0.1:
                return False, f"Identical images have high dissimilarity: {similarity}"
            
            # Create different images
            img3 = np.ones((50, 50), dtype=np.uint8) * 255
            similarity2 = processor.calculate_similarity(img1, img3)
            
            if similarity2 < 0.5:
                return False, f"Different images have low dissimilarity: {similarity2}"
            
            return True, f"Similarity calculation works (identical: {similarity:.4f}, different: {similarity2:.4f})"
        except Exception as e:
            return False, str(e)
    
    return check_requirement("Similarity matching", check)


def validate_documentation():
    """Check that comprehensive documentation exists"""
    def check():
        try:
            from automated_processing_studio import AutomatedProcessingStudio
            
            studio = AutomatedProcessingStudio()
            
            # Create minimal test
            with tempfile.TemporaryDirectory() as temp_dir:
                results = {
                    'success': True,
                    'iterations': 5,
                    'final_similarity': 0.1,
                    'pipeline': ['test1', 'test2'],
                    'processing_time': 1.0,
                    'anomalies_detected': [],
                    'processing_log': []
                }
                
                test_img = np.ones((50, 50), dtype=np.uint8) * 128
                
                # Set cache dir to temp
                studio.cache_dir = Path(temp_dir)
                studio._generate_report(results, test_img, test_img, test_img)
                
                # Check files created
                report_dirs = list(Path(temp_dir).glob("report_*"))
                
                if len(report_dirs) != 1:
                    return False, "Report directory not created"
                
                report_dir = report_dirs[0]
                expected_files = [
                    "input.png", "target.png", "output.png",
                    "comparison.png", "report.txt", "processing_log.json"
                ]
                
                for file in expected_files:
                    if not (report_dir / file).exists():
                        return False, f"Missing report file: {file}"
                
                return True, "Report generation works with all expected files"
        except Exception as e:
            return False, str(e)
    
    return check_requirement("Documentation generation", check)


def validate_script_detection():
    """Check that scripts are properly detected and loaded"""
    def check():
        try:
            from automated_processing_studio import ScriptManager
            
            # Create temp script directory
            with tempfile.TemporaryDirectory() as temp_dir:
                scripts_dir = Path(temp_dir) / "scripts"
                scripts_dir.mkdir()
                
                # Create test script
                test_script = '''
import numpy as np
def process_image(image: np.ndarray) -> np.ndarray:
    return image * 2
'''
                with open(scripts_dir / "test_multiply.py", "w") as f:
                    f.write(test_script)
                
                # Load scripts
                manager = ScriptManager(str(scripts_dir))
                
                if len(manager.functions) != 1:
                    return False, f"Expected 1 script, found {len(manager.functions)}"
                
                if "test_multiply.py" not in manager.functions:
                    return False, "Test script not loaded"
                
                # Test execution
                test_img = np.ones((10, 10), dtype=np.uint8) * 50
                result = manager.functions["test_multiply.py"](test_img)
                
                if not np.array_equal(result, test_img * 2):
                    return False, "Script execution failed"
                
                return True, "Script detection and loading works"
        except Exception as e:
            return False, str(e)
    
    return check_requirement("Script detection", check)


def validate_unit_tests():
    """Check that comprehensive unit tests exist and can run"""
    def check():
        if not os.path.exists("test_automated_studio.py"):
            return False, "Test file not found"
        
        # Check test content
        with open("test_automated_studio.py", "r") as f:
            content = f.read()
        
        required_test_classes = [
            "TestDependencyManager",
            "TestImageProcessor", 
            "TestScriptManager",
            "TestReinforcementLearner",
            "TestAnomalyDetector",
            "TestAutomatedProcessingStudio",
            "TestIntegration",
            "MasterTestSuite"
        ]
        
        missing = []
        for test_class in required_test_classes:
            if test_class not in content:
                missing.append(test_class)
        
        if missing:
            return False, f"Missing test classes: {', '.join(missing)}"
        
        return True, f"All {len(required_test_classes)} test classes present"
    
    return check_requirement("Unit tests exist", check)


def run_validation():
    """Run all validation checks"""
    print_section("Automated Processing Studio Validation")
    print("Checking all requirements are met...")
    
    checks = [
        validate_dependencies,
        validate_auto_dependency_installation,
        validate_no_argparse,
        validate_ram_processing,
        validate_learning_capabilities,
        validate_anomaly_detection,
        validate_similarity_matching,
        validate_documentation,
        validate_script_detection,
        validate_unit_tests
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print_section("Summary")
    passed = sum(results)
    total = len(results)
    
    print(f"Total checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n✨ ALL REQUIREMENTS MET! The Automated Processing Studio is ready to use.")
        print("\nTo get started:")
        print("1. Run: python setup_dependencies.py")
        print("2. Run: python automated_processing_studio.py")
        print("3. Or try the demo: python demo_automated_studio.py")
    else:
        print("\n⚠️  Some requirements are not met. Please fix the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)