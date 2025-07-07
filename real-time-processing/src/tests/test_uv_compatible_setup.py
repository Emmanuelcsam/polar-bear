#!/usr/bin/env python3
"""
Comprehensive test suite for uv_compatible_setup.py
Tests all functions with rigorous edge cases
"""

import unittest
import subprocess
import sys
import os
import platform
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
import importlib.util

# Import the module to test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tools.uv_compatible_setup import (
    Colors, print_header, print_success, print_error, print_warning, print_info,
    check_uv, install_with_uv, check_packages, test_opencv, create_test_script
)


class TestColors(unittest.TestCase):
    """Test the Colors class"""
    
    def test_colors_attributes(self):
        """Test that Colors has all required attributes"""
        expected_attrs = ['HEADER', 'OKBLUE', 'OKGREEN', 'WARNING', 
                         'FAIL', 'ENDC', 'BOLD', 'UNDERLINE']
        
        for attr in expected_attrs:
            self.assertTrue(hasattr(Colors, attr))
            self.assertIsInstance(getattr(Colors, attr), str)
            
    def test_colors_ansi_codes(self):
        """Test that Colors contains valid ANSI codes"""
        self.assertEqual(Colors.HEADER, '\033[95m')
        self.assertEqual(Colors.OKGREEN, '\033[92m')
        self.assertEqual(Colors.ENDC, '\033[0m')
        self.assertEqual(Colors.BOLD, '\033[1m')


class TestPrintFunctions(unittest.TestCase):
    """Test the print helper functions"""
    
    @patch('builtins.print')
    def test_print_header(self, mock_print):
        """Test print_header function"""
        print_header("Test Header")
        
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("Test Header", args)
        self.assertIn(Colors.HEADER, args)
        self.assertIn(Colors.BOLD, args)
        self.assertIn("=" * 60, args)
        
    @patch('builtins.print')
    def test_print_success(self, mock_print):
        """Test print_success function"""
        print_success("Success message")
        
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("✓", args)
        self.assertIn("Success message", args)
        self.assertIn(Colors.OKGREEN, args)
        
    @patch('builtins.print')
    def test_print_error(self, mock_print):
        """Test print_error function"""
        print_error("Error message")
        
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("✗", args)
        self.assertIn("Error message", args)
        self.assertIn(Colors.FAIL, args)
        
    @patch('builtins.print')
    def test_print_warning(self, mock_print):
        """Test print_warning function"""
        print_warning("Warning message")
        
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("⚠", args)
        self.assertIn("Warning message", args)
        self.assertIn(Colors.WARNING, args)
        
    @patch('builtins.print')
    def test_print_info(self, mock_print):
        """Test print_info function"""
        print_info("Info message")
        
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("ℹ", args)
        self.assertIn("Info message", args)
        self.assertIn(Colors.OKBLUE, args)


class TestCheckUv(unittest.TestCase):
    """Test the check_uv function"""
    
    @patch('subprocess.run')
    def test_check_uv_installed(self, mock_run):
        """Test check_uv when uv is installed"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="uv 0.1.0"
        )
        
        with patch('builtins.print'):
            result = check_uv()
            
        self.assertTrue(result)
        mock_run.assert_called_with(
            ['uv', '--version'],
            capture_output=True,
            text=True
        )
        
    @patch('subprocess.run')
    def test_check_uv_not_installed(self, mock_run):
        """Test check_uv when uv is not installed"""
        mock_run.side_effect = FileNotFoundError()
        
        with patch('builtins.print'):
            result = check_uv()
            
        self.assertFalse(result)
        
    @patch('subprocess.run')
    def test_check_uv_error(self, mock_run):
        """Test check_uv when command fails"""
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Error"
        )
        
        with patch('builtins.print'):
            result = check_uv()
            
        self.assertFalse(result)


class TestInstallWithUv(unittest.TestCase):
    """Test the install_with_uv function"""
    
    @patch('subprocess.run')
    def test_install_with_uv_success(self, mock_run):
        """Test install_with_uv with successful installation"""
        mock_run.return_value = Mock(returncode=0)
        
        packages = ['numpy', 'opencv-python']
        
        with patch('builtins.print'):
            result = install_with_uv(packages)
            
        self.assertTrue(result)
        
        # Should call uv pip install
        mock_run.assert_called_with(
            ['uv', 'pip', 'install'] + packages,
            capture_output=True,
            text=True
        )
        
    @patch('subprocess.run')
    def test_install_with_uv_empty_packages(self, mock_run):
        """Test install_with_uv with no packages"""
        packages = []
        
        with patch('builtins.print'):
            result = install_with_uv(packages)
            
        self.assertTrue(result)
        mock_run.assert_not_called()
        
    @patch('subprocess.run')
    def test_install_with_uv_failure(self, mock_run):
        """Test install_with_uv with installation failure"""
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Package not found"
        )
        
        packages = ['invalid-package']
        
        with patch('builtins.print'):
            result = install_with_uv(packages)
            
        self.assertFalse(result)
        
    @patch('subprocess.run')
    def test_install_with_uv_single_package(self, mock_run):
        """Test install_with_uv with single package string"""
        mock_run.return_value = Mock(returncode=0)
        
        with patch('builtins.print'):
            result = install_with_uv('numpy')
            
        self.assertTrue(result)
        
        # Should convert single package to list
        mock_run.assert_called_with(
            ['uv', 'pip', 'install', 'numpy'],
            capture_output=True,
            text=True
        )
        
    @patch('subprocess.run')
    def test_install_with_uv_with_output(self, mock_run):
        """Test install_with_uv captures and displays output"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Successfully installed numpy-1.24.0"
        )
        
        with patch('builtins.print') as mock_print:
            result = install_with_uv(['numpy'])
            
        self.assertTrue(result)
        
        # Should print the installation output
        printed_text = ' '.join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Successfully installed", printed_text)


class TestCheckPackages(unittest.TestCase):
    """Test the check_packages function"""
    
    @patch('importlib.util.find_spec')
    def test_check_packages_all_installed(self, mock_find_spec):
        """Test check_packages when all packages are installed"""
        # Mock all packages as installed
        mock_find_spec.return_value = Mock()  # Non-None means found
        
        packages = ['numpy', 'cv2', 'psutil']
        
        with patch('builtins.print'):
            result = check_packages(packages)
            
        self.assertTrue(result)
        self.assertEqual(mock_find_spec.call_count, len(packages))
        
    @patch('importlib.util.find_spec')
    def test_check_packages_some_missing(self, mock_find_spec):
        """Test check_packages when some packages are missing"""
        # Mock some packages as missing
        def find_spec_side_effect(name):
            if name == 'cv2':
                return None  # Not found
            return Mock()  # Found
            
        mock_find_spec.side_effect = find_spec_side_effect
        
        packages = ['numpy', 'cv2', 'psutil']
        
        with patch('builtins.print'):
            result = check_packages(packages)
            
        self.assertFalse(result)
        
    @patch('importlib.util.find_spec')
    def test_check_packages_empty_list(self, mock_find_spec):
        """Test check_packages with empty package list"""
        packages = []
        
        with patch('builtins.print'):
            result = check_packages(packages)
            
        self.assertTrue(result)  # No packages to check
        mock_find_spec.assert_not_called()
        
    @patch('importlib.util.find_spec')
    def test_check_packages_with_aliases(self, mock_find_spec):
        """Test check_packages handles package aliases correctly"""
        # Mock find_spec to handle aliases
        def find_spec_side_effect(name):
            # opencv-python installs as cv2
            if name == 'opencv-python':
                # Should check for cv2 instead
                return None
            elif name == 'cv2':
                return Mock()
            return Mock()
            
        mock_find_spec.side_effect = find_spec_side_effect
        
        packages = ['numpy', 'opencv-python']
        
        with patch('builtins.print'):
            # The function should handle the opencv-python -> cv2 mapping
            result = check_packages(packages)
            
        # Should check for cv2 when opencv-python is in the list
        calls = [call.args[0] for call in mock_find_spec.call_args_list]
        self.assertIn('cv2', calls)


class TestTestOpencv(unittest.TestCase):
    """Test the test_opencv function"""
    
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    @patch('cv2.GaussianBlur')
    @patch('cv2.Canny')
    def test_test_opencv_success(self, mock_canny, mock_blur, mock_cvtcolor, mock_imread):
        """Test test_opencv with successful OpenCV operations"""
        # Mock successful operations
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = test_image
        mock_cvtcolor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_blur.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_canny.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        with patch('builtins.print'):
            result = test_opencv()
            
        self.assertTrue(result)
        
        # Verify all operations were called
        mock_imread.assert_called_once()
        mock_cvtcolor.assert_called_once()
        mock_blur.assert_called_once()
        mock_canny.assert_called_once()
        
    @patch('cv2.imread')
    def test_test_opencv_import_error(self, mock_imread):
        """Test test_opencv when cv2 import fails"""
        mock_imread.side_effect = ImportError("No module named 'cv2'")
        
        with patch('builtins.print'):
            result = test_opencv()
            
        self.assertFalse(result)
        
    @patch('cv2.imread')
    def test_test_opencv_operation_error(self, mock_imread):
        """Test test_opencv when OpenCV operation fails"""
        mock_imread.side_effect = Exception("OpenCV error")
        
        with patch('builtins.print'):
            result = test_opencv()
            
        self.assertFalse(result)
        
    @patch('cv2.imread')
    @patch('numpy.any')
    def test_test_opencv_edge_detection_check(self, mock_any, mock_imread):
        """Test test_opencv checks edge detection results"""
        # Mock operations
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = test_image
        
        # Mock that edges were detected
        mock_any.return_value = True
        
        with patch('cv2.cvtColor'):
            with patch('cv2.GaussianBlur'):
                with patch('cv2.Canny'):
                    with patch('builtins.print'):
                        result = test_opencv()
                        
        self.assertTrue(result)
        mock_any.assert_called()  # Should check if edges were found


class TestCreateTestScript(unittest.TestCase):
    """Test the create_test_script function"""
    
    @patch('builtins.open', unittest.mock.mock_open())
    def test_create_test_script_success(self, mock_open):
        """Test create_test_script creates file successfully"""
        with patch('builtins.print'):
            create_test_script()
            
        # Should open file for writing
        mock_open.assert_called_once_with('test_geometry_detection.py', 'w')
        
        # Should write content
        mock_file = mock_open()
        self.assertGreater(mock_file.write.call_count, 0)
        
        # Check written content
        written_content = ''.join(call.args[0] for call in mock_file.write.call_args_list)
        
        # Should contain key elements
        self.assertIn('#!/usr/bin/env python3', written_content)
        self.assertIn('import cv2', written_content)
        self.assertIn('import numpy', written_content)
        self.assertIn('from integrated_geometry_system import', written_content)
        self.assertIn('def test_basic_detection():', written_content)
        self.assertIn('if __name__ == "__main__":', written_content)
        
    @patch('builtins.open', unittest.mock.mock_open())
    def test_create_test_script_contains_tests(self, mock_open):
        """Test that created script contains proper test functions"""
        with patch('builtins.print'):
            create_test_script()
            
        written_content = ''.join(
            call.args[0] for call in mock_open().write.call_args_list
        )
        
        # Check for test functions
        self.assertIn('test_basic_detection', written_content)
        self.assertIn('test_camera_backends', written_content)
        self.assertIn('test_performance', written_content)
        
        # Check for shape creation
        self.assertIn('cv2.circle', written_content)
        self.assertIn('cv2.rectangle', written_content)
        
        # Check for assertions
        self.assertIn('assert', written_content)
        self.assertIn('print("All tests passed!")', written_content)
        
    @patch('builtins.open')
    def test_create_test_script_io_error(self, mock_open):
        """Test create_test_script handles IO errors"""
        mock_open.side_effect = IOError("Cannot write file")
        
        with patch('builtins.print') as mock_print:
            # Should not raise exception
            create_test_script()
            
        # Should print error message
        error_printed = any('Error' in str(call) for call in mock_print.call_args_list)
        self.assertTrue(error_printed)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for uv setup scenarios"""
    
    @patch('subprocess.run')
    def test_full_setup_flow_success(self, mock_run):
        """Test complete successful setup flow"""
        # Mock uv check success
        mock_run.side_effect = [
            Mock(returncode=0, stdout="uv 0.1.0"),  # check_uv
            Mock(returncode=0),  # install_with_uv
        ]
        
        # Mock package checks
        with patch('importlib.util.find_spec', return_value=Mock()):
            with patch('uv_compatible_setup.test_opencv', return_value=True):
                with patch('uv_compatible_setup.create_test_script'):
                    with patch('builtins.print'):
                        # Simulate main execution
                        has_uv = check_uv()
                        self.assertTrue(has_uv)
                        
                        packages = ['opencv-python', 'numpy', 'psutil']
                        install_success = install_with_uv(packages)
                        self.assertTrue(install_success)
                        
                        check_success = check_packages(['cv2', 'numpy', 'psutil'])
                        self.assertTrue(check_success)
                        
                        opencv_works = test_opencv()
                        self.assertTrue(opencv_works)
                        
                        create_test_script()
                        
    @patch('subprocess.run')
    def test_setup_flow_no_uv(self, mock_run):
        """Test setup flow when uv is not available"""
        # Mock uv not found
        mock_run.side_effect = FileNotFoundError()
        
        with patch('builtins.print') as mock_print:
            has_uv = check_uv()
            
        self.assertFalse(has_uv)
        
        # Should print installation instructions
        printed_text = ' '.join(str(call) for call in mock_print.call_args_list)
        self.assertIn('curl', printed_text)  # Installation command
        self.assertIn('astral-sh', printed_text)  # Installation URL
        
    @patch('subprocess.run')
    @patch('importlib.util.find_spec')
    def test_setup_flow_partial_packages(self, mock_find_spec, mock_run):
        """Test setup flow with some packages already installed"""
        # Mock uv available
        mock_run.return_value = Mock(returncode=0, stdout="uv 0.1.0")
        
        # Mock some packages installed
        def find_spec_side_effect(name):
            if name == 'numpy':
                return Mock()  # Installed
            return None  # Not installed
            
        mock_find_spec.side_effect = find_spec_side_effect
        
        with patch('builtins.print'):
            # Check packages
            all_installed = check_packages(['numpy', 'cv2', 'psutil'])
            
        self.assertFalse(all_installed)  # Not all packages installed
        
    def test_script_content_validity(self):
        """Test that generated test script content is valid Python"""
        with patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            with patch('builtins.print'):
                create_test_script()
                
        # Get written content
        written_content = ''.join(
            call.args[0] for call in mock_open().write.call_args_list
        )
        
        # Try to compile the script
        try:
            compile(written_content, 'test_script.py', 'exec')
            compilation_success = True
        except SyntaxError:
            compilation_success = False
            
        self.assertTrue(compilation_success, "Generated script has syntax errors")


if __name__ == '__main__':
    unittest.main(verbosity=2)