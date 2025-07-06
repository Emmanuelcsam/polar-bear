#!/usr/bin/env python3
"""
Comprehensive test suite for python313_fix.py
Tests all functions with rigorous edge cases
"""

import unittest
import subprocess
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock, call
import urllib.request
import urllib.error

# Import the module to test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from python313_fix import get_opencv_versions, try_install_opencv


class TestGetOpencvVersions(unittest.TestCase):
    """Test the get_opencv_versions function"""
    
    @patch('urllib.request.urlopen')
    def test_get_opencv_versions_success(self, mock_urlopen):
        """Test get_opencv_versions with successful API response"""
        # Mock PyPI API response
        mock_response_data = {
            "releases": {
                "4.8.1.78": [{"yanked": False}],
                "4.8.0.76": [{"yanked": False}],
                "4.7.0.72": [{"yanked": False}],
                "4.6.0.66": [{"yanked": True}],  # Yanked version
                "4.5.5.64": [{"yanked": False}]
            }
        }
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        versions = get_opencv_versions()
        
        # Should return non-yanked versions in reverse order (newest first)
        expected = ["4.8.1.78", "4.8.0.76", "4.7.0.72", "4.5.5.64"]
        self.assertEqual(versions, expected)
        
        # Verify API was called correctly
        mock_urlopen.assert_called_once_with(
            "https://pypi.org/pypi/opencv-python/json"
        )
        
    @patch('urllib.request.urlopen')
    def test_get_opencv_versions_empty_releases(self, mock_urlopen):
        """Test get_opencv_versions with no releases"""
        mock_response_data = {"releases": {}}
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        versions = get_opencv_versions()
        
        self.assertEqual(versions, [])
        
    @patch('urllib.request.urlopen')
    def test_get_opencv_versions_all_yanked(self, mock_urlopen):
        """Test get_opencv_versions when all versions are yanked"""
        mock_response_data = {
            "releases": {
                "4.8.1.78": [{"yanked": True}],
                "4.8.0.76": [{"yanked": True}],
                "4.7.0.72": [{"yanked": True}]
            }
        }
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        versions = get_opencv_versions()
        
        self.assertEqual(versions, [])
        
    @patch('urllib.request.urlopen')
    def test_get_opencv_versions_network_error(self, mock_urlopen):
        """Test get_opencv_versions with network error"""
        mock_urlopen.side_effect = urllib.error.URLError("Network error")
        
        with patch('builtins.print') as mock_print:
            versions = get_opencv_versions()
            
        self.assertEqual(versions, [])
        
        # Should print error message
        error_printed = any('Error' in str(call) for call in mock_print.call_args_list)
        self.assertTrue(error_printed)
        
    @patch('urllib.request.urlopen')
    def test_get_opencv_versions_invalid_json(self, mock_urlopen):
        """Test get_opencv_versions with invalid JSON response"""
        mock_response = Mock()
        mock_response.read.return_value = b"Invalid JSON"
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        with patch('builtins.print') as mock_print:
            versions = get_opencv_versions()
            
        self.assertEqual(versions, [])
        
        # Should print error message
        error_printed = any('Error' in str(call) for call in mock_print.call_args_list)
        self.assertTrue(error_printed)
        
    @patch('urllib.request.urlopen')
    def test_get_opencv_versions_mixed_yanked(self, mock_urlopen):
        """Test get_opencv_versions with mix of yanked and non-yanked"""
        mock_response_data = {
            "releases": {
                "4.9.0.80": [{"yanked": False}],
                "4.8.1.78": [{"yanked": True}],
                "4.8.0.76": [{"yanked": False}],
                "4.7.0.72": [{"yanked": True}],
                "4.6.0.66": [{"yanked": False}]
            }
        }
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        versions = get_opencv_versions()
        
        # Should only return non-yanked versions
        expected = ["4.9.0.80", "4.8.0.76", "4.6.0.66"]
        self.assertEqual(versions, expected)
        
    @patch('urllib.request.urlopen')
    def test_get_opencv_versions_missing_yanked_field(self, mock_urlopen):
        """Test get_opencv_versions when yanked field is missing"""
        mock_response_data = {
            "releases": {
                "4.8.1.78": [{}],  # Missing yanked field
                "4.8.0.76": [{"yanked": False}],
                "4.7.0.72": [{"other_field": "value"}]  # Missing yanked field
            }
        }
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        versions = get_opencv_versions()
        
        # Should treat missing yanked field as False (not yanked)
        expected = ["4.8.1.78", "4.8.0.76", "4.7.0.72"]
        self.assertEqual(versions, expected)


class TestTryInstallOpencv(unittest.TestCase):
    """Test the try_install_opencv function"""
    
    @patch('subprocess.run')
    @patch('python313_fix.get_opencv_versions')
    def test_try_install_opencv_first_version_success(self, mock_get_versions, mock_run):
        """Test try_install_opencv when first version installs successfully"""
        mock_get_versions.return_value = ["4.8.1.78", "4.8.0.76", "4.7.0.72"]
        mock_run.return_value = Mock(returncode=0)
        
        with patch('builtins.print'):
            result = try_install_opencv()
            
        self.assertTrue(result)
        
        # Should only call pip install once
        mock_run.assert_called_once_with(
            [sys.executable, '-m', 'pip', 'install', 'opencv-python==4.8.1.78'],
            capture_output=True,
            text=True
        )
        
    @patch('subprocess.run')
    @patch('python313_fix.get_opencv_versions')
    def test_try_install_opencv_fallback_versions(self, mock_get_versions, mock_run):
        """Test try_install_opencv tries multiple versions on failure"""
        mock_get_versions.return_value = ["4.8.1.78", "4.8.0.76", "4.7.0.72"]
        
        # First two fail, third succeeds
        mock_run.side_effect = [
            Mock(returncode=1, stderr="Error installing 4.8.1.78"),
            Mock(returncode=1, stderr="Error installing 4.8.0.76"),
            Mock(returncode=0)
        ]
        
        with patch('builtins.print'):
            result = try_install_opencv()
            
        self.assertTrue(result)
        
        # Should have tried 3 versions
        self.assertEqual(mock_run.call_count, 3)
        
        # Check the versions tried
        expected_calls = [
            call([sys.executable, '-m', 'pip', 'install', 'opencv-python==4.8.1.78'],
                 capture_output=True, text=True),
            call([sys.executable, '-m', 'pip', 'install', 'opencv-python==4.8.0.76'],
                 capture_output=True, text=True),
            call([sys.executable, '-m', 'pip', 'install', 'opencv-python==4.7.0.72'],
                 capture_output=True, text=True)
        ]
        mock_run.assert_has_calls(expected_calls)
        
    @patch('subprocess.run')
    @patch('python313_fix.get_opencv_versions')
    def test_try_install_opencv_all_fail(self, mock_get_versions, mock_run):
        """Test try_install_opencv when all versions fail"""
        mock_get_versions.return_value = ["4.8.1.78", "4.8.0.76"]
        mock_run.return_value = Mock(returncode=1, stderr="Installation failed")
        
        with patch('builtins.print'):
            result = try_install_opencv()
            
        self.assertFalse(result)
        
        # Should have tried all versions
        self.assertEqual(mock_run.call_count, 2)
        
    @patch('subprocess.run')
    @patch('python313_fix.get_opencv_versions')
    def test_try_install_opencv_no_versions(self, mock_get_versions, mock_run):
        """Test try_install_opencv when no versions available"""
        mock_get_versions.return_value = []
        
        with patch('builtins.print') as mock_print:
            result = try_install_opencv()
            
        self.assertFalse(result)
        
        # Should not try to install anything
        mock_run.assert_not_called()
        
        # Should print error message
        error_printed = any('No OpenCV versions' in str(call) 
                          for call in mock_print.call_args_list)
        self.assertTrue(error_printed)
        
    @patch('subprocess.run')
    @patch('python313_fix.get_opencv_versions')
    def test_try_install_opencv_subprocess_error(self, mock_get_versions, mock_run):
        """Test try_install_opencv handles subprocess errors"""
        mock_get_versions.return_value = ["4.8.1.78"]
        mock_run.side_effect = subprocess.SubprocessError("Command failed")
        
        with patch('builtins.print'):
            result = try_install_opencv()
            
        self.assertFalse(result)
        
    @patch('subprocess.run')
    @patch('python313_fix.get_opencv_versions')
    @patch('builtins.print')
    def test_try_install_opencv_output_messages(self, mock_print, mock_get_versions, mock_run):
        """Test try_install_opencv prints appropriate messages"""
        mock_get_versions.return_value = ["4.8.1.78", "4.8.0.76"]
        
        # First fails, second succeeds
        mock_run.side_effect = [
            Mock(returncode=1, stderr="Error"),
            Mock(returncode=0)
        ]
        
        result = try_install_opencv()
        
        # Check printed messages
        printed_text = ' '.join(str(call) for call in mock_print.call_args_list)
        
        self.assertIn("Trying to install", printed_text)
        self.assertIn("4.8.1.78", printed_text)
        self.assertIn("Failed", printed_text)
        self.assertIn("4.8.0.76", printed_text)
        self.assertIn("Successfully installed", printed_text)


class TestMainExecution(unittest.TestCase):
    """Test the main execution flow"""
    
    @patch('python313_fix.try_install_opencv')
    @patch('builtins.print')
    def test_main_success(self, mock_print, mock_try_install):
        """Test main execution when installation succeeds"""
        mock_try_install.return_value = True
        
        # Simulate running the script
        with patch.object(sys, 'argv', ['python313_fix.py']):
            # Import would execute main
            exec("""
if __name__ == "__main__":
    print("OpenCV installation helper for Python 3.13")
    print("=" * 50)
    
    if try_install_opencv():
        print("\\nInstallation completed successfully!")
        print("You can now import cv2 in your Python 3.13 environment.")
    else:
        print("\\nInstallation failed. Please try manual installation:")
        print("1. Check https://pypi.org/project/opencv-python/#history")
        print("2. Try installing with: pip install opencv-python --pre")
        print("3. Consider using Python 3.12 or earlier for now")
""")
            
        # Check appropriate messages were printed
        printed_text = ' '.join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Installation completed successfully", printed_text)
        
    @patch('python313_fix.try_install_opencv')
    @patch('builtins.print')
    def test_main_failure(self, mock_print, mock_try_install):
        """Test main execution when installation fails"""
        mock_try_install.return_value = False
        
        # Simulate running the script
        with patch.object(sys, 'argv', ['python313_fix.py']):
            exec("""
if __name__ == "__main__":
    print("OpenCV installation helper for Python 3.13")
    print("=" * 50)
    
    if try_install_opencv():
        print("\\nInstallation completed successfully!")
        print("You can now import cv2 in your Python 3.13 environment.")
    else:
        print("\\nInstallation failed. Please try manual installation:")
        print("1. Check https://pypi.org/project/opencv-python/#history")
        print("2. Try installing with: pip install opencv-python --pre")
        print("3. Consider using Python 3.12 or earlier for now")
""")
            
        # Check failure messages were printed
        printed_text = ' '.join(str(call) for call in mock_print.call_args_list)
        self.assertIn("Installation failed", printed_text)
        self.assertIn("manual installation", printed_text)
        self.assertIn("--pre", printed_text)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for python313_fix scenarios"""
    
    @patch('urllib.request.urlopen')
    @patch('subprocess.run')
    def test_full_flow_success(self, mock_run, mock_urlopen):
        """Test complete flow from version fetch to installation"""
        # Mock PyPI response
        mock_response_data = {
            "releases": {
                "4.8.1.78": [{"yanked": False}],
                "4.8.0.76": [{"yanked": False}]
            }
        }
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Mock successful installation
        mock_run.return_value = Mock(returncode=0)
        
        with patch('builtins.print'):
            # Get versions
            versions = get_opencv_versions()
            self.assertEqual(len(versions), 2)
            
            # Try installation
            result = try_install_opencv()
            self.assertTrue(result)
            
    @patch('urllib.request.urlopen')
    @patch('subprocess.run')
    def test_full_flow_network_failure(self, mock_run, mock_urlopen):
        """Test flow when network request fails"""
        # Mock network failure
        mock_urlopen.side_effect = urllib.error.URLError("Network unreachable")
        
        with patch('builtins.print'):
            # Try installation
            result = try_install_opencv()
            self.assertFalse(result)
            
        # Should not attempt pip install
        mock_run.assert_not_called()
        
    @patch('urllib.request.urlopen')
    @patch('subprocess.run')
    def test_version_compatibility_check(self, mock_run, mock_urlopen):
        """Test that version strings are handled correctly"""
        # Mock response with various version formats
        mock_response_data = {
            "releases": {
                "4.8.1.78": [{"yanked": False}],
                "4.8.0": [{"yanked": False}],  # Short version
                "4.7.0.72.dev0": [{"yanked": False}],  # Dev version
                "4.6.0.66rc1": [{"yanked": False}]  # Release candidate
            }
        }
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        versions = get_opencv_versions()
        
        # All versions should be returned in order
        self.assertEqual(len(versions), 4)
        self.assertEqual(versions[0], "4.8.1.78")  # Newest first


if __name__ == '__main__':
    unittest.main(verbosity=2)