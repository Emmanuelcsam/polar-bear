"""
Unit tests for auto_installer_refactored module
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
from unittest.mock import patch, MagicMock
import subprocess
import auto_installer_refactored as auto_installer


class TestAutoInstaller(unittest.TestCase):
    """Test cases for auto_installer_refactored module"""
    
    def test_check_library_installed_success(self):
        """Test successful library check"""
        # Test with a library that should be installed (os)
        result = auto_installer.check_library_installed('os')
        self.assertTrue(result)
    
    def test_check_library_installed_failure(self):
        """Test failed library check"""
        # Test with a non-existent library
        result = auto_installer.check_library_installed('nonexistent_library_12345')
        self.assertFalse(result)
    
    def test_check_library_with_dash(self):
        """Test library name with dash conversion"""
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = True
            auto_installer.check_library_installed('test-library')
            mock_import.assert_called_with('test_library')
    
    @patch('subprocess.check_call')
    def test_install_library_success(self, mock_subprocess):
        """Test successful library installation"""
        mock_subprocess.return_value = None
        result = auto_installer.install_library('test_library')
        self.assertTrue(result)
        mock_subprocess.assert_called_once()
    
    @patch('subprocess.check_call')
    def test_install_library_failure(self, mock_subprocess):
        """Test failed library installation"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'pip')
        result = auto_installer.install_library('test_library')
        self.assertFalse(result)
    
    @patch('auto_installer_refactored.check_library_installed')
    @patch('auto_installer_refactored.install_library')
    def test_auto_install_dependencies(self, mock_install, mock_check):
        """Test auto-install dependencies function"""
        # Setup mocks
        mock_check.side_effect = [True, False, False]  # First lib installed, others not
        mock_install.side_effect = [True, False]  # Second installs successfully, third fails
        
        libraries = ['lib1', 'lib2', 'lib3']
        results = auto_installer.auto_install_dependencies(libraries)
        
        self.assertEqual(results['lib1'], 'already_installed')
        self.assertEqual(results['lib2'], 'installed')
        self.assertEqual(results['lib3'], 'failed')
    
    def test_auto_install_dependencies_default_list(self):
        """Test auto-install with default library list"""
        with patch('auto_installer_refactored.check_library_installed') as mock_check:
            mock_check.return_value = True  # All libraries already installed
            
            results = auto_installer.auto_install_dependencies()
            
            # Should use default list
            self.assertIn('numpy', results)
            self.assertIn('torch', results)
            self.assertEqual(len(results), 6)  # Default list has 6 libraries


if __name__ == '__main__':
    unittest.main()
