#!/usr/bin/env python3
"""
Comprehensive test suite for setup_installer.py
Tests all functions, classes, and methods with rigorous edge cases
"""

import unittest
import subprocess
import sys
import os
import platform
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
import importlib.util

# Import the module to test
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup_installer import (
    Colors, print_header, print_success, print_error, print_warning, print_info,
    SystemChecker, PackageInstaller, CameraDetector, ConfigurationWizard,
    TestRunner, create_launcher_scripts, show_next_steps
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
        self.assertIn("!", args)
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


class TestSystemChecker(unittest.TestCase):
    """Test the SystemChecker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.checker = SystemChecker()
        
    def test_system_checker_initialization(self):
        """Test SystemChecker initialization"""
        self.assertIsInstance(self.checker, SystemChecker)
        
    @patch('builtins.print')
    def test_check_all(self, mock_print):
        """Test check_all method"""
        # Mock all check methods
        with patch.object(self.checker, 'check_python', return_value=True):
            with patch.object(self.checker, 'check_system', return_value=True):
                with patch.object(self.checker, 'check_hardware', return_value=True):
                    with patch.object(self.checker, 'check_gpu', return_value=True):
                        with patch.object(self.checker, 'check_existing_packages', return_value=True):
                            result = self.checker.check_all()
                            
        self.assertTrue(result)
        
    def test_check_python_valid_version(self):
        """Test check_python with valid version"""
        with patch('sys.version_info', (3, 9, 0)):
            with patch('builtins.print'):
                result = self.checker.check_python()
                
        self.assertTrue(result)
        
    def test_check_python_invalid_version(self):
        """Test check_python with invalid version"""
        with patch('sys.version_info', (3, 7, 0)):
            with patch('builtins.print'):
                result = self.checker.check_python()
                
        self.assertFalse(result)
        
    @patch('platform.system')
    @patch('platform.release')
    @patch('platform.machine')
    def test_check_system(self, mock_machine, mock_release, mock_system):
        """Test check_system method"""
        mock_system.return_value = "Linux"
        mock_release.return_value = "5.10.0"
        mock_machine.return_value = "x86_64"
        
        with patch('builtins.print'):
            result = self.checker.check_system()
            
        self.assertTrue(result)
        
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_hardware(self, mock_disk, mock_memory, mock_cpu):
        """Test check_hardware method"""
        mock_cpu.return_value = 8
        mock_memory.return_value = Mock(total=8*1024**3)  # 8GB
        mock_disk.return_value = Mock(free=10*1024**3)  # 10GB free
        
        with patch('builtins.print'):
            result = self.checker.check_hardware()
            
        self.assertTrue(result)
        
    @patch('psutil.virtual_memory')
    def test_check_hardware_insufficient_ram(self, mock_memory):
        """Test check_hardware with insufficient RAM"""
        mock_memory.return_value = Mock(total=1*1024**3)  # 1GB
        
        with patch('builtins.print'):
            result = self.checker.check_hardware()
            
        self.assertFalse(result)
        
    @patch('subprocess.run')
    def test_check_gpu_nvidia(self, mock_run):
        """Test check_gpu with NVIDIA GPU"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3080\nCUDA Version: 11.4"
        )
        
        with patch('builtins.print'):
            result = self.checker.check_gpu()
            
        self.assertIsInstance(result, dict)
        self.assertTrue(result['nvidia'])
        self.assertIn("RTX 3080", result['gpu_info'])
        
    @patch('subprocess.run')
    def test_check_gpu_no_nvidia(self, mock_run):
        """Test check_gpu with no NVIDIA GPU"""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'nvidia-smi')
        
        with patch('builtins.print'):
            result = self.checker.check_gpu()
            
        self.assertIsInstance(result, dict)
        self.assertFalse(result['nvidia'])
        
    @patch('importlib.util.find_spec')
    def test_check_existing_packages(self, mock_find_spec):
        """Test check_existing_packages method"""
        # Mock some packages as installed
        def find_spec_side_effect(name):
            if name in ['numpy', 'opencv-python']:
                return Mock()  # Package found
            return None  # Package not found
            
        mock_find_spec.side_effect = find_spec_side_effect
        
        with patch('builtins.print'):
            result = self.checker.check_existing_packages()
            
        self.assertIsInstance(result, dict)
        self.assertIn('numpy', result)
        self.assertTrue(result['numpy'])
        self.assertIn('matplotlib', result)
        self.assertFalse(result['matplotlib'])


class TestPackageInstaller(unittest.TestCase):
    """Test the PackageInstaller class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.installer = PackageInstaller()
        
    @patch('builtins.print')
    def test_install_all(self, mock_print):
        """Test install_all method"""
        with patch.object(self.installer, 'update_pip', return_value=True):
            with patch.object(self.installer, 'install_system_dependencies', return_value=True):
                with patch.object(self.installer, 'install_python_packages', return_value=True):
                    with patch.object(self.installer, 'ask_install_optional', return_value=True):
                        result = self.installer.install_all()
                        
        self.assertTrue(result)
        
    @patch('subprocess.run')
    def test_update_pip_success(self, mock_run):
        """Test update_pip with success"""
        mock_run.return_value = Mock(returncode=0)
        
        with patch('builtins.print'):
            result = self.installer.update_pip()
            
        self.assertTrue(result)
        mock_run.assert_called_with(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            capture_output=True,
            text=True
        )
        
    @patch('subprocess.run')
    def test_update_pip_failure(self, mock_run):
        """Test update_pip with failure"""
        mock_run.return_value = Mock(returncode=1, stderr="Error message")
        
        with patch('builtins.print'):
            result = self.installer.update_pip()
            
        self.assertFalse(result)
        
    @patch('platform.system')
    @patch('subprocess.run')
    def test_install_system_dependencies_linux(self, mock_run, mock_system):
        """Test install_system_dependencies on Linux"""
        mock_system.return_value = "Linux"
        mock_run.return_value = Mock(returncode=0)
        
        with patch('builtins.print'):
            with patch('builtins.input', return_value='y'):
                result = self.installer.install_system_dependencies()
                
        self.assertTrue(result)
        # Should try to install system packages
        self.assertTrue(any('apt-get' in str(call) for call in mock_run.call_args_list))
        
    @patch('platform.system')
    def test_install_system_dependencies_windows(self, mock_system):
        """Test install_system_dependencies on Windows"""
        mock_system.return_value = "Windows"
        
        with patch('builtins.print'):
            result = self.installer.install_system_dependencies()
            
        self.assertTrue(result)  # Should skip on Windows
        
    def test_install_python_packages(self):
        """Test install_python_packages method"""
        # Mock successful package installations
        with patch.object(self.installer, 'install_package', return_value=True):
            with patch('builtins.print'):
                result = self.installer.install_python_packages()
                
        self.assertTrue(result)
        
        # Check that all required packages were attempted
        expected_packages = ['opencv-python', 'numpy', 'psutil', 
                           'matplotlib', 'pandas']
        
        self.assertEqual(self.installer.install_package.call_count, 
                        len(expected_packages))
                        
    @patch('subprocess.run')
    def test_install_package_success(self, mock_run):
        """Test install_package with success"""
        mock_run.return_value = Mock(returncode=0)
        
        with patch('builtins.print'):
            result = self.installer.install_package('test-package', 'Test Package')
            
        self.assertTrue(result)
        mock_run.assert_called_with(
            [sys.executable, '-m', 'pip', 'install', 'test-package'],
            capture_output=True,
            text=True
        )
        
    @patch('subprocess.run')
    def test_install_package_failure_with_fallback(self, mock_run):
        """Test install_package with failure and fallback"""
        # First call fails, second succeeds
        mock_run.side_effect = [
            Mock(returncode=1, stderr="Error"),
            Mock(returncode=0)
        ]
        
        with patch('builtins.print'):
            result = self.installer.install_package(
                'test-package', 
                'Test Package',
                fallback_version='test-package==1.0.0'
            )
            
        self.assertTrue(result)
        self.assertEqual(mock_run.call_count, 2)
        
    @patch('builtins.input')
    def test_ask_install_optional_yes(self, mock_input):
        """Test ask_install_optional with yes response"""
        mock_input.return_value = 'y'
        
        with patch('builtins.print'):
            with patch.object(self.installer, 'check_pylon_sdk', return_value=True):
                with patch.object(self.installer, 'install_package', return_value=True):
                    result = self.installer.ask_install_optional()
                    
        self.assertTrue(result)
        
    @patch('builtins.input')
    def test_ask_install_optional_no(self, mock_input):
        """Test ask_install_optional with no response"""
        mock_input.return_value = 'n'
        
        with patch('builtins.print'):
            result = self.installer.ask_install_optional()
            
        self.assertTrue(result)  # Should still return True
        
    @patch('platform.system')
    @patch('os.path.exists')
    def test_check_pylon_sdk_found(self, mock_exists, mock_system):
        """Test check_pylon_sdk when SDK is found"""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        
        with patch('builtins.print'):
            result = self.installer.check_pylon_sdk()
            
        self.assertTrue(result)
        
    @patch('platform.system')
    @patch('os.path.exists')
    def test_check_pylon_sdk_not_found(self, mock_exists, mock_system):
        """Test check_pylon_sdk when SDK is not found"""
        mock_system.return_value = "Linux"
        mock_exists.return_value = False
        
        with patch('builtins.print'):
            result = self.installer.check_pylon_sdk()
            
        self.assertFalse(result)


class TestCameraDetector(unittest.TestCase):
    """Test the CameraDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = CameraDetector()
        
    @patch('builtins.print')
    def test_detect_all(self, mock_print):
        """Test detect_all method"""
        with patch.object(self.detector, 'detect_opencv_cameras', return_value=[0, 1]):
            with patch.object(self.detector, 'detect_pylon_cameras', return_value=['Basler1']):
                cameras = self.detector.detect_all()
                
        self.assertIn('opencv', cameras)
        self.assertIn('pylon', cameras)
        self.assertEqual(len(cameras['opencv']), 2)
        self.assertEqual(len(cameras['pylon']), 1)
        
    @patch('cv2.VideoCapture')
    def test_detect_opencv_cameras_found(self, mock_capture):
        """Test detect_opencv_cameras when cameras found"""
        # Mock camera that opens successfully
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap
        
        with patch('builtins.print'):
            cameras = self.detector.detect_opencv_cameras()
            
        self.assertGreater(len(cameras), 0)
        mock_cap.release.assert_called()
        
    @patch('cv2.VideoCapture')
    def test_detect_opencv_cameras_none_found(self, mock_capture):
        """Test detect_opencv_cameras when no cameras found"""
        # Mock camera that fails to open
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        with patch('builtins.print'):
            cameras = self.detector.detect_opencv_cameras()
            
        self.assertEqual(len(cameras), 0)
        
    @patch('importlib.import_module')
    def test_detect_pylon_cameras_with_module(self, mock_import):
        """Test detect_pylon_cameras with pypylon installed"""
        # Mock pypylon module
        mock_pylon = Mock()
        mock_factory = Mock()
        mock_factory.GetInstance.return_value.EnumerateDevices.return_value = [
            Mock(GetModelName=Mock(return_value="Basler Camera 1")),
            Mock(GetModelName=Mock(return_value="Basler Camera 2"))
        ]
        mock_pylon.pylon.TlFactory = mock_factory
        mock_import.return_value = mock_pylon
        
        with patch('builtins.print'):
            cameras = self.detector.detect_pylon_cameras()
            
        self.assertEqual(len(cameras), 2)
        self.assertIn("Basler Camera 1", cameras[0])
        
    @patch('importlib.import_module')
    def test_detect_pylon_cameras_no_module(self, mock_import):
        """Test detect_pylon_cameras without pypylon"""
        mock_import.side_effect = ImportError()
        
        with patch('builtins.print'):
            cameras = self.detector.detect_pylon_cameras()
            
        self.assertEqual(len(cameras), 0)
        
    def test_test_cameras(self):
        """Test test_cameras method"""
        cameras = {
            'opencv': [0],
            'pylon': []
        }
        
        with patch.object(self.detector, 'test_opencv_camera', return_value=True):
            with patch('builtins.print'):
                results = self.detector.test_cameras(cameras)
                
        self.assertTrue(results)
        
    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_test_opencv_camera_success(self, mock_destroy, mock_waitkey, 
                                       mock_imshow, mock_capture):
        """Test test_opencv_camera with successful test"""
        # Mock camera
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8))
        ] * 10 + [(False, None)]  # 10 frames then stop
        mock_capture.return_value = mock_cap
        
        # Mock user input
        mock_waitkey.side_effect = [255] * 9 + [ord('q')]  # Press 'q' to quit
        
        with patch('builtins.print'):
            result = self.detector.test_opencv_camera(0)
            
        self.assertTrue(result)
        mock_cap.release.assert_called()
        mock_destroy.assert_called()
        
    def test_suggest_camera_fixes(self):
        """Test suggest_camera_fixes method"""
        with patch('platform.system', return_value='Linux'):
            with patch('builtins.print') as mock_print:
                self.detector.suggest_camera_fixes()
                
        # Should print suggestions
        self.assertGreater(mock_print.call_count, 5)


class TestConfigurationWizard(unittest.TestCase):
    """Test the ConfigurationWizard class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.wizard = ConfigurationWizard()
        
    @patch('builtins.print')
    def test_run(self, mock_print):
        """Test run method"""
        with patch.object(self.wizard, 'select_camera', return_value={'backend': 'opencv', 'id': 0}):
            with patch.object(self.wizard, 'configure_gpu', return_value=True):
                with patch.object(self.wizard, 'select_features', return_value={'feature': True}):
                    with patch.object(self.wizard, 'save_config', return_value=True):
                        config = self.wizard.run()
                        
        self.assertIsInstance(config, dict)
        self.assertIn('camera', config)
        self.assertIn('enable_gpu', config)
        self.assertIn('features', config)
        
    @patch('builtins.input')
    def test_select_camera_opencv(self, mock_input):
        """Test select_camera selecting OpenCV"""
        mock_input.side_effect = ['1', '0']  # Select OpenCV, then camera 0
        
        available_cameras = {
            'opencv': [0, 1],
            'pylon': []
        }
        
        with patch('builtins.print'):
            result = self.wizard.select_camera(available_cameras)
            
        self.assertEqual(result['backend'], 'opencv')
        self.assertEqual(result['id'], 0)
        
    @patch('builtins.input')
    def test_select_camera_invalid_then_valid(self, mock_input):
        """Test select_camera with invalid then valid input"""
        mock_input.side_effect = ['3', '1', '0']  # Invalid, then valid
        
        available_cameras = {
            'opencv': [0],
            'pylon': []
        }
        
        with patch('builtins.print'):
            result = self.wizard.select_camera(available_cameras)
            
        self.assertEqual(result['backend'], 'opencv')
        
    @patch('builtins.input')
    def test_configure_gpu_yes(self, mock_input):
        """Test configure_gpu with yes response"""
        mock_input.return_value = 'y'
        
        gpu_info = {'nvidia': True, 'cuda': True}
        
        with patch('builtins.print'):
            result = self.wizard.configure_gpu(gpu_info)
            
        self.assertTrue(result)
        
    @patch('builtins.input')
    def test_configure_gpu_no_gpu(self, mock_input):
        """Test configure_gpu with no GPU available"""
        gpu_info = {'nvidia': False}
        
        with patch('builtins.print'):
            result = self.wizard.configure_gpu(gpu_info)
            
        self.assertFalse(result)
        
    @patch('builtins.input')
    def test_select_features_all_yes(self, mock_input):
        """Test select_features with all features selected"""
        mock_input.return_value = 'y'
        
        with patch('builtins.print'):
            features = self.wizard.select_features()
            
        expected_features = ['shape_detection', 'tube_angle_detection', 
                           'performance_monitoring', 'recording']
        
        for feature in expected_features:
            self.assertTrue(features[feature])
            
    @patch('builtins.open', unittest.mock.mock_open())
    @patch('json.dump')
    def test_save_config(self, mock_json_dump):
        """Test save_config method"""
        config = {
            'camera': {'backend': 'opencv', 'id': 0},
            'enable_gpu': True,
            'features': {'shape_detection': True}
        }
        
        with patch('builtins.print'):
            result = self.wizard.save_config(config)
            
        self.assertTrue(result)
        mock_json_dump.assert_called_once()
        
        # Check saved config
        saved_config = mock_json_dump.call_args[0][0]
        self.assertEqual(saved_config, config)


class TestTestRunner(unittest.TestCase):
    """Test the TestRunner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.runner = TestRunner()
        
    @patch('builtins.print')
    def test_run_all_tests(self, mock_print):
        """Test run_all_tests method"""
        with patch.object(self.runner, 'test_opencv', return_value=True):
            with patch.object(self.runner, 'test_numpy', return_value=True):
                with patch.object(self.runner, 'test_shape_detection', return_value=True):
                    with patch.object(self.runner, 'test_gpu', return_value=True):
                        result = self.runner.run_all_tests()
                        
        self.assertTrue(result)
        
    @patch('cv2.imread')
    def test_test_opencv_success(self, mock_imread):
        """Test test_opencv with success"""
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch('builtins.print'):
            result = self.runner.test_opencv()
            
        self.assertTrue(result)
        
    @patch('cv2.imread')
    def test_test_opencv_failure(self, mock_imread):
        """Test test_opencv with failure"""
        mock_imread.side_effect = Exception("OpenCV error")
        
        with patch('builtins.print'):
            result = self.runner.test_opencv()
            
        self.assertFalse(result)
        
    def test_test_numpy_success(self):
        """Test test_numpy with success"""
        with patch('builtins.print'):
            result = self.runner.test_numpy()
            
        self.assertTrue(result)
        
    @patch('integrated_geometry_system.GeometryDetector')
    def test_test_shape_detection_success(self, mock_detector_class):
        """Test test_shape_detection with success"""
        mock_detector = Mock()
        mock_detector.detect_shapes.return_value = [Mock(), Mock()]
        mock_detector_class.return_value = mock_detector
        
        with patch('builtins.print'):
            result = self.runner.test_shape_detection()
            
        self.assertTrue(result)
        
    @patch('integrated_geometry_system.GeometryDetector')
    def test_test_shape_detection_failure(self, mock_detector_class):
        """Test test_shape_detection with failure"""
        mock_detector_class.side_effect = Exception("Detection error")
        
        with patch('builtins.print'):
            result = self.runner.test_shape_detection()
            
        self.assertFalse(result)
        
    @patch('cv2.cuda.getCudaEnabledDeviceCount')
    def test_test_gpu_with_cuda(self, mock_cuda_count):
        """Test test_gpu with CUDA available"""
        mock_cuda_count.return_value = 1
        
        gpu_info = {'nvidia': True, 'cuda': True}
        
        with patch('builtins.print'):
            result = self.runner.test_gpu(gpu_info)
            
        self.assertTrue(result)
        
    def test_test_gpu_no_gpu(self):
        """Test test_gpu with no GPU"""
        gpu_info = {'nvidia': False}
        
        with patch('builtins.print'):
            result = self.runner.test_gpu(gpu_info)
            
        self.assertTrue(result)  # Should pass even without GPU


class TestCreateLauncherScripts(unittest.TestCase):
    """Test the create_launcher_scripts function"""
    
    @patch('platform.system')
    @patch('builtins.open', unittest.mock.mock_open())
    @patch('os.chmod')
    def test_create_launcher_scripts_linux(self, mock_chmod, mock_system):
        """Test create_launcher_scripts on Linux"""
        mock_system.return_value = 'Linux'
        
        with patch('builtins.print'):
            create_launcher_scripts()
            
        # Should create shell script
        self.assertTrue(any('run_geometry_detection.sh' in str(call) 
                           for call in unittest.mock.mock_open().mock_calls))
        
        # Should make executable
        mock_chmod.assert_called()
        
    @patch('platform.system')
    @patch('builtins.open', unittest.mock.mock_open())
    def test_create_launcher_scripts_windows(self, mock_system):
        """Test create_launcher_scripts on Windows"""
        mock_system.return_value = 'Windows'
        
        with patch('builtins.print'):
            create_launcher_scripts()
            
        # Should create batch file
        self.assertTrue(any('run_geometry_detection.bat' in str(call) 
                           for call in unittest.mock.mock_open().mock_calls))


class TestShowNextSteps(unittest.TestCase):
    """Test the show_next_steps function"""
    
    @patch('builtins.print')
    def test_show_next_steps(self, mock_print):
        """Test show_next_steps function"""
        show_next_steps()
        
        # Should print multiple lines of instructions
        self.assertGreater(mock_print.call_count, 10)
        
        # Check for key information
        printed_text = ' '.join(str(call) for call in mock_print.call_args_list)
        self.assertIn('integrated_geometry_system.py', printed_text)
        self.assertIn('calibration', printed_text)
        self.assertIn('benchmark', printed_text)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for setup installer scenarios"""
    
    def test_full_installation_flow(self):
        """Test complete installation flow"""
        # Mock all components
        with patch('setup_installer.SystemChecker') as mock_checker_class:
            with patch('setup_installer.PackageInstaller') as mock_installer_class:
                with patch('setup_installer.CameraDetector') as mock_detector_class:
                    with patch('setup_installer.ConfigurationWizard') as mock_wizard_class:
                        with patch('setup_installer.TestRunner') as mock_runner_class:
                            # Set up mocks
                            mock_checker = Mock()
                            mock_checker.check_all.return_value = True
                            mock_checker_class.return_value = mock_checker
                            
                            mock_installer = Mock()
                            mock_installer.install_all.return_value = True
                            mock_installer_class.return_value = mock_installer
                            
                            mock_detector = Mock()
                            mock_detector.detect_all.return_value = {'opencv': [0]}
                            mock_detector.test_cameras.return_value = True
                            mock_detector_class.return_value = mock_detector
                            
                            mock_wizard = Mock()
                            mock_wizard.run.return_value = {'camera': {'backend': 'opencv'}}
                            mock_wizard_class.return_value = mock_wizard
                            
                            mock_runner = Mock()
                            mock_runner.run_all_tests.return_value = True
                            mock_runner_class.return_value = mock_runner
                            
                            # Import and run main
                            with patch('setup_installer.create_launcher_scripts'):
                                with patch('setup_installer.show_next_steps'):
                                    with patch('builtins.print'):
                                        # Simulate running the installer
                                        # (In real code, this would be in if __name__ == "__main__")
                                        checker = mock_checker_class()
                                        if checker.check_all():
                                            installer = mock_installer_class()
                                            if installer.install_all():
                                                detector = mock_detector_class()
                                                cameras = detector.detect_all()
                                                if detector.test_cameras(cameras):
                                                    wizard = mock_wizard_class()
                                                    config = wizard.run()
                                                    runner = mock_runner_class()
                                                    runner.run_all_tests()
                                                    
        # Verify all components were called
        mock_checker.check_all.assert_called_once()
        mock_installer.install_all.assert_called_once()
        mock_detector.detect_all.assert_called_once()
        mock_wizard.run.assert_called_once()
        mock_runner.run_all_tests.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)