"""
Unit tests for the main application module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import json
import tempfile
import shutil
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "current-process"))

from app import DataAcquisition, EnhancedApplication
from detection import Defect
from config_manager import get_config


class TestDataAcquisition:
    """Test DataAcquisition class"""
    
    @pytest.fixture
    def data_acquisition(self):
        """Create DataAcquisition instance"""
        return DataAcquisition()
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image"""
        return np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_zones(self):
        """Create sample zone masks"""
        h, w = 200, 200
        zones = {
            'core': np.zeros((h, w), dtype=np.uint8),
            'cladding': np.zeros((h, w), dtype=np.uint8),
            'ferrule': np.zeros((h, w), dtype=np.uint8)
        }
        
        # Create circular zones
        center = (100, 100)
        cv2.circle(zones['core'], center, 30, 255, -1)
        cv2.circle(zones['cladding'], center, 60, 255, -1)
        cv2.circle(zones['cladding'], center, 30, 0, -1)
        cv2.circle(zones['ferrule'], center, 90, 255, -1)
        cv2.circle(zones['ferrule'], center, 60, 0, -1)
        
        return zones
    
    @pytest.fixture
    def sample_defects(self):
        """Create sample defects"""
        return [
            Defect(
                type="scratch",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.8,
                confidence=0.9,
                zone="core",
                area=400,
                detection_method="test_method"
            ),
            Defect(
                type="pit",
                location=(150, 150),
                bbox=(145, 145, 10, 10),
                severity=0.3,
                confidence=0.7,
                zone="cladding",
                area=100,
                detection_method="test_method"
            )
        ]
    
    def test_initialization(self, data_acquisition):
        """Test DataAcquisition initialization"""
        assert data_acquisition.config is not None
    
    def test_process_results(self, data_acquisition, sample_image, sample_zones, sample_defects):
        """Test processing results"""
        image_path = Path("test_image.jpg")
        
        result = data_acquisition.process_results(
            image_path, sample_image, sample_zones, sample_defects
        )
        
        assert isinstance(result, dict)
        assert 'report' in result
        assert 'visualizations' in result
        assert 'pass' in result
        
        # Check report structure
        report = result['report']
        assert report['image_path'] == str(image_path)
        assert 'timestamp' in report
        assert report['total_defects'] == 2
        assert len(report['defects']) == 2
        assert 'metrics' in report
        assert 'zones' in report
    
    def test_determine_pass_fail_pass(self, data_acquisition):
        """Test pass/fail determination - passing case"""
        # Few defects with low severity
        defects = [
            Defect(
                type="contamination",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.3,
                confidence=0.9,
                zone="cladding",
                area=400
            )
        ]
        
        result = data_acquisition._determine_pass_fail(defects)
        assert result is True
    
    def test_determine_pass_fail_too_many_defects(self, data_acquisition):
        """Test pass/fail determination - too many defects"""
        # More than 5 defects
        defects = []
        for i in range(6):
            defects.append(Defect(
                type="contamination",
                location=(100 + i*10, 100),
                bbox=(90 + i*10, 90, 20, 20),
                severity=0.3,
                confidence=0.9,
                zone="cladding",
                area=400
            ))
        
        result = data_acquisition._determine_pass_fail(defects)
        assert result is False
    
    def test_determine_pass_fail_high_severity(self, data_acquisition):
        """Test pass/fail determination - high severity defect"""
        defects = [
            Defect(
                type="contamination",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.8,  # Above threshold
                confidence=0.9,
                zone="cladding",
                area=400
            )
        ]
        
        result = data_acquisition._determine_pass_fail(defects)
        assert result is False
    
    def test_determine_pass_fail_critical_type(self, data_acquisition):
        """Test pass/fail determination - critical defect type"""
        defects = [
            Defect(
                type="fiber_damage",  # Critical type
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.3,
                confidence=0.9,
                zone="core",
                area=400
            )
        ]
        
        result = data_acquisition._determine_pass_fail(defects)
        assert result is False
    
    def test_generate_visualizations(self, data_acquisition, sample_image, sample_zones, sample_defects):
        """Test visualization generation"""
        visualizations = data_acquisition._generate_visualizations(
            sample_image, sample_zones, sample_defects
        )
        
        assert isinstance(visualizations, dict)
        assert 'defect_overlay' in visualizations
        assert 'zone_overlay' in visualizations
        
        # Check overlay shapes
        assert visualizations['defect_overlay'].shape == sample_image.shape
        assert visualizations['zone_overlay'].shape == sample_image.shape
    
    def test_generate_visualizations_with_heatmap(self, data_acquisition, sample_image, sample_zones, sample_defects):
        """Test visualization with heatmap generation"""
        data_acquisition.config.visualization.generate_heatmaps = True
        
        visualizations = data_acquisition._generate_visualizations(
            sample_image, sample_zones, sample_defects
        )
        
        assert 'defect_heatmap' in visualizations
        assert visualizations['defect_heatmap'].shape == sample_image.shape
    
    def test_calculate_zone_properties(self, data_acquisition, sample_zones):
        """Test zone property calculation"""
        zone_info = data_acquisition._calculate_zone_properties(sample_zones)
        
        assert isinstance(zone_info, dict)
        assert 'core' in zone_info
        assert 'cladding' in zone_info
        assert 'ferrule' in zone_info
        
        # Check core properties
        core_info = zone_info['core']
        assert 'center' in core_info
        assert 'radius' in core_info
        assert 'diameter' in core_info
        assert core_info['diameter'] == core_info['radius'] * 2
    
    def test_calculate_zone_properties_empty_zone(self, data_acquisition):
        """Test zone property calculation with empty zone"""
        zones = {
            'core': np.zeros((100, 100), dtype=np.uint8),  # Empty
            'cladding': np.zeros((100, 100), dtype=np.uint8)
        }
        
        zone_info = data_acquisition._calculate_zone_properties(zones)
        
        assert zone_info['core'] is None
        assert zone_info['cladding'] is None
    
    def test_calculate_metrics(self, data_acquisition, sample_image, sample_zones, sample_defects):
        """Test metrics calculation"""
        metrics = data_acquisition._calculate_metrics(
            sample_image, sample_zones, sample_defects
        )
        
        assert isinstance(metrics, dict)
        assert 'total_area' in metrics
        assert 'defect_density' in metrics
        assert 'average_severity' in metrics
        assert 'zone_defects' in metrics
        assert 'zone_measurements' in metrics
        
        # Check calculations
        assert metrics['total_area'] == sample_image.shape[0] * sample_image.shape[1]
        assert metrics['average_severity'] == pytest.approx(0.55, rel=0.01)  # (0.8 + 0.3) / 2
    
    def test_calculate_metrics_with_concentricity(self, data_acquisition, sample_image, sample_zones, sample_defects):
        """Test metrics calculation including concentricity"""
        metrics = data_acquisition._calculate_metrics(
            sample_image, sample_zones, sample_defects
        )
        
        # Should calculate concentricity between core and cladding
        assert 'concentricity_offset_pixels' in metrics
        assert 'concentricity_percentage' in metrics
        assert metrics['concentricity_offset_pixels'] >= 0
        assert metrics['concentricity_percentage'] >= 0
    
    def test_calculate_metrics_no_defects(self, data_acquisition, sample_image, sample_zones):
        """Test metrics calculation with no defects"""
        metrics = data_acquisition._calculate_metrics(
            sample_image, sample_zones, []
        )
        
        assert metrics['average_severity'] == 0
        assert metrics['defect_density'] == 0
        
        # Zone defects should be empty
        for zone_name in sample_zones:
            assert metrics['zone_defects'][zone_name]['count'] == 0
            assert metrics['zone_defects'][zone_name]['types'] == []


class TestEnhancedApplication:
    """Test EnhancedApplication class"""
    
    @pytest.fixture
    def app(self):
        """Create application instance"""
        config = get_config()
        config.interactive_mode = False
        with patch('app.EnhancedProcessor'), \
             patch('app.EnhancedSeparator'), \
             patch('app.EnhancedDetector'), \
             patch('app.DataAcquisition'):
            return EnhancedApplication()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, app):
        """Test application initialization"""
        assert app.config_manager is not None
        assert app.config is not None
        assert app.processor is not None
        assert app.separator is not None
        assert app.detector is not None
        assert app.data_acquisition is not None
    
    def test_setup_output_directory(self):
        """Test output directory setup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_config()
            config.output_dir = Path(tmpdir) / "output"
            
            with patch('app.EnhancedProcessor'), \
                 patch('app.EnhancedSeparator'), \
                 patch('app.EnhancedDetector'), \
                 patch('app.DataAcquisition'):
                app = EnhancedApplication()
            
            # Check directories were created
            assert config.output_dir.exists()
            assert (config.output_dir / 'passed').exists()
            assert (config.output_dir / 'failed').exists()
            assert (config.output_dir / 'reports').exists()
            assert (config.output_dir / 'visualizations').exists()
    
    def test_select_mode(self, app):
        """Test mode selection"""
        # Test each mode
        test_cases = [
            ('1', 'batch'),
            ('2', 'single'),
            ('3', 'realtime'),
            ('4', 'test'),
            ('5', 'config'),
            ('6', 'quit'),
            ('invalid', 'quit')  # Invalid input defaults to quit
        ]
        
        for input_value, expected_mode in test_cases:
            with patch('builtins.input', return_value=input_value):
                mode = app._select_mode()
                assert mode == expected_mode
    
    def test_process_image_success(self, app, temp_dir):
        """Test successful image processing"""
        # Create test image
        image_path = temp_dir / "test.jpg"
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), test_image)
        
        # Mock pipeline components
        app.processor.process_image = Mock(return_value={'original': test_image})
        app.separator.separate_zones = Mock(return_value={
            'core': np.ones((100, 100), dtype=np.uint8),
            'cladding': np.zeros((100, 100), dtype=np.uint8),
            'ferrule': np.zeros((100, 100), dtype=np.uint8)
        })
        app.detector.detect_defects = Mock(return_value=[])
        app.data_acquisition.process_results = Mock(return_value={
            'pass': True,
            'report': {'total_defects': 0},
            'visualizations': {}
        })
        
        result = app._process_image(image_path)
        
        assert result is not None
        assert result['pass'] is True
        
        # Verify pipeline was called
        app.processor.process_image.assert_called_once_with(image_path)
        app.separator.separate_zones.assert_called_once()
        app.detector.detect_defects.assert_called_once()
        app.data_acquisition.process_results.assert_called_once()
    
    def test_process_image_file_not_found(self, app):
        """Test processing non-existent image"""
        result = app._process_image(Path("nonexistent.jpg"))
        assert result is None
    
    def test_process_image_error_handling(self, app, temp_dir):
        """Test error handling during processing"""
        image_path = temp_dir / "test.jpg"
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), test_image)
        
        # Mock processor to raise error
        app.processor.process_image = Mock(side_effect=Exception("Processing error"))
        
        result = app._process_image(image_path)
        assert result is None
    
    def test_save_results(self, app, temp_dir):
        """Test saving results"""
        app.config.output_dir = temp_dir
        app._setup_output_directory()
        
        image_path = Path("test.jpg")
        result = {
            'pass': True,
            'report': {
                'total_defects': 0,
                'timestamp': datetime.now().isoformat()
            },
            'visualizations': {
                'defect_overlay': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            }
        }
        
        # Mock config
        app.config.visualization.generate_overlays = True
        
        # Save results
        app._save_results(image_path, result)
        
        # Check files were created
        reports = list((temp_dir / 'reports').glob('*.json'))
        assert len(reports) == 1
        
        # Check report content
        with open(reports[0], 'r') as f:
            saved_report = json.load(f)
            assert saved_report['total_defects'] == 0
    
    def test_process_batch(self, app, temp_dir):
        """Test batch processing"""
        # Create test images
        image_files = []
        for i in range(3):
            image_path = temp_dir / f"test_{i}.jpg"
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), test_image)
            image_files.append(image_path)
        
        # Mock process_image
        results = [
            {'pass': True, 'report': {'total_defects': 0}, 'visualizations': {}},
            {'pass': False, 'report': {'total_defects': 2}, 'visualizations': {}},
            None  # Error case
        ]
        app._process_image = Mock(side_effect=results)
        
        # Process batch
        app._process_batch(image_files)
        
        # Verify all images were processed
        assert app._process_image.call_count == 3
    
    def test_run_batch_mode(self, app, temp_dir):
        """Test batch mode execution"""
        # Create test directory with images
        for i in range(2):
            image_path = temp_dir / f"test_{i}.jpg"
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), test_image)
        
        # Mock user inputs and processing
        with patch('builtins.input', side_effect=[str(temp_dir), 'yes']):
            with patch.object(app, '_process_batch') as mock_process:
                app._run_batch_mode()
                
                # Verify batch processing was called
                mock_process.assert_called_once()
                args = mock_process.call_args[0][0]
                assert len(args) == 2  # Should find 2 images
    
    def test_run_batch_mode_no_images(self, app, temp_dir):
        """Test batch mode with no images"""
        with patch('builtins.input', return_value=str(temp_dir)):
            app._run_batch_mode()
            # Should handle gracefully
    
    def test_run_single_mode(self, app, temp_dir):
        """Test single image mode"""
        # Create test image
        image_path = temp_dir / "test.jpg"
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), test_image)
        
        # Mock processing
        mock_result = {
            'pass': True,
            'report': {'total_defects': 1},
            'visualizations': {}
        }
        app._process_image = Mock(return_value=mock_result)
        
        # Mock user inputs
        with patch('builtins.input', side_effect=[str(image_path), 'no']):
            app._run_single_mode()
        
        app._process_image.assert_called_once_with(image_path)
    
    def test_run_single_mode_show_visualizations(self, app, temp_dir):
        """Test single mode with visualization display"""
        # Create test image
        image_path = temp_dir / "test.jpg"
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), test_image)
        
        # Mock processing
        mock_result = {
            'pass': True,
            'report': {'total_defects': 1},
            'visualizations': {
                'overlay': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            }
        }
        app._process_image = Mock(return_value=mock_result)
        app._display_visualizations = Mock()
        
        # Mock user inputs
        with patch('builtins.input', side_effect=[str(image_path), 'yes']):
            app._run_single_mode()
        
        app._display_visualizations.assert_called_once_with(mock_result['visualizations'])
    
    def test_run_realtime_mode(self, app):
        """Test real-time mode"""
        app.config.processing.realtime_enabled = True
        
        # Mock RealtimeProcessor
        with patch('app.RealtimeProcessor') as MockProcessor:
            mock_processor = MockProcessor.return_value
            
            # Mock user input
            with patch('builtins.input', return_value='0'):
                app._run_realtime_mode()
            
            # Verify realtime processor was used
            mock_processor.start.assert_called_once_with(0)
            mock_processor.display_loop.assert_called_once()
            mock_processor.stop.assert_called_once()
    
    def test_run_realtime_mode_not_enabled(self, app):
        """Test real-time mode when not enabled"""
        app.config.processing.realtime_enabled = False
        
        # Mock user input to not enable
        with patch('builtins.input', side_effect=['no']):
            app._run_realtime_mode()
            # Should return without starting
    
    def test_run_test_mode_unit_tests(self, app):
        """Test running unit tests"""
        # Mock subprocess
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "Tests passed"
            mock_run.return_value.stderr = ""
            
            with patch('builtins.input', return_value='1'):
                app._run_test_mode()
            
            # Verify pytest was called
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert '-m' in args
            assert 'pytest' in args
    
    def test_run_test_mode_test_image(self, app, temp_dir):
        """Test running with test image"""
        # Create test image in expected location
        test_image_dir = Path(app.__module__).parent / "test_image"
        test_image_dir.mkdir(exist_ok=True)
        test_image_path = test_image_dir / "img(303).jpg"
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_image_path), test_image)
        
        try:
            # Mock processing
            mock_result = {
                'pass': True,
                'report': {
                    'total_defects': 1,
                    'metrics': {
                        'total_area': 10000,
                        'defect_density': 0.1,
                        'average_severity': 0.5,
                        'zone_measurements': {
                            'core': {
                                'diameter_pixels': 60,
                                'center': (50, 50),
                                'area_pixels': 2827
                            }
                        },
                        'zone_defects': {
                            'core': {'count': 1, 'types': ['scratch']}
                        },
                        'concentricity_offset_pixels': 2.5,
                        'concentricity_percentage': 5.0
                    }
                },
                'visualizations': {}
            }
            app._process_image = Mock(return_value=mock_result)
            
            # Run test mode
            with patch('builtins.input', side_effect=['2', 'no']):
                app._run_test_mode()
            
            app._process_image.assert_called_once()
            
        finally:
            # Cleanup
            if test_image_path.exists():
                test_image_path.unlink()
            if test_image_dir.exists():
                test_image_dir.rmdir()
    
    def test_reconfigure(self, app):
        """Test reconfiguration"""
        # Mock get_config_manager
        with patch('app.get_config_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.get_config.return_value = app.config
            mock_get_config.return_value = mock_config_manager
            
            # Mock component constructors
            with patch('app.EnhancedProcessor') as MockProcessor, \
                 patch('app.EnhancedSeparator') as MockSeparator, \
                 patch('app.EnhancedDetector') as MockDetector:
                
                app._reconfigure()
                
                # Verify components were recreated
                MockProcessor.assert_called()
                MockSeparator.assert_called()
                MockDetector.assert_called()
    
    def test_display_visualizations(self, app):
        """Test visualization display"""
        visualizations = {
            'overlay1': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            'overlay2': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        }
        
        # Mock cv2 functions
        with patch('cv2.imshow') as mock_imshow, \
             patch('cv2.waitKey', return_value=27), \
             patch('cv2.destroyAllWindows') as mock_destroy:
            
            app._display_visualizations(visualizations)
            
            # Verify images were shown
            assert mock_imshow.call_count == 2
            mock_destroy.assert_called_once()
    
    def test_run_main_loop(self, app):
        """Test main run loop"""
        # Mock mode selection to return different modes then quit
        with patch.object(app, '_select_mode', side_effect=['batch', 'quit']):
            with patch.object(app, '_run_batch_mode'):
                app.run()
                
                # Verify batch mode was called
                app._run_batch_mode.assert_called_once()


def test_main_function():
    """Test main entry point"""
    with patch('app.EnhancedApplication') as MockApp:
        mock_app = MockApp.return_value
        
        # Import and run main
        from app import main
        main()
        
        # Verify app was created and run
        MockApp.assert_called_once()
        mock_app.run.assert_called_once()


def test_main_keyboard_interrupt():
    """Test keyboard interrupt handling"""
    with patch('app.EnhancedApplication') as MockApp:
        mock_app = MockApp.return_value
        mock_app.run.side_effect = KeyboardInterrupt()
        
        # Import and run main
        from app import main
        main()  # Should handle gracefully


def test_main_general_exception():
    """Test general exception handling"""
    with patch('app.EnhancedApplication') as MockApp:
        mock_app = MockApp.return_value
        mock_app.run.side_effect = Exception("Test error")
        
        # Import and run main
        from app import main
        with patch('builtins.print'):  # Suppress print output
            main()  # Should handle gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])