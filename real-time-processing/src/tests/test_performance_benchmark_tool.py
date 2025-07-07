#!/usr/bin/env python3
"""
Comprehensive test suite for performance_benchmark_tool.py
Tests all functions, classes, and methods with rigorous edge cases
"""

import unittest
import numpy as np
import cv2
import json
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import pandas as pd
import matplotlib.pyplot as plt

# Import the module to test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tools.performance_benchmark_tool import (
    BenchmarkResult, BenchmarkScenario, SimpleShapesScenario,
    ComplexScenario, PerformanceBenchmark, run_quick_benchmark
)


class TestBenchmarkResult(unittest.TestCase):
    """Test the BenchmarkResult dataclass"""
    
    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult"""
        result = BenchmarkResult(
            scenario_name="Test Scenario",
            method="Standard",
            avg_detection_time=0.025,
            avg_fps=40.0,
            total_shapes_detected=150,
            accuracy_score=0.95,
            cpu_usage=45.0,
            memory_usage=512.0,
            gpu_usage=25.0,
            parameters={"threshold": 100}
        )
        
        self.assertEqual(result.scenario_name, "Test Scenario")
        self.assertEqual(result.method, "Standard")
        self.assertEqual(result.avg_detection_time, 0.025)
        self.assertEqual(result.avg_fps, 40.0)
        self.assertEqual(result.total_shapes_detected, 150)
        self.assertEqual(result.accuracy_score, 0.95)
        self.assertEqual(result.cpu_usage, 45.0)
        self.assertEqual(result.memory_usage, 512.0)
        self.assertEqual(result.gpu_usage, 25.0)
        self.assertEqual(result.parameters["threshold"], 100)
        
    def test_benchmark_result_optional_fields(self):
        """Test BenchmarkResult with optional fields None"""
        result = BenchmarkResult(
            scenario_name="Test",
            method="Basic",
            avg_detection_time=0.1,
            avg_fps=10.0,
            total_shapes_detected=10,
            accuracy_score=0.8
        )
        
        self.assertIsNone(result.cpu_usage)
        self.assertIsNone(result.memory_usage)
        self.assertIsNone(result.gpu_usage)
        self.assertIsNone(result.parameters)
        
    def test_benchmark_result_as_dict(self):
        """Test converting BenchmarkResult to dict"""
        result = BenchmarkResult(
            scenario_name="Test",
            method="GPU",
            avg_detection_time=0.01,
            avg_fps=100.0,
            total_shapes_detected=200,
            accuracy_score=0.99,
            gpu_usage=75.0
        )
        
        result_dict = asdict(result)
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['scenario_name'], "Test")
        self.assertEqual(result_dict['gpu_usage'], 75.0)


class TestBenchmarkScenario(unittest.TestCase):
    """Test the BenchmarkScenario base class"""
    
    def test_benchmark_scenario_abstract(self):
        """Test that BenchmarkScenario is abstract"""
        with self.assertRaises(TypeError):
            # Cannot instantiate abstract class
            scenario = BenchmarkScenario("Test", 100)
            
    def test_benchmark_scenario_subclass(self):
        """Test creating a subclass of BenchmarkScenario"""
        class TestScenario(BenchmarkScenario):
            def generate_frames(self):
                for i in range(self.num_frames):
                    yield np.zeros((100, 100, 3), dtype=np.uint8)
                    
        scenario = TestScenario("TestScenario", 5)
        self.assertEqual(scenario.name, "TestScenario")
        self.assertEqual(scenario.num_frames, 5)
        
        # Test frame generation
        frames = list(scenario.generate_frames())
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0].shape, (100, 100, 3))


class TestSimpleShapesScenario(unittest.TestCase):
    """Test the SimpleShapesScenario class"""
    
    def test_simple_shapes_scenario_creation(self):
        """Test creating SimpleShapesScenario"""
        scenario = SimpleShapesScenario(10)
        self.assertEqual(scenario.name, "Simple Shapes")
        self.assertEqual(scenario.num_frames, 10)
        
    def test_simple_shapes_generate_frames(self):
        """Test SimpleShapesScenario.generate_frames()"""
        scenario = SimpleShapesScenario(5)
        frames = list(scenario.generate_frames())
        
        self.assertEqual(len(frames), 5)
        
        # Check each frame
        for frame in frames:
            self.assertEqual(frame.shape, (600, 800, 3))
            self.assertEqual(frame.dtype, np.uint8)
            # Should have some non-black pixels (shapes drawn)
            self.assertGreater(np.sum(frame), 0)
            
    def test_simple_shapes_different_frames(self):
        """Test that SimpleShapesScenario generates different frames"""
        scenario = SimpleShapesScenario(3)
        frames = list(scenario.generate_frames())
        
        # Frames should be different (random shapes)
        # Compare first two frames
        if len(frames) >= 2:
            diff = cv2.absdiff(frames[0], frames[1])
            self.assertGreater(np.sum(diff), 0)
            
    def test_simple_shapes_has_shapes(self):
        """Test that frames contain detectable shapes"""
        scenario = SimpleShapesScenario(1)
        frame = next(scenario.generate_frames())
        
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Should have at least some shapes
        self.assertGreater(len(contours), 0)


class TestComplexScenario(unittest.TestCase):
    """Test the ComplexScenario class"""
    
    def test_complex_scenario_creation(self):
        """Test creating ComplexScenario"""
        scenario = ComplexScenario(10)
        self.assertEqual(scenario.name, "Complex Scene")
        self.assertEqual(scenario.num_frames, 10)
        
    def test_complex_scenario_generate_frames(self):
        """Test ComplexScenario.generate_frames()"""
        scenario = ComplexScenario(5)
        frames = list(scenario.generate_frames())
        
        self.assertEqual(len(frames), 5)
        
        # Check each frame
        for frame in frames:
            self.assertEqual(frame.shape, (600, 800, 3))
            self.assertEqual(frame.dtype, np.uint8)
            # Should have shapes and noise
            self.assertGreater(np.sum(frame), 0)
            
    def test_complex_scenario_has_noise(self):
        """Test that complex scenario adds noise"""
        scenario = ComplexScenario(1)
        frame = next(scenario.generate_frames())
        
        # Check for noise by looking at pixel variation
        # Even in solid areas, there should be some variation
        roi = frame[10:20, 10:20]
        std_dev = np.std(roi)
        self.assertGreater(std_dev, 0)  # Should have some noise
        
    def test_complex_scenario_motion_blur(self):
        """Test that complex scenario might have motion blur"""
        scenario = ComplexScenario(2)
        frames = list(scenario.generate_frames())
        
        # Frames should be slightly different due to motion
        if len(frames) >= 2:
            diff = cv2.absdiff(frames[0], frames[1])
            self.assertGreater(np.sum(diff), 0)


class TestPerformanceBenchmark(unittest.TestCase):
    """Test the PerformanceBenchmark class"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('performance_benchmark_tool.GeometryDetector'):
            self.benchmark = PerformanceBenchmark()
            
    def test_performance_benchmark_initialization(self):
        """Test PerformanceBenchmark initialization"""
        self.assertIsNotNone(self.benchmark.detector)
        self.assertEqual(len(self.benchmark.results), 0)
        
    @patch('platform.system')
    @patch('platform.processor')
    @patch('platform.machine')
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_get_hardware_info(self, mock_memory, mock_cpu_count, 
                               mock_machine, mock_processor, mock_system):
        """Test _get_hardware_info method"""
        # Set up mocks
        mock_system.return_value = "Linux"
        mock_processor.return_value = "Intel i7"
        mock_machine.return_value = "x86_64"
        mock_cpu_count.return_value = 8
        mock_memory.return_value = Mock(total=16 * 1024**3)  # 16 GB
        
        with patch('subprocess.run') as mock_run:
            # Mock nvidia-smi output
            mock_run.return_value = Mock(
                stdout="NVIDIA GeForce RTX 3080",
                returncode=0
            )
            
            info = self.benchmark._get_hardware_info()
            
        self.assertEqual(info['platform'], "Linux")
        self.assertEqual(info['processor'], "Intel i7")
        self.assertEqual(info['architecture'], "x86_64")
        self.assertEqual(info['cpu_count'], 8)
        self.assertEqual(info['ram_gb'], 16.0)
        self.assertIn("RTX 3080", info['gpu'])
        
    @patch('subprocess.run')
    def test_get_hardware_info_no_gpu(self, mock_run):
        """Test _get_hardware_info with no GPU"""
        mock_run.side_effect = Exception("No nvidia-smi")
        
        info = self.benchmark._get_hardware_info()
        self.assertEqual(info['gpu'], "No GPU detected")
        
    def test_run_benchmark_simple(self):
        """Test run_benchmark with simple scenario"""
        scenario = SimpleShapesScenario(3)
        
        # Mock detector methods
        self.benchmark.detector.config = Mock(enable_gpu=False)
        self.benchmark.detector.detect_shapes = Mock(
            return_value=[Mock() for _ in range(5)]  # 5 shapes
        )
        
        with patch('psutil.Process') as mock_process:
            mock_proc = Mock()
            mock_proc.cpu_percent.return_value = 50.0
            mock_proc.memory_info.return_value = Mock(rss=100 * 1024 * 1024)
            mock_process.return_value = mock_proc
            
            result = self.benchmark.run_benchmark(scenario, "Test Method", 
                                                  {"param": "value"})
            
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.scenario_name, "Simple Shapes")
        self.assertEqual(result.method, "Test Method")
        self.assertEqual(result.total_shapes_detected, 15)  # 5 shapes * 3 frames
        self.assertGreater(result.avg_fps, 0)
        self.assertAlmostEqual(result.cpu_usage, 50.0)
        self.assertAlmostEqual(result.memory_usage, 100.0)
        
    def test_run_benchmark_with_warmup(self):
        """Test run_benchmark with warmup frames"""
        scenario = SimpleShapesScenario(10)
        
        self.benchmark.detector.detect_shapes = Mock(return_value=[])
        
        with patch('psutil.Process'):
            result = self.benchmark.run_benchmark(scenario, "Test", warmup=5)
            
        # Should process all frames even with warmup
        self.assertEqual(self.benchmark.detector.detect_shapes.call_count, 10)
        
    def test_run_all_benchmarks(self):
        """Test run_all_benchmarks method"""
        # Mock the actual benchmark running
        with patch.object(self.benchmark, 'run_benchmark') as mock_run:
            mock_result = BenchmarkResult(
                scenario_name="Test",
                method="Standard",
                avg_detection_time=0.02,
                avg_fps=50.0,
                total_shapes_detected=100,
                accuracy_score=0.9
            )
            mock_run.return_value = mock_result
            
            self.benchmark.run_all_benchmarks(scenarios=[SimpleShapesScenario(5)])
            
        # Should have results
        self.assertGreater(len(self.benchmark.results), 0)
        
    @patch('matplotlib.pyplot.show')
    def test_visualize_results(self, mock_show):
        """Test visualize_results method"""
        # Add some results
        self.benchmark.results = [
            BenchmarkResult(
                scenario_name="Simple",
                method="Standard",
                avg_detection_time=0.02,
                avg_fps=50.0,
                total_shapes_detected=100,
                accuracy_score=0.9,
                cpu_usage=40.0,
                memory_usage=500.0
            ),
            BenchmarkResult(
                scenario_name="Simple",
                method="GPU",
                avg_detection_time=0.01,
                avg_fps=100.0,
                total_shapes_detected=100,
                accuracy_score=0.95,
                cpu_usage=30.0,
                memory_usage=600.0,
                gpu_usage=60.0
            )
        ]
        
        # Test visualization
        self.benchmark.visualize_results()
        
        # Should have created plots
        mock_show.assert_called()
        
    def test_visualize_results_empty(self):
        """Test visualize_results with no results"""
        # Should not raise error
        self.benchmark.visualize_results()
        
    def test_export_results_json(self):
        """Test export_results method with JSON format"""
        # Add test results
        self.benchmark.results = [
            BenchmarkResult(
                scenario_name="Test",
                method="Standard",
                avg_detection_time=0.02,
                avg_fps=50.0,
                total_shapes_detected=100,
                accuracy_score=0.9
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                         delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            self.benchmark.export_results(tmp_path)
            
            # Verify file was created and contains data
            with open(tmp_path, 'r') as f:
                data = json.load(f)
                
            self.assertIn('results', data)
            self.assertIn('hardware_info', data)
            self.assertIn('summary', data)
            self.assertEqual(len(data['results']), 1)
            self.assertEqual(data['results'][0]['scenario_name'], "Test")
            
        finally:
            os.unlink(tmp_path)
            
    def test_export_results_csv(self):
        """Test export_results with CSV format"""
        self.benchmark.results = [
            BenchmarkResult(
                scenario_name="Test",
                method="Standard",
                avg_detection_time=0.02,
                avg_fps=50.0,
                total_shapes_detected=100,
                accuracy_score=0.9
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', 
                                         delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            self.benchmark.export_results(tmp_path)
            
            # Verify CSV was created
            df = pd.read_csv(tmp_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]['scenario_name'], "Test")
            self.assertEqual(df.iloc[0]['avg_fps'], 50.0)
            
        finally:
            os.unlink(tmp_path)
            
    def test_generate_report(self):
        """Test _generate_report method"""
        # Add diverse results
        self.benchmark.results = [
            BenchmarkResult(
                scenario_name="Simple",
                method="Standard",
                avg_detection_time=0.02,
                avg_fps=50.0,
                total_shapes_detected=100,
                accuracy_score=0.9,
                cpu_usage=40.0,
                memory_usage=500.0
            ),
            BenchmarkResult(
                scenario_name="Simple",
                method="GPU",
                avg_detection_time=0.01,
                avg_fps=100.0,
                total_shapes_detected=105,
                accuracy_score=0.95,
                cpu_usage=30.0,
                memory_usage=600.0,
                gpu_usage=60.0
            ),
            BenchmarkResult(
                scenario_name="Complex",
                method="Standard",
                avg_detection_time=0.04,
                avg_fps=25.0,
                total_shapes_detected=200,
                accuracy_score=0.85,
                cpu_usage=60.0,
                memory_usage=700.0
            )
        ]
        
        report = self.benchmark._generate_report()
        
        # Check report structure
        self.assertIn("=" * 80, report)
        self.assertIn("PERFORMANCE BENCHMARK REPORT", report)
        self.assertIn("Hardware Information", report)
        self.assertIn("Best Performers", report)
        self.assertIn("Method Comparison", report)
        self.assertIn("Scenario Analysis", report)
        
        # Check content
        self.assertIn("Fastest Average FPS: GPU", report)
        self.assertIn("Most Accurate: GPU", report)
        self.assertIn("Simple", report)
        self.assertIn("Complex", report)
        
    def test_generate_report_empty(self):
        """Test _generate_report with no results"""
        report = self.benchmark._generate_report()
        self.assertIn("No benchmark results available", report)


class TestRunQuickBenchmark(unittest.TestCase):
    """Test the run_quick_benchmark function"""
    
    @patch('performance_benchmark_tool.PerformanceBenchmark')
    def test_run_quick_benchmark_default(self, mock_benchmark_class):
        """Test run_quick_benchmark with default parameters"""
        mock_benchmark = Mock()
        mock_benchmark_class.return_value = mock_benchmark
        
        # Mock benchmark results
        mock_results = [
            BenchmarkResult(
                scenario_name="Simple",
                method="Standard",
                avg_detection_time=0.02,
                avg_fps=50.0,
                total_shapes_detected=100,
                accuracy_score=0.9
            )
        ]
        mock_benchmark.results = mock_results
        
        results = run_quick_benchmark()
        
        # Verify benchmark was created and run
        mock_benchmark_class.assert_called_once()
        mock_benchmark.run_all_benchmarks.assert_called_once()
        
        # Check scenarios used
        call_args = mock_benchmark.run_all_benchmarks.call_args
        scenarios = call_args[1]['scenarios']
        self.assertEqual(len(scenarios), 2)  # Simple and Complex
        
        # Verify results returned
        self.assertEqual(results, mock_results)
        
    @patch('performance_benchmark_tool.PerformanceBenchmark')
    def test_run_quick_benchmark_custom_frames(self, mock_benchmark_class):
        """Test run_quick_benchmark with custom frame count"""
        mock_benchmark = Mock()
        mock_benchmark_class.return_value = mock_benchmark
        mock_benchmark.results = []
        
        run_quick_benchmark(num_frames=20)
        
        # Check frame count in scenarios
        call_args = mock_benchmark.run_all_benchmarks.call_args
        scenarios = call_args[1]['scenarios']
        
        for scenario in scenarios:
            self.assertEqual(scenario.num_frames, 20)
            
    @patch('performance_benchmark_tool.PerformanceBenchmark')
    @patch('matplotlib.pyplot.show')
    def test_run_quick_benchmark_visualize(self, mock_show, mock_benchmark_class):
        """Test run_quick_benchmark with visualization"""
        mock_benchmark = Mock()
        mock_benchmark_class.return_value = mock_benchmark
        mock_benchmark.results = []
        
        run_quick_benchmark(visualize=True)
        
        # Should call visualize_results
        mock_benchmark.visualize_results.assert_called_once()
        
    @patch('performance_benchmark_tool.PerformanceBenchmark')
    def test_run_quick_benchmark_export(self, mock_benchmark_class):
        """Test run_quick_benchmark with export"""
        mock_benchmark = Mock()
        mock_benchmark_class.return_value = mock_benchmark
        mock_benchmark.results = []
        
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            run_quick_benchmark(export_path=tmp.name)
            
            # Should call export_results
            mock_benchmark.export_results.assert_called_once_with(tmp.name)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete benchmark scenarios"""
    
    def test_scenario_frame_generation_consistency(self):
        """Test that scenarios generate consistent frame counts"""
        scenarios = [
            SimpleShapesScenario(10),
            ComplexScenario(10)
        ]
        
        for scenario in scenarios:
            frames = list(scenario.generate_frames())
            self.assertEqual(len(frames), 10)
            
            # All frames should have same dimensions
            shapes = [frame.shape for frame in frames]
            self.assertEqual(len(set(shapes)), 1)
            
    def test_benchmark_result_serialization(self):
        """Test BenchmarkResult can be serialized properly"""
        result = BenchmarkResult(
            scenario_name="Test",
            method="GPU",
            avg_detection_time=0.015,
            avg_fps=66.67,
            total_shapes_detected=150,
            accuracy_score=0.92,
            cpu_usage=35.5,
            memory_usage=1024.0,
            gpu_usage=55.0,
            parameters={"threshold": 100, "enable_gpu": True}
        )
        
        # Test JSON serialization
        json_str = json.dumps(asdict(result))
        loaded = json.loads(json_str)
        
        self.assertEqual(loaded['scenario_name'], "Test")
        self.assertAlmostEqual(loaded['avg_fps'], 66.67)
        self.assertEqual(loaded['parameters']['threshold'], 100)
        
    @patch('performance_benchmark_tool.GeometryDetector')
    def test_full_benchmark_workflow(self, mock_detector_class):
        """Test complete benchmark workflow"""
        # Set up mock detector
        mock_detector = Mock()
        mock_detector.config = Mock(enable_gpu=False)
        mock_detector.detect_shapes = Mock(
            side_effect=lambda x: [Mock() for _ in range(3)]
        )
        mock_detector_class.return_value = mock_detector
        
        # Create benchmark
        benchmark = PerformanceBenchmark()
        
        # Run simple benchmark
        scenario = SimpleShapesScenario(5)
        
        with patch('psutil.Process') as mock_process:
            mock_proc = Mock()
            mock_proc.cpu_percent.return_value = 45.0
            mock_proc.memory_info.return_value = Mock(rss=200 * 1024 * 1024)
            mock_process.return_value = mock_proc
            
            result = benchmark.run_benchmark(scenario, "Test Method")
            
        # Verify result
        self.assertEqual(result.scenario_name, "Simple Shapes")
        self.assertEqual(result.total_shapes_detected, 15)  # 3 shapes * 5 frames
        self.assertGreater(result.avg_fps, 0)
        
        # Test export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                         delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            benchmark.export_results(tmp_path)
            
            # Verify export
            with open(tmp_path, 'r') as f:
                data = json.load(f)
                
            self.assertEqual(len(data['results']), 1)
            self.assertEqual(data['results'][0]['total_shapes_detected'], 15)
            
        finally:
            os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)