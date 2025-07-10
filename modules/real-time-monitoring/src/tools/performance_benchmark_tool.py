#!/usr/bin/env python3
"""
Performance Benchmark and Comparison Tool
========================================

Comprehensive benchmarking tool for the Integrated Geometry Detection System.
Compares different detection methods, hardware configurations, and optimization settings.

Features:
- Multi-method comparison (CPU vs GPU vs optimized algorithms)
- Automated test suite with synthetic and real-world scenarios
- Detailed performance metrics and visualization
- Export results for analysis

Usage: python performance_benchmark_tool.py
"""

import cv2
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import psutil
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import sys

# Import from integrated system
from src.core.integrated_geometry_system import (
    GeometryDetector,
    ShapeType,
    Config,
    setup_logging,
    PerformanceMetrics
)

@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result data"""
    method_name: str
    hardware_info: Dict
    
    # Performance metrics
    avg_fps: float
    min_fps: float
    max_fps: float
    std_fps: float
    
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Resource usage
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_memory_mb: float
    max_memory_mb: float
    avg_gpu_percent: float
    max_gpu_percent: float
    avg_gpu_memory_mb: float
    
    # Detection metrics
    total_shapes_detected: int
    avg_shapes_per_frame: float
    detection_accuracy: float
    false_positives: int
    false_negatives: int
    
    # Test conditions
    frame_count: int
    resolution: Tuple[int, int]
    test_duration_seconds: float
    timestamp: str

class BenchmarkScenario:
    """Test scenario for benchmarking"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.frames = []
        self.ground_truth = []
    
    def generate_frames(self, count: int, resolution: Tuple[int, int]) -> List[np.ndarray]:
        """Generate test frames - override in subclasses"""
        raise NotImplementedError

class SimpleShapesScenario(BenchmarkScenario):
    """Simple geometric shapes scenario"""
    
    def __init__(self):
        super().__init__(
            "Simple Shapes",
            "Basic geometric shapes with varying counts and sizes"
        )
    
    def generate_frames(self, count: int, resolution: Tuple[int, int]) -> List[np.ndarray]:
        """Generate frames with simple shapes"""
        width, height = resolution
        frames = []
        ground_truths = []
        
        for i in range(count):
            # Create blank frame
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            frame_truth = []
            
            # Random number of shapes
            num_shapes = np.random.randint(5, 20)
            
            for _ in range(num_shapes):
                shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
                color = (np.random.randint(0, 255), 
                        np.random.randint(0, 255),
                        np.random.randint(0, 255))
                
                if shape_type == 'circle':
                    center = (np.random.randint(50, width-50), 
                             np.random.randint(50, height-50))
                    radius = np.random.randint(20, min(100, width//10))
                    cv2.circle(frame, center, radius, color, -1)
                    frame_truth.append({
                        'type': 'circle',
                        'center': center,
                        'radius': radius
                    })
                
                elif shape_type == 'rectangle':
                    x = np.random.randint(50, width-150)
                    y = np.random.randint(50, height-150)
                    w = np.random.randint(50, min(150, width//5))
                    h = np.random.randint(50, min(150, height//5))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
                    frame_truth.append({
                        'type': 'rectangle',
                        'bbox': (x, y, w, h)
                    })
                
                elif shape_type == 'triangle':
                    pts = np.array([
                        [np.random.randint(50, width-50), 
                         np.random.randint(50, height-50)]
                        for _ in range(3)
                    ], np.int32)
                    cv2.fillPoly(frame, [pts], color)
                    frame_truth.append({
                        'type': 'triangle',
                        'vertices': pts.tolist()
                    })
            
            frames.append(frame)
            ground_truths.append(frame_truth)
        
        self.frames = frames
        self.ground_truth = ground_truths
        return frames

class ComplexScenario(BenchmarkScenario):
    """Complex scenario with overlapping shapes and noise"""
    
    def __init__(self):
        super().__init__(
            "Complex Scene",
            "Overlapping shapes with noise and varying lighting"
        )
    
    def generate_frames(self, count: int, resolution: Tuple[int, int]) -> List[np.ndarray]:
        """Generate complex frames"""
        width, height = resolution
        frames = []
        
        for i in range(count):
            # Create gradient background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            gradient = np.linspace(100, 200, width)
            frame[:, :] = gradient[np.newaxis, :, np.newaxis]
            
            # Add noise
            noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise)
            
            # Add overlapping shapes
            num_shapes = np.random.randint(15, 30)
            
            for _ in range(num_shapes):
                shape_type = np.random.choice(['circle', 'ellipse', 'polygon'])
                color = (np.random.randint(50, 255), 
                        np.random.randint(50, 255),
                        np.random.randint(50, 255))
                
                if shape_type == 'circle':
                    center = (np.random.randint(0, width), 
                             np.random.randint(0, height))
                    radius = np.random.randint(10, 80)
                    cv2.circle(frame, center, radius, color, -1)
                
                elif shape_type == 'ellipse':
                    center = (np.random.randint(0, width), 
                             np.random.randint(0, height))
                    axes = (np.random.randint(20, 100), 
                           np.random.randint(20, 100))
                    angle = np.random.randint(0, 180)
                    cv2.ellipse(frame, center, axes, angle, 0, 360, color, -1)
                
                elif shape_type == 'polygon':
                    num_vertices = np.random.randint(5, 8)
                    pts = np.array([
                        [np.random.randint(0, width), 
                         np.random.randint(0, height)]
                        for _ in range(num_vertices)
                    ], np.int32)
                    cv2.fillPoly(frame, [pts], color)
            
            # Apply blur to some regions
            if i % 3 == 0:
                blur_region = frame[height//4:3*height//4, width//4:3*width//4]
                blurred = cv2.GaussianBlur(blur_region, (15, 15), 0)
                frame[height//4:3*height//4, width//4:3*width//4] = blurred
            
            frames.append(frame)
        
        self.frames = frames
        return frames

class PerformanceBenchmark:
    """Main benchmarking system"""
    
    def __init__(self):
        self.logger = setup_logging("PerformanceBenchmark")
        self.results = []
        self.scenarios = [
            SimpleShapesScenario(),
            ComplexScenario()
        ]
        
        # Test configurations
        self.resolutions = [
            (640, 480),
            (1280, 720),
            (1920, 1080)
        ]
        
        # Detection methods to test
        self.methods = [
            ('CPU', {'use_gpu': False, 'max_threads': 1}),
            ('CPU_Multi', {'use_gpu': False, 'max_threads': 4}),
            ('GPU', {'use_gpu': True, 'max_threads': 4}),
        ]
        
        # Hardware info
        self.hardware_info = self._get_hardware_info()
    
    def _get_hardware_info(self) -> Dict:
        """Collect hardware information"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version.split()[0],
            'opencv_version': cv2.__version__
        }
        
        # GPU info
        try:
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            info['gpu_count'] = gpu_count
            
            if gpu_count > 0:
                # Try to get GPU name
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', 
                                           '--format=csv,noheader'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        info['gpu_name'] = result.stdout.strip()
                except:
                    info['gpu_name'] = f"CUDA GPU ({gpu_count} device(s))"
        except:
            info['gpu_count'] = 0
        
        return info
    
    def run_benchmark(self, method_name: str, config: Dict, 
                     scenario: BenchmarkScenario, 
                     resolution: Tuple[int, int]) -> BenchmarkResult:
        """Run benchmark for specific configuration"""
        self.logger.info(f"Running benchmark: {method_name} - {scenario.name} - {resolution}")
        
        # Configure detector
        Config.MAX_THREADS = config.get('max_threads', 4)
        detector = GeometryDetector(use_gpu=config.get('use_gpu', False))
        
        # Generate test frames
        frames = scenario.generate_frames(100, resolution)
        
        # Metrics storage
        fps_values = []
        latencies = []
        cpu_usage = []
        memory_usage = []
        gpu_usage = []
        gpu_memory = []
        shapes_per_frame = []
        
        # Resource monitor
        process = psutil.Process()
        
        # Warm up
        for _ in range(5):
            detector.detect_shapes(frames[0])
        
        # Run benchmark
        start_time = time.time()
        
        for i, frame in enumerate(frames):
            # Start timing
            frame_start = time.time()
            
            # Get resource usage before
            cpu_before = process.cpu_percent()
            mem_before = process.memory_info().rss / (1024**2)  # MB
            
            # Detect shapes
            shapes = detector.detect_shapes(frame)
            
            # End timing
            frame_end = time.time()
            frame_time = frame_end - frame_start
            
            # Get resource usage after
            cpu_after = process.cpu_percent()
            mem_after = process.memory_info().rss / (1024**2)
            
            # Store metrics
            if i > 0:  # Skip first frame for FPS calculation
                fps = 1.0 / frame_time
                fps_values.append(fps)
                latencies.append(frame_time * 1000)  # ms
                cpu_usage.append((cpu_before + cpu_after) / 2)
                memory_usage.append(mem_after)
                shapes_per_frame.append(len(shapes))
            
            # GPU metrics if available
            if config.get('use_gpu', False):
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage.append(gpus[0].load * 100)
                        gpu_memory.append(gpus[0].memoryUsed)
                except:
                    pass
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        result = BenchmarkResult(
            method_name=f"{method_name}_{scenario.name}_{resolution[0]}x{resolution[1]}",
            hardware_info=self.hardware_info,
            
            # FPS metrics
            avg_fps=np.mean(fps_values) if fps_values else 0,
            min_fps=np.min(fps_values) if fps_values else 0,
            max_fps=np.max(fps_values) if fps_values else 0,
            std_fps=np.std(fps_values) if fps_values else 0,
            
            # Latency metrics
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            min_latency_ms=np.min(latencies) if latencies else 0,
            max_latency_ms=np.max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            
            # Resource usage
            avg_cpu_percent=np.mean(cpu_usage) if cpu_usage else 0,
            max_cpu_percent=np.max(cpu_usage) if cpu_usage else 0,
            avg_memory_mb=np.mean(memory_usage) if memory_usage else 0,
            max_memory_mb=np.max(memory_usage) if memory_usage else 0,
            avg_gpu_percent=np.mean(gpu_usage) if gpu_usage else 0,
            max_gpu_percent=np.max(gpu_usage) if gpu_usage else 0,
            avg_gpu_memory_mb=np.mean(gpu_memory) if gpu_memory else 0,
            
            # Detection metrics
            total_shapes_detected=sum(shapes_per_frame),
            avg_shapes_per_frame=np.mean(shapes_per_frame) if shapes_per_frame else 0,
            detection_accuracy=0.0,  # Would need ground truth comparison
            false_positives=0,
            false_negatives=0,
            
            # Test conditions
            frame_count=len(frames),
            resolution=resolution,
            test_duration_seconds=total_time,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return result
    
    def run_all_benchmarks(self):
        """Run all benchmark combinations"""
        total_tests = len(self.methods) * len(self.scenarios) * len(self.resolutions)
        current_test = 0
        
        print(f"\nRunning {total_tests} benchmark tests...")
        print("="*60)
        
        for method_name, config in self.methods:
            # Skip GPU tests if not available
            if config.get('use_gpu', False) and self.hardware_info.get('gpu_count', 0) == 0:
                self.logger.warning(f"Skipping {method_name} - No GPU available")
                continue
            
            for scenario in self.scenarios:
                for resolution in self.resolutions:
                    current_test += 1
                    print(f"\nTest {current_test}/{total_tests}: {method_name} - "
                          f"{scenario.name} - {resolution[0]}x{resolution[1]}")
                    
                    try:
                        result = self.run_benchmark(method_name, config, 
                                                   scenario, resolution)
                        self.results.append(result)
                        
                        # Print summary
                        print(f"  FPS: {result.avg_fps:.1f} "
                              f"(min: {result.min_fps:.1f}, max: {result.max_fps:.1f})")
                        print(f"  Latency: {result.avg_latency_ms:.1f}ms "
                              f"(p95: {result.p95_latency_ms:.1f}ms)")
                        print(f"  CPU: {result.avg_cpu_percent:.1f}% "
                              f"Memory: {result.avg_memory_mb:.0f}MB")
                        if result.avg_gpu_percent > 0:
                            print(f"  GPU: {result.avg_gpu_percent:.1f}% "
                                  f"GPU Memory: {result.avg_gpu_memory_mb:.0f}MB")
                        
                    except Exception as e:
                        self.logger.error(f"Benchmark failed: {e}")
                        import traceback
                        traceback.print_exc()
        
        print("\n" + "="*60)
        print("Benchmarking complete!")
    
    def visualize_results(self):
        """Create visualization of benchmark results"""
        if not self.results:
            self.logger.warning("No results to visualize")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Geometry Detection Performance Benchmark Results', fontsize=16)
        
        # 1. FPS comparison by method
        ax = axes[0, 0]
        methods = df['method_name'].str.split('_').str[0].unique()
        fps_by_method = [df[df['method_name'].str.startswith(m)]['avg_fps'].values 
                         for m in methods]
        
        bp = ax.boxplot(fps_by_method, labels=methods)
        ax.set_ylabel('FPS')
        ax.set_title('FPS by Detection Method')
        ax.grid(True, alpha=0.3)
        
        # 2. FPS vs Resolution
        ax = axes[0, 1]
        for method in methods:
            method_df = df[df['method_name'].str.startswith(method)]
            resolutions = [f"{r[0]}x{r[1]}" for r in method_df['resolution']]
            ax.plot(resolutions, method_df['avg_fps'], 'o-', label=method, linewidth=2)
        
        ax.set_xlabel('Resolution')
        ax.set_ylabel('FPS')
        ax.set_title('FPS vs Resolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Latency distribution
        ax = axes[0, 2]
        latency_data = [df[df['method_name'].str.startswith(m)]['avg_latency_ms'].values 
                        for m in methods]
        
        bp = ax.boxplot(latency_data, labels=methods)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Processing Latency by Method')
        ax.grid(True, alpha=0.3)
        
        # 4. Resource usage
        ax = axes[1, 0]
        x = np.arange(len(methods))
        width = 0.35
        
        cpu_means = [df[df['method_name'].str.startswith(m)]['avg_cpu_percent'].mean() 
                     for m in methods]
        mem_means = [df[df['method_name'].str.startswith(m)]['avg_memory_mb'].mean() / 10  # Scale for visibility
                     for m in methods]
        
        bars1 = ax.bar(x - width/2, cpu_means, width, label='CPU %', color='skyblue')
        bars2 = ax.bar(x + width/2, mem_means, width, label='Memory (MB/10)', color='lightcoral')
        
        ax.set_ylabel('Usage')
        ax.set_title('Resource Usage by Method')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Shapes detected
        ax = axes[1, 1]
        shapes_by_method = [df[df['method_name'].str.startswith(m)]['avg_shapes_per_frame'].values 
                           for m in methods]
        
        bp = ax.boxplot(shapes_by_method, labels=methods)
        ax.set_ylabel('Shapes per Frame')
        ax.set_title('Detection Count by Method')
        ax.grid(True, alpha=0.3)
        
        # 6. Performance summary table
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        for method in methods:
            method_df = df[df['method_name'].str.startswith(method)]
            summary_data.append([
                method,
                f"{method_df['avg_fps'].mean():.1f}",
                f"{method_df['avg_latency_ms'].mean():.1f}",
                f"{method_df['avg_cpu_percent'].mean():.1f}",
                f"{method_df['avg_gpu_percent'].mean():.1f}" if method == 'GPU' else "N/A"
            ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Method', 'Avg FPS', 'Avg Latency (ms)', 'CPU %', 'GPU %'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Visualization saved to {filename}")
        
        plt.show()
    
    def export_results(self):
        """Export results to various formats"""
        if not self.results:
            self.logger.warning("No results to export")
            return
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Export to JSON
        json_filename = f"benchmark_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        self.logger.info(f"Results exported to {json_filename}")
        
        # Export to CSV
        csv_filename = f"benchmark_results_{timestamp}.csv"
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(csv_filename, index=False)
        self.logger.info(f"Results exported to {csv_filename}")
        
        # Generate detailed report
        self._generate_report(timestamp)
    
    def _generate_report(self, timestamp: str):
        """Generate detailed performance report"""
        report_filename = f"benchmark_report_{timestamp}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GEOMETRY DETECTION PERFORMANCE BENCHMARK REPORT\n")
            f.write("="*80 + "\n\n")
            
            # System information
            f.write("SYSTEM INFORMATION\n")
            f.write("-"*40 + "\n")
            for key, value in self.hardware_info.items():
                f.write(f"{key:20}: {value}\n")
            f.write("\n")
            
            # Test summary
            f.write("TEST SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Total tests run: {len(self.results)}\n")
            f.write(f"Test scenarios: {', '.join([s.name for s in self.scenarios])}\n")
            f.write(f"Resolutions tested: {', '.join([f'{r[0]}x{r[1]}' for r in self.resolutions])}\n")
            f.write(f"Methods tested: {', '.join([m[0] for m in self.methods])}\n\n")
            
            # Performance summary by method
            f.write("PERFORMANCE SUMMARY BY METHOD\n")
            f.write("-"*40 + "\n")
            
            df = pd.DataFrame([asdict(r) for r in self.results])
            
            for method_name, _ in self.methods:
                method_results = df[df['method_name'].str.startswith(method_name)]
                if len(method_results) == 0:
                    continue
                
                f.write(f"\n{method_name}:\n")
                f.write(f"  Average FPS: {method_results['avg_fps'].mean():.2f} "
                       f"(±{method_results['avg_fps'].std():.2f})\n")
                f.write(f"  Average Latency: {method_results['avg_latency_ms'].mean():.2f}ms "
                       f"(±{method_results['avg_latency_ms'].std():.2f}ms)\n")
                f.write(f"  CPU Usage: {method_results['avg_cpu_percent'].mean():.1f}% "
                       f"(max: {method_results['max_cpu_percent'].max():.1f}%)\n")
                f.write(f"  Memory Usage: {method_results['avg_memory_mb'].mean():.0f}MB "
                       f"(max: {method_results['max_memory_mb'].max():.0f}MB)\n")
                
                if method_name == 'GPU':
                    f.write(f"  GPU Usage: {method_results['avg_gpu_percent'].mean():.1f}% "
                           f"(max: {method_results['max_gpu_percent'].max():.1f}%)\n")
                    f.write(f"  GPU Memory: {method_results['avg_gpu_memory_mb'].mean():.0f}MB\n")
            
            # Best configurations
            f.write("\n\nBEST CONFIGURATIONS\n")
            f.write("-"*40 + "\n")
            
            # Highest FPS
            best_fps = df.loc[df['avg_fps'].idxmax()]
            f.write(f"Highest FPS: {best_fps['avg_fps']:.1f} - {best_fps['method_name']}\n")
            
            # Lowest latency
            best_latency = df.loc[df['avg_latency_ms'].idxmin()]
            f.write(f"Lowest Latency: {best_latency['avg_latency_ms']:.1f}ms - "
                   f"{best_latency['method_name']}\n")
            
            # Most efficient (FPS per CPU %)
            df['efficiency'] = df['avg_fps'] / (df['avg_cpu_percent'] + 1)
            best_efficiency = df.loc[df['efficiency'].idxmax()]
            f.write(f"Best Efficiency: {best_efficiency['efficiency']:.2f} FPS/CPU% - "
                   f"{best_efficiency['method_name']}\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            
            if self.hardware_info.get('gpu_count', 0) > 0:
                gpu_results = df[df['method_name'].str.startswith('GPU')]
                cpu_results = df[df['method_name'].str.startswith('CPU')]
                
                if len(gpu_results) > 0 and len(cpu_results) > 0:
                    gpu_speedup = gpu_results['avg_fps'].mean() / cpu_results['avg_fps'].mean()
                    f.write(f"• GPU provides {gpu_speedup:.1f}x speedup over single-threaded CPU\n")
                    
                    if gpu_speedup > 1.5:
                        f.write("• Recommend using GPU acceleration for production\n")
                    else:
                        f.write("• GPU speedup is minimal - consider CPU-only deployment\n")
            else:
                f.write("• No GPU detected - using multi-threaded CPU processing\n")
            
            # Resolution recommendations
            f.write("\n• Performance by resolution:\n")
            for res in self.resolutions:
                res_str = f"{res[0]}x{res[1]}"
                res_results = df[df['resolution'].apply(lambda x: x == res)]
                avg_fps = res_results['avg_fps'].mean()
                f.write(f"  - {res_str}: {avg_fps:.1f} FPS average\n")
            
            f.write("\n")
        
        self.logger.info(f"Detailed report saved to {report_filename}")

def run_quick_benchmark():
    """Run a quick benchmark with default settings"""
    print("\nQuick Benchmark Mode")
    print("="*60)
    print("Running quick performance test...")
    
    # Simple test
    detector_cpu = GeometryDetector(use_gpu=False)
    detector_gpu = GeometryDetector(use_gpu=True) if cv2.cuda.getCudaEnabledDeviceCount() > 0 else None
    
    # Create test frame
    test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    
    # Add various shapes
    cv2.circle(test_frame, (640, 360), 100, (0, 0, 255), -1)
    cv2.rectangle(test_frame, (100, 100), (300, 300), (0, 255, 0), -1)
    pts = np.array([[500, 100], [600, 300], [400, 300]], np.int32)
    cv2.fillPoly(test_frame, [pts], (255, 0, 0))
    
    # Benchmark CPU
    print("\nCPU Performance:")
    times = []
    for _ in range(50):
        start = time.time()
        shapes = detector_cpu.detect_shapes(test_frame)
        times.append(time.time() - start)
    
    avg_time = np.mean(times[10:])  # Skip warmup
    print(f"  Average time: {avg_time*1000:.2f}ms")
    print(f"  FPS: {1/avg_time:.1f}")
    print(f"  Shapes detected: {len(shapes)}")
    
    # Benchmark GPU if available
    if detector_gpu:
        print("\nGPU Performance:")
        times = []
        for _ in range(50):
            start = time.time()
            shapes = detector_gpu.detect_shapes(test_frame)
            times.append(time.time() - start)
        
        avg_time = np.mean(times[10:])
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  FPS: {1/avg_time:.1f}")
        print(f"  Shapes detected: {len(shapes)}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Benchmark Tool')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark')
    parser.add_argument('--full', action='store_true',
                       help='Run full benchmark suite')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize existing results')
    parser.add_argument('--results', type=str,
                       help='Path to existing results JSON file')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GEOMETRY DETECTION PERFORMANCE BENCHMARK TOOL")
    print("="*60)
    
    if args.quick or (not args.full and not args.visualize):
        run_quick_benchmark()
    
    if args.full:
        print("\nStarting full benchmark suite...")
        print("This may take several minutes.\n")
        
        benchmark = PerformanceBenchmark()
        benchmark.run_all_benchmarks()
        benchmark.visualize_results()
        benchmark.export_results()
        
        print("\nBenchmark complete!")
    
    if args.visualize and args.results:
        print(f"\nLoading results from {args.results}...")
        benchmark = PerformanceBenchmark()
        
        with open(args.results, 'r') as f:
            data = json.load(f)
            benchmark.results = [BenchmarkResult(**item) for item in data]
        
        benchmark.visualize_results()

if __name__ == "__main__":
    main()
