#!/usr/bin/env python3
"""
Geometry Detection Performance Benchmark & Comparison Tool
=========================================================

This tool benchmarks and compares different geometric detection methods,
including classical computer vision, GPU acceleration, and deep learning approaches.

Features:
- Performance metrics (FPS, accuracy, latency)
- Method comparison (CPU vs GPU vs ML)
- Visualization of results
- Export benchmark data

Author: Computer Vision Benchmark System
Date: 2025
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any
import json
import psutil
import GPUtil
from dataclasses import dataclass, asdict
import threading
import queue
from abc import ABC, abstractmethod

@dataclass
class BenchmarkResult:
    """Store benchmark results for a detection method"""
    method_name: str
    fps: float
    avg_latency: float
    min_latency: float
    max_latency: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    shapes_detected: int
    false_positives: int
    false_negatives: int

class ShapeDetectorBase(ABC):
    """Abstract base class for shape detectors"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect shapes in frame"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get detector name"""
        pass

class ClassicalCVDetector(ShapeDetectorBase):
    """Classical computer vision shape detector"""
    
    def __init__(self):
        self.name = "Classical CV (CPU)"
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect shapes using classical CV methods"""
        shapes = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get shape properties
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            shape = {
                'type': self.classify_shape(approx),
                'vertices': len(approx),
                'area': area,
                'center': center,
                'bbox': (x, y, w, h),
                'contour': contour
            }
            shapes.append(shape)
        
        return shapes
    
    def classify_shape(self, approx: np.ndarray) -> str:
        """Classify shape based on vertices"""
        vertices = len(approx)
        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            return 'rectangle'
        elif vertices == 5:
            return 'pentagon'
        elif vertices == 6:
            return 'hexagon'
        elif vertices > 6:
            return 'circle'
        else:
            return 'unknown'
    
    def get_name(self) -> str:
        return self.name

class GPUAcceleratedDetector(ShapeDetectorBase):
    """GPU-accelerated shape detector using CUDA"""
    
    def __init__(self):
        self.name = "GPU Accelerated (CUDA)"
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if not self.cuda_available:
            print("CUDA not available, falling back to CPU")
            self.name += " [CPU Fallback]"
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect shapes using GPU acceleration"""
        shapes = []
        
        if self.cuda_available:
            # Upload to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            
            # Gaussian blur on GPU
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)
            
            # Canny edge detection on GPU
            gpu_canny = cv2.cuda.createCannyEdgeDetector(50, 150)
            gpu_edges = gpu_canny.detect(gpu_blurred)
            
            # Download results
            edges = gpu_edges.download()
        else:
            # CPU fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours (CPU only)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            shape = {
                'type': self.classify_shape(approx),
                'vertices': len(approx),
                'area': area,
                'center': center,
                'bbox': (x, y, w, h),
                'contour': contour
            }
            shapes.append(shape)
        
        return shapes
    
    def classify_shape(self, approx: np.ndarray) -> str:
        """Classify shape based on vertices"""
        vertices = len(approx)
        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            return 'rectangle'
        elif vertices == 5:
            return 'pentagon'
        elif vertices == 6:
            return 'hexagon'
        elif vertices > 6:
            return 'circle'
        else:
            return 'unknown'
    
    def get_name(self) -> str:
        return self.name

class HoughTransformDetector(ShapeDetectorBase):
    """Detector using Hough transforms for lines and circles"""
    
    def __init__(self):
        self.name = "Hough Transform"
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect shapes using Hough transforms"""
        shapes = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                shape = {
                    'type': 'line',
                    'vertices': 2,
                    'area': 0,
                    'center': center,
                    'bbox': (min(x1, x2), min(y1, y2), 
                            abs(x2 - x1), abs(y2 - y1)),
                    'length': length,
                    'endpoints': [(x1, y1), (x2, y2)]
                }
                shapes.append(shape)
        
        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                  param1=100, param2=30, minRadius=10, maxRadius=200)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                area = np.pi * radius * radius
                
                shape = {
                    'type': 'circle',
                    'vertices': 0,
                    'area': area,
                    'center': center,
                    'bbox': (center[0] - radius, center[1] - radius,
                            2 * radius, 2 * radius),
                    'radius': radius
                }
                shapes.append(shape)
        
        return shapes
    
    def get_name(self) -> str:
        return self.name

class PerformanceBenchmark:
    """Performance benchmarking system"""
    
    def __init__(self):
        self.detectors = []
        self.results = []
        self.ground_truth = None
        self.frame_buffer = []
        
    def add_detector(self, detector: ShapeDetectorBase):
        """Add a detector to benchmark"""
        self.detectors.append(detector)
    
    def generate_test_frames(self, num_frames: int = 100) -> List[np.ndarray]:
        """Generate synthetic test frames with known shapes"""
        frames = []
        ground_truth = []
        
        for i in range(num_frames):
            # Create blank frame
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
            frame_shapes = []
            
            # Add random shapes
            num_shapes = np.random.randint(5, 15)
            
            for _ in range(num_shapes):
                shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
                color = (np.random.randint(0, 255), 
                        np.random.randint(0, 255),
                        np.random.randint(0, 255))
                
                if shape_type == 'circle':
                    center = (np.random.randint(50, 1230), 
                             np.random.randint(50, 670))
                    radius = np.random.randint(20, 100)
                    cv2.circle(frame, center, radius, color, -1)
                    frame_shapes.append({
                        'type': 'circle',
                        'center': center,
                        'radius': radius
                    })
                
                elif shape_type == 'rectangle':
                    x = np.random.randint(50, 1100)
                    y = np.random.randint(50, 600)
                    w = np.random.randint(50, 150)
                    h = np.random.randint(50, 150)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
                    frame_shapes.append({
                        'type': 'rectangle',
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    })
                
                elif shape_type == 'triangle':
                    pts = np.array([
                        [np.random.randint(50, 1230), np.random.randint(50, 670)],
                        [np.random.randint(50, 1230), np.random.randint(50, 670)],
                        [np.random.randint(50, 1230), np.random.randint(50, 670)]
                    ], np.int32)
                    cv2.fillPoly(frame, [pts], color)
                    center = np.mean(pts, axis=0).astype(int)
                    frame_shapes.append({
                        'type': 'triangle',
                        'vertices': pts.tolist(),
                        'center': tuple(center)
                    })
            
            frames.append(frame)
            ground_truth.append(frame_shapes)
        
        self.ground_truth = ground_truth
        return frames
    
    def measure_system_resources(self) -> Dict[str, float]:
        """Measure current system resource usage"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU usage (if available)
        gpu_percent = 0
        gpu_memory = 0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                gpu_memory = gpu.memoryUtil * 100
        except:
            pass
        
        return {
            'cpu': cpu_percent,
            'memory': memory_percent,
            'gpu': gpu_percent,
            'gpu_memory': gpu_memory
        }
    
    def calculate_accuracy_metrics(self, detected: List[Dict], 
                                 ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        # Simple matching based on center distance and type
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        matched_gt = set()
        
        for det in detected:
            matched = False
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue
                
                # Check type
                if det['type'] != gt['type']:
                    continue
                
                # Check center distance
                det_center = det['center']
                gt_center = gt['center']
                dist = np.sqrt((det_center[0] - gt_center[0])**2 + 
                             (det_center[1] - gt_center[1])**2)
                
                if dist < 50:  # 50 pixel threshold
                    true_positives += 1
                    matched_gt.add(i)
                    matched = True
                    break
            
            if not matched:
                false_positives += 1
        
        false_negatives = len(ground_truth) - len(matched_gt)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / len(ground_truth) if ground_truth else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def benchmark_detector(self, detector: ShapeDetectorBase, 
                         frames: List[np.ndarray]) -> BenchmarkResult:
        """Benchmark a single detector"""
        print(f"\nBenchmarking {detector.get_name()}...")
        
        latencies = []
        all_detected = []
        resource_usage = []
        
        # Warm up
        for _ in range(5):
            detector.detect(frames[0])
        
        # Benchmark
        start_time = time.time()
        
        for i, frame in enumerate(frames):
            # Measure resources before
            resources_before = self.measure_system_resources()
            
            # Time detection
            detect_start = time.time()
            detected_shapes = detector.detect(frame)
            detect_end = time.time()
            
            # Measure resources after
            resources_after = self.measure_system_resources()
            
            # Store results
            latency = (detect_end - detect_start) * 1000  # ms
            latencies.append(latency)
            all_detected.append(detected_shapes)
            
            # Average resource usage
            avg_resources = {
                key: (resources_before[key] + resources_after[key]) / 2
                for key in resources_before
            }
            resource_usage.append(avg_resources)
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(frames)}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        fps = len(frames) / total_time
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        # Average resource usage
        avg_cpu = np.mean([r['cpu'] for r in resource_usage])
        avg_memory = np.mean([r['memory'] for r in resource_usage])
        avg_gpu = np.mean([r['gpu'] for r in resource_usage])
        avg_gpu_memory = np.mean([r['gpu_memory'] for r in resource_usage])
        
        # Calculate accuracy metrics
        total_shapes = 0
        accuracy_metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        if self.ground_truth:
            all_metrics = []
            for detected, gt in zip(all_detected, self.ground_truth):
                metrics = self.calculate_accuracy_metrics(detected, gt)
                all_metrics.append(metrics)
                total_shapes += len(detected)
            
            # Average metrics
            for key in accuracy_metrics:
                accuracy_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        result = BenchmarkResult(
            method_name=detector.get_name(),
            fps=fps,
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            gpu_usage=avg_gpu,
            gpu_memory=avg_gpu_memory,
            accuracy=accuracy_metrics['accuracy'],
            precision=accuracy_metrics['precision'],
            recall=accuracy_metrics['recall'],
            f1_score=accuracy_metrics['f1_score'],
            shapes_detected=total_shapes // len(frames),
            false_positives=int(accuracy_metrics['false_positives']),
            false_negatives=int(accuracy_metrics['false_negatives'])
        )
        
        return result
    
    def run_benchmark(self, num_frames: int = 100):
        """Run full benchmark suite"""
        print("Generating test frames...")
        frames = self.generate_test_frames(num_frames)
        self.frame_buffer = frames
        
        print(f"Generated {len(frames)} test frames")
        print(f"Running benchmarks on {len(self.detectors)} detectors...")
        
        self.results = []
        
        for detector in self.detectors:
            result = self.benchmark_detector(detector, frames)
            self.results.append(result)
            print(f"  Completed: {result.method_name}")
            print(f"    FPS: {result.fps:.2f}")
            print(f"    Latency: {result.avg_latency:.2f}ms")
            print(f"    Accuracy: {result.accuracy:.2%}")
    
    def visualize_results(self):
        """Create visualization of benchmark results"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Geometry Detection Benchmark Results', fontsize=16)
        
        # Extract data
        methods = [r.method_name for r in self.results]
        fps_values = [r.fps for r in self.results]
        latencies = [r.avg_latency for r in self.results]
        accuracies = [r.accuracy * 100 for r in self.results]
        cpu_usage = [r.cpu_usage for r in self.results]
        gpu_usage = [r.gpu_usage for r in self.results]
        f1_scores = [r.f1_score for r in self.results]
        
        # 1. FPS comparison
        ax = axes[0, 0]
        bars = ax.bar(methods, fps_values, color='green')
        ax.set_title('Frames Per Second (FPS)')
        ax.set_ylabel('FPS')
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, fps_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom')
        
        # 2. Latency comparison
        ax = axes[0, 1]
        bars = ax.bar(methods, latencies, color='orange')
        ax.set_title('Average Latency')
        ax.set_ylabel('Latency (ms)')
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        for bar, val in zip(bars, latencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom')
        
        # 3. Accuracy comparison
        ax = axes[0, 2]
        bars = ax.bar(methods, accuracies, color='blue')
        ax.set_title('Detection Accuracy')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 110)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        for bar, val in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom')
        
        # 4. Resource usage
        ax = axes[1, 0]
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cpu_usage, width, label='CPU', color='red')
        bars2 = ax.bar(x + width/2, gpu_usage, width, label='GPU', color='purple')
        
        ax.set_title('Resource Usage')
        ax.set_ylabel('Usage (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        
        # 5. F1 Score
        ax = axes[1, 1]
        bars = ax.bar(methods, f1_scores, color='teal')
        ax.set_title('F1 Score')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1.1)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        for bar, val in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom')
        
        # 6. Overall comparison radar chart
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create text summary
        summary_text = "Performance Summary:\n\n"
        for result in self.results:
            summary_text += f"{result.method_name}:\n"
            summary_text += f"  Speed: {result.fps:.1f} FPS\n"
            summary_text += f"  Accuracy: {result.accuracy:.1%}\n"
            summary_text += f"  Efficiency: {result.f1_score:.3f}\n\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_results(self, filename: str = 'benchmark_results.json'):
        """Export benchmark results to file"""
        if not self.results:
            print("No results to export")
            return
        
        # Convert results to dict
        results_dict = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_frames': len(self.frame_buffer),
            'results': [asdict(r) for r in self.results]
        }
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        # Also save to CSV
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(filename.replace('.json', '.csv'), index=False)
        
        print(f"Results exported to {filename} and {filename.replace('.json', '.csv')}")
    
    def compare_on_real_video(self, video_path: str = 0, max_frames: int = 300):
        """Compare detectors on real video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Failed to open video")
            return
        
        print("\nComparing on real video...")
        
        # Create comparison window
        window_name = 'Detector Comparison'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_count = 0
        detector_index = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get current detector
            detector = self.detectors[detector_index]
            
            # Detect shapes
            start_time = time.time()
            shapes = detector.detect(frame)
            detect_time = (time.time() - start_time) * 1000
            
            # Draw results
            output = frame.copy()
            
            # Draw shapes
            for shape in shapes:
                if 'contour' in shape:
                    cv2.drawContours(output, [shape['contour']], -1, (0, 255, 0), 2)
                
                # Draw bounding box
                if 'bbox' in shape:
                    x, y, w, h = shape['bbox']
                    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)
                
                # Draw center
                if 'center' in shape:
                    cv2.circle(output, shape['center'], 5, (0, 0, 255), -1)
                
                # Label
                if 'center' in shape and 'type' in shape:
                    cv2.putText(output, shape['type'], 
                               (shape['center'][0] - 30, shape['center'][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Add info
            info_text = [
                f"Method: {detector.get_name()}",
                f"Shapes: {len(shapes)}",
                f"Time: {detect_time:.1f}ms",
                f"FPS: {1000/detect_time:.1f}",
                "",
                "Press SPACE to switch detector",
                "Press Q to quit"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(output, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
            
            cv2.imshow(window_name, output)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                detector_index = (detector_index + 1) % len(self.detectors)
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main benchmark function"""
    # Create benchmark
    benchmark = PerformanceBenchmark()
    
    # Add detectors
    benchmark.add_detector(ClassicalCVDetector())
    benchmark.add_detector(GPUAcceleratedDetector())
    benchmark.add_detector(HoughTransformDetector())
    
    # Run benchmark
    print("Starting Geometry Detection Performance Benchmark")
    print("=" * 50)
    
    # Synthetic benchmark
    benchmark.run_benchmark(num_frames=100)
    
    # Visualize results
    benchmark.visualize_results()
    
    # Export results
    benchmark.export_results()
    
    # Real video comparison
    print("\nPress any key to start real video comparison...")
    cv2.waitKey(0)
    benchmark.compare_on_real_video(video_path=0, max_frames=300)
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()
