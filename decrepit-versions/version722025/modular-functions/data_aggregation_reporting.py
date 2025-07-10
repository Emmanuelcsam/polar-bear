#!/usr/bin/env python3
"""
Data Aggregation and Reporting - Standalone Module
Extracted from fiber optic defect detection system
Aggregates results and generates comprehensive reports
"""

import os
import json
import csv
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import glob


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays"""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
        except ImportError:
            pass
        return super().default(obj)


@dataclass
class ProcessingResult:
    """Represents results from processing a single image"""
    image_path: str
    timestamp: str
    success: bool
    method: str
    center: Optional[Tuple[float, float]] = None
    core_radius: Optional[float] = None
    cladding_radius: Optional[float] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class QualityMetrics:
    """Quality metrics for analysis results"""
    total_processed: int = 0
    successful_detections: int = 0
    failed_detections: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_processing_time: float = 0.0
    avg_core_radius: float = 0.0
    avg_cladding_radius: float = 0.0
    std_core_radius: float = 0.0
    std_cladding_radius: float = 0.0


class DataAggregator:
    """Aggregates and analyzes processing results"""
    
    def __init__(self):
        self.results: List[ProcessingResult] = []
        self.quality_metrics: Optional[QualityMetrics] = None
    
    def load_results_from_directory(self, results_dir: str, pattern: str = "*.json"):
        """
        Load results from JSON files in a directory.
        
        Args:
            results_dir (str): Directory containing result files
            pattern (str): File pattern to match (default: "*.json")
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        json_files = list(results_path.glob(pattern))
        print(f"Found {len(json_files)} result files in {results_dir}")
        
        loaded_count = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to ProcessingResult
                result = self._dict_to_processing_result(data)
                if result:
                    self.results.append(result)
                    loaded_count += 1
                    
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
        
        print(f"Successfully loaded {loaded_count} results")
    
    def _dict_to_processing_result(self, data: Dict[str, Any]) -> Optional[ProcessingResult]:
        """Convert dictionary data to ProcessingResult"""
        try:
            # Extract required fields
            image_path = data.get('image_path', '')
            method = data.get('method', 'unknown')
            success = data.get('success', False)
            
            # Extract optional fields
            center = data.get('center')
            if center and isinstance(center, (list, tuple)) and len(center) == 2:
                center = (float(center[0]), float(center[1]))
            else:
                center = None
            
            core_radius = data.get('core_radius')
            if core_radius is not None:
                core_radius = float(core_radius)
            
            cladding_radius = data.get('cladding_radius')
            if cladding_radius is not None:
                cladding_radius = float(cladding_radius)
            
            confidence = float(data.get('confidence', 0.0))
            processing_time = float(data.get('processing_time', 0.0))
            error = data.get('error')
            
            # Handle timestamp
            timestamp = data.get('timestamp', datetime.now().isoformat())
            
            # Additional data
            additional_data = {k: v for k, v in data.items() 
                             if k not in ['image_path', 'method', 'success', 'center', 
                                        'core_radius', 'cladding_radius', 'confidence', 
                                        'processing_time', 'error', 'timestamp']}
            
            return ProcessingResult(
                image_path=image_path,
                timestamp=timestamp,
                success=success,
                method=method,
                center=center,
                core_radius=core_radius,
                cladding_radius=cladding_radius,
                confidence=confidence,
                processing_time=processing_time,
                error=error,
                additional_data=additional_data if additional_data else None
            )
            
        except Exception as e:
            print(f"Warning: Failed to parse result data: {e}")
            return None
    
    def add_result(self, result: ProcessingResult):
        """Add a single result to the aggregator"""
        self.results.append(result)
    
    def calculate_quality_metrics(self) -> QualityMetrics:
        """Calculate quality metrics from loaded results"""
        if not self.results:
            return QualityMetrics()
        
        successful_results = [r for r in self.results if r.success]
        
        # Basic counts
        total_processed = len(self.results)
        successful_detections = len(successful_results)
        failed_detections = total_processed - successful_detections
        success_rate = successful_detections / total_processed if total_processed > 0 else 0.0
        
        # Average metrics for successful results
        confidences = [r.confidence for r in successful_results if r.confidence is not None]
        processing_times = [r.processing_time for r in self.results if r.processing_time is not None]
        core_radii = [r.core_radius for r in successful_results if r.core_radius is not None]
        cladding_radii = [r.cladding_radius for r in successful_results if r.cladding_radius is not None]
        
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
        avg_core_radius = statistics.mean(core_radii) if core_radii else 0.0
        avg_cladding_radius = statistics.mean(cladding_radii) if cladding_radii else 0.0
        
        # Standard deviations
        std_core_radius = statistics.stdev(core_radii) if len(core_radii) > 1 else 0.0
        std_cladding_radius = statistics.stdev(cladding_radii) if len(cladding_radii) > 1 else 0.0
        
        self.quality_metrics = QualityMetrics(
            total_processed=total_processed,
            successful_detections=successful_detections,
            failed_detections=failed_detections,
            success_rate=success_rate,
            avg_confidence=avg_confidence,
            avg_processing_time=avg_processing_time,
            avg_core_radius=avg_core_radius,
            avg_cladding_radius=avg_cladding_radius,
            std_core_radius=std_core_radius,
            std_cladding_radius=std_cladding_radius
        )
        
        return self.quality_metrics
    
    def get_method_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics broken down by method"""
        method_results = {}
        
        for result in self.results:
            method = result.method
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)
        
        method_performance = {}
        for method, results in method_results.items():
            successful = [r for r in results if r.success]
            
            method_performance[method] = {
                'total_attempts': len(results),
                'successful': len(successful),
                'success_rate': len(successful) / len(results) if results else 0.0,
                'avg_confidence': statistics.mean([r.confidence for r in successful if r.confidence]) if successful else 0.0,
                'avg_processing_time': statistics.mean([r.processing_time for r in results if r.processing_time]) if results else 0.0
            }
        
        return method_performance
    
    def get_outliers(self, metric: str = 'confidence', threshold: float = 2.0) -> List[ProcessingResult]:
        """
        Find outliers in the results based on a specific metric.
        
        Args:
            metric (str): Metric to analyze ('confidence', 'core_radius', 'cladding_radius', 'processing_time')
            threshold (float): Number of standard deviations for outlier detection
            
        Returns:
            List[ProcessingResult]: Results that are outliers
        """
        successful_results = [r for r in self.results if r.success]
        
        # Extract metric values
        if metric == 'confidence':
            values = [r.confidence for r in successful_results if r.confidence is not None]
        elif metric == 'core_radius':
            values = [r.core_radius for r in successful_results if r.core_radius is not None]
        elif metric == 'cladding_radius':
            values = [r.cladding_radius for r in successful_results if r.cladding_radius is not None]
        elif metric == 'processing_time':
            values = [r.processing_time for r in self.results if r.processing_time is not None]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if len(values) < 2:
            return []
        
        # Calculate mean and standard deviation
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        # Find outliers
        outliers = []
        for result in self.results:
            if metric == 'confidence' and result.confidence is not None:
                value = result.confidence
            elif metric == 'core_radius' and result.core_radius is not None:
                value = result.core_radius
            elif metric == 'cladding_radius' and result.cladding_radius is not None:
                value = result.cladding_radius
            elif metric == 'processing_time' and result.processing_time is not None:
                value = result.processing_time
            else:
                continue
            
            if abs(value - mean_val) > threshold * std_val:
                outliers.append(result)
        
        return outliers
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        if not self.quality_metrics:
            self.calculate_quality_metrics()
        
        method_performance = self.get_method_performance()
        
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_results_analyzed': len(self.results),
            'quality_metrics': {
                'success_rate': self.quality_metrics.success_rate,
                'avg_confidence': self.quality_metrics.avg_confidence,
                'avg_processing_time': self.quality_metrics.avg_processing_time,
                'avg_core_radius': self.quality_metrics.avg_core_radius,
                'avg_cladding_radius': self.quality_metrics.avg_cladding_radius,
                'std_core_radius': self.quality_metrics.std_core_radius,
                'std_cladding_radius': self.quality_metrics.std_cladding_radius
            },
            'method_performance': method_performance,
            'failed_analyses': [
                {
                    'image_path': r.image_path,
                    'method': r.method,
                    'error': r.error,
                    'timestamp': r.timestamp
                }
                for r in self.results if not r.success
            ]
        }
        
        return report
    
    def export_to_csv(self, output_file: str):
        """Export results to CSV format"""
        fieldnames = [
            'image_path', 'timestamp', 'method', 'success', 'center_x', 'center_y',
            'core_radius', 'cladding_radius', 'confidence', 'processing_time', 'error'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'image_path': result.image_path,
                    'timestamp': result.timestamp,
                    'method': result.method,
                    'success': result.success,
                    'center_x': result.center[0] if result.center else None,
                    'center_y': result.center[1] if result.center else None,
                    'core_radius': result.core_radius,
                    'cladding_radius': result.cladding_radius,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time,
                    'error': result.error
                }
                writer.writerow(row)
    
    def create_html_report(self, output_file: str):
        """Create an HTML report with visualizations"""
        if not self.quality_metrics:
            self.calculate_quality_metrics()
        
        method_performance = self.get_method_performance()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fiber Optic Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c5aa0; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .error-list {{ max-height: 300px; overflow-y: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Fiber Optic Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total images analyzed: {len(self.results)}</p>
    </div>
    
    <div class="section">
        <h2>Quality Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{self.quality_metrics.success_rate:.1%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.quality_metrics.avg_confidence:.3f}</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.quality_metrics.avg_processing_time:.3f}s</div>
                <div class="metric-label">Avg Processing Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.quality_metrics.avg_core_radius:.1f}px</div>
                <div class="metric-label">Avg Core Radius</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.quality_metrics.avg_cladding_radius:.1f}px</div>
                <div class="metric-label">Avg Cladding Radius</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Method Performance</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>Total Attempts</th>
                <th>Successful</th>
                <th>Success Rate</th>
                <th>Avg Confidence</th>
                <th>Avg Processing Time</th>
            </tr>
        """
        
        for method, perf in method_performance.items():
            html_content += f"""
            <tr>
                <td>{method}</td>
                <td>{perf['total_attempts']}</td>
                <td>{perf['successful']}</td>
                <td>{perf['success_rate']:.1%}</td>
                <td>{perf['avg_confidence']:.3f}</td>
                <td>{perf['avg_processing_time']:.3f}s</td>
            </tr>
            """
        
        html_content += """
        </table>
    </div>
    """
        
        # Add failures section if there are any
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            html_content += f"""
    <div class="section">
        <h2>Failed Analyses ({len(failed_results)} total)</h2>
        <div class="error-list">
            <table>
                <tr>
                    <th>Image</th>
                    <th>Method</th>
                    <th>Error</th>
                    <th>Timestamp</th>
                </tr>
            """
            
            for result in failed_results:
                html_content += f"""
                <tr>
                    <td>{os.path.basename(result.image_path)}</td>
                    <td>{result.method}</td>
                    <td class="failure">{result.error or 'Unknown error'}</td>
                    <td>{result.timestamp}</td>
                </tr>
                """
            
            html_content += """
            </table>
        </div>
    </div>
            """
        
        html_content += """
</body>
</html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Command line interface for data aggregation and reporting"""
    parser = argparse.ArgumentParser(description='Data Aggregation and Reporting for Fiber Optic Analysis')
    parser.add_argument('results_dir', help='Directory containing result JSON files')
    parser.add_argument('--pattern', default='*.json', help='File pattern to match (default: *.json)')
    parser.add_argument('--output-dir', default='reports', help='Output directory for reports (default: reports)')
    parser.add_argument('--csv', action='store_true', help='Generate CSV report')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--json', action='store_true', help='Generate JSON summary report')
    parser.add_argument('--outliers', help='Find outliers for metric (confidence, core_radius, cladding_radius, processing_time)')
    parser.add_argument('--threshold', type=float, default=2.0, help='Outlier threshold in standard deviations (default: 2.0)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create aggregator and load results
    aggregator = DataAggregator()
    print(f"Loading results from: {args.results_dir}")
    aggregator.load_results_from_directory(args.results_dir, args.pattern)
    
    if not aggregator.results:
        print("No results found to analyze")
        return 1
    
    # Calculate quality metrics
    print("Calculating quality metrics...")
    metrics = aggregator.calculate_quality_metrics()
    
    # Print basic statistics
    print(f"\n=== Analysis Summary ===")
    print(f"Total processed: {metrics.total_processed}")
    print(f"Successful: {metrics.successful_detections}")
    print(f"Failed: {metrics.failed_detections}")
    print(f"Success rate: {metrics.success_rate:.1%}")
    print(f"Average confidence: {metrics.avg_confidence:.3f}")
    print(f"Average processing time: {metrics.avg_processing_time:.3f}s")
    print(f"Average core radius: {metrics.avg_core_radius:.1f}px")
    print(f"Average cladding radius: {metrics.avg_cladding_radius:.1f}px")
    
    # Method performance
    if args.verbose:
        print(f"\n=== Method Performance ===")
        method_perf = aggregator.get_method_performance()
        for method, perf in method_perf.items():
            print(f"{method}:")
            print(f"  Success rate: {perf['success_rate']:.1%} ({perf['successful']}/{perf['total_attempts']})")
            print(f"  Avg confidence: {perf['avg_confidence']:.3f}")
            print(f"  Avg processing time: {perf['avg_processing_time']:.3f}s")
    
    # Find outliers
    if args.outliers:
        print(f"\n=== Outliers ({args.outliers}) ===")
        outliers = aggregator.get_outliers(args.outliers, args.threshold)
        if outliers:
            print(f"Found {len(outliers)} outliers:")
            for outlier in outliers:
                print(f"  {os.path.basename(outlier.image_path)}: {getattr(outlier, args.outliers, 'N/A')}")
        else:
            print("No outliers found")
    
    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.csv:
        csv_file = os.path.join(args.output_dir, f"analysis_results_{timestamp}.csv")
        aggregator.export_to_csv(csv_file)
        print(f"CSV report saved: {csv_file}")
    
    if args.html:
        html_file = os.path.join(args.output_dir, f"analysis_report_{timestamp}.html")
        aggregator.create_html_report(html_file)
        print(f"HTML report saved: {html_file}")
    
    if args.json:
        json_file = os.path.join(args.output_dir, f"analysis_summary_{timestamp}.json")
        summary = aggregator.generate_summary_report()
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=4, cls=NumpyEncoder)
        print(f"JSON summary saved: {json_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
