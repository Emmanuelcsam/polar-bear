#!/usr/bin/env python3
"""
Unified Reporting Module for Fiber Optic Inspection
==================================================
This module combines all reporting functionality for fiber optic defect analysis:
- CSV report generation
- Data aggregation and statistics
- Individual image reports
- Pass/fail criteria evaluation
- Comprehensive reporting with visualizations
"""

import argparse
import csv
import datetime
import glob
import json
import logging
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures and Utilities
# ============================================================================

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


# ============================================================================
# CSV Report Generator
# ============================================================================

class CSVReportGenerator:
    """
    Standalone CSV report generator for fiber optic inspection results.
    """
    
    def __init__(self):
        """Initialize the CSV report generator."""
        pass
    
    def generate_defect_csv_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Path
    ) -> bool:
        """
        Generate a detailed CSV report of defects.
        
        Args:
            analysis_results: Dictionary containing analysis results with defects
            output_path: Path where the CSV file will be saved
            
        Returns:
            True if report was generated successfully, False otherwise
        """
        try:
            defects_list = analysis_results.get("characterized_defects", [])
            
            if not defects_list:
                logger.info(f"No defects to report for {output_path.name}. Creating empty CSV with headers.")
                return self._create_empty_csv(output_path)
            
            # Create DataFrame from defects
            df = pd.DataFrame(defects_list)
            
            # Define desired column order
            report_columns = [
                "defect_id", "zone", "classification", "confidence_score",
                "centroid_x_px", "centroid_y_px",
                "length_um", "width_um", "effective_diameter_um", "area_um2",
                "length_px", "width_px", "area_px",
                "aspect_ratio_oriented",
                "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
                "rotated_rect_center_px", "rotated_rect_angle_deg"
            ]
            
            # Select only columns that exist in the DataFrame
            final_columns = [col for col in report_columns if col in df.columns]
            df_report = df[final_columns]
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df_report.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Defect CSV report saved successfully to '{output_path}'.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save defect CSV report to '{output_path}': {e}")
            return False
    
    def _create_empty_csv(self, output_path: Path) -> bool:
        """
        Create an empty CSV file with standard headers.
        
        Args:
            output_path: Path where the empty CSV will be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cols = [
                "defect_id", "zone", "classification", "confidence_score",
                "centroid_x_px", "centroid_y_px", "area_px", "length_px", "width_px",
                "aspect_ratio_oriented", "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
                "area_um2", "length_um", "width_um", "effective_diameter_um",
                "rotated_rect_center_px", "rotated_rect_angle_deg"
            ]
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame([], columns=cols)
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Empty defect report CSV (with headers) saved to '{output_path}'.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save empty defect report to '{output_path}': {e}")
            return False
    
    def generate_summary_csv_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Path,
        include_metadata: bool = True
    ) -> bool:
        """
        Generate a summary CSV report with overall statistics.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_path: Path where the summary CSV will be saved
            include_metadata: Whether to include metadata in the report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            defects_list = analysis_results.get("characterized_defects", [])
            
            # Calculate summary statistics
            summary_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_defects": len(defects_list),
                "overall_status": analysis_results.get("overall_status", "UNKNOWN"),
                "scratches": len([d for d in defects_list if d.get("classification") == "Scratch"]),
                "pits_digs": len([d for d in defects_list if d.get("classification") == "Pit/Dig"]),
                "other_defects": len([d for d in defects_list if d.get("classification") not in ["Scratch", "Pit/Dig"]]),
            }
            
            # Add failure reasons if available
            failure_reasons = analysis_results.get("failure_reasons", [])
            summary_data["failure_reasons"] = "; ".join(failure_reasons) if failure_reasons else "None"
            
            # Add zone-based statistics
            zone_stats = self._calculate_zone_statistics(defects_list)
            summary_data.update(zone_stats)
            
            # Add size statistics
            size_stats = self._calculate_size_statistics(defects_list)
            summary_data.update(size_stats)
            
            # Create DataFrame and save
            df_summary = pd.DataFrame([summary_data])
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df_summary.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Summary CSV report saved to '{output_path}'.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save summary CSV report to '{output_path}': {e}")
            return False
    
    def _calculate_zone_statistics(self, defects_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics by zone."""
        zone_counts = {}
        zones = ["Core", "Cladding", "Adhesive", "Contact", "Unknown"]
        
        for zone in zones:
            zone_defects = [d for d in defects_list if d.get("zone") == zone]
            zone_counts[f"{zone.lower()}_defects"] = len(zone_defects)
        
        return zone_counts
    
    def _calculate_size_statistics(self, defects_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate size-based statistics."""
        if not defects_list:
            return {
                "avg_defect_area_um2": 0,
                "max_defect_area_um2": 0,
                "min_defect_area_um2": 0,
                "avg_confidence": 0,
                "max_confidence": 0,
                "min_confidence": 0
            }
        
        areas = [d.get("area_um2", 0) for d in defects_list if d.get("area_um2") is not None]
        confidences = [d.get("confidence_score", 0) for d in defects_list if d.get("confidence_score") is not None]
        
        return {
            "avg_defect_area_um2": sum(areas) / len(areas) if areas else 0,
            "max_defect_area_um2": max(areas) if areas else 0,
            "min_defect_area_um2": min(areas) if areas else 0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0
        }
    
    def generate_comparative_report(
        self,
        multiple_results: List[Dict[str, Any]],
        output_path: Path,
        image_names: Optional[List[str]] = None
    ) -> bool:
        """
        Generate a comparative CSV report for multiple images.
        
        Args:
            multiple_results: List of analysis results from multiple images
            output_path: Path where the comparative CSV will be saved
            image_names: Optional list of image names for identification
            
        Returns:
            True if successful, False otherwise
        """
        try:
            comparative_data = []
            
            for i, results in enumerate(multiple_results):
                image_name = image_names[i] if image_names and i < len(image_names) else f"Image_{i+1}"
                defects_list = results.get("characterized_defects", [])
                
                # Calculate statistics for this image
                summary_data = {
                    "image_name": image_name,
                    "total_defects": len(defects_list),
                    "overall_status": results.get("overall_status", "UNKNOWN"),
                    "scratches": len([d for d in defects_list if d.get("classification") == "Scratch"]),
                    "pits_digs": len([d for d in defects_list if d.get("classification") == "Pit/Dig"]),
                }
                
                # Add zone statistics
                zone_stats = self._calculate_zone_statistics(defects_list)
                summary_data.update(zone_stats)
                
                # Add size statistics
                size_stats = self._calculate_size_statistics(defects_list)
                summary_data.update(size_stats)
                
                comparative_data.append(summary_data)
            
            # Create DataFrame and save
            df_comparative = pd.DataFrame(comparative_data)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df_comparative.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Comparative CSV report saved to '{output_path}'.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save comparative CSV report to '{output_path}': {e}")
            return False
    
    def load_analysis_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load analysis results from JSON file.
        
        Args:
            file_path: Path to JSON file containing analysis results
            
        Returns:
            Analysis results dictionary or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded analysis results from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load analysis results from {file_path}: {e}")
            return None


# ============================================================================
# Data Aggregator
# ============================================================================

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
            timestamp = data.get('timestamp', datetime.datetime.now().isoformat())
            
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
            'generation_timestamp': datetime.datetime.now().isoformat(),
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
        <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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


# ============================================================================
# Pass/Fail Rules Engine
# ============================================================================

class PassFailRulesEngine:
    """
    Advanced rules engine for fiber optic inspection pass/fail evaluation.
    """
    
    def __init__(self, rules_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the rules engine with configuration.
        
        Args:
            rules_config: Dictionary containing rules configuration
        """
        self.rules_config = rules_config or self._get_default_rules()
        
    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default IEC 61300-3-35 compliant rules."""
        return {
            "single_mode_pc": {
                "Core": {
                    "max_scratches": 0,
                    "max_defects": 0,
                    "max_defect_size_um": 3,
                    "max_total_defect_area_um2": 0,
                    "critical_zone": True
                },
                "Cladding": {
                    "max_scratches": 5,
                    "max_scratches_gt_5um": 0,
                    "max_defects": 5,
                    "max_defect_size_um": 10,
                    "max_total_defect_area_um2": 100,
                    "critical_zone": False
                }
            },
            "multi_mode_pc": {
                "Core": {
                    "max_scratches": 1,
                    "max_scratch_length_um": 10,
                    "max_defects": 3,
                    "max_defect_size_um": 5,
                    "max_total_defect_area_um2": 25,
                    "critical_zone": True
                },
                "Cladding": {
                    "max_scratches": "unlimited",
                    "max_defects": "unlimited",
                    "max_defect_size_um": 20,
                    "max_total_defect_area_um2": 200,
                    "critical_zone": False
                }
            }
        }
    
    def evaluate_zone_rules(self, 
                           defects: List[Dict[str, Any]], 
                           zone_name: str,
                           fiber_type: str,
                           zone_rules: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Evaluate pass/fail rules for a specific zone.
        
        Args:
            defects: List of defects in the zone
            zone_name: Name of the zone
            fiber_type: Type of fiber
            zone_rules: Rules configuration for the zone
            
        Returns:
            Tuple of (status, failure_reasons)
        """
        status = "PASS"
        failure_reasons = []
        
        # Separate defects by classification
        scratches = [d for d in defects if d.get("classification") == "Scratch"]
        pits_digs = [d for d in defects if d.get("classification") in ["Pit", "Dig"]]
        all_non_scratch = [d for d in defects if d.get("classification") != "Scratch"]
        
        # Check critical zone rule (any defect fails)
        if zone_rules.get("critical_zone", False) and len(defects) > 0:
            status = "FAIL"
            failure_reasons.append(f"Zone '{zone_name}': Critical zone contains {len(defects)} defect(s)")
            return status, failure_reasons
        
        # Check scratch count
        max_scratches = zone_rules.get("max_scratches")
        if isinstance(max_scratches, int) and len(scratches) > max_scratches:
            status = "FAIL"
            failure_reasons.append(
                f"Zone '{zone_name}': Too many scratches ({len(scratches)} > {max_scratches})"
            )
        
        # Check pit/dig count
        max_defects = zone_rules.get("max_defects")
        if isinstance(max_defects, int) and len(pits_digs) > max_defects:
            status = "FAIL"
            failure_reasons.append(
                f"Zone '{zone_name}': Too many pits/digs ({len(pits_digs)} > {max_defects})"
            )
        
        # Check individual defect sizes
        max_defect_size_um = zone_rules.get("max_defect_size_um")
        max_scratch_length_um = zone_rules.get("max_scratch_length_um")
        
        for defect in defects:
            defect_type = defect.get("classification", "Unknown")
            primary_dimension_um = defect.get("length_um")
            
            if primary_dimension_um is None:
                continue
            
            # Determine size limit based on defect type
            if defect_type == "Scratch":
                size_limit = max_scratch_length_um if max_scratch_length_um is not None else max_defect_size_um
            else:
                size_limit = max_defect_size_um
            
            if size_limit is not None and primary_dimension_um > size_limit:
                status = "FAIL"
                reason = (f"Zone '{zone_name}': {defect_type} '{defect.get('defect_id', 'Unknown')}' "
                         f"size ({primary_dimension_um:.2f}µm) exceeds limit ({size_limit}µm)")
                failure_reasons.append(reason)
        
        # Check total defect area
        max_total_area = zone_rules.get("max_total_defect_area_um2")
        if max_total_area is not None:
            total_area = sum(d.get("area_um2", 0) for d in defects)
            if total_area > max_total_area:
                status = "FAIL"
                failure_reasons.append(
                    f"Zone '{zone_name}': Total defect area ({total_area:.2f}µm²) exceeds limit ({max_total_area}µm²)"
                )
        
        # Check scratches greater than threshold
        scratches_gt_threshold = zone_rules.get("max_scratches_gt_5um")
        if scratches_gt_threshold is not None:
            large_scratches = [s for s in scratches if s.get("length_um", 0) > 5.0]
            if len(large_scratches) > scratches_gt_threshold:
                status = "FAIL"
                failure_reasons.append(
                    f"Zone '{zone_name}': Too many scratches > 5µm ({len(large_scratches)} > {scratches_gt_threshold})"
                )
        
        return status, failure_reasons
    
    def apply_pass_fail_rules(self, 
                             characterized_defects: List[Dict[str, Any]],
                             fiber_type: str) -> Tuple[str, List[str]]:
        """
        Apply comprehensive pass/fail rules to characterized defects.
        
        Args:
            characterized_defects: List of characterized defect dictionaries
            fiber_type: Fiber type key (e.g., "single_mode_pc")
            
        Returns:
            Tuple of (overall_status, failure_reasons)
        """
        overall_status = "PASS"
        all_failure_reasons = []
        
        # Get rules for this fiber type
        fiber_rules = self.rules_config.get(fiber_type)
        if not fiber_rules:
            error_msg = f"No pass/fail rules defined for fiber type '{fiber_type}'"
            logger.error(error_msg)
            return "ERROR_CONFIG", [error_msg]
        
        # Group defects by zone
        defects_by_zone = {}
        for defect in characterized_defects:
            zone_name = defect.get("zone", "Unknown")
            if zone_name not in defects_by_zone:
                defects_by_zone[zone_name] = []
            defects_by_zone[zone_name].append(defect)
        
        # Evaluate each zone
        for zone_name, zone_rules in fiber_rules.items():
            zone_defects = defects_by_zone.get(zone_name, [])
            
            zone_status, zone_reasons = self.evaluate_zone_rules(
                zone_defects, zone_name, fiber_type, zone_rules
            )
            
            if zone_status == "FAIL":
                overall_status = "FAIL"
                all_failure_reasons.extend(zone_reasons)
        
        # Log results
        if overall_status == "PASS":
            logger.info(f"Pass/Fail evaluation for '{fiber_type}': PASS")
        else:
            logger.warning(f"Pass/Fail evaluation for '{fiber_type}': FAIL - {len(all_failure_reasons)} reason(s)")
        
        return overall_status, all_failure_reasons
    
    def get_zone_statistics(self, 
                           defects: List[Dict[str, Any]],
                           zone_name: str) -> Dict[str, Any]:
        """
        Calculate detailed statistics for a zone.
        
        Args:
            defects: List of defects in the zone
            zone_name: Name of the zone
            
        Returns:
            Dictionary with zone statistics
        """
        zone_defects = [d for d in defects if d.get("zone") == zone_name]
        
        if not zone_defects:
            return {
                "zone_name": zone_name,
                "total_defects": 0,
                "scratches": 0,
                "pits_digs": 0,
                "total_area_um2": 0.0,
                "max_defect_size_um": 0.0,
                "avg_defect_size_um": 0.0
            }
        
        # Separate by type
        scratches = [d for d in zone_defects if d.get("classification") == "Scratch"]
        pits_digs = [d for d in zone_defects if d.get("classification") in ["Pit", "Dig"]]
        
        # Calculate statistics
        total_area = sum(d.get("area_um2", 0) for d in zone_defects)
        defect_sizes = [d.get("length_um", d.get("length_px", 0)) for d in zone_defects]
        
        stats = {
            "zone_name": zone_name,
            "total_defects": len(zone_defects),
            "scratches": len(scratches),
            "pits_digs": len(pits_digs),
            "total_area_um2": float(total_area),
            "max_defect_size_um": float(max(defect_sizes) if defect_sizes else 0),
            "avg_defect_size_um": float(sum(defect_sizes) / len(defect_sizes) if defect_sizes else 0)
        }
        
        return stats
    
    def generate_detailed_report(self, 
                                characterized_defects: List[Dict[str, Any]],
                                fiber_type: str) -> Dict[str, Any]:
        """
        Generate a comprehensive inspection report.
        
        Args:
            characterized_defects: List of characterized defects
            fiber_type: Fiber type key
            
        Returns:
            Detailed inspection report dictionary
        """
        # Apply pass/fail rules
        overall_status, failure_reasons = self.apply_pass_fail_rules(characterized_defects, fiber_type)
        
        # Calculate zone statistics
        zone_stats = {}
        fiber_rules = self.rules_config.get(fiber_type, {})
        
        for zone_name in fiber_rules.keys():
            zone_stats[zone_name] = self.get_zone_statistics(characterized_defects, zone_name)
        
        # Overall statistics
        total_defects = len(characterized_defects)
        total_scratches = len([d for d in characterized_defects if d.get("classification") == "Scratch"])
        total_pits_digs = len([d for d in characterized_defects if d.get("classification") in ["Pit", "Dig"]])
        
        report = {
            "fiber_type": fiber_type,
            "overall_status": overall_status,
            "failure_reasons": failure_reasons,
            "summary": {
                "total_defects": total_defects,
                "total_scratches": total_scratches,
                "total_pits_digs": total_pits_digs,
                "pass_fail_status": overall_status
            },
            "zone_statistics": zone_stats,
            "detailed_defects": characterized_defects
        }
        
        return report


# ============================================================================
# Pass/Fail Criteria Functions
# ============================================================================

def apply_pass_fail_criteria_v1(defects: List[Dict[str, Any]], 
                             pixels_per_micron: float or None) -> Dict[str, Any]:
    """Apply IEC 61300-3-35 based pass/fail criteria (version 1)"""
    # Zone-specific criteria (in microns)
    zone_criteria = {
        'core': {'max_scratch_width_um': 3, 'max_dig_diameter_um': 2, 'max_count': 5},
        'cladding': {'max_scratch_width_um': 5, 'max_dig_diameter_um': 5, 'max_count': None},
        'ferrule': {'max_defect_um': 25, 'max_count': None},
        'adhesive': {'max_defect_um': 50, 'max_count': None}
    }
    
    status = 'PASS'
    failures = []
    
    if not pixels_per_micron:
        return {
            'status': 'INCONCLUSIVE',
            'failures': ['Cannot determine pass/fail status without pixel to micron conversion.'],
            'total_defects': len(defects)
        }

    defects_by_zone = {name: [] for name in zone_criteria.keys()}
    for defect in defects:
        zone_name = defect.get('zone_name', defect.get('zone', '')).lower()
        if zone_name in defects_by_zone:
            defects_by_zone[zone_name].append(defect)
    
    for zone_name, criteria in zone_criteria.items():
        zone_defects = defects_by_zone.get(zone_name, [])
        
        if criteria.get('max_count') and len(zone_defects) > criteria['max_count']:
            status = 'FAIL'
            failures.append(f"{zone_name}: Too many defects ({len(zone_defects)} > {criteria['max_count']})")
        
        for defect in zone_defects:
            defect_type = defect.get('defect_type', defect.get('type', defect.get('classification', ''))).lower()
            if 'scratch' in defect_type and 'max_scratch_width_um' in criteria:
                width_um = defect.get('minor_dimension_um', defect.get('width_um', 0))
                if width_um > criteria['max_scratch_width_um']:
                    status = 'FAIL'
                    failures.append(f"{zone_name} scratch width > {criteria['max_scratch_width_um']}µm")
            elif 'dig' in defect_type or 'pit' in defect_type and 'max_dig_diameter_um' in criteria:
                diameter_um = defect.get('major_dimension_um', defect.get('diameter_um', defect.get('effective_diameter_um', 0)))
                if diameter_um > criteria['max_dig_diameter_um']:
                    status = 'FAIL'
                    failures.append(f"{zone_name} dig diameter > {criteria['max_dig_diameter_um']}µm")
            elif 'max_defect_um' in criteria:
                size_um = defect.get('major_dimension_um', defect.get('length_um', defect.get('effective_diameter_um', 0)))
                if size_um > criteria['max_defect_um']:
                    status = 'FAIL'
                    failures.append(f"{zone_name} defect size > {criteria['max_defect_um']}µm")

    return {
        'status': status,
        'failures': list(set(failures)), # Unique failures
        'defects_by_zone': {k: len(v) for k, v in defects_by_zone.items()},
        'total_defects': len(defects)
    }


def apply_pass_fail_criteria_v3(defects_df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    Apply IEC-61300 based pass/fail criteria (version 3 - DataFrame based)
    """
    zones_criteria = {
        "core": {"max_defect_um": 3},
        "cladding": {"max_defect_um": 10},
        "ferrule": {"max_defect_um": 20}
    }
    
    status = "PASS"
    failure_reasons = []
    
    if defects_df.empty:
        return status, failure_reasons

    for zone_name, zone_params in zones_criteria.items():
        zone_defects = defects_df[defects_df["zone"] == zone_name]
        
        if zone_defects.empty:
            continue
        
        # Check dig sizes
        digs = zone_defects[zone_defects["type"] == "dig"]
        if not digs.empty:
            max_dig_diameter = digs["diameter_um"].max()
            if max_dig_diameter > zone_params["max_defect_um"]:
                status = "FAIL"
                failure_reasons.append(
                    f"{zone_name}: Dig diameter {max_dig_diameter:.1f}μm exceeds limit {zone_params['max_defect_um']}μm"
                )
        
        # Check scratch lengths
        scratches = zone_defects[zone_defects["type"] == "scratch"]
        if not scratches.empty:
            max_scratch_length = scratches["length_um"].max()
            # Scratches are often evaluated by width, but here we use length as a proxy
            if max_scratch_length > zone_params["max_defect_um"] * 5: # Allow longer scratches
                status = "FAIL"
                failure_reasons.append(
                    f"{zone_name}: Scratch length {max_scratch_length:.1f}μm exceeds limit"
                )
        
        # Check total defect count (example criteria)
        if zone_name == "core" and len(zone_defects) > 5:
            status = "FAIL"
            failure_reasons.append(f"Core: Too many defects ({len(zone_defects)} > 5)")
            
    return status, failure_reasons


# ============================================================================
# Visual Report Generation
# ============================================================================

def generate_annotated_image(
    original_bgr_image: np.ndarray,
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    zone_masks: Dict[str, np.ndarray],
    output_path: str,
    report_timestamp: Optional[datetime.datetime] = None
) -> bool:
    """
    Generate and save an annotated image showing zones, defects, and pass/fail status.
    
    Args:
        original_bgr_image: The original BGR image
        analysis_results: Dictionary containing characterized defects and pass/fail status
        localization_data: Dictionary with fiber localization info
        zone_masks: Dictionary of binary masks for each zone
        output_path: Path where the annotated image will be saved
        report_timestamp: Optional datetime for displaying on the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        annotated_image = original_bgr_image.copy()
        
        # Default zone colors
        zone_color_map = {
            "Core": (255, 0, 0),      # Blue
            "Cladding": (0, 255, 0),  # Green
            "Adhesive": (0, 255, 255), # Yellow
            "Contact": (255, 0, 255)   # Magenta
        }
        
        # Draw zones
        zone_outline_thickness = 2
        for zone_name, zone_mask_np in zone_masks.items():
            color = zone_color_map.get(zone_name, (128, 128, 128))
            contours, _ = cv2.findContours(zone_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_image, contours, -1, color, zone_outline_thickness)
            
            # Add zone labels
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(annotated_image, zone_name, (cx-20, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw defects
        defect_outline_thickness = 2
        for defect in analysis_results.get('characterized_defects', []):
            defect_color = (0, 0, 255)  # Red for defects
            
            if 'contour_points_px' in defect:
                contour_points = np.array(defect['contour_points_px']).astype(np.int32)
                cv2.drawContours(annotated_image, [contour_points], -1, defect_color, defect_outline_thickness)
                
                # Add defect ID label
                cx = int(defect.get('centroid_x_px', 0))
                cy = int(defect.get('centroid_y_px', 0))
                defect_id = defect.get('defect_id', '')
                cv2.putText(annotated_image, str(defect_id), (cx, cy), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, defect_color, 1)
        
        # Add pass/fail stamp
        pass_fail_status = analysis_results.get('overall_pass_fail', analysis_results.get('overall_status', 'UNKNOWN'))
        stamp_color = (0, 255, 0) if pass_fail_status == 'PASS' else (0, 0, 255)
        cv2.putText(annotated_image, pass_fail_status, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, stamp_color, 3)
        
        # Add timestamp if provided
        if report_timestamp:
            timestamp_str = report_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_image, timestamp_str, (50, annotated_image.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save the annotated image
        success = cv2.imwrite(str(output_path), annotated_image)
        if success:
            logger.info(f"Annotated image saved to {output_path}")
        else:
            logger.error(f"Failed to save annotated image to {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error generating annotated image: {e}")
        return False


def generate_polar_defect_distribution(
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    output_path: str,
    bins: int = 36
) -> bool:
    """
    Generate a polar histogram showing the angular distribution of defects.
    
    Args:
        analysis_results: Dictionary containing characterized defects
        localization_data: Dictionary with fiber localization info
        output_path: Path where the plot will be saved
        bins: Number of angular bins for the histogram
        
    Returns:
        True if successful, False otherwise
    """
    try:
        defects = analysis_results.get('characterized_defects', [])
        
        if not defects:
            logger.warning("No defects found for polar distribution plot")
            return False
        
        # Get fiber center
        fiber_center_x = localization_data.get('fiber_center_x_px', 0)
        fiber_center_y = localization_data.get('fiber_center_y_px', 0)
        
        # Calculate angles for each defect
        angles = []
        for defect in defects:
            defect_x = defect.get('centroid_x_px', 0)
            defect_y = defect.get('centroid_y_px', 0)
            
            # Calculate angle from fiber center to defect
            dx = defect_x - fiber_center_x
            dy = defect_y - fiber_center_y
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # Create polar histogram
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Create histogram
        hist, bin_edges = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
        
        # Plot bars
        theta = bin_edges[:-1] + np.diff(bin_edges) / 2
        width = 2 * np.pi / bins
        bars = ax.bar(theta, hist, width=width, alpha=0.7)
        
        # Customize plot
        ax.set_title('Angular Distribution of Defects', y=1.08)
        if hasattr(ax, 'set_theta_zero_location'):
            ax.set_theta_zero_location('N')  # 0 degrees at top
        if hasattr(ax, 'set_theta_direction'):
            ax.set_theta_direction(-1)       # Clockwise
        ax.set_ylim(0, max(hist) * 1.1 if hist.size > 0 else 1)
        
        # Add statistics
        total_defects = len(defects)
        ax.text(0.02, 0.98, f'Total Defects: {total_defects}', 
               transform=ax.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Polar distribution plot saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating polar distribution plot: {e}")
        return False


def generate_comprehensive_report(
    original_bgr_image: np.ndarray,
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    zone_masks: Dict[str, np.ndarray],
    calibration_data: Dict[str, Any],
    output_dir: str,
    base_filename: str = "inspection_report"
) -> Dict[str, str]:
    """
    Generate a comprehensive report including annotated image, CSV, and polar plot.
    
    Args:
        original_bgr_image: The original BGR image
        analysis_results: Dictionary containing characterized defects
        localization_data: Dictionary with fiber localization info
        zone_masks: Dictionary of binary masks for each zone
        calibration_data: Dictionary with calibration information
        output_dir: Directory where files will be saved
        base_filename: Base filename for all generated files
        
    Returns:
        Dictionary mapping report type to file path
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Generate all reports
    reports = {}
    
    # Annotated image
    img_path = output_dir_path / f"{base_filename}_{timestamp_str}_annotated.png"
    if generate_annotated_image(original_bgr_image, analysis_results, localization_data, 
                               zone_masks, str(img_path), timestamp):
        reports['annotated_image'] = str(img_path)
    
    # CSV report
    csv_path = output_dir_path / f"{base_filename}_{timestamp_str}_defects.csv"
    csv_generator = CSVReportGenerator()
    if csv_generator.generate_defect_csv_report(analysis_results, csv_path):
        reports['csv_report'] = str(csv_path)
    
    # Polar distribution plot
    polar_path = output_dir_path / f"{base_filename}_{timestamp_str}_polar.png"
    if generate_polar_defect_distribution(analysis_results, localization_data, str(polar_path)):
        reports['polar_plot'] = str(polar_path)
    
    return reports


# ============================================================================
# Main Function and CLI
# ============================================================================

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Unified Reporting Module for Fiber Optic Inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate CSV report from analysis results
  %(prog)s csv --analysis results.json --output report.csv

  # Aggregate results from multiple analyses
  %(prog)s aggregate results_dir/ --csv --html --json

  # Apply pass/fail criteria
  %(prog)s passfail --analysis results.json --fiber-type single_mode_pc

  # Generate demo reports
  %(prog)s demo --output-dir demo_reports
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # CSV report generation
    csv_parser = subparsers.add_parser('csv', help='Generate CSV reports')
    csv_parser.add_argument('--analysis', help='Path to JSON file with analysis results')
    csv_parser.add_argument('--output', required=True, help='Path for output CSV file')
    csv_parser.add_argument('--report-type', choices=['detailed', 'summary', 'comparative'], 
                           default='detailed', help='Type of report to generate')
    csv_parser.add_argument('--demo', action='store_true', help='Generate demo report with sample data')
    
    # Data aggregation
    agg_parser = subparsers.add_parser('aggregate', help='Aggregate and analyze multiple results')
    agg_parser.add_argument('results_dir', help='Directory containing result JSON files')
    agg_parser.add_argument('--pattern', default='*.json', help='File pattern to match')
    agg_parser.add_argument('--output-dir', default='reports', help='Output directory for reports')
    agg_parser.add_argument('--csv', action='store_true', help='Generate CSV report')
    agg_parser.add_argument('--html', action='store_true', help='Generate HTML report')
    agg_parser.add_argument('--json', action='store_true', help='Generate JSON summary report')
    agg_parser.add_argument('--outliers', help='Find outliers for metric')
    agg_parser.add_argument('--threshold', type=float, default=2.0, help='Outlier threshold')
    
    # Pass/fail evaluation
    pf_parser = subparsers.add_parser('passfail', help='Apply pass/fail criteria')
    pf_parser.add_argument('--analysis', required=True, help='Path to analysis results JSON')
    pf_parser.add_argument('--fiber-type', default='single_mode_pc', 
                          choices=['single_mode_pc', 'multi_mode_pc'], 
                          help='Fiber type for pass/fail criteria')
    pf_parser.add_argument('--output', help='Output path for detailed report JSON')
    
    # Demo mode
    demo_parser = subparsers.add_parser('demo', help='Generate demo reports with sample data')
    demo_parser.add_argument('--output-dir', default='demo_reports', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute commands
    if args.command == 'csv':
        generator = CSVReportGenerator()
        
        if args.demo:
            # Create sample data
            analysis_results = create_sample_data()
        else:
            if not args.analysis:
                logger.error("Either --analysis or --demo must be specified")
                return 1
            
            analysis_results = generator.load_analysis_results(args.analysis)
            if analysis_results is None:
                return 1
        
        output_path = Path(args.output)
        
        if args.report_type == 'detailed':
            success = generator.generate_defect_csv_report(analysis_results, output_path)
        elif args.report_type == 'summary':
            success = generator.generate_summary_csv_report(analysis_results, output_path)
        elif args.report_type == 'comparative':
            # For demo, use the same data multiple times
            if args.demo:
                multiple_results = [analysis_results, analysis_results]
                image_names = ["Sample_Image_1", "Sample_Image_2"]
            else:
                multiple_results = [analysis_results]
                image_names = ["Single_Image"]
            
            success = generator.generate_comparative_report(multiple_results, output_path, image_names)
        
        if success:
            logger.info(f"Successfully generated {args.report_type} CSV report: {output_path}")
            return 0
        else:
            return 1
    
    elif args.command == 'aggregate':
        aggregator = DataAggregator()
        
        try:
            aggregator.load_results_from_directory(args.results_dir, args.pattern)
            
            if not aggregator.results:
                logger.error("No results found to analyze")
                return 1
            
            metrics = aggregator.calculate_quality_metrics()
            
            # Print summary
            print(f"\n=== Analysis Summary ===")
            print(f"Total processed: {metrics.total_processed}")
            print(f"Success rate: {metrics.success_rate:.1%}")
            print(f"Average confidence: {metrics.avg_confidence:.3f}")
            
            # Generate reports
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(args.output_dir, exist_ok=True)
            
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
            
            if args.outliers:
                outliers = aggregator.get_outliers(args.outliers, args.threshold)
                if outliers:
                    print(f"\nFound {len(outliers)} outliers for {args.outliers}")
                else:
                    print(f"\nNo outliers found for {args.outliers}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error during aggregation: {e}")
            return 1
    
    elif args.command == 'passfail':
        try:
            # Load analysis results
            with open(args.analysis, 'r') as f:
                analysis_results = json.load(f)
            
            defects = analysis_results.get('characterized_defects', [])
            
            # Apply pass/fail rules
            engine = PassFailRulesEngine()
            report = engine.generate_detailed_report(defects, args.fiber_type)
            
            # Print summary
            print(f"\n=== Pass/Fail Evaluation ===")
            print(f"Fiber Type: {args.fiber_type}")
            print(f"Overall Status: {report['overall_status']}")
            print(f"Total Defects: {report['summary']['total_defects']}")
            
            if report['failure_reasons']:
                print("\nFailure Reasons:")
                for reason in report['failure_reasons']:
                    print(f"  - {reason}")
            
            # Save detailed report if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=4)
                print(f"\nDetailed report saved to: {args.output}")
            
            return 0 if report['overall_status'] == 'PASS' else 1
            
        except Exception as e:
            logger.error(f"Error during pass/fail evaluation: {e}")
            return 1
    
    elif args.command == 'demo':
        logger.info("Generating demo reports...")
        
        # Create demo data and reports
        demo_data = create_demo_data()
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate various reports
        csv_gen = CSVReportGenerator()
        
        # Detailed CSV
        csv_path = Path(args.output_dir) / "demo_defects.csv"
        csv_gen.generate_defect_csv_report(demo_data['analysis_results'], csv_path)
        
        # Summary CSV
        summary_path = Path(args.output_dir) / "demo_summary.csv"
        csv_gen.generate_summary_csv_report(demo_data['analysis_results'], summary_path)
        
        # Pass/fail report
        engine = PassFailRulesEngine()
        pf_report = engine.generate_detailed_report(
            demo_data['analysis_results']['characterized_defects'], 
            'single_mode_pc'
        )
        
        pf_path = Path(args.output_dir) / "demo_passfail.json"
        with open(pf_path, 'w') as f:
            json.dump(pf_report, f, indent=4)
        
        # Visual reports if image data available
        if 'image' in demo_data:
            reports = generate_comprehensive_report(
                demo_data['image'],
                demo_data['analysis_results'],
                demo_data['localization_data'],
                demo_data['zone_masks'],
                demo_data['calibration_data'],
                args.output_dir,
                "demo"
            )
            
            print(f"\nGenerated demo reports in {args.output_dir}:")
            for report_type, path in reports.items():
                print(f"  - {report_type}: {path}")
        
        print(f"\nDemo reports generated in: {args.output_dir}")
        return 0
    
    return 0


def create_sample_data():
    """Create sample analysis data for testing."""
    return {
        "image_filename": "test_image.png",
        "overall_status": "FAIL",
        "total_defect_count": 3,
        "failure_reasons": ["Core defect exceeds size limit", "Critical scratch detected"],
        "characterized_defects": [
            {
                "defect_id": "D1_S1",
                "zone": "Core",
                "classification": "Scratch",
                "confidence_score": 0.95,
                "centroid_x_px": 190,
                "centroid_y_px": 140,
                "area_px": 50,
                "length_px": 22,
                "width_px": 2.5,
                "aspect_ratio_oriented": 8.8,
                "bbox_x_px": 180,
                "bbox_y_px": 130,
                "bbox_w_px": 15,
                "bbox_h_px": 25,
                "area_um2": 12.5,
                "length_um": 11.0,
                "width_um": 1.25,
                "effective_diameter_um": 4.0
            },
            {
                "defect_id": "D1_L1",
                "zone": "Cladding",
                "classification": "Scratch",
                "confidence_score": 0.88,
                "centroid_x_px": 230,
                "centroid_y_px": 170,
                "area_px": 30,
                "length_px": 10,
                "width_px": 3,
                "area_um2": 7.5,
                "length_um": 5.0,
                "width_um": 1.5
            },
            {
                "defect_id": "D2_P2",
                "zone": "Contact",
                "classification": "Pit/Dig",
                "confidence_score": 0.70,
                "centroid_x_px": 150,
                "centroid_y_px": 100,
                "area_px": 20,
                "length_px": 5,
                "width_px": 4,
                "bbox_x_px": 148,
                "bbox_y_px": 98,
                "bbox_w_px": 5,
                "bbox_h_px": 5,
                "area_um2": 5.0,
                "effective_diameter_um": 2.51
            }
        ]
    }


def create_demo_data():
    """Create comprehensive demo data including image data."""
    # Create synthetic test image
    test_image = np.ones((400, 400, 3), dtype=np.uint8) * 128
    
    # Draw a circle to simulate fiber
    cv2.circle(test_image, (200, 200), 100, (200, 200, 200), -1)
    
    # Analysis results
    analysis_results = create_sample_data()
    
    # Add contour points for visualization
    for defect in analysis_results['characterized_defects']:
        cx = defect['centroid_x_px']
        cy = defect['centroid_y_px']
        w = defect.get('bbox_w_px', 10)
        h = defect.get('bbox_h_px', 10)
        defect['contour_points_px'] = [
            [cx - w//2, cy - h//2],
            [cx + w//2, cy - h//2],
            [cx + w//2, cy + h//2],
            [cx - w//2, cy + h//2]
        ]
    
    # Localization data
    localization_data = {
        'fiber_center_x_px': 200,
        'fiber_center_y_px': 200,
        'core_radius_px': 50,
        'cladding_radius_px': 100
    }
    
    # Zone masks
    zone_masks = {}
    for zone_name, radius in [('Core', 50), ('Cladding', 100), ('Adhesive', 120), ('Contact', 150)]:
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(mask, (200, 200), radius, (255,), -1)
        if zone_name != 'Core':
            prev_radius = 50 if zone_name == 'Cladding' else (100 if zone_name == 'Adhesive' else 120)
            cv2.circle(mask, (200, 200), prev_radius, (0,), -1)
        zone_masks[zone_name] = mask
    
    # Calibration data
    calibration_data = {
        'um_per_px': 0.5,
        'calibration_method': 'synthetic_test'
    }
    
    return {
        'image': test_image,
        'analysis_results': analysis_results,
        'localization_data': localization_data,
        'zone_masks': zone_masks,
        'calibration_data': calibration_data
    }


if __name__ == "__main__":
    sys.exit(main())