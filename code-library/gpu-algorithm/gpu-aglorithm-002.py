#!/usr/bin/env python3
"""
GPU-Accelerated Data Acquisition Module
Aggregates detection results and performs clustering analysis
"""

import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from gpu_utils import GPUManager, gpu_accelerated, log_gpu_memory

# Configure logging
logger = logging.getLogger('DataAcquisitionGPU')

# Try to import RAPIDS for GPU clustering
try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    logger.warning("RAPIDS cuML not available, will use CPU clustering")
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler


@dataclass
class AggregatedDefect:
    """Aggregated defect information from clustering"""
    center: Tuple[float, float]
    size: float
    severity: str
    confidence: float
    region: str
    cluster_size: int  # Number of defects in this cluster
    contributing_methods: List[str]
    features: Dict[str, float]


@dataclass
class DataAcquisitionResult:
    """Final result from data acquisition"""
    aggregated_defects: List[AggregatedDefect]
    quality_metrics: Dict[str, float]
    pass_fail_status: str
    detailed_report: Dict[str, Any]
    visualizations: Dict[str, np.ndarray]
    processing_time: float


class DataAcquisitionGPU:
    """GPU-accelerated data acquisition and analysis"""
    
    def __init__(self, config: Optional[Dict] = None, force_cpu: bool = False):
        """Initialize GPU-accelerated data acquisition"""
        self.config = config or {}
        self.gpu_manager = GPUManager(force_cpu=force_cpu)
        self.use_rapids = RAPIDS_AVAILABLE and self.gpu_manager.use_gpu and not force_cpu
        self.logger = logging.getLogger('DataAcquisitionGPU')
        
        # Clustering parameters
        self.clustering_eps = self.config.get('clustering_eps', 20)  # pixels
        self.clustering_min_samples = self.config.get('clustering_min_samples', 2)
        
        # Quality thresholds
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'perfect': 95,
            'good': 85,
            'acceptable': 70,
            'poor': 50
        })
        
        self.logger.info(f"Initialized DataAcquisitionGPU with GPU={self.gpu_manager.use_gpu}, "
                        f"RAPIDS={self.use_rapids}")
    
    def aggregate_results(self, detection_results: List[Dict[str, Any]], 
                         original_image: Optional[np.ndarray] = None,
                         separation_result: Optional[Dict[str, Any]] = None) -> DataAcquisitionResult:
        """
        Aggregate detection results from multiple sources
        
        Args:
            detection_results: List of detection results
            original_image: Original image for visualization
            separation_result: Separation result with masks
            
        Returns:
            DataAcquisitionResult with aggregated analysis
        """
        start_time = time.time()
        self.logger.info(f"Starting data aggregation for {len(detection_results)} results")
        
        # Extract all defects
        all_defects = self._extract_all_defects(detection_results)
        
        if not all_defects:
            self.logger.info("No defects found, returning perfect result")
            return self._create_perfect_result(start_time)
        
        # Cluster defects
        aggregated_defects = self._cluster_defects_gpu(all_defects)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(aggregated_defects)
        
        # Determine pass/fail status
        pass_fail_status = self._determine_pass_fail(quality_metrics)
        
        # Generate detailed report
        detailed_report = self._generate_detailed_report(
            aggregated_defects, quality_metrics, detection_results
        )
        
        # Generate visualizations
        visualizations = {}
        if original_image is not None:
            visualizations = self._generate_visualizations_gpu(
                original_image, aggregated_defects, separation_result
            )
        
        processing_time = time.time() - start_time
        
        result = DataAcquisitionResult(
            aggregated_defects=aggregated_defects,
            quality_metrics=quality_metrics,
            pass_fail_status=pass_fail_status,
            detailed_report=detailed_report,
            visualizations=visualizations,
            processing_time=processing_time
        )
        
        self.logger.info(f"Data acquisition completed in {processing_time:.2f}s - "
                        f"Status: {pass_fail_status}, "
                        f"Aggregated defects: {len(aggregated_defects)}")
        log_gpu_memory()
        
        return result
    
    def _extract_all_defects(self, detection_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all defects from detection results"""
        all_defects = []
        
        for i, result in enumerate(detection_results):
            method_name = f"method_{i}"
            
            # Handle different result formats
            if isinstance(result, dict):
                # Extract defects from different possible keys
                for key in ['defects', 'anomalies', 'detected_defects']:
                    if key in result:
                        defects = result[key]
                        if isinstance(defects, list):
                            for defect in defects:
                                defect_dict = defect if isinstance(defect, dict) else asdict(defect)
                                defect_dict['source_method'] = method_name
                                all_defects.append(defect_dict)
                        break
                
                # Also check for region-specific results
                for region in ['core', 'cladding', 'ferrule']:
                    if f'{region}_defects' in result:
                        defects = result[f'{region}_defects']
                        if isinstance(defects, list):
                            for defect in defects:
                                defect_dict = defect if isinstance(defect, dict) else asdict(defect)
                                defect_dict['source_method'] = method_name
                                defect_dict['region'] = region
                                all_defects.append(defect_dict)
        
        self.logger.info(f"Extracted {len(all_defects)} total defects")
        return all_defects
    
    @gpu_accelerated
    def _cluster_defects_gpu(self, defects: List[Dict[str, Any]]) -> List[AggregatedDefect]:
        """Cluster nearby defects using GPU acceleration"""
        if not defects:
            return []
        
        # Extract defect locations
        locations = np.array([
            defect.get('location', defect.get('center', [0, 0]))
            for defect in defects
        ])
        
        if len(locations) == 0:
            return []
        
        # Perform clustering
        if self.use_rapids:
            # GPU clustering with RAPIDS
            clusterer = cuDBSCAN(
                eps=self.clustering_eps,
                min_samples=self.clustering_min_samples
            )
            
            # Transfer to GPU
            import cupy as cp
            locations_gpu = cp.asarray(locations)
            
            # Fit clustering
            labels = clusterer.fit_predict(locations_gpu)
            
            # Transfer back to CPU
            labels = cp.asnumpy(labels)
        else:
            # CPU clustering
            clusterer = DBSCAN(
                eps=self.clustering_eps,
                min_samples=self.clustering_min_samples
            )
            labels = clusterer.fit_predict(locations)
        
        # Aggregate defects by cluster
        aggregated = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                # Treat each noise point as individual defect
                noise_indices = np.where(labels == -1)[0]
                for idx in noise_indices:
                    defect = defects[idx]
                    aggregated.append(self._create_aggregated_defect([defect]))
            else:
                # Aggregate cluster
                cluster_indices = np.where(labels == label)[0]
                cluster_defects = [defects[i] for i in cluster_indices]
                aggregated.append(self._create_aggregated_defect(cluster_defects))
        
        self.logger.info(f"Clustered {len(defects)} defects into {len(aggregated)} aggregated defects")
        return aggregated
    
    def _create_aggregated_defect(self, cluster_defects: List[Dict[str, Any]]) -> AggregatedDefect:
        """Create aggregated defect from cluster"""
        # Calculate center
        locations = [d.get('location', d.get('center', [0, 0])) for d in cluster_defects]
        center = np.mean(locations, axis=0)
        
        # Calculate size (sum of individual sizes)
        total_size = sum(d.get('size', d.get('area', 10)) for d in cluster_defects)
        
        # Determine severity (use highest)
        severities = [d.get('severity', 'LOW') for d in cluster_defects]
        severity_order = ['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        severity = max(severities, key=lambda x: severity_order.index(x) if x in severity_order else 0)
        
        # Calculate confidence (average)
        confidences = [d.get('confidence', 0.5) for d in cluster_defects]
        confidence = np.mean(confidences)
        
        # Determine region (use most common)
        regions = [d.get('region', 'unknown') for d in cluster_defects]
        region = max(set(regions), key=regions.count)
        
        # Get contributing methods
        methods = list(set(d.get('source_method', 'unknown') for d in cluster_defects))
        
        # Aggregate features
        aggregated_features = {}
        for defect in cluster_defects:
            features = defect.get('features', {})
            for key, value in features.items():
                if key not in aggregated_features:
                    aggregated_features[key] = []
                aggregated_features[key].append(value)
        
        # Average features
        averaged_features = {
            key: float(np.mean(values))
            for key, values in aggregated_features.items()
        }
        
        return AggregatedDefect(
            center=tuple(center),
            size=float(total_size),
            severity=severity,
            confidence=float(confidence),
            region=region,
            cluster_size=len(cluster_defects),
            contributing_methods=methods,
            features=averaged_features
        )
    
    def _calculate_quality_metrics(self, aggregated_defects: List[AggregatedDefect]) -> Dict[str, float]:
        """Calculate overall quality metrics"""
        metrics = {
            'total_defects': len(aggregated_defects),
            'critical_defects': 0,
            'high_severity_defects': 0,
            'medium_severity_defects': 0,
            'low_severity_defects': 0,
            'negligible_defects': 0,
            'core_defects': 0,
            'cladding_defects': 0,
            'ferrule_defects': 0,
            'total_defect_area': 0.0,
            'average_confidence': 0.0,
            'quality_score': 100.0
        }
        
        if not aggregated_defects:
            return metrics
        
        # Count defects by severity and region
        for defect in aggregated_defects:
            # Severity counts
            severity_key = f"{defect.severity.lower()}_defects"
            if severity_key in metrics:
                metrics[severity_key] += 1
            
            # Region counts
            region_key = f"{defect.region}_defects"
            if region_key in metrics:
                metrics[region_key] += 1
            
            # Total area
            metrics['total_defect_area'] += defect.size
        
        # Average confidence
        metrics['average_confidence'] = np.mean([d.confidence for d in aggregated_defects])
        
        # Calculate quality score
        score = 100.0
        
        # Penalties by severity
        severity_penalties = {
            'critical_defects': 20.0,
            'high_severity_defects': 10.0,
            'medium_severity_defects': 5.0,
            'low_severity_defects': 2.0,
            'negligible_defects': 0.5
        }
        
        for severity, penalty in severity_penalties.items():
            score -= metrics[severity] * penalty
        
        # Additional penalty for defect area
        area_penalty = min(20.0, metrics['total_defect_area'] / 1000.0)
        score -= area_penalty
        
        metrics['quality_score'] = max(0.0, score)
        
        return metrics
    
    def _determine_pass_fail(self, quality_metrics: Dict[str, float]) -> str:
        """Determine pass/fail status based on quality metrics"""
        quality_score = quality_metrics['quality_score']
        critical_defects = quality_metrics['critical_defects']
        
        # Automatic fail conditions
        if critical_defects > 0:
            return 'FAIL - Critical defects found'
        
        # Quality-based determination
        if quality_score >= self.quality_thresholds['perfect']:
            return 'PASS - Perfect quality'
        elif quality_score >= self.quality_thresholds['good']:
            return 'PASS - Good quality'
        elif quality_score >= self.quality_thresholds['acceptable']:
            return 'PASS - Acceptable quality'
        elif quality_score >= self.quality_thresholds['poor']:
            return 'MARGINAL - Poor quality'
        else:
            return 'FAIL - Quality below threshold'
    
    def _generate_detailed_report(self, aggregated_defects: List[AggregatedDefect],
                                 quality_metrics: Dict[str, float],
                                 detection_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_aggregated_defects': len(aggregated_defects),
                'quality_score': quality_metrics['quality_score'],
                'pass_fail_status': self._determine_pass_fail(quality_metrics)
            },
            'quality_metrics': quality_metrics,
            'defect_distribution': {
                'by_severity': {
                    'CRITICAL': quality_metrics['critical_defects'],
                    'HIGH': quality_metrics['high_severity_defects'],
                    'MEDIUM': quality_metrics['medium_severity_defects'],
                    'LOW': quality_metrics['low_severity_defects'],
                    'NEGLIGIBLE': quality_metrics['negligible_defects']
                },
                'by_region': {
                    'core': quality_metrics['core_defects'],
                    'cladding': quality_metrics['cladding_defects'],
                    'ferrule': quality_metrics['ferrule_defects']
                }
            },
            'aggregated_defects': [asdict(d) for d in aggregated_defects],
            'processing_info': {
                'gpu_used': self.gpu_manager.use_gpu,
                'rapids_used': self.use_rapids,
                'clustering_parameters': {
                    'eps': self.clustering_eps,
                    'min_samples': self.clustering_min_samples
                }
            }
        }
        
        return report
    
    @gpu_accelerated
    def _generate_visualizations_gpu(self, original_image: np.ndarray,
                                    aggregated_defects: List[AggregatedDefect],
                                    separation_result: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Generate visualization images using GPU acceleration"""
        visualizations = {}
        
        # Transfer image to GPU
        img_gpu = self.gpu_manager.array_to_gpu(original_image)
        xp = self.gpu_manager.get_array_module(img_gpu)
        
        # 1. Defect overlay visualization
        defect_overlay = self._create_defect_overlay_gpu(img_gpu, aggregated_defects)
        visualizations['defect_overlay'] = self.gpu_manager.array_to_cpu(defect_overlay)
        
        # 2. Heatmap visualization
        heatmap = self._create_defect_heatmap_gpu(img_gpu.shape[:2], aggregated_defects)
        visualizations['defect_heatmap'] = self.gpu_manager.array_to_cpu(heatmap)
        
        # 3. Region-specific visualizations
        if separation_result and 'masks' in separation_result:
            masks = separation_result['masks']
            region_viz = self._create_region_visualization_gpu(img_gpu, masks, aggregated_defects)
            visualizations['region_analysis'] = self.gpu_manager.array_to_cpu(region_viz)
        
        # 4. Quality map
        quality_map = self._create_quality_map_gpu(img_gpu.shape[:2], aggregated_defects)
        visualizations['quality_map'] = self.gpu_manager.array_to_cpu(quality_map)
        
        return visualizations
    
    @gpu_accelerated
    def _create_defect_overlay_gpu(self, image: Union[np.ndarray, 'cp.ndarray'],
                                  defects: List[AggregatedDefect]) -> Union[np.ndarray, 'cp.ndarray']:
        """Create defect overlay visualization using GPU"""
        xp = self.gpu_manager.get_array_module(image)
        
        # Create overlay
        overlay = image.copy()
        
        # Color mapping for severity
        severity_colors = {
            'CRITICAL': (255, 0, 0),      # Red
            'HIGH': (255, 128, 0),        # Orange
            'MEDIUM': (255, 255, 0),      # Yellow
            'LOW': (0, 255, 255),         # Cyan
            'NEGLIGIBLE': (128, 128, 128) # Gray
        }
        
        # Draw defects
        for defect in defects:
            cx, cy = int(defect.center[0]), int(defect.center[1])
            radius = int(np.sqrt(defect.size / np.pi))
            color = severity_colors.get(defect.severity, (255, 255, 255))
            
            # Draw circle (simplified for GPU)
            y, x = xp.ogrid[:image.shape[0], :image.shape[1]]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            # Apply color with transparency
            alpha = 0.6
            for c in range(3):
                overlay[:, :, c] = xp.where(
                    mask,
                    overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                    overlay[:, :, c]
                )
        
        return overlay.astype(xp.uint8)
    
    @gpu_accelerated
    def _create_defect_heatmap_gpu(self, shape: Tuple[int, int],
                                  defects: List[AggregatedDefect]) -> Union[np.ndarray, 'cp.ndarray']:
        """Create defect density heatmap using GPU"""
        xp = self.gpu_manager.get_array_module(self.gpu_manager.array_to_gpu(np.array([0])))
        
        # Create heatmap
        heatmap = xp.zeros(shape, dtype=xp.float32)
        
        # Add Gaussian for each defect
        for defect in defects:
            cx, cy = int(defect.center[0]), int(defect.center[1])
            
            # Severity-based intensity
            severity_weights = {
                'CRITICAL': 1.0,
                'HIGH': 0.8,
                'MEDIUM': 0.6,
                'LOW': 0.4,
                'NEGLIGIBLE': 0.2
            }
            intensity = severity_weights.get(defect.severity, 0.5)
            
            # Size-based sigma
            sigma = max(10, np.sqrt(defect.size))
            
            # Create Gaussian
            y, x = xp.ogrid[:shape[0], :shape[1]]
            gaussian = xp.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            
            # Add to heatmap
            heatmap += gaussian * intensity * defect.confidence
        
        # Normalize and convert to color
        if xp.max(heatmap) > 0:
            heatmap = heatmap / xp.max(heatmap)
        
        # Apply colormap (simple jet colormap)
        heatmap_color = xp.zeros((*shape, 3), dtype=xp.uint8)
        
        # Blue to cyan
        mask1 = heatmap < 0.25
        heatmap_color[mask1, 0] = 0
        heatmap_color[mask1, 1] = (heatmap[mask1] * 4 * 255).astype(xp.uint8)
        heatmap_color[mask1, 2] = 255
        
        # Cyan to green
        mask2 = (heatmap >= 0.25) & (heatmap < 0.5)
        heatmap_color[mask2, 0] = 0
        heatmap_color[mask2, 1] = 255
        heatmap_color[mask2, 2] = ((0.5 - heatmap[mask2]) * 4 * 255).astype(xp.uint8)
        
        # Green to yellow
        mask3 = (heatmap >= 0.5) & (heatmap < 0.75)
        heatmap_color[mask3, 0] = ((heatmap[mask3] - 0.5) * 4 * 255).astype(xp.uint8)
        heatmap_color[mask3, 1] = 255
        heatmap_color[mask3, 2] = 0
        
        # Yellow to red
        mask4 = heatmap >= 0.75
        heatmap_color[mask4, 0] = 255
        heatmap_color[mask4, 1] = ((1.0 - heatmap[mask4]) * 4 * 255).astype(xp.uint8)
        heatmap_color[mask4, 2] = 0
        
        return heatmap_color
    
    @gpu_accelerated
    def _create_region_visualization_gpu(self, image: Union[np.ndarray, 'cp.ndarray'],
                                       masks: Dict[str, np.ndarray],
                                       defects: List[AggregatedDefect]) -> Union[np.ndarray, 'cp.ndarray']:
        """Create region-specific defect visualization"""
        xp = self.gpu_manager.get_array_module(image)
        
        # Create visualization
        viz = xp.zeros_like(image)
        
        # Region colors
        region_colors = {
            'core': (255, 0, 0),      # Red
            'cladding': (0, 255, 0),  # Green
            'ferrule': (0, 0, 255)    # Blue
        }
        
        # Apply region coloring
        for region_name, mask in masks.items():
            if region_name in region_colors:
                # Convert mask to GPU if needed
                mask_gpu = self.gpu_manager.array_to_gpu(mask) if isinstance(mask, np.ndarray) else mask
                color = region_colors[region_name]
                
                for c in range(3):
                    viz[:, :, c] = xp.where(mask_gpu > 0, color[c], viz[:, :, c])
        
        # Overlay defects
        for defect in defects:
            cx, cy = int(defect.center[0]), int(defect.center[1])
            radius = max(5, int(np.sqrt(defect.size / np.pi)))
            
            # Draw white circle for defects
            y, x = xp.ogrid[:image.shape[0], :image.shape[1]]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            viz[mask] = 255  # White for defects
        
        return viz
    
    @gpu_accelerated
    def _create_quality_map_gpu(self, shape: Tuple[int, int],
                               defects: List[AggregatedDefect]) -> Union[np.ndarray, 'cp.ndarray']:
        """Create local quality map"""
        xp = self.gpu_manager.get_array_module(self.gpu_manager.array_to_gpu(np.array([0])))
        
        # Start with perfect quality
        quality_map = xp.ones(shape, dtype=xp.float32) * 100.0
        
        # Reduce quality around defects
        for defect in defects:
            cx, cy = int(defect.center[0]), int(defect.center[1])
            
            # Impact radius based on severity
            severity_radius = {
                'CRITICAL': 50,
                'HIGH': 40,
                'MEDIUM': 30,
                'LOW': 20,
                'NEGLIGIBLE': 10
            }
            radius = severity_radius.get(defect.severity, 20)
            
            # Quality reduction
            severity_impact = {
                'CRITICAL': 50,
                'HIGH': 30,
                'MEDIUM': 20,
                'LOW': 10,
                'NEGLIGIBLE': 5
            }
            impact = severity_impact.get(defect.severity, 10)
            
            # Apply Gaussian quality reduction
            y, x = xp.ogrid[:shape[0], :shape[1]]
            distance_sq = (x - cx)**2 + (y - cy)**2
            gaussian = xp.exp(-distance_sq / (2 * radius**2))
            
            quality_map -= gaussian * impact * defect.confidence
        
        # Clamp to valid range
        quality_map = xp.clip(quality_map, 0, 100)
        
        # Convert to color image (green=good, red=bad)
        quality_color = xp.zeros((*shape, 3), dtype=xp.uint8)
        
        # Red channel increases as quality decreases
        quality_color[:, :, 0] = ((100 - quality_map) * 2.55).astype(xp.uint8)
        
        # Green channel increases as quality increases
        quality_color[:, :, 1] = (quality_map * 2.55).astype(xp.uint8)
        
        # Blue channel constant low
        quality_color[:, :, 2] = 32
        
        return quality_color
    
    def _create_perfect_result(self, start_time: float) -> DataAcquisitionResult:
        """Create result for perfect quality (no defects)"""
        return DataAcquisitionResult(
            aggregated_defects=[],
            quality_metrics={
                'total_defects': 0,
                'critical_defects': 0,
                'high_severity_defects': 0,
                'medium_severity_defects': 0,
                'low_severity_defects': 0,
                'negligible_defects': 0,
                'core_defects': 0,
                'cladding_defects': 0,
                'ferrule_defects': 0,
                'total_defect_area': 0.0,
                'average_confidence': 0.0,
                'quality_score': 100.0
            },
            pass_fail_status='PASS - Perfect quality',
            detailed_report={
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_aggregated_defects': 0,
                    'quality_score': 100.0,
                    'pass_fail_status': 'PASS - Perfect quality'
                }
            },
            visualizations={},
            processing_time=time.time() - start_time
        )
    
    def save_results(self, result: DataAcquisitionResult, output_dir: str):
        """Save acquisition results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed report
        report_path = output_path / 'acquisition_report.json'
        with open(report_path, 'w') as f:
            json.dump(result.detailed_report, f, indent=2)
        
        # Save visualizations
        for name, image in result.visualizations.items():
            viz_path = output_path / f'visualization_{name}.png'
            cv2.imwrite(str(viz_path), image)
        
        # Save summary
        summary_path = output_path / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Quality Score: {result.quality_metrics['quality_score']:.1f}%\n")
            f.write(f"Status: {result.pass_fail_status}\n")
            f.write(f"Total Defects: {result.quality_metrics['total_defects']}\n")
            f.write(f"Critical Defects: {result.quality_metrics['critical_defects']}\n")
            f.write(f"Processing Time: {result.processing_time:.2f}s\n")
        
        self.logger.info(f"Saved acquisition results to {output_path}")


def aggregate_detection_results(detection_results: List[Dict[str, Any]],
                              original_image: Optional[np.ndarray] = None,
                              config: Optional[Dict] = None,
                              force_cpu: bool = False) -> DataAcquisitionResult:
    """
    Aggregate detection results
    
    Args:
        detection_results: List of detection results
        original_image: Original image for visualization
        config: Configuration dictionary
        force_cpu: Force CPU mode
        
    Returns:
        DataAcquisitionResult
    """
    acquisition = DataAcquisitionGPU(config, force_cpu)
    return acquisition.aggregate_results(detection_results, original_image)


if __name__ == "__main__":
    # Test the GPU data acquisition
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_acquisition_gpu.py <detection_results.json> [--cpu]")
        sys.exit(1)
    
    force_cpu = '--cpu' in sys.argv
    
    # Load test detection results
    with open(sys.argv[1], 'r') as f:
        detection_results = [json.load(f)]
    
    # Run aggregation
    result = aggregate_detection_results(detection_results, force_cpu=force_cpu)
    
    print(f"Data acquisition completed!")
    print(f"Quality score: {result.quality_metrics['quality_score']:.1f}%")
    print(f"Status: {result.pass_fail_status}")
    print(f"Aggregated defects: {len(result.aggregated_defects)}")
    print(f"Processing time: {result.processing_time:.2f}s")