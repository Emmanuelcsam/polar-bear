#!/usr/bin/env python3
"""
Defect Clustering and Analysis Module
Advanced defect aggregation, clustering, and spatial analysis.
Extracted and optimized from data_acquisition.py DefectAggregator.
"""

import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
import hashlib


class DefectClusterAnalyzer:
    """
    Advanced defect clustering and spatial analysis system.
    Aggregates defects from multiple sources and performs intelligent clustering.
    """
    
    def __init__(self, 
                 clustering_eps: float = 30.0, 
                 min_cluster_size: int = 1,
                 image_shape: Optional[Tuple[int, int]] = None):
        """
        Initialize the defect cluster analyzer.
        
        Args:
            clustering_eps: DBSCAN epsilon parameter (distance threshold)
            min_cluster_size: Minimum number of defects to form a cluster
            image_shape: (height, width) of the reference image
        """
        self.clustering_eps = clustering_eps
        self.min_cluster_size = min_cluster_size
        self.image_shape = image_shape
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.defects = []
        self.clusters = []
        self.merged_defects = []
        self.data_integrity_log = []
    
    def add_defect(self, defect: Dict) -> bool:
        """
        Add a single defect to the analyzer with validation.
        
        Args:
            defect: Defect dictionary with required fields
            
        Returns:
            True if defect was successfully added, False otherwise
        """
        if not self._validate_defect(defect):
            return False
        
        # Add unique ID if not present
        if 'unique_id' not in defect:
            defect_str = f"{defect.get('location_xy', [0, 0])}{defect.get('defect_type', 'unknown')}"
            defect['unique_id'] = hashlib.md5(defect_str.encode()).hexdigest()[:8]
        
        self.defects.append(defect)
        return True
    
    def add_defects_from_list(self, defects: List[Dict]) -> int:
        """
        Add multiple defects from a list.
        
        Args:
            defects: List of defect dictionaries
            
        Returns:
            Number of successfully added defects
        """
        added_count = 0
        for defect in defects:
            if self.add_defect(defect):
                added_count += 1
        
        self.logger.info(f"Added {added_count}/{len(defects)} defects")
        return added_count
    
    def load_defects_from_json(self, json_path: str, source_name: Optional[str] = None) -> int:
        """
        Load defects from a JSON detection report.
        
        Args:
            json_path: Path to JSON file containing detection results
            source_name: Optional source identifier
            
        Returns:
            Number of successfully loaded defects
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            defects = data.get('defects', [])
            
            # Add source information
            for defect in defects:
                if source_name:
                    defect['source'] = source_name
                defect['source_file'] = json_path
            
            return self.add_defects_from_list(defects)
            
        except Exception as e:
            self.logger.error(f"Error loading defects from {json_path}: {e}")
            self.data_integrity_log.append({
                'file': json_path,
                'issue': f'Failed to load: {e}',
                'severity': 'ERROR'
            })
            return 0
    
    def load_defects_from_directory(self, directory: str, pattern: str = "*_report.json") -> int:
        """
        Load defects from all JSON files in a directory.
        
        Args:
            directory: Path to directory containing JSON files
            pattern: File pattern to match (default: "*_report.json")
            
        Returns:
            Total number of successfully loaded defects
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            self.logger.error(f"Directory does not exist: {directory}")
            return 0
        
        json_files = list(dir_path.rglob(pattern))
        total_loaded = 0
        
        for json_file in json_files:
            source_name = json_file.stem.replace('_report', '')
            loaded = self.load_defects_from_json(str(json_file), source_name)
            total_loaded += loaded
        
        self.logger.info(f"Loaded {total_loaded} defects from {len(json_files)} files")
        return total_loaded
    
    def _validate_defect(self, defect: Dict) -> bool:
        """Validate defect structure and data."""
        required_fields = ['location_xy']
        
        for field in required_fields:
            if field not in defect:
                self.data_integrity_log.append({
                    'defect_id': defect.get('unique_id', 'unknown'),
                    'issue': f'Missing required field: {field}',
                    'severity': 'ERROR'
                })
                return False
        
        # Validate location
        location = defect['location_xy']
        if not isinstance(location, (list, tuple)) or len(location) != 2:
            self.data_integrity_log.append({
                'defect_id': defect.get('unique_id', 'unknown'),
                'issue': 'Invalid location_xy format',
                'severity': 'ERROR'
            })
            return False
        
        # Validate coordinates are numeric
        try:
            x, y = float(location[0]), float(location[1])
            
            # Check bounds if image shape is known
            if self.image_shape:
                h, w = self.image_shape
                if not (0 <= x <= w and 0 <= y <= h):
                    self.data_integrity_log.append({
                        'defect_id': defect.get('unique_id', 'unknown'),
                        'issue': f'Coordinates out of bounds: ({x}, {y})',
                        'severity': 'WARNING'
                    })
                    # Don't reject, just warn
            
        except (ValueError, TypeError):
            self.data_integrity_log.append({
                'defect_id': defect.get('unique_id', 'unknown'),
                'issue': 'Non-numeric coordinates',
                'severity': 'ERROR'
            })
            return False
        
        return True
    
    def cluster_defects(self, custom_eps: Optional[float] = None) -> List[Dict]:
        """
        Cluster defects using DBSCAN algorithm.
        
        Args:
            custom_eps: Optional custom epsilon parameter
            
        Returns:
            List of cluster dictionaries
        """
        if not self.defects:
            self.logger.warning("No defects to cluster")
            return []
        
        eps = custom_eps if custom_eps is not None else self.clustering_eps
        
        # Extract coordinates
        coords = []
        valid_defects = []
        
        for defect in self.defects:
            try:
                x, y = float(defect['location_xy'][0]), float(defect['location_xy'][1])
                coords.append([x, y])
                valid_defects.append(defect)
            except:
                self.logger.warning(f"Skipping defect with invalid coordinates: {defect.get('unique_id', 'unknown')}")
        
        if not coords:
            self.logger.error("No valid coordinates for clustering")
            return []
        
        coords = np.array(coords)
        
        # Adaptive clustering based on defect density
        if len(coords) > 100:
            # Use larger epsilon for high-density scenarios
            eps = eps * 1.5
            self.logger.info(f"High defect density detected, using adaptive eps: {eps}")
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=self.min_cluster_size).fit(coords)
        
        # Group defects by cluster
        cluster_groups = defaultdict(list)
        for defect, label in zip(valid_defects, clustering.labels_):
            cluster_groups[label].append(defect)
        
        # Process clusters
        clusters = []
        total_clusters = len(set(clustering.labels_) - {-1})
        noise_points = np.sum(clustering.labels_ == -1)
        
        self.logger.info(f"Clustering results: {total_clusters} clusters, {noise_points} noise points")
        
        for cluster_id, cluster_defects in cluster_groups.items():
            if cluster_id == -1:
                # Noise points - create individual clusters
                for defect in cluster_defects:
                    cluster = self._create_single_defect_cluster(defect)
                    clusters.append(cluster)
            else:
                # Create merged cluster
                cluster = self._merge_defect_cluster(cluster_defects, cluster_id)
                clusters.append(cluster)
        
        self.clusters = clusters
        return clusters
    
    def _create_single_defect_cluster(self, defect: Dict) -> Dict:
        """Create a cluster from a single defect (noise point)."""
        x, y = defect['location_xy']
        
        cluster = {
            'cluster_id': f"single_{defect.get('unique_id', 'unknown')}",
            'defect_count': 1,
            'center': [float(x), float(y)],
            'extent': [1, 1],  # Single pixel extent
            'total_area': defect.get('area_px', 1),
            'severity_max': defect.get('severity', 'LOW'),
            'severity_avg': defect.get('severity', 'LOW'),
            'confidence_avg': defect.get('confidence', 0.5),
            'defect_types': [defect.get('defect_type', 'UNKNOWN')],
            'constituent_defects': [defect],
            'is_clustered': False,
            'spatial_density': 1.0
        }
        
        return cluster
    
    def _merge_defect_cluster(self, defects: List[Dict], cluster_id: int) -> Dict:
        """Merge multiple defects into a single cluster."""
        # Extract coordinates and properties
        coordinates = []
        areas = []
        confidences = []
        severities = []
        defect_types = []
        
        for defect in defects:
            x, y = defect['location_xy']
            coordinates.append([x, y])
            areas.append(defect.get('area_px', 1))
            confidences.append(defect.get('confidence', 0.5))
            severities.append(defect.get('severity', 'LOW'))
            defect_types.append(defect.get('defect_type', 'UNKNOWN'))
        
        coordinates = np.array(coordinates)
        
        # Calculate cluster properties
        center = np.mean(coordinates, axis=0)
        extent = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
        total_area = sum(areas)
        avg_confidence = np.mean(confidences)
        
        # Determine dominant severity
        severity_order = ['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        severity_indices = [severity_order.index(s) if s in severity_order else 1 for s in severities]
        max_severity = severity_order[max(severity_indices)]
        avg_severity_idx = int(np.mean(severity_indices))
        avg_severity = severity_order[min(avg_severity_idx, len(severity_order) - 1)]
        
        # Calculate spatial density
        if extent[0] > 0 and extent[1] > 0:
            cluster_area = extent[0] * extent[1]
            spatial_density = len(defects) / cluster_area
        else:
            spatial_density = len(defects)  # Point cluster
        
        cluster = {
            'cluster_id': f"cluster_{cluster_id}",
            'defect_count': len(defects),
            'center': center.tolist(),
            'extent': extent.tolist(),
            'total_area': total_area,
            'severity_max': max_severity,
            'severity_avg': avg_severity,
            'confidence_avg': float(avg_confidence),
            'defect_types': list(set(defect_types)),
            'constituent_defects': defects,
            'is_clustered': True,
            'spatial_density': float(spatial_density)
        }
        
        return cluster
    
    def create_defect_heatmap(self, 
                             image_shape: Optional[Tuple[int, int]] = None,
                             sigma: float = 20.0,
                             normalize: bool = True) -> np.ndarray:
        """
        Create a heatmap showing defect density.
        
        Args:
            image_shape: (height, width) of the output heatmap
            sigma: Gaussian blur sigma for smoothing
            normalize: Whether to normalize values to [0, 1]
            
        Returns:
            Heatmap as numpy array
        """
        if image_shape is None and self.image_shape is None:
            raise ValueError("Image shape must be provided either during initialization or as parameter")
        
        shape = image_shape or self.image_shape
        h, w = shape
        
        # Create empty heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Add defects to heatmap
        for defect in self.defects:
            try:
                x, y = defect['location_xy']
                x, y = int(x), int(y)
                
                # Check bounds
                if 0 <= x < w and 0 <= y < h:
                    # Weight by confidence and area
                    weight = defect.get('confidence', 0.5) * np.sqrt(defect.get('area_px', 1))
                    heatmap[y, x] += weight
                    
            except:
                continue
        
        # Apply Gaussian smoothing
        if sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize if requested
        if normalize and np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def analyze_spatial_distribution(self) -> Dict[str, Any]:
        """
        Analyze the spatial distribution of defects.
        
        Returns:
            Dictionary containing spatial analysis results
        """
        if not self.defects:
            return {'error': 'No defects to analyze'}
        
        # Extract coordinates
        coordinates = []
        for defect in self.defects:
            try:
                x, y = defect['location_xy']
                coordinates.append([float(x), float(y)])
            except:
                continue
        
        if not coordinates:
            return {'error': 'No valid coordinates'}
        
        coordinates = np.array(coordinates)
        
        # Basic statistics
        center_of_mass = np.mean(coordinates, axis=0)
        std_dev = np.std(coordinates, axis=0)
        extent = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
        
        # Calculate distances from center
        distances = np.sqrt(np.sum((coordinates - center_of_mass)**2, axis=1))
        
        # Quadrant analysis (if image shape is known)
        quadrant_counts = None
        if self.image_shape:
            h, w = self.image_shape
            cx, cy = w / 2, h / 2
            
            quadrants = {
                'Q1': np.sum((coordinates[:, 0] >= cx) & (coordinates[:, 1] < cy)),  # Top-right
                'Q2': np.sum((coordinates[:, 0] < cx) & (coordinates[:, 1] < cy)),   # Top-left
                'Q3': np.sum((coordinates[:, 0] < cx) & (coordinates[:, 1] >= cy)),  # Bottom-left
                'Q4': np.sum((coordinates[:, 0] >= cx) & (coordinates[:, 1] >= cy))  # Bottom-right
            }
            quadrant_counts = quadrants
        
        analysis = {
            'total_defects': len(self.defects),
            'valid_coordinates': len(coordinates),
            'center_of_mass': center_of_mass.tolist(),
            'std_deviation': std_dev.tolist(),
            'spatial_extent': extent.tolist(),
            'mean_distance_from_center': float(np.mean(distances)),
            'max_distance_from_center': float(np.max(distances)),
            'distance_std': float(np.std(distances)),
            'quadrant_distribution': quadrant_counts
        }
        
        return analysis
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Dictionary containing full analysis results
        """
        # Basic statistics
        total_defects = len(self.defects)
        total_clusters = len(self.clusters)
        
        # Defect type analysis
        defect_types = defaultdict(int)
        severity_counts = defaultdict(int)
        confidence_values = []
        
        for defect in self.defects:
            defect_types[defect.get('defect_type', 'UNKNOWN')] += 1
            severity_counts[defect.get('severity', 'LOW')] += 1
            confidence_values.append(defect.get('confidence', 0.5))
        
        # Cluster analysis
        cluster_sizes = [cluster['defect_count'] for cluster in self.clusters]
        clustered_defects = sum(cluster_sizes)
        
        # Spatial analysis
        spatial_analysis = self.analyze_spatial_distribution()
        
        report = {
            'summary': {
                'total_defects': total_defects,
                'total_clusters': total_clusters,
                'clustered_defects': clustered_defects,
                'clustering_efficiency': clustered_defects / total_defects if total_defects > 0 else 0,
                'avg_confidence': float(np.mean(confidence_values)) if confidence_values else 0.0,
                'data_integrity_issues': len(self.data_integrity_log)
            },
            'defect_types': dict(defect_types),
            'severity_distribution': dict(severity_counts),
            'cluster_statistics': {
                'cluster_count': total_clusters,
                'avg_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
                'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                'single_defect_clusters': sum(1 for size in cluster_sizes if size == 1)
            },
            'spatial_analysis': spatial_analysis,
            'data_integrity_log': self.data_integrity_log
        }
        
        return report
    
    def save_results(self, output_dir: str, include_heatmap: bool = True):
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
            include_heatmap: Whether to generate and save heatmap
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary report
        report = self.generate_summary_report()
        with open(output_path / "defect_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save clusters
        cluster_data = {
            'clusters': self.clusters,
            'clustering_parameters': {
                'eps': self.clustering_eps,
                'min_samples': self.min_cluster_size
            }
        }
        with open(output_path / "defect_clusters.json", 'w') as f:
            json.dump(cluster_data, f, indent=2, default=str)
        
        # Save heatmap if requested
        if include_heatmap and self.image_shape:
            heatmap = self.create_defect_heatmap()
            heatmap_normalized = (heatmap * 255).astype(np.uint8)
            cv2.imwrite(str(output_path / "defect_heatmap.png"), heatmap_normalized)
            
            # Save heatmap data
            np.save(output_path / "defect_heatmap.npy", heatmap)
        
        self.logger.info(f"Results saved to: {output_path}")


def main():
    """Test the DefectClusterAnalyzer functionality."""
    print("Testing DefectClusterAnalyzer...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create analyzer
    analyzer = DefectClusterAnalyzer(
        clustering_eps=50.0,
        min_cluster_size=2,
        image_shape=(500, 500)
    )
    
    # Create synthetic defects for testing
    print("Creating synthetic defects...")
    synthetic_defects = []
    
    # Cluster 1: Top-left region
    for i in range(10):
        x = np.random.normal(100, 20)
        y = np.random.normal(100, 20)
        defect = {
            'location_xy': [x, y],
            'defect_type': 'SCRATCH',
            'severity': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
            'confidence': np.random.uniform(0.5, 0.9),
            'area_px': np.random.randint(5, 50)
        }
        synthetic_defects.append(defect)
    
    # Cluster 2: Bottom-right region
    for i in range(8):
        x = np.random.normal(400, 15)
        y = np.random.normal(400, 15)
        defect = {
            'location_xy': [x, y],
            'defect_type': 'DIG',
            'severity': np.random.choice(['MEDIUM', 'HIGH', 'CRITICAL']),
            'confidence': np.random.uniform(0.6, 0.95),
            'area_px': np.random.randint(10, 100)
        }
        synthetic_defects.append(defect)
    
    # Scattered noise defects
    for i in range(5):
        x = np.random.uniform(50, 450)
        y = np.random.uniform(50, 450)
        defect = {
            'location_xy': [x, y],
            'defect_type': 'CONTAMINATION',
            'severity': 'LOW',
            'confidence': np.random.uniform(0.3, 0.7),
            'area_px': np.random.randint(1, 20)
        }
        synthetic_defects.append(defect)
    
    # Add defects to analyzer
    added_count = analyzer.add_defects_from_list(synthetic_defects)
    print(f"Added {added_count} synthetic defects")
    
    # Perform clustering
    print("Performing clustering...")
    clusters = analyzer.cluster_defects()
    print(f"Generated {len(clusters)} clusters")
    
    # Analyze spatial distribution
    spatial_analysis = analyzer.analyze_spatial_distribution()
    print(f"Spatial analysis complete. Center of mass: {spatial_analysis['center_of_mass']}")
    
    # Generate summary report
    report = analyzer.generate_summary_report()
    print(f"Analysis complete. Total defects: {report['summary']['total_defects']}")
    print(f"Defect types: {report['defect_types']}")
    print(f"Clustering efficiency: {report['summary']['clustering_efficiency']:.2%}")
    
    # Save results
    analyzer.save_results("test_defect_analysis", include_heatmap=True)
    print("Results saved to 'test_defect_analysis' directory")
    
    # Test loading from directory (if user has detection results)
    test_dir = input("\nEnter path to directory with detection JSON files (or press Enter to skip): ").strip()
    if test_dir and Path(test_dir).exists():
        analyzer2 = DefectClusterAnalyzer(image_shape=(500, 500))
        loaded = analyzer2.load_defects_from_directory(test_dir)
        if loaded > 0:
            clusters2 = analyzer2.cluster_defects()
            print(f"Loaded {loaded} defects from {test_dir}, created {len(clusters2)} clusters")


if __name__ == "__main__":
    main()
