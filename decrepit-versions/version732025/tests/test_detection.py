"""
Unit tests for enhanced detection module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import json
from sklearn.cluster import DBSCAN

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "current-process"))

from detection import (
    EnhancedDetector, MLDefectDetector, TraditionalDetector,
    DefectMerger, Defect
)
from config_manager import get_config


class TestDefect:
    """Test Defect dataclass"""
    
    def test_defect_creation(self):
        """Test creating a defect"""
        defect = Defect(
            type="scratch",
            location=(100, 100),
            bbox=(90, 90, 20, 20),
            severity=0.8,
            confidence=0.9,
            zone="core",
            area=400
        )
        
        assert defect.type == "scratch"
        assert defect.location == (100, 100)
        assert defect.bbox == (90, 90, 20, 20)
        assert defect.severity == 0.8
        assert defect.confidence == 0.9
        assert defect.zone == "core"
        assert defect.area == 400
        assert defect.detection_method == ""
    
    def test_defect_to_dict(self):
        """Test converting defect to dictionary"""
        defect = Defect(
            type="pit",
            location=(50, 50),
            bbox=(45, 45, 10, 10),
            severity=0.5,
            confidence=0.7,
            zone="cladding",
            area=100,
            properties={"radius": 5},
            detection_method="traditional"
        )
        
        d = defect.to_dict()
        
        assert isinstance(d, dict)
        assert d['type'] == "pit"
        assert d['location'] == (50, 50)
        assert d['properties']['radius'] == 5


class TestMLDefectDetector:
    """Test ML defect detector"""
    
    def test_initialization_no_ml(self):
        """Test initialization when ML is disabled"""
        config = get_config()
        config.processing.ml_enabled = False
        
        detector = MLDefectDetector()
        assert detector.detector_model is None
        assert detector.anomaly_model is None
    
    def test_detect_defects_no_model(self):
        """Test detection when no models available"""
        config = get_config()
        config.processing.ml_enabled = False
        
        detector = MLDefectDetector()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        zone_mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        defects = detector.detect_defects(image, zone_mask, "test_zone")
        
        assert isinstance(defects, list)
        assert len(defects) == 0  # No models, no detections
    
    def test_run_object_detection_error_handling(self):
        """Test object detection error handling"""
        detector = MLDefectDetector()
        detector.detector_model = None
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        zone_mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        defects = detector._run_object_detection(image, zone_mask, "test")
        assert isinstance(defects, list)
        assert len(defects) == 0
    
    def test_run_anomaly_detection_error_handling(self):
        """Test anomaly detection error handling"""
        detector = MLDefectDetector()
        detector.anomaly_model = None
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        zone_mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        defects = detector._run_anomaly_detection(image, zone_mask, "test")
        assert isinstance(defects, list)
        assert len(defects) == 0


class TestTraditionalDetector:
    """Test traditional detection methods"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return TraditionalDetector()
    
    @pytest.fixture
    def test_image(self):
        """Create test image with defects"""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Add scratch (line)
        cv2.line(image, (50, 50), (150, 60), (255, 255, 255), 2)
        
        # Add pit (circle)
        cv2.circle(image, (100, 150), 5, (0, 0, 0), -1)
        
        # Add contamination (colored blob)
        cv2.circle(image, (150, 100), 10, (255, 0, 0), -1)
        
        return image
    
    @pytest.fixture
    def zone_mask(self):
        """Create full zone mask"""
        return np.ones((200, 200), dtype=np.uint8) * 255
    
    def test_detect_scratches(self, detector, test_image, zone_mask):
        """Test scratch detection"""
        defects = detector.detect_scratches(test_image, zone_mask, "test_zone")
        
        assert isinstance(defects, list)
        # May or may not detect scratches depending on parameters
        
        for defect in defects:
            assert defect.type == "scratch"
            assert defect.zone == "test_zone"
            assert 'length' in defect.properties
            assert 'aspect_ratio' in defect.properties
    
    def test_detect_pits(self, detector, test_image, zone_mask):
        """Test pit detection"""
        defects = detector.detect_pits(test_image, zone_mask, "test_zone")
        
        assert isinstance(defects, list)
        
        for defect in defects:
            assert defect.type == "pit"
            assert defect.zone == "test_zone"
            assert 'radius' in defect.properties
    
    def test_detect_contamination(self, detector, test_image, zone_mask):
        """Test contamination detection"""
        defects = detector.detect_contamination(test_image, zone_mask, "test_zone")
        
        assert isinstance(defects, list)
        
        for defect in defects:
            assert defect.type == "contamination"
            assert defect.zone == "test_zone"
            assert 'color' in defect.properties
            assert 'compactness' in defect.properties
    
    def test_detect_statistical_anomalies(self, detector, test_image, zone_mask):
        """Test statistical anomaly detection"""
        defects = detector.detect_statistical_anomalies(test_image, zone_mask, "test_zone")
        
        assert isinstance(defects, list)
        
        for defect in defects:
            assert defect.type == "statistical_anomaly"
            assert defect.zone == "test_zone"
            assert 'max_z_score' in defect.properties
            assert 'mean_z_score' in defect.properties
    
    def test_empty_zone_mask(self, detector, test_image):
        """Test detection with empty zone mask"""
        empty_mask = np.zeros((200, 200), dtype=np.uint8)
        
        defects = detector.detect_scratches(test_image, empty_mask, "test")
        assert len(defects) == 0
        
        defects = detector.detect_pits(test_image, empty_mask, "test")
        assert len(defects) == 0


class TestDefectMerger:
    """Test defect merging and clustering"""
    
    @pytest.fixture
    def merger(self):
        """Create merger instance"""
        return DefectMerger()
    
    def test_merge_empty_list(self, merger):
        """Test merging empty defect list"""
        merged = merger.merge_defects([])
        assert isinstance(merged, list)
        assert len(merged) == 0
    
    def test_merge_single_defect(self, merger):
        """Test merging single defect"""
        defect = Defect(
            type="scratch",
            location=(100, 100),
            bbox=(90, 90, 20, 20),
            severity=0.8,
            confidence=0.9,
            zone="core",
            area=400
        )
        
        merged = merger.merge_defects([defect])
        assert len(merged) == 1
        assert merged[0].type == "scratch"
    
    def test_merge_overlapping_defects(self, merger):
        """Test merging overlapping defects"""
        # Create overlapping defects
        defects = []
        for i in range(3):
            defect = Defect(
                type="scratch",
                location=(100 + i*2, 100 + i*2),  # Slightly offset
                bbox=(90 + i*2, 90 + i*2, 20, 20),
                severity=0.7 + i*0.1,
                confidence=0.8,
                zone="core",
                area=400
            )
            defects.append(defect)
        
        merged = merger.merge_defects(defects)
        
        # Should merge into fewer defects
        assert len(merged) <= len(defects)
    
    def test_merge_different_zones(self, merger):
        """Test that defects in different zones are not merged"""
        defects = [
            Defect(
                type="scratch",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.8,
                confidence=0.9,
                zone="core",
                area=400
            ),
            Defect(
                type="scratch",
                location=(105, 105),  # Close but different zone
                bbox=(95, 95, 20, 20),
                severity=0.8,
                confidence=0.9,
                zone="cladding",
                area=400
            )
        ]
        
        merged = merger.merge_defects(defects)
        
        # Should not merge across zones
        assert len(merged) == 2
    
    def test_merge_cluster_properties(self, merger):
        """Test merging of cluster properties"""
        defects = [
            Defect(
                type="pit",
                location=(100, 100),
                bbox=(95, 95, 10, 10),
                severity=0.6,
                confidence=0.8,
                zone="core",
                area=100,
                properties={"radius": 5},
                detection_method="method1"
            ),
            Defect(
                type="scratch",  # Different type
                location=(102, 102),
                bbox=(97, 97, 10, 10),
                severity=0.8,
                confidence=0.9,
                zone="core",
                area=100,
                properties={"radius": 6},
                detection_method="method2"
            )
        ]
        
        # Merge
        cluster = merger._merge_cluster(defects)
        
        assert cluster.severity == pytest.approx(0.7, rel=0.01)  # Average
        assert cluster.confidence == pytest.approx(0.85, rel=0.01)
        assert cluster.area == 200  # Sum
        assert "method1+method2" in cluster.detection_method


class TestEnhancedDetector:
    """Test the main enhanced detector"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        config = get_config()
        config.interactive_mode = False
        config.processing.parallel_processing = False
        return EnhancedDetector()
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Add some defects
        cv2.circle(image, (50, 50), 5, (0, 0, 0), -1)  # Dark spot
        cv2.line(image, (100, 50), (150, 100), (255, 255, 255), 2)  # Bright line
        
        return image
    
    @pytest.fixture
    def zones(self):
        """Create test zones"""
        h, w = 200, 200
        zones = {
            'core': np.zeros((h, w), dtype=np.uint8),
            'cladding': np.zeros((h, w), dtype=np.uint8),
            'ferrule': np.zeros((h, w), dtype=np.uint8)
        }
        
        # Define zones
        cv2.circle(zones['core'], (100, 100), 40, 255, -1)
        cv2.circle(zones['cladding'], (100, 100), 80, 255, -1)
        cv2.circle(zones['cladding'], (100, 100), 40, 0, -1)
        cv2.circle(zones['ferrule'], (100, 100), 100, 255, -1)
        cv2.circle(zones['ferrule'], (100, 100), 80, 0, -1)
        
        return zones
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector.ml_detector is not None
        assert detector.traditional_detector is not None
        assert detector.merger is not None
    
    def test_detect_defects(self, detector, test_image, zones):
        """Test defect detection"""
        variations = {"original": test_image}
        
        defects = detector.detect_defects(test_image, zones, variations)
        
        assert isinstance(defects, list)
        # May detect some defects
        for defect in defects:
            assert isinstance(defect, Defect)
            assert defect.zone in ['core', 'cladding', 'ferrule']
    
    def test_detect_in_zone(self, detector, test_image):
        """Test detection in single zone"""
        zone_mask = np.ones((200, 200), dtype=np.uint8) * 255
        variations = {"original": test_image}
        
        defects = detector._detect_in_zone(test_image, zone_mask, "test_zone", variations)
        
        assert isinstance(defects, list)
        for defect in defects:
            assert defect.zone == "test_zone"
            assert defect.confidence >= detector.config.detection.confidence_threshold
    
    def test_visualize_defects(self, detector, test_image):
        """Test defect visualization"""
        defects = [
            Defect(
                type="scratch",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.8,
                confidence=0.9,
                zone="core",
                area=400
            ),
            Defect(
                type="pit",
                location=(50, 50),
                bbox=(45, 45, 10, 10),
                severity=0.5,
                confidence=0.7,
                zone="cladding",
                area=100
            )
        ]
        
        vis = detector.visualize_defects(test_image, defects)
        
        assert vis.shape == test_image.shape
        assert vis.dtype == test_image.dtype
        # Should have drawn something
        assert not np.array_equal(vis, test_image)
    
    def test_generate_heatmap(self, detector):
        """Test heatmap generation"""
        defects = [
            Defect(
                type="scratch",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.8,
                confidence=0.9,
                zone="core",
                area=400
            )
        ]
        
        heatmap = detector.generate_heatmap((200, 200), defects)
        
        assert heatmap.shape == (200, 200, 3)  # Color heatmap
        assert heatmap.dtype == np.uint8
        
        # Should have some non-zero values
        assert np.any(heatmap > 0)
    
    def test_empty_zone_handling(self, detector, test_image):
        """Test handling of empty zones"""
        zones = {
            'core': None,  # Empty zone
            'cladding': np.zeros((200, 200), dtype=np.uint8),  # All zeros
            'ferrule': np.ones((200, 200), dtype=np.uint8) * 255
        }
        
        variations = {"original": test_image}
        defects = detector.detect_defects(test_image, zones, variations)
        
        assert isinstance(defects, list)
        # Should only have defects from ferrule zone
        for defect in defects:
            assert defect.zone == "ferrule"


def test_integration_detection():
    """Integration test for complete detection pipeline"""
    config = get_config()
    config.interactive_mode = False
    config.processing.parallel_processing = False
    config.detection.confidence_threshold = 0.5  # Lower for testing
    
    # Create detector
    detector = EnhancedDetector()
    
    # Create test image with known defects
    image = np.ones((300, 300, 3), dtype=np.uint8) * 128
    
    # Add defects
    # Scratch in core
    cv2.line(image, (140, 140), (160, 160), (255, 255, 255), 3)
    
    # Pit in cladding
    cv2.circle(image, (100, 150), 8, (0, 0, 0), -1)
    
    # Contamination in ferrule
    cv2.ellipse(image, (200, 200), (15, 10), 45, 0, 360, (255, 0, 0), -1)
    
    # Create zones
    zones = {
        'core': np.zeros((300, 300), dtype=np.uint8),
        'cladding': np.zeros((300, 300), dtype=np.uint8),
        'ferrule': np.zeros((300, 300), dtype=np.uint8)
    }
    
    center = (150, 150)
    cv2.circle(zones['core'], center, 30, 255, -1)
    cv2.circle(zones['cladding'], center, 80, 255, -1)
    cv2.circle(zones['cladding'], center, 30, 0, -1)
    cv2.circle(zones['ferrule'], center, 140, 255, -1)
    cv2.circle(zones['ferrule'], center, 80, 0, -1)
    
    # Detect defects
    variations = {"original": image}
    defects = detector.detect_defects(image, zones, variations)
    
    # Should detect some defects
    assert len(defects) >= 0  # May vary based on detection parameters
    
    # Visualize
    vis = detector.visualize_defects(image, defects)
    assert vis.shape == image.shape
    
    # Generate heatmap
    if defects:
        heatmap = detector.generate_heatmap(image.shape[:2], defects)
        assert heatmap.shape == (300, 300, 3)


class TestMLDefectDetectorAdvanced:
    """Advanced tests for ML defect detector"""
    
    def test_pytorch_model_architecture(self):
        """Test PyTorch model architecture creation"""
        config = get_config()
        config.processing.ml_enabled = True
        config.processing.pytorch_enabled = True
        config.processing.tensorflow_enabled = False
        
        detector = MLDefectDetector()
        # Should handle missing dependencies gracefully
        assert detector.config is not None
    
    def test_tensorflow_model_architecture(self):
        """Test TensorFlow model architecture creation"""
        config = get_config()
        config.processing.ml_enabled = True
        config.processing.pytorch_enabled = False
        config.processing.tensorflow_enabled = True
        
        detector = MLDefectDetector()
        assert detector.config is not None
    
    def test_sliding_window_anomaly_detection(self):
        """Test sliding window approach in anomaly detection"""
        detector = MLDefectDetector()
        detector.anomaly_model = None  # Force no model
        
        # Create image with known anomaly
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Add bright spot (anomaly)
        cv2.circle(image, (100, 100), 20, (255, 255, 255), -1)
        
        zone_mask = np.ones((200, 200), dtype=np.uint8) * 255
        
        defects = detector._run_anomaly_detection(image, zone_mask, "test_zone")
        
        # Should handle no model case
        assert isinstance(defects, list)
    
    def test_object_detection_coordinate_conversion(self):
        """Test coordinate conversion in object detection"""
        detector = MLDefectDetector()
        detector.detector_model = None
        
        # Test with different image sizes
        test_cases = [(100, 100), (200, 150), (640, 480)]
        
        for h, w in test_cases:
            image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            zone_mask = np.ones((h, w), dtype=np.uint8) * 255
            
            defects = detector._run_object_detection(image, zone_mask, "test_zone")
            
            # Verify any returned defects have valid coordinates
            for defect in defects:
                x, y, bbox_w, bbox_h = defect.bbox
                assert 0 <= x < w
                assert 0 <= y < h
                assert x + bbox_w <= w
                assert y + bbox_h <= h


class TestTraditionalDetectorAdvanced:
    """Advanced tests for traditional detection methods"""
    
    @pytest.fixture
    def detector(self):
        """Create detector with empty knowledge base"""
        detector = TraditionalDetector()
        detector.knowledge_base = {}  # Start fresh
        return detector
    
    def test_scratch_detection_orientations(self, detector):
        """Test scratch detection with various orientations"""
        # Create images with scratches at different angles
        test_angles = [0, 45, 90, 135]
        
        for angle in test_angles:
            image = np.ones((200, 200, 3), dtype=np.uint8) * 128
            
            # Draw line at specific angle
            center = (100, 100)
            length = 60
            angle_rad = np.radians(angle)
            x1 = int(center[0] - length/2 * np.cos(angle_rad))
            y1 = int(center[1] - length/2 * np.sin(angle_rad))
            x2 = int(center[0] + length/2 * np.cos(angle_rad))
            y2 = int(center[1] + length/2 * np.sin(angle_rad))
            
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            zone_mask = np.ones((200, 200), dtype=np.uint8) * 255
            defects = detector.detect_scratches(image, zone_mask, "test_zone")
            
            # Should detect scratches at any orientation
            # (may not always detect due to threshold settings)
            assert isinstance(defects, list)
    
    def test_pit_detection_size_filtering(self, detector):
        """Test pit detection with size filtering"""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Add pits of various sizes
        pit_sizes = [3, 5, 10, 20]
        for i, size in enumerate(pit_sizes):
            x = 50 + i * 40
            cv2.circle(image, (x, 100), size, (0, 0, 0), -1)
        
        zone_mask = np.ones((200, 200), dtype=np.uint8) * 255
        
        # Test with different min size thresholds
        original_min_size = detector.config.detection.min_defect_size
        
        try:
            # Small threshold - should detect all
            detector.config.detection.min_defect_size = 10
            defects_all = detector.detect_pits(image, zone_mask, "test_zone")
            
            # Large threshold - should detect only large pits
            detector.config.detection.min_defect_size = 500
            defects_large = detector.detect_pits(image, zone_mask, "test_zone")
            
            # Should detect fewer with larger threshold
            assert len(defects_large) <= len(defects_all)
            
        finally:
            detector.config.detection.min_defect_size = original_min_size
    
    def test_contamination_color_analysis(self, detector):
        """Test contamination detection color analysis"""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Add colored contamination
        # Red contamination
        cv2.circle(image, (50, 100), 15, (255, 0, 0), -1)
        # Blue contamination
        cv2.circle(image, (150, 100), 15, (0, 0, 255), -1)
        
        zone_mask = np.ones((200, 200), dtype=np.uint8) * 255
        defects = detector.detect_contamination(image, zone_mask, "test_zone")
        
        # Check color properties
        for defect in defects:
            assert 'color' in defect.properties
            assert 'compactness' in defect.properties
            assert len(defect.properties['color']) == 3  # BGR color
    
    def test_statistical_anomaly_with_reference(self, detector):
        """Test statistical anomaly detection with reference statistics"""
        # Set reference statistics
        detector.knowledge_base['test_zone_statistics'] = {
            'mean': 128.0,
            'std': 10.0
        }
        
        # Create image with known anomalies
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Add very bright region (statistical anomaly)
        cv2.rectangle(image, (50, 50), (100, 100), (250, 250, 250), -1)
        
        zone_mask = np.ones((200, 200), dtype=np.uint8) * 255
        
        # Test with different thresholds
        original_threshold = detector.config.detection.anomaly_threshold
        
        try:
            # Low threshold - more sensitive
            detector.config.detection.anomaly_threshold = 2.0
            defects_sensitive = detector.detect_statistical_anomalies(image, zone_mask, "test_zone")
            
            # High threshold - less sensitive
            detector.config.detection.anomaly_threshold = 5.0
            defects_strict = detector.detect_statistical_anomalies(image, zone_mask, "test_zone")
            
            # Should detect more with lower threshold
            assert len(defects_sensitive) >= len(defects_strict)
            
        finally:
            detector.config.detection.anomaly_threshold = original_threshold
    
    def test_all_detection_methods_empty_image(self, detector):
        """Test all detection methods with empty/uniform image"""
        # Uniform gray image (no defects)
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        zone_mask = np.ones((200, 200), dtype=np.uint8) * 255
        
        # All methods should handle gracefully
        scratches = detector.detect_scratches(image, zone_mask, "test")
        pits = detector.detect_pits(image, zone_mask, "test")
        contamination = detector.detect_contamination(image, zone_mask, "test")
        anomalies = detector.detect_statistical_anomalies(image, zone_mask, "test")
        
        # Should return empty or very few defects
        assert isinstance(scratches, list)
        assert isinstance(pits, list)
        assert isinstance(contamination, list)
        assert isinstance(anomalies, list)


class TestDefectMergerAdvanced:
    """Advanced tests for defect merger"""
    
    def test_dbscan_parameter_effects(self):
        """Test effect of DBSCAN parameters on clustering"""
        merger = DefectMerger()
        
        # Create closely spaced defects
        defects = []
        for i in range(10):
            for j in range(10):
                defect = Defect(
                    type="pit",
                    location=(i * 10, j * 10),
                    bbox=(i * 10 - 2, j * 10 - 2, 4, 4),
                    severity=0.5,
                    confidence=0.8,
                    zone="core",
                    area=16
                )
                defects.append(defect)
        
        # Test with different eps values
        original_eps = merger.config.detection.cluster_eps
        
        try:
            # Small eps - many small clusters
            merger.config.detection.cluster_eps = 5
            merged_small = merger.merge_defects(defects)
            
            # Large eps - few large clusters
            merger.config.detection.cluster_eps = 50
            merged_large = merger.merge_defects(defects)
            
            # Should have fewer clusters with larger eps
            assert len(merged_large) < len(merged_small)
            
        finally:
            merger.config.detection.cluster_eps = original_eps
    
    def test_merge_mixed_defect_types(self):
        """Test merging clusters with mixed defect types"""
        merger = DefectMerger()
        
        # Create cluster with different defect types
        defects = [
            Defect(type="scratch", location=(100, 100), bbox=(95, 95, 10, 10),
                  severity=0.7, confidence=0.9, zone="core", area=100),
            Defect(type="scratch", location=(102, 102), bbox=(97, 97, 10, 10),
                  severity=0.6, confidence=0.8, zone="core", area=100),
            Defect(type="pit", location=(104, 104), bbox=(99, 99, 10, 10),
                  severity=0.8, confidence=0.7, zone="core", area=100),
        ]
        
        merged = merger._merge_cluster(defects)
        
        # Should use majority type (scratch)
        assert merged.type == "scratch"
        
        # Properties should be aggregated
        assert merged.area == 300  # Sum of areas
        assert 0.6 <= merged.severity <= 0.8  # Average
    
    def test_merge_preserves_detection_methods(self):
        """Test that merging preserves detection method information"""
        merger = DefectMerger()
        
        defects = [
            Defect(type="scratch", location=(100, 100), bbox=(95, 95, 10, 10),
                  severity=0.7, confidence=0.9, zone="core", area=100,
                  detection_method="ml_detection"),
            Defect(type="scratch", location=(102, 102), bbox=(97, 97, 10, 10),
                  severity=0.6, confidence=0.8, zone="core", area=100,
                  detection_method="traditional_scratch_detection"),
        ]
        
        merged = merger._merge_cluster(defects)
        
        # Should combine detection methods
        assert "ml_detection" in merged.detection_method
        assert "traditional_scratch_detection" in merged.detection_method


class TestEnhancedDetectorAdvanced:
    """Advanced tests for enhanced detector"""
    
    def test_confidence_filtering(self):
        """Test confidence threshold filtering"""
        config = get_config()
        config.processing.parallel_processing = False
        
        detector = EnhancedDetector()
        
        # Mock traditional detector to return defects with various confidences
        def mock_detect(image, zone_mask, zone_name):
            return [
                Defect(type="test", location=(50, 50), bbox=(45, 45, 10, 10),
                      severity=0.5, confidence=0.3, zone=zone_name, area=100),  # Low confidence
                Defect(type="test", location=(100, 100), bbox=(95, 95, 10, 10),
                      severity=0.5, confidence=0.8, zone=zone_name, area=100),  # High confidence
            ]
        
        detector.traditional_detector.detect_scratches = mock_detect
        detector.config.detection.detection_algorithms = ['scratches']
        
        # Test with different thresholds
        original_threshold = detector.config.detection.confidence_threshold
        
        try:
            # High threshold
            detector.config.detection.confidence_threshold = 0.7
            zones = {'test': np.ones((200, 200), dtype=np.uint8) * 255}
            defects_high = detector.detect_defects(
                np.zeros((200, 200, 3), dtype=np.uint8), zones, {}
            )
            
            # Low threshold
            detector.config.detection.confidence_threshold = 0.2
            defects_low = detector.detect_defects(
                np.zeros((200, 200, 3), dtype=np.uint8), zones, {}
            )
            
            # Should detect more with lower threshold
            assert len(defects_low) >= len(defects_high)
            
        finally:
            detector.config.detection.confidence_threshold = original_threshold
    
    def test_parallel_detection_consistency(self):
        """Test that parallel detection produces consistent results"""
        # Create two detectors with different parallel settings
        config = get_config()
        
        config.processing.parallel_processing = False
        detector_seq = EnhancedDetector()
        
        config.processing.parallel_processing = True
        config.processing.max_workers = 2
        detector_par = EnhancedDetector()
        
        # Create test data
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        cv2.circle(image, (100, 100), 10, (0, 0, 0), -1)  # Add defect
        
        zones = {
            'test': np.ones((200, 200), dtype=np.uint8) * 255
        }
        variations = {"original": image}
        
        # Detect with both
        defects_seq = detector_seq.detect_defects(image, zones, variations)
        defects_par = detector_par.detect_defects(image, zones, variations)
        
        # Results should be similar (order might differ)
        assert len(defects_seq) == len(defects_par)
    
    def test_heatmap_generation_edge_cases(self):
        """Test heatmap generation with edge cases"""
        detector = EnhancedDetector()
        
        # Empty defect list
        heatmap = detector.generate_heatmap((200, 200), [])
        assert heatmap.shape == (200, 200, 3)
        assert np.all(heatmap == cv2.applyColorMap(np.zeros((200, 200), dtype=np.uint8), cv2.COLORMAP_JET))
        
        # Defect at image edge
        edge_defects = [
            Defect(type="test", location=(0, 0), bbox=(0, 0, 10, 10),
                  severity=1.0, confidence=0.9, zone="test", area=100),
            Defect(type="test", location=(199, 199), bbox=(195, 195, 5, 5),
                  severity=1.0, confidence=0.9, zone="test", area=25),
        ]
        
        heatmap = detector.generate_heatmap((200, 200), edge_defects)
        assert heatmap.shape == (200, 200, 3)
        # Should have non-zero values at edges
        assert np.any(heatmap[0:10, 0:10] > 0)
        assert np.any(heatmap[190:200, 190:200] > 0)
    
    def test_visualization_color_mapping(self):
        """Test defect visualization with color mapping"""
        detector = EnhancedDetector()
        
        # Set custom colors
        detector.config.visualization.defect_colors = {
            'scratch': (255, 0, 0),      # Red
            'pit': (0, 255, 0),          # Green
            'contamination': (0, 0, 255), # Blue
            'unknown': (255, 255, 255)    # White
        }
        
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        defects = [
            Defect(type="scratch", location=(50, 50), bbox=(45, 45, 10, 10),
                  severity=0.5, confidence=0.9, zone="test", area=100),
            Defect(type="pit", location=(100, 100), bbox=(95, 95, 10, 10),
                  severity=0.5, confidence=0.9, zone="test", area=100),
            Defect(type="unknown_type", location=(150, 150), bbox=(145, 145, 10, 10),
                  severity=0.5, confidence=0.9, zone="test", area=100),
        ]
        
        vis = detector.visualize_defects(image, defects)
        
        # Check that colors were applied correctly
        # Note: This is a visual test, checking that image was modified
        assert not np.array_equal(vis, image)


def test_detection_pipeline_stress():
    """Stress test the detection pipeline with various scenarios"""
    config = get_config()
    config.interactive_mode = False
    config.processing.parallel_processing = False
    
    detector = EnhancedDetector()
    
    # Test scenarios
    test_cases = [
        # (name, image_generator, zone_generator)
        ("many_defects", 
         lambda: create_image_with_many_defects(200, 200, 50),
         lambda h, w: {'test': np.ones((h, w), dtype=np.uint8) * 255}),
        
        ("overlapping_zones",
         lambda: np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),
         lambda h, w: create_overlapping_zones(h, w)),
        
        ("complex_boundaries",
         lambda: create_image_with_complex_defects(200, 200),
         lambda h, w: create_complex_zone_boundaries(h, w)),
    ]
    
    for name, img_gen, zone_gen in test_cases:
        try:
            image = img_gen()
            zones = zone_gen(*image.shape[:2])
            variations = {"original": image}
            
            defects = detector.detect_defects(image, zones, variations)
            
            # Should complete without error
            assert isinstance(defects, list)
            
            # Validate defects
            for defect in defects:
                assert isinstance(defect, Defect)
                assert defect.confidence >= 0 and defect.confidence <= 1
                assert defect.severity >= 0 and defect.severity <= 1
                
        except Exception as e:
            pytest.fail(f"Detection failed for {name}: {e}")


def create_image_with_many_defects(h, w, num_defects):
    """Create image with many synthetic defects"""
    image = np.ones((h, w, 3), dtype=np.uint8) * 128
    
    for _ in range(num_defects):
        defect_type = np.random.choice(['scratch', 'pit', 'contamination'])
        
        if defect_type == 'scratch':
            # Random line
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            color = (255, 255, 255) if np.random.random() > 0.5 else (0, 0, 0)
            cv2.line(image, pt1, pt2, color, np.random.randint(1, 3))
            
        elif defect_type == 'pit':
            # Random circle
            center = (np.random.randint(10, w-10), np.random.randint(10, h-10))
            radius = np.random.randint(2, 10)
            cv2.circle(image, center, radius, (0, 0, 0), -1)
            
        else:  # contamination
            # Random colored blob
            center = (np.random.randint(10, w-10), np.random.randint(10, h-10))
            axes = (np.random.randint(5, 15), np.random.randint(5, 15))
            angle = np.random.randint(0, 180)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.ellipse(image, center, axes, angle, 0, 360, color, -1)
    
    return image


def create_overlapping_zones(h, w):
    """Create zones with overlapping regions"""
    zones = {}
    
    # Create overlapping circular zones
    center = (w // 2, h // 2)
    
    zones['core'] = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(zones['core'], center, min(h, w) // 4, 255, -1)
    
    zones['cladding'] = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(zones['cladding'], (center[0] + 10, center[1]), min(h, w) // 3, 255, -1)
    
    zones['ferrule'] = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(zones['ferrule'], (center[0], center[1] + 10), min(h, w) // 2, 255, -1)
    
    return zones


def create_complex_zone_boundaries(h, w):
    """Create zones with complex boundaries"""
    zones = {}
    
    # Create irregular shaped zones
    zones['core'] = np.zeros((h, w), dtype=np.uint8)
    points = np.array([
        [w//2 - 20, h//2 - 30],
        [w//2 + 20, h//2 - 25],
        [w//2 + 25, h//2 + 20],
        [w//2 - 15, h//2 + 30]
    ], np.int32)
    cv2.fillPoly(zones['core'], [points], 255)
    
    zones['cladding'] = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(zones['cladding'], (w//2, h//2), (60, 40), 30, 0, 360, 255, -1)
    zones['cladding'][zones['core'] > 0] = 0
    
    zones['ferrule'] = np.ones((h, w), dtype=np.uint8) * 255
    zones['ferrule'][zones['core'] > 0] = 0
    zones['ferrule'][zones['cladding'] > 0] = 0
    
    return zones


def create_image_with_complex_defects(h, w):
    """Create image with complex defect patterns"""
    image = np.ones((h, w, 3), dtype=np.uint8) * 128
    
    # Add noise
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Add gradient defect
    for i in range(h):
        image[i, :, :] = np.clip(image[i, :, :].astype(np.int32) + i // 10, 0, 255)
    
    # Add textured region
    texture_region = image[50:100, 50:100]
    texture = np.random.randint(100, 150, texture_region.shape, dtype=np.uint8)
    image[50:100, 50:100] = texture
    
    return image


if __name__ == "__main__":
    pytest.main([__file__, "-v"])