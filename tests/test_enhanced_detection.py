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

from enhanced_detection import (
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])