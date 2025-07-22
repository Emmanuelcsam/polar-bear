"""
Machine Learning Dataset Builder for Fiber Optic Inspection
Ingests images, performs unsupervised analysis, and creates structured datasets
"""

import cv2
import numpy as np
import pandas as pd
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats, ndimage
from scipy.spatial import distance
import hashlib
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZoneSegmenter:
    """Segment fiber end-face into zones: core, cladding, ferrule"""
    
    def __init__(self, expected_core_radius_um: float = 9.0,
                 expected_cladding_radius_um: float = 62.5,
                 pixels_per_um: float = 2.0):
        self.expected_core_r = expected_core_radius_um * pixels_per_um
        self.expected_clad_r = expected_cladding_radius_um * pixels_per_um
        self.px_per_um = pixels_per_um
    
    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment image into fiber zones
        
        Returns:
            Dictionary with masks for each zone
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find fiber center using Hough circles
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=int(self.expected_clad_r * 0.8),
            maxRadius=int(self.expected_clad_r * 1.2)
        )
        
        if circles is None:
            # Fallback: use image center
            h, w = gray.shape
            center = (w // 2, h // 2)
            clad_radius = min(h, w) // 3
            core_radius = clad_radius // 7
        else:
            circles = np.uint16(np.around(circles))
            center = (circles[0, 0, 0], circles[0, 0, 1])
            clad_radius = circles[0, 0, 2]
            core_radius = int(clad_radius * (self.expected_core_r / self.expected_clad_r))
        
        # Create masks
        h, w = gray.shape
        masks = {
            'core': np.zeros((h, w), dtype=np.uint8),
            'cladding': np.zeros((h, w), dtype=np.uint8),
            'ferrule': np.zeros((h, w), dtype=np.uint8),
            'adhesive': np.zeros((h, w), dtype=np.uint8)
        }
        
        # Core mask
        cv2.circle(masks['core'], center, core_radius, 255, -1)
        
        # Cladding mask (excluding core)
        cv2.circle(masks['cladding'], center, clad_radius, 255, -1)
        masks['cladding'] = cv2.bitwise_and(masks['cladding'], 
                                           cv2.bitwise_not(masks['core']))
        
        # Adhesive zone (thin ring outside cladding)
        adhesive_outer = int(clad_radius * 1.1)
        cv2.circle(masks['adhesive'], center, adhesive_outer, 255, -1)
        cv2.circle(masks['adhesive'], center, clad_radius, 0, -1)
        
        # Ferrule (everything else)
        all_fiber = cv2.bitwise_or(masks['core'], masks['cladding'])
        all_fiber = cv2.bitwise_or(all_fiber, masks['adhesive'])
        masks['ferrule'] = cv2.bitwise_not(all_fiber)
        
        # Store geometry
        self.geometry = {
            'center': center,
            'core_radius': core_radius,
            'cladding_radius': clad_radius,
            'core_offset': 0.0  # Could calculate if core center differs
        }
        
        return masks

class DefectDetector:
    """Unsupervised defect detection using image processing"""
    
    def __init__(self):
        self.min_defect_area = 10  # pixels
        self.scratch_aspect_ratio = 3.0
    
    def detect_anomalies(self, image: np.ndarray, 
                        masks: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Detect defects using multiple methods
        
        Returns:
            List of defect dictionaries
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        defects = []
        
        # Method 1: Adaptive threshold for contamination
        contamination = self._detect_contamination(gray)
        
        # Method 2: Edge detection for scratches
        scratches = self._detect_scratches(gray)
        
        # Method 3: Statistical outliers for pits
        pits = self._detect_pits(gray)
        
        # Combine all defect masks
        all_defects = cv2.bitwise_or(contamination, scratches)
        all_defects = cv2.bitwise_or(all_defects, pits)
        
        # Find contours and classify
        contours, _ = cv2.findContours(all_defects, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_defect_area:
                continue
            
            # Get defect properties
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            
            # Classify defect type
            if aspect_ratio >= self.scratch_aspect_ratio:
                defect_type = "SCRATCH"
            elif self._is_circular(contour):
                defect_type = "PIT"
            else:
                defect_type = "CONTAMINATION"
            
            # Determine zone
            zone = self._get_zone(cx, cy, masks)
            
            # Calculate severity
            severity = self._calculate_severity(area, zone, defect_type)
            
            defects.append({
                'id': f'DEF_{idx:04d}',
                'type': defect_type,
                'bbox': [int(x), int(y), int(w), int(h)],
                'centroid': [int(cx), int(cy)],
                'area': int(area),
                'aspect_ratio': float(aspect_ratio),
                'zone': zone,
                'severity': severity,
                'contour': contour
            })
        
        return defects
    
    def _detect_contamination(self, gray: np.ndarray) -> np.ndarray:
        """Detect contamination using adaptive thresholding"""
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _detect_scratches(self, gray: np.ndarray) -> np.ndarray:
        """Detect scratches using directional filters"""
        scratch_mask = np.zeros_like(gray)
        
        # Apply directional filters at multiple angles
        for angle in range(0, 180, 15):
            # Create line kernel
            kernel_size = 15
            kernel = self._create_line_kernel(angle, kernel_size)
            
            # Filter image
            filtered = cv2.filter2D(gray, -1, kernel)
            
            # Threshold
            _, thresh = cv2.threshold(filtered, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Accumulate
            scratch_mask = cv2.bitwise_or(scratch_mask, thresh)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, kernel)
        
        return scratch_mask
    
    def _detect_pits(self, gray: np.ndarray) -> np.ndarray:
        """Detect pits using blob detection"""
        # Invert image (pits are usually dark)
        inverted = cv2.bitwise_not(gray)
        
        # Blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(inverted)
        
        # Create mask from keypoints
        pit_mask = np.zeros_like(gray)
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            cv2.circle(pit_mask, (x, y), r, 255, -1)
        
        return pit_mask
    
    def _create_line_kernel(self, angle: float, length: int) -> np.ndarray:
        """Create line-shaped kernel at given angle"""
        angle_rad = np.deg2rad(angle)
        center = length // 2
        kernel = np.zeros((length, length))
        
        for i in range(length):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
        
        return kernel / (np.sum(kernel) + 1e-8)
    
    def _is_circular(self, contour) -> bool:
        """Check if contour is circular"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity > 0.7
    
    def _get_zone(self, x: int, y: int, masks: Dict[str, np.ndarray]) -> str:
        """Determine which zone a point belongs to"""
        for zone, mask in masks.items():
            if mask[y, x] > 0:
                return zone.upper()
        return "UNKNOWN"
    
    def _calculate_severity(self, area: float, zone: str, defect_type: str) -> str:
        """Calculate defect severity based on size and location"""
        # Zone-specific thresholds
        if zone == "CORE":
            if area > 50: return "HIGH"
            elif area > 20: return "MEDIUM"
            else: return "LOW"
        elif zone == "CLADDING":
            if area > 100: return "HIGH"
            elif area > 50: return "MEDIUM"
            else: return "LOW"
        else:  # FERRULE
            if area > 200: return "HIGH"
            elif area > 100: return "MEDIUM"
            else: return "LOW"

class FeatureExtractor:
    """Extract comprehensive features from images and regions"""
    
    def extract_intensity_stats(self, image: np.ndarray, 
                               mask: np.ndarray) -> Dict[str, float]:
        """Extract intensity statistics for a region"""
        pixels = image[mask > 0]
        if len(pixels) == 0:
            return {}
        
        return {
            'mean': float(np.mean(pixels)),
            'std': float(np.std(pixels)),
            'min': float(np.min(pixels)),
            'max': float(np.max(pixels)),
            'median': float(np.median(pixels)),
            'q25': float(np.percentile(pixels, 25)),
            'q75': float(np.percentile(pixels, 75)),
            'skewness': float(stats.skew(pixels)),
            'kurtosis': float(stats.kurtosis(pixels))
        }
    
    def extract_texture_features(self, image: np.ndarray, 
                                mask: np.ndarray) -> Dict[str, float]:
        """Extract texture features using GLCM"""
        # Simplified texture features
        pixels = image[mask > 0]
        if len(pixels) == 0:
            return {}
        
        # Local variance
        kernel = np.ones((5, 5)) / 25
        local_mean = cv2.filter2D(image, -1, kernel)
        local_var = cv2.filter2D((image - local_mean)**2, -1, kernel)
        
        var_pixels = local_var[mask > 0]
        
        return {
            'entropy': float(stats.entropy(pixels)),
            'local_variance_mean': float(np.mean(var_pixels)),
            'local_variance_std': float(np.std(var_pixels))
        }
    
    def extract_gradient_features(self, image: np.ndarray,
                                 mask: np.ndarray) -> Dict[str, float]:
        """Extract gradient-based features"""
        # Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        mag_pixels = grad_mag[mask > 0]
        if len(mag_pixels) == 0:
            return {}
        
        return {
            'gradient_mean': float(np.mean(mag_pixels)),
            'gradient_std': float(np.std(mag_pixels)),
            'gradient_max': float(np.max(mag_pixels))
        }

class DefectClusterer:
    """Cluster defects to assign labels"""
    
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    
    def cluster_defects(self, defects: List[Dict]) -> List[Dict]:
        """Cluster defects and assign labels"""
        if len(defects) < self.n_clusters:
            # Not enough defects to cluster
            for d in defects:
                d['cluster'] = 0
            return defects
        
        # Extract features for clustering
        features = []
        for d in defects:
            features.append([
                d['area'],
                d['aspect_ratio'],
                1 if d['zone'] == 'CORE' else 2 if d['zone'] == 'CLADDING' else 3,
                1 if d['type'] == 'SCRATCH' else 2 if d['type'] == 'PIT' else 3
            ])
        
        # Normalize and cluster
        features_scaled = self.scaler.fit_transform(features)
        clusters = self.clusterer.fit_predict(features_scaled)
        
        # Assign clusters
        for d, cluster in zip(defects, clusters):
            d['cluster'] = int(cluster)
        
        return defects

class DatasetBuilder:
    """Main class to build ML datasets from fiber images"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.segmenter = ZoneSegmenter()
        self.detector = DefectDetector()
        self.feature_extractor = FeatureExtractor()
        self.clusterer = DefectClusterer()
        
        # Create output subdirectories
        self.pixel_dir = self.output_dir / "pixel_level"
        self.region_dir = self.output_dir / "region_level"
        self.image_dir = self.output_dir / "image_level"
        self.defect_lib_dir = self.output_dir / "defect_library"
        
        for d in [self.pixel_dir, self.region_dir, self.image_dir, self.defect_lib_dir]:
            d.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.output_dir / "dataset.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                filename TEXT,
                center_x REAL,
                center_y REAL,
                core_radius REAL,
                cladding_radius REAL,
                has_defects INTEGER,
                total_defects INTEGER,
                processing_time REAL
            )
        """)
        
        # Defects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS defects (
                defect_id TEXT PRIMARY KEY,
                image_id TEXT,
                defect_type TEXT,
                zone TEXT,
                severity TEXT,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                area REAL,
                aspect_ratio REAL,
                cluster INTEGER,
                FOREIGN KEY (image_id) REFERENCES images(image_id)
            )
        """)
        
        # Zone features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zone_features (
                feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT,
                zone TEXT,
                mean_intensity REAL,
                std_intensity REAL,
                entropy REAL,
                gradient_mean REAL,
                defect_count INTEGER,
                defect_area_ratio REAL,
                FOREIGN KEY (image_id) REFERENCES images(image_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def process_image(self, image_path: Path) -> Dict:
        """Process a single image"""
        logger.info(f"Processing {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load {image_path}")
            return None
        
        # Generate image ID
        image_id = hashlib.md5(image_path.name.encode()).hexdigest()[:16]
        
        # Segment zones
        masks = self.segmenter.segment(image)
        geometry = self.segmenter.geometry
        
        # Detect defects
        defects = self.detector.detect_anomalies(image, masks)
        
        # Extract features for each zone
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        zone_features = {}
        
        for zone, mask in masks.items():
            if np.sum(mask) == 0:
                continue
            
            features = {}
            features.update(self.feature_extractor.extract_intensity_stats(gray, mask))
            features.update(self.feature_extractor.extract_texture_features(gray, mask))
            features.update(self.feature_extractor.extract_gradient_features(gray, mask))
            
            # Add defect statistics
            zone_defects = [d for d in defects if d['zone'].lower() == zone]
            features['defect_count'] = len(zone_defects)
            features['defect_area_ratio'] = sum(d['area'] for d in zone_defects) / np.sum(mask)
            
            zone_features[zone] = features
        
        # Save masks
        self._save_masks(image_id, masks, defects)
        
        # Save to database
        self._save_to_database(image_id, image_path.name, geometry, defects, zone_features)
        
        # Save defect patches
        self._save_defect_patches(image_id, image, defects)
        
        return {
            'image_id': image_id,
            'geometry': geometry,
            'defect_count': len(defects),
            'zones': zone_features
        }
    
    def _save_masks(self, image_id: str, masks: Dict[str, np.ndarray], 
                    defects: List[Dict]):
        """Save segmentation masks"""
        # Zone mask (multi-class)
        h, w = next(iter(masks.values())).shape
        zone_mask = np.zeros((h, w), dtype=np.uint8)
        zone_mask[masks['core'] > 0] = 1
        zone_mask[masks['cladding'] > 0] = 2
        zone_mask[masks['ferrule'] > 0] = 3
        
        # Defect mask
        defect_mask = np.zeros((h, w), dtype=np.uint8)
        for d in defects:
            cv2.drawContours(defect_mask, [d['contour']], -1, 
                           {'SCRATCH': 1, 'PIT': 2, 'CONTAMINATION': 3}[d['type']], -1)
        
        # Save
        np.save(self.pixel_dir / f"{image_id}_zones.npy", zone_mask)
        np.save(self.pixel_dir / f"{image_id}_defects.npy", defect_mask)
    
    def _save_to_database(self, image_id: str, filename: str, 
                         geometry: Dict, defects: List[Dict], 
                         zone_features: Dict):
        """Save results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert image record
        cursor.execute("""
            INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image_id, filename,
            geometry['center'][0], geometry['center'][1],
            geometry['core_radius'], geometry['cladding_radius'],
            1 if defects else 0, len(defects), 0.0
        ))
        
        # Insert defects
        for d in defects:
            cursor.execute("""
                INSERT OR REPLACE INTO defects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{image_id}_{d['id']}", image_id,
                d['type'], d['zone'], d['severity'],
                d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3],
                d['area'], d['aspect_ratio'], d.get('cluster', -1)
            ))
        
        # Insert zone features
        for zone, features in zone_features.items():
            cursor.execute("""
                INSERT INTO zone_features 
                (image_id, zone, mean_intensity, std_intensity, entropy, 
                 gradient_mean, defect_count, defect_area_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_id, zone,
                features.get('mean', 0), features.get('std', 0),
                features.get('entropy', 0), features.get('gradient_mean', 0),
                features.get('defect_count', 0), features.get('defect_area_ratio', 0)
            ))
        
        conn.commit()
        conn.close()
    
    def _save_defect_patches(self, image_id: str, image: np.ndarray, 
                            defects: List[Dict]):
        """Save individual defect patches"""
        for d in defects:
            x, y, w, h = d['bbox']
            # Add padding
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image.shape[1], x + w + pad)
            y2 = min(image.shape[0], y + h + pad)
            
            patch = image[y1:y2, x1:x2]
            
            # Save by type
            type_dir = self.defect_lib_dir / d['type'].lower()
            type_dir.mkdir(exist_ok=True)
            
            patch_path = type_dir / f"{image_id}_{d['id']}.png"
            cv2.imwrite(str(patch_path), patch)
    
    def build_dataset(self, image_dir: Path):
        """Build complete dataset from directory of images"""
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        logger.info(f"Found {len(image_paths)} images to process")
        
        all_defects = []
        
        # Process each image
        for img_path in tqdm(image_paths, desc="Processing images"):
            result = self.process_image(img_path)
            if result:
                # Collect defects for clustering
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM defects WHERE image_id = ?
                """, (result['image_id'],))
                img_defects = cursor.fetchall()
                conn.close()
                
                all_defects.extend(img_defects)
        
        # Cluster all defects
        if all_defects:
            logger.info("Clustering defects...")
            # Convert back to dict format for clustering
            defect_dicts = []
            for d in all_defects:
                defect_dicts.append({
                    'defect_id': d[0],
                    'type': d[2],
                    'zone': d[3],
                    'area': d[9],
                    'aspect_ratio': d[10]
                })
            
            # Cluster
            clustered = self.clusterer.cluster_defects(defect_dicts)
            
            # Update database with clusters
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for d in clustered:
                cursor.execute("""
                    UPDATE defects SET cluster = ? WHERE defect_id = ?
                """, (d['cluster'], d['defect_id']))
            conn.commit()
            conn.close()
        
        # Generate summary CSV files
        self._generate_summaries()
        
        logger.info("Dataset building complete!")
    
    def _generate_summaries(self):
        """Generate summary CSV files"""
        conn = sqlite3.connect(self.db_path)
        
        # Image-level summary
        image_df = pd.read_sql_query("""
            SELECT * FROM images
        """, conn)
        image_df.to_csv(self.image_dir / "image_summary.csv", index=False)
        
        # Defect summary
        defect_df = pd.read_sql_query("""
            SELECT * FROM defects
        """, conn)
        defect_df.to_csv(self.region_dir / "defect_summary.csv", index=False)
        
        # Zone features
        zone_df = pd.read_sql_query("""
            SELECT * FROM zone_features
        """, conn)
        zone_df.to_csv(self.region_dir / "zone_features.csv", index=False)
        
        conn.close()

# Example usage
if __name__ == "__main__":
    # Build dataset
    builder = DatasetBuilder(Path("./fiber_dataset"))
    builder.build_dataset(Path("./raw_images"))
    
    print("Dataset built successfully!")
    print(f"Output directory: {builder.output_dir}")
    print(f"Database: {builder.db_path}")
    print(f"Pixel masks: {builder.pixel_dir}")
    print(f"Defect library: {builder.defect_lib_dir}")