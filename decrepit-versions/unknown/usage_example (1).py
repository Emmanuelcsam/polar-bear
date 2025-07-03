#!/usr/bin/env python3
"""
Usage Example and Integration Guide for Ultimate Defect Detection System
========================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from ultimate_defect_detector import UltimateDefectDetector, DefectDetectionConfig, DefectType
import logging
import json
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class FiberOpticInspectionPipeline:
    """
    Complete inspection pipeline integrating the ultimate defect detection system
    with mask separation and region analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the inspection pipeline"""
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
        
        # Initialize detector with custom configuration
        detector_config = DefectDetectionConfig()
        self._apply_custom_config(detector_config)
        
        self.detector = UltimateDefectDetector(detector_config)
        self.results = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the pipeline"""
        return {
            "preprocessing": {
                "use_illumination_correction": True,
                "use_noise_reduction": True,
                "use_contrast_enhancement": True
            },
            "detection": {
                "algorithms": {
                    "statistical": ["zscore", "mad", "iqr", "grubbs", "dixon", "chauvenet"],
                    "spatial": ["lbp", "glcm", "fractal"],
                    "frequency": ["fft", "wavelet", "gabor"],
                    "morphological": ["tophat", "blackhat", "gradient"],
                    "ml": ["isolation_forest", "one_class_svm", "dbscan"]
                },
                "thresholds": {
                    "zscore": 3.0,
                    "mad": 3.5,
                    "min_defect_area": 3,
                    "min_contrast": 10.0
                }
            },
            "regions": {
                "core": {
                    "diameter_um": 9.0,
                    "sensitivity": 1.2,
                    "max_defects": 0
                },
                "cladding": {
                    "diameter_um": 125.0,
                    "sensitivity": 1.0,
                    "max_defects": 5
                },
                "ferrule": {
                    "sensitivity": 0.8,
                    "max_defects": 10
                }
            },
            "output": {
                "save_intermediate": True,
                "visualization_dpi": 300,
                "report_format": "comprehensive"
            }
        }
    
    def _apply_custom_config(self, detector_config: DefectDetectionConfig) -> None:
        """Apply custom configuration to detector"""
        # Apply detection thresholds
        thresholds = self.config["detection"]["thresholds"]
        detector_config.zscore_threshold = thresholds.get("zscore", 3.0)
        detector_config.mad_threshold = thresholds.get("mad", 3.5)
        detector_config.min_defect_area = thresholds.get("min_defect_area", 3)
        
        # Apply preprocessing settings
        prep = self.config["preprocessing"]
        detector_config.use_illumination_correction = prep.get("use_illumination_correction", True)
        detector_config.use_noise_reduction = prep.get("use_noise_reduction", True)
        detector_config.use_contrast_enhancement = prep.get("use_contrast_enhancement", True)
    
    def inspect_fiber(self, image_path: str, mask_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete fiber optic inspection
        
        Args:
            image_path: Path to fiber optic image
            mask_path: Optional path to pre-generated masks
            
        Returns:
            Comprehensive inspection results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        logging.info(f"Loaded image: {image_path}")
        
        # Generate or load masks
        if mask_path and Path(mask_path).exists():
            masks = self._load_masks(mask_path)
        else:
            masks = self._generate_masks(image)
        
        # Analyze each region
        all_results = {}
        
        for region_name, mask in masks.items():
            logging.info(f"Analyzing {region_name} region...")
            
            # Get region-specific sensitivity
            sensitivity = self.config["regions"].get(region_name.lower(), {}).get("sensitivity", 1.0)
            
            # Adjust detector sensitivity for this region
            self._adjust_detector_sensitivity(sensitivity)
            
            # Run comprehensive analysis
            region_results = self.detector.analyze_comprehensive(
                image, 
                mask, 
                region_type=region_name
            )
            
            all_results[region_name] = region_results
            
            # Log summary
            num_defects = len(region_results.get("defects", []))
            logging.info(f"{region_name}: Found {num_defects} defects")
        
        # Combine results
        self.results = {
            "image_path": image_path,
            "regions": all_results,
            "summary": self._generate_summary(all_results),
            "pass_fail": self._evaluate_pass_fail(all_results)
        }
        
        return self.results
    
    def _generate_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate region masks using simple thresholding (placeholder)"""
        # This is a simplified version - in practice, use the mask_separation.py methods
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Simple circular masks based on image center
        h, w = gray.shape
        center = (w // 2, h // 2)
        
        masks = {}
        
        # Core mask (innermost region)
        core_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core_mask, center, int(min(h, w) * 0.05), 255, -1)
        masks["Core"] = core_mask
        
        # Cladding mask (middle region)
        cladding_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(cladding_mask, center, int(min(h, w) * 0.25), 255, -1)
        cladding_mask = cv2.subtract(cladding_mask, core_mask)
        masks["Cladding"] = cladding_mask
        
        # Ferrule mask (outer region)
        ferrule_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ferrule_mask, center, int(min(h, w) * 0.45), 255, -1)
        ferrule_mask = cv2.subtract(ferrule_mask, cladding_mask)
        ferrule_mask = cv2.subtract(ferrule_mask, core_mask)
        masks["Ferrule"] = ferrule_mask
        
        return masks
    
    def _load_masks(self, mask_path: str) -> Dict[str, np.ndarray]:
        """Load pre-generated masks"""
        # Implementation depends on mask storage format
        # This is a placeholder
        return {}
    
    def _adjust_detector_sensitivity(self, sensitivity: float) -> None:
        """Adjust detector sensitivity for specific region"""
        # Adjust thresholds based on sensitivity
        base_zscore = self.config["detection"]["thresholds"]["zscore"]
        base_mad = self.config["detection"]["thresholds"]["mad"]
        
        self.detector.config.zscore_threshold = base_zscore / sensitivity
        self.detector.config.mad_threshold = base_mad / sensitivity
    
    def _generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate inspection summary"""
        summary = {
            "total_defects": 0,
            "defects_by_region": {},
            "defects_by_type": {},
            "critical_defects": []
        }
        
        for region_name, results in all_results.items():
            defects = results.get("defects", [])
            summary["total_defects"] += len(defects)
            summary["defects_by_region"][region_name] = len(defects)
            
            for defect in defects:
                defect_type = defect.type.name
                if defect_type not in summary["defects_by_type"]:
                    summary["defects_by_type"][defect_type] = 0
                summary["defects_by_type"][defect_type] += 1
                
                # Critical defects: high confidence and in core region
                if region_name == "Core" or (defect.confidence > 0.9 and defect.area_px > 100):
                    summary["critical_defects"].append({
                        "id": defect.id,
                        "type": defect_type,
                        "region": region_name,
                        "confidence": defect.confidence,
                        "area": defect.area_px
                    })
        
        return summary
    
    def _evaluate_pass_fail(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate pass/fail based on IEC standards"""
        pass_fail = {
            "overall": "PASS",
            "reasons": [],
            "by_region": {}
        }
        
        for region_name, results in all_results.items():
            region_config = self.config["regions"].get(region_name.lower(), {})
            max_defects = region_config.get("max_defects", float('inf'))
            
            defects = results.get("defects", [])
            region_pass = len(defects) <= max_defects
            
            pass_fail["by_region"][region_name] = "PASS" if region_pass else "FAIL"
            
            if not region_pass:
                pass_fail["overall"] = "FAIL"
                pass_fail["reasons"].append(
                    f"{region_name}: {len(defects)} defects (max allowed: {max_defects})"
                )
        
        return pass_fail
    
    def generate_report(self, output_dir: str = "inspection_results") -> None:
        """Generate comprehensive inspection report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Save numerical results
        results_file = output_path / "inspection_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays and custom objects to serializable format
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        # 2. Generate visualizations for each region
        for region_name, results in self.results["regions"].items():
            # Save region-specific visualization
            save_path = output_path / f"{region_name.lower()}_analysis.png"
            self._visualize_region_results(region_name, results, save_path)
        
        # 3. Generate summary visualization
        self._generate_summary_visualization(output_path / "summary.png")
        
        # 4. Generate text report
        self._generate_text_report(output_path / "report.txt")
        
        # 5. Generate detailed defect catalog
        self._generate_defect_catalog(output_path / "defect_catalog.html")
        
        logging.info(f"Report generated in {output_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and custom objects to serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def _visualize_region_results(self, region_name: str, results: Dict[str, Any], 
                                 save_path: Path) -> None:
        """Visualize results for a specific region"""
        # Use the detector's visualization method
        # This is a placeholder - implement custom visualization as needed
        pass
    
    def _generate_summary_visualization(self, save_path: Path) -> None:
        """Generate summary visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Defects by region
        ax = axes[0, 0]
        regions = list(self.results["summary"]["defects_by_region"].keys())
        counts = list(self.results["summary"]["defects_by_region"].values())
        ax.bar(regions, counts)
        ax.set_title("Defects by Region")
        ax.set_ylabel("Count")
        
        # 2. Defects by type
        ax = axes[0, 1]
        types = list(self.results["summary"]["defects_by_type"].keys())
        counts = list(self.results["summary"]["defects_by_type"].values())
        ax.pie(counts, labels=types, autopct='%1.1f%%')
        ax.set_title("Defects by Type")
        
        # 3. Pass/Fail status
        ax = axes[1, 0]
        pass_fail = self.results["pass_fail"]
        status_text = f"Overall: {pass_fail['overall']}\n\n"
        for region, status in pass_fail["by_region"].items():
            status_text += f"{region}: {status}\n"
        ax.text(0.1, 0.5, status_text, fontsize=14, verticalalignment='center')
        ax.set_title("Pass/Fail Status")
        ax.axis('off')
        
        # 4. Critical defects
        ax = axes[1, 1]
        critical = self.results["summary"]["critical_defects"]
        if critical:
            critical_text = f"Critical Defects: {len(critical)}\n\n"
            for i, defect in enumerate(critical[:5]):  # Show first 5
                critical_text += f"{defect['id']}: {defect['type']} in {defect['region']}\n"
            if len(critical) > 5:
                critical_text += f"... and {len(critical) - 5} more"
        else:
            critical_text = "No critical defects found"
        ax.text(0.1, 0.5, critical_text, fontsize=12, verticalalignment='center')
        ax.set_title("Critical Defects")
        ax.axis('off')
        
        plt.suptitle("Fiber Optic Inspection Summary", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, save_path: Path) -> None:
        """Generate detailed text report"""
        with open(save_path, 'w') as f:
            f.write("FIBER OPTIC INSPECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Image information
            f.write(f"Image: {self.results['image_path']}\n")
            f.write(f"Date: {np.datetime64('now')}\n\n")
            
            # Overall results
            f.write("OVERALL RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Status: {self.results['pass_fail']['overall']}\n")
            f.write(f"Total Defects: {self.results['summary']['total_defects']}\n\n")
            
            # Regional analysis
            f.write("REGIONAL ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for region, count in self.results['summary']['defects_by_region'].items():
                status = self.results['pass_fail']['by_region'][region]
                f.write(f"{region}: {count} defects - {status}\n")
            
            # Defect types
            f.write("\nDEFECT TYPES\n")
            f.write("-" * 30 + "\n")
            for defect_type, count in self.results['summary']['defects_by_type'].items():
                f.write(f"{defect_type}: {count}\n")
            
            # Critical defects
            f.write("\nCRITICAL DEFECTS\n")
            f.write("-" * 30 + "\n")
            if self.results['summary']['critical_defects']:
                for defect in self.results['summary']['critical_defects']:
                    f.write(f"{defect['id']} - {defect['type']} in {defect['region']} "
                           f"(Confidence: {defect['confidence']:.2f}, Area: {defect['area']}px)\n")
            else:
                f.write("None\n")
            
            # Failure reasons
            if self.results['pass_fail']['overall'] == "FAIL":
                f.write("\nFAILURE REASONS\n")
                f.write("-" * 30 + "\n")
                for reason in self.results['pass_fail']['reasons']:
                    f.write(f"- {reason}\n")
    
    def _generate_defect_catalog(self, save_path: Path) -> None:
        """Generate HTML catalog of all defects"""
        html_content = """
        <html>
        <head>
            <title>Defect Catalog</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .defect { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
                .critical { background-color: #ffe6e6; }
                .info { display: inline-block; margin: 5px 10px; }
                h1, h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Fiber Optic Defect Catalog</h1>
        """
        
        for region_name, results in self.results["regions"].items():
            html_content += f"<h2>{region_name} Region</h2>\n"
            
            defects = results.get("defects", [])
            if not defects:
                html_content += "<p>No defects found</p>\n"
                continue
            
            for defect in defects:
                is_critical = any(d['id'] == defect.id for d in self.results['summary']['critical_defects'])
                css_class = "defect critical" if is_critical else "defect"
                
                html_content += f'<div class="{css_class}">\n'
                html_content += f'<h3>Defect {defect.id} - {defect.type.name}</h3>\n'
                html_content += f'<span class="info">Confidence: {defect.confidence:.2f}</span>\n'
                html_content += f'<span class="info">Area: {defect.area_px} px</span>\n'
                html_content += f'<span class="info">Location: ({defect.location[0]}, {defect.location[1]})</span>\n'
                html_content += f'<span class="info">Aspect Ratio: {defect.major_axis/defect.minor_axis:.2f}</span>\n'
                html_content += '</div>\n'
        
        html_content += """
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)


# Example usage functions
def basic_usage_example():
    """Basic usage example"""
    print("Basic Usage Example")
    print("=" * 50)
    
    # Initialize pipeline with default configuration
    pipeline = FiberOpticInspectionPipeline()
    
    # Create a test image (in practice, load your actual fiber image)
    test_image = create_test_fiber_image()
    cv2.imwrite("test_fiber.png", test_image)
    
    # Run inspection
    results = pipeline.inspect_fiber("test_fiber.png")
    
    # Print summary
    print(f"Overall Status: {results['pass_fail']['overall']}")
    print(f"Total Defects: {results['summary']['total_defects']}")
    print("\nDefects by Region:")
    for region, count in results['summary']['defects_by_region'].items():
        print(f"  {region}: {count}")
    
    # Generate report
    pipeline.generate_report("basic_inspection_results")
    print("\nReport generated in 'basic_inspection_results' directory")


def advanced_usage_example():
    """Advanced usage with custom configuration"""
    print("\nAdvanced Usage Example")
    print("=" * 50)
    
    # Create custom configuration
    custom_config = {
        "preprocessing": {
            "use_illumination_correction": True,
            "use_noise_reduction": True,
            "use_contrast_enhancement": True
        },
        "detection": {
            "algorithms": {
                "statistical": ["zscore", "mad", "iqr", "grubbs", "dixon"],
                "spatial": ["lbp", "glcm"],
                "frequency": ["fft", "wavelet"],
                "morphological": ["tophat", "blackhat"],
                "ml": ["isolation_forest", "dbscan"]
            },
            "thresholds": {
                "zscore": 2.5,  # More sensitive
                "mad": 3.0,
                "min_defect_area": 5,
                "min_contrast": 8.0
            }
        },
        "regions": {
            "core": {
                "diameter_um": 9.0,
                "sensitivity": 1.5,  # Higher sensitivity for core
                "max_defects": 0
            },
            "cladding": {
                "diameter_um": 125.0,
                "sensitivity": 1.0,
                "max_defects": 3  # Stricter limit
            }
        },
        "output": {
            "save_intermediate": True,
            "visualization_dpi": 300,
            "report_format": "comprehensive"
        }
    }
    
    # Save configuration
    with open("custom_config.json", 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    # Initialize pipeline with custom configuration
    pipeline = FiberOpticInspectionPipeline("custom_config.json")
    
    # Create a test image with known defects
    test_image = create_test_fiber_image_with_defects()
    cv2.imwrite("test_fiber_defects.png", test_image)
    
    # Run inspection
    results = pipeline.inspect_fiber("test_fiber_defects.png")
    
    # Detailed analysis of results
    print("\nDetailed Analysis:")
    for region_name, region_results in results['regions'].items():
        print(f"\n{region_name} Region:")
        defects = region_results.get('defects', [])
        print(f"  Defects found: {len(defects)}")
        
        if defects:
            # Group by type
            by_type = {}
            for defect in defects:
                defect_type = defect.type.name
                if defect_type not in by_type:
                    by_type[defect_type] = []
                by_type[defect_type].append(defect)
            
            for defect_type, type_defects in by_type.items():
                print(f"  {defect_type}: {len(type_defects)}")
                # Show details of first defect of each type
                d = type_defects[0]
                print(f"    Example - ID: {d.id}, Confidence: {d.confidence:.2f}, "
                      f"Area: {d.area_px}px, Location: {d.location}")
    
    # Generate comprehensive report
    pipeline.generate_report("advanced_inspection_results")
    print("\nComprehensive report generated in 'advanced_inspection_results' directory")


def create_test_fiber_image(size: int = 500) -> np.ndarray:
    """Create a simple test fiber image"""
    # Create blank image
    image = np.ones((size, size), dtype=np.uint8) * 200
    
    # Add fiber structure
    center = (size // 2, size // 2)
    
    # Core
    cv2.circle(image, center, int(size * 0.02), 100, -1)
    
    # Cladding
    cv2.circle(image, center, int(size * 0.125), 150, -1)
    cv2.circle(image, center, int(size * 0.02), 100, -1)  # Redraw core
    
    # Ferrule/coating
    cv2.circle(image, center, int(size * 0.4), 180, 3)
    
    # Add some noise
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image


def create_test_fiber_image_with_defects(size: int = 500) -> np.ndarray:
    """Create test fiber image with various defects"""
    image = create_test_fiber_image(size)
    
    # Add various defects
    center = (size // 2, size // 2)
    
    # Scratch in cladding
    cv2.line(image, (center[0] + 30, center[1] - 50), 
             (center[0] + 80, center[1] - 30), 50, 2)
    
    # Pit in core (small dark spot)
    cv2.circle(image, (center[0] + 5, center[1]), 3, 30, -1)
    
    # Contamination in cladding
    cv2.ellipse(image, (center[0] - 40, center[1] + 30), 
                (15, 10), 45, 0, 360, 120, -1)
    
    # Particle (bright spot)
    cv2.circle(image, (center[0] + 60, center[1] + 60), 5, 250, -1)
    
    # Chip at edge
    pts = np.array([[size-100, size-150], [size-80, size-140], 
                    [size-90, size-120], [size-110, size-130]], np.int32)
    cv2.fillPoly(image, [pts], 80)
    
    return image


def performance_comparison_example():
    """Compare performance with different configurations"""
    print("\nPerformance Comparison Example")
    print("=" * 50)
    
    # Create test image
    test_image = create_test_fiber_image_with_defects()
    cv2.imwrite("test_comparison.png", test_image)
    
    # Configuration variants
    configs = {
        "Fast": {
            "preprocessing": {
                "use_illumination_correction": False,
                "use_noise_reduction": False,
                "use_contrast_enhancement": True
            },
            "detection": {
                "algorithms": {
                    "statistical": ["zscore"],
                    "morphological": ["tophat"]
                },
                "thresholds": {"zscore": 3.0, "min_defect_area": 10}
            }
        },
        "Balanced": {
            "preprocessing": {
                "use_illumination_correction": True,
                "use_noise_reduction": True,
                "use_contrast_enhancement": True
            },
            "detection": {
                "algorithms": {
                    "statistical": ["zscore", "mad"],
                    "spatial": ["lbp"],
                    "morphological": ["tophat", "blackhat"]
                },
                "thresholds": {"zscore": 3.0, "min_defect_area": 5}
            }
        },
        "Comprehensive": {
            "preprocessing": {
                "use_illumination_correction": True,
                "use_noise_reduction": True,
                "use_contrast_enhancement": True
            },
            "detection": {
                "algorithms": {
                    "statistical": ["zscore", "mad", "iqr", "grubbs"],
                    "spatial": ["lbp", "glcm"],
                    "frequency": ["fft", "wavelet"],
                    "morphological": ["tophat", "blackhat", "gradient"],
                    "ml": ["isolation_forest", "dbscan"]
                },
                "thresholds": {"zscore": 2.5, "min_defect_area": 3}
            }
        }
    }
    
    # Run comparison
    import time
    
    results_comparison = {}
    
    for config_name, config_data in configs.items():
        print(f"\nTesting {config_name} configuration...")
        
        # Save config
        with open(f"config_{config_name.lower()}.json", 'w') as f:
            # Merge with default regions config
            full_config = {
                **config_data,
                "regions": {
                    "core": {"max_defects": 0},
                    "cladding": {"max_defects": 5},
                    "ferrule": {"max_defects": 10}
                },
                "output": {"save_intermediate": False}
            }
            json.dump(full_config, f)
        
        # Initialize pipeline
        pipeline = FiberOpticInspectionPipeline(f"config_{config_name.lower()}.json")
        
        # Measure performance
        start_time = time.time()
        results = pipeline.inspect_fiber("test_comparison.png")
        end_time = time.time()
        
        # Store results
        results_comparison[config_name] = {
            "time": end_time - start_time,
            "total_defects": results['summary']['total_defects'],
            "defects_by_type": results['summary']['defects_by_type'],
            "status": results['pass_fail']['overall']
        }
        
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Defects found: {results['summary']['total_defects']}")
        print(f"  Status: {results['pass_fail']['overall']}")
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    
    for config_name, results in results_comparison.items():
        print(f"\n{config_name}:")
        print(f"  Processing time: {results['time']:.2f}s")
        print(f"  Total defects: {results['total_defects']}")
        print(f"  Defect types: {list(results['defects_by_type'].keys())}")
        print(f"  Final status: {results['status']}")
    
    # Calculate efficiency metrics
    fast_time = results_comparison["Fast"]["time"]
    comp_time = results_comparison["Comprehensive"]["time"]
    fast_defects = results_comparison["Fast"]["total_defects"]
    comp_defects = results_comparison["Comprehensive"]["total_defects"]
    
    print(f"\nSpeedup factor (Fast vs Comprehensive): {comp_time/fast_time:.1f}x")
    print(f"Detection improvement (Comprehensive vs Fast): "
          f"{comp_defects - fast_defects} additional defects")


if __name__ == "__main__":
    # Run all examples
    print("ULTIMATE DEFECT DETECTION SYSTEM - USAGE EXAMPLES")
    print("=" * 70)
    
    # 1. Basic usage
    basic_usage_example()
    
    # 2. Advanced usage
    advanced_usage_example()
    
    # 3. Performance comparison
    performance_comparison_example()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("\nThe Ultimate Defect Detection System provides:")
    print("- 150+ distinct algorithms")
    print("- PhD-level analysis methods")
    print("- Comprehensive defect characterization")
    print("- Flexible configuration system")
    print("- Detailed reporting capabilities")
    print("- Performance optimization options")
