#!/usr/bin/env python3

import logging
import os
from pathlib import Path
import time

# Import all modular components
from config import OmniConfig, NumpyEncoder
from utils import load_image, get_timestamp
from feature_extraction import extract_ultra_comprehensive_features
from comparison import compute_exhaustive_comparison, compute_image_structural_comparison
from defect_detection import (
    detect_specific_defects, compute_local_anomaly_map, find_anomaly_regions
)
from reference_model import (
    load_knowledge_base, build_minimal_reference, save_knowledge_base,
    build_comprehensive_reference_model
)
from anomaly_detection import (
    detect_anomalies_comprehensive, analyze_end_face, convert_to_pipeline_format,
    confidence_to_severity
)
from visualization import visualize_comprehensive_results, save_simple_anomaly_image
from report_generation import generate_detailed_report
from defect_mask import create_defect_mask


class OmniFiberAnalyzer:
    """The ultimate fiber optic anomaly detection system - modular version."""
    
    def __init__(self, config: OmniConfig):
        # Store configuration object containing all analysis parameters
        self.config = config
        # Set knowledge base path, defaulting to "fiber_anomaly_kb.json" if not specified
        self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
        # Initialize empty reference model structure for storing learned patterns
        self.reference_model = {
            'features': [],              # List of feature dictionaries from reference images
            'statistical_model': None,   # Statistical parameters (mean, std, covariance)
            'archetype_image': None,     # Median image representing typical fiber
            'feature_names': [],         # List of feature names in consistent order
            'comparison_results': {},    # Cached comparison results
            'learned_thresholds': {},    # Learned anomaly detection thresholds
            'timestamp': None           # When model was created/updated
        }
        # Initialize metadata storage for current image being processed
        self.current_metadata = None
        # Create logger instance for this class
        self.logger = logging.getLogger(__name__)
        # Attempt to load existing knowledge base from disk
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load previously saved knowledge base from JSON."""
        loaded_model = load_knowledge_base(self.knowledge_base_path)
        if loaded_model:
            self.reference_model = loaded_model
    
    def save_knowledge_base(self):
        """Save current knowledge base to JSON."""
        return save_knowledge_base(self.reference_model, self.knowledge_base_path)
    
    def build_reference_model(self, ref_dir):
        """Build a comprehensive reference model from a directory of images."""
        self.reference_model = build_comprehensive_reference_model(ref_dir, self.config)
        if self.reference_model:
            self.save_knowledge_base()
            return True
        return False
    
    def analyze_end_face(self, image_path: str, output_dir: str):
        """Main analysis method - compatible with pipeline expectations"""
        return analyze_end_face(image_path, output_dir, self.config, self.reference_model)
    
    def detect_anomalies_comprehensive(self, test_path):
        """Perform exhaustive anomaly detection on a test image."""
        return detect_anomalies_comprehensive(test_path, self.reference_model, self.config)


def main():
    """Main execution function for standalone testing."""
    # Print banner
    print("\n" + "="*80)
    print("OMNIRIBER ANALYZER - MODULAR DETECTION SYSTEM (v1.5)".center(80))
    print("="*80)
    print("\nThis modular system provides the same functionality as the original")
    print("detection.py but split into focused, reusable components.\n")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s'
    )
    
    # Create default configuration
    config = OmniConfig()
    
    # Initialize analyzer with configuration
    analyzer = OmniFiberAnalyzer(config)
    
    # Interactive testing loop
    while True:
        print("\nAvailable operations:")
        print("1. Analyze single image")
        print("2. Build reference model from directory")
        print("3. Test individual modules")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Analyze single image
            test_path = input("\nEnter path to test image: ").strip()
            test_path = test_path.strip('"\'')
            
            if not os.path.isfile(test_path):
                print(f"✗ File not found: {test_path}")
                continue
            
            output_dir = f"detection_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
            
            print(f"\nAnalyzing {test_path}...")
            result = analyzer.analyze_end_face(test_path, output_dir)
            
            if result:
                print(f"\n✓ Analysis completed successfully!")
                print(f"Results saved to: {output_dir}/")
            else:
                print(f"\n✗ Analysis failed for {test_path}")
        
        elif choice == '2':
            # Build reference model
            ref_dir = input("\nEnter path to reference images directory: ").strip()
            ref_dir = ref_dir.strip('"\'')
            
            if not os.path.isdir(ref_dir):
                print(f"✗ Directory not found: {ref_dir}")
                continue
            
            print(f"\nBuilding reference model from {ref_dir}...")
            success = analyzer.build_reference_model(ref_dir)
            
            if success:
                print("✓ Reference model built successfully!")
            else:
                print("✗ Failed to build reference model")
        
        elif choice == '3':
            # Test individual modules
            print("\nModule testing options:")
            print("1. Test feature extraction")
            print("2. Test image loading")
            print("3. Test statistical functions")
            print("4. Back to main menu")
            
            module_choice = input("\nEnter module test choice (1-4): ").strip()
            
            if module_choice == '1':
                # Test feature extraction
                test_path = input("\nEnter path to test image: ").strip()
                test_path = test_path.strip('"\'')
                
                if os.path.isfile(test_path):
                    image = load_image(test_path)
                    if image is not None:
                        print("Testing feature extraction...")
                        features, feature_names = extract_ultra_comprehensive_features(image)
                        print(f"✓ Extracted {len(features)} features")
                        print(f"Feature names: {len(feature_names)}")
                    else:
                        print("✗ Failed to load image")
                else:
                    print(f"✗ File not found: {test_path}")
            
            elif module_choice == '2':
                # Test image loading
                test_path = input("\nEnter path to test image: ").strip()
                test_path = test_path.strip('"\'')
                
                if os.path.isfile(test_path):
                    image = load_image(test_path)
                    if image is not None:
                        print(f"✓ Image loaded successfully!")
                        print(f"Shape: {image.shape}")
                        print(f"Data type: {image.dtype}")
                    else:
                        print("✗ Failed to load image")
                else:
                    print(f"✗ File not found: {test_path}")
            
            elif module_choice == '3':
                # Test statistical functions
                import numpy as np
                from statistical_functions import compute_skewness, compute_kurtosis, compute_entropy
                
                # Generate test data
                test_data = np.random.normal(0, 1, 1000)
                print("Testing statistical functions...")
                print(f"Skewness: {compute_skewness(test_data):.4f}")
                print(f"Kurtosis: {compute_kurtosis(test_data):.4f}")
                print(f"Entropy: {compute_entropy(test_data):.4f}")
                print("✓ Statistical functions working correctly")
        
        elif choice == '4':
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")
    
    # Exit message
    print("\nThank you for using the Modular OmniFiber Analyzer!")


# Entry point for script execution
if __name__ == "__main__":
    main() 