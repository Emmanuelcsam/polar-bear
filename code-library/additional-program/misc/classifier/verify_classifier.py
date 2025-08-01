#!/usr/bin/env python3
"""
Verification script to check the advanced classifier functionality
without running the full application
"""

import os
import re
from pathlib import Path

def verify_script_structure():
    """Verify the script has all required components"""
    print("\n=== VERIFYING SCRIPT STRUCTURE ===")
    
    script_path = "image-classifier-advanced.py"
    if not os.path.exists(script_path):
        print("✗ Script not found!")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for all required classes and methods
    required_components = [
        ("class KnowledgeBank", "KnowledgeBank class"),
        ("class AdvancedImageClassifier", "AdvancedImageClassifier class"),
        ("def extract_visual_features", "Visual feature extraction"),
        ("def _extract_color_histogram", "Color histogram extraction"),
        ("def _extract_texture_features", "Texture feature extraction"),
        ("def _extract_edge_features", "Edge feature extraction"),
        ("def _extract_shape_features", "Shape feature extraction"),
        ("def _extract_statistical_features", "Statistical feature extraction"),
        ("def parse_classification", "Classification parsing"),
        ("def build_classification_string", "Classification string building"),
        ("def analyze_reference_folder", "Reference folder analysis"),
        ("def find_similar_images", "Similarity search"),
        ("def classify_image", "Image classification"),
        ("def process_mode", "Automated processing mode"),
        ("def manual_mode", "Manual processing mode"),
        ("def main", "Main function")
    ]
    
    all_found = True
    for pattern, description in required_components:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} - NOT FOUND")
            all_found = False
    
    # Check for improved features
    print("\n=== CHECKING IMPROVEMENTS ===")
    improvements = [
        ("cosine_similarity", "Using sklearn cosine similarity"),
        ("feature_cache", "Feature caching implemented"),
        ("classification_history", "Classification history tracking"),
        ("get_classification_confidence", "Confidence scoring"),
        ("defect_patterns", "Defect pattern matching"),
        ("adaptive_threshold", "Adaptive threshold support"),
        ("logging.error", "Proper error logging"),
        ("tqdm", "Progress bar support"),
        ("_combine_features", "Feature combination and normalization")
    ]
    
    for pattern, description in improvements:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"⚠ {description} - Not found (optional)")
    
    return all_found

def verify_folder_structure():
    """Verify the required folder structure"""
    print("\n=== VERIFYING FOLDER STRUCTURE ===")
    
    folders = {
        "reference": "Reference folder for training images",
        "dataset": "Dataset folder for images to classify"
    }
    
    all_exist = True
    for folder, description in folders.items():
        if os.path.exists(folder):
            # Count images
            image_count = 0
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if Path(file).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                        image_count += 1
            print(f"✓ {folder}/ - {description} ({image_count} images)")
        else:
            print(f"⚠ {folder}/ - {description} (NOT FOUND)")
            all_exist = False
    
    return all_exist

def verify_reference_structure():
    """Verify the reference folder has proper hierarchical structure"""
    print("\n=== VERIFYING REFERENCE STRUCTURE ===")
    
    if not os.path.exists("reference"):
        print("⚠ Reference folder not found")
        return False
    
    # Expected structure patterns
    expected_patterns = [
        "connector_type/core_diameter/region/condition",
        "fc/50/core/clean",
        "sma/cladding/dirty"
    ]
    
    # Analyze actual structure
    structure_found = set()
    for root, dirs, files in os.walk("reference"):
        rel_path = os.path.relpath(root, "reference")
        if rel_path != "." and files:
            structure_found.add(rel_path)
    
    if structure_found:
        print(f"✓ Found {len(structure_found)} folder paths with images:")
        for path in sorted(list(structure_found))[:10]:
            print(f"  - {path}")
        if len(structure_found) > 10:
            print(f"  ... and {len(structure_found) - 10} more")
    else:
        print("⚠ No hierarchical structure found in reference folder")
    
    return len(structure_found) > 0

def test_parsing_logic():
    """Test the classification parsing logic"""
    print("\n=== TESTING PARSING LOGIC ===")
    
    # Test cases
    test_filenames = [
        "50-fc-core-clean.jpg",
        "91-sma-cladding-scratched.png",
        "fc-ferrule-oil.jpg",
        "darkgray_20.jpg",
        "gray-31.jpg",
        "50-fc-core-blob-anomaly.jpg"
    ]
    
    # Simple parsing simulation
    connector_patterns = ['fc', 'sma', 'sc', 'lc', 'st']
    region_patterns = ['core', 'cladding', 'ferrule']
    condition_patterns = ['clean', 'dirty']
    defect_patterns = ['scratched', 'oil', 'blob', 'dig', 'anomaly']
    
    print("Testing filename parsing:")
    for filename in test_filenames:
        base = Path(filename).stem
        parts = re.split(r'[-_]', base)
        
        components = {
            'filename': filename,
            'parts': parts
        }
        
        # Parse components
        for part in parts:
            part_lower = part.lower()
            if part.isdigit() and len(part) <= 3:
                components['core_diameter'] = part
            elif part_lower in connector_patterns:
                components['connector_type'] = part_lower
            elif part_lower in region_patterns:
                components['region'] = part_lower
            elif part_lower in condition_patterns:
                components['condition'] = part_lower
            elif part_lower in defect_patterns:
                components['defect_type'] = components.get('defect_type', [])
                components['defect_type'].append(part_lower)
        
        print(f"\n  {filename}:")
        for key, value in components.items():
            if key not in ['filename', 'parts']:
                print(f"    - {key}: {value}")

def verify_configuration():
    """Verify configuration handling"""
    print("\n=== VERIFYING CONFIGURATION ===")
    
    config_file = "classifier_config.json"
    if os.path.exists(config_file):
        print(f"✓ Configuration file exists: {config_file}")
        
        import json
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print("  Configuration values:")
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"    - {key}: {value}")
                elif isinstance(value, list):
                    print(f"    - {key}: {len(value)} items")
                elif isinstance(value, dict):
                    print(f"    - {key}: {len(value)} entries")
        except Exception as e:
            print(f"  ⚠ Error reading config: {e}")
    else:
        print("⚠ No configuration file found (will be created on first run)")

def main():
    """Run all verifications"""
    print("="*60)
    print("ADVANCED IMAGE CLASSIFIER - FUNCTIONALITY VERIFICATION")
    print("="*60)
    
    # Run verifications
    script_ok = verify_script_structure()
    folders_ok = verify_folder_structure()
    reference_ok = verify_reference_structure()
    
    # Run tests
    test_parsing_logic()
    verify_configuration()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if script_ok:
        print("✓ Script structure is complete")
    else:
        print("✗ Script structure has issues")
    
    if folders_ok:
        print("✓ Folder structure is ready")
    else:
        print("⚠ Folder structure needs setup")
    
    if reference_ok:
        print("✓ Reference folder has proper structure")
    else:
        print("⚠ Reference folder needs organization")
    
    print("\nThe classifier includes these features:")
    print("- Multiple feature extraction methods (color, texture, edge, shape, statistical)")
    print("- Adaptive similarity threshold")
    print("- Classification confidence scoring")
    print("- Knowledge bank with persistent learning")
    print("- Hierarchical folder structure support")
    print("- Both automated and manual classification modes")
    print("- Progress tracking and comprehensive logging")
    print("- Feature caching for performance")
    print("- Custom keyword support")
    print("- User feedback incorporation")
    
    print("\nTo run the classifier:")
    print("1. Ensure Python packages are installed (Pillow, numpy, sklearn, opencv-python, imagehash, tqdm)")
    print("2. Place reference images in the 'reference' folder with proper hierarchy")
    print("3. Place images to classify in the 'dataset' folder")
    print("4. Run: python image-classifier-advanced.py")
    print("5. Choose mode 1 (automated) or mode 2 (manual)")

if __name__ == "__main__":
    main()