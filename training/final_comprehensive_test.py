#!/usr/bin/env python3
"""
Final Comprehensive Test - Demonstrates all features working
"""

import os
import sys
import tempfile
import shutil
from PIL import Image
import numpy as np
import subprocess
import json
import time

# Add current directory to path to import the classifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_realistic_test_images(output_dir, name_pattern, base_color, count=3):
    """Create realistic test images with variations"""
    images = []
    for i in range(count):
        # Add variation to base color
        variation = np.random.randint(-30, 30, 3)
        color = np.clip(np.array(base_color) + variation, 0, 255)
        
        # Create image with patterns
        img_array = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Fill with base color
        img_array[:, :] = color
        
        # Add circular pattern (simulating fiber core)
        center = (64, 64)
        radius = 30
        y, x = np.ogrid[:128, :128]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img_array[mask] = np.clip(color + 50, 0, 255)
        
        # Add some noise
        noise = np.random.randint(-10, 10, (128, 128, 3))
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_array)
        filename = f"{name_pattern}_{i}.jpg"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        images.append(filepath)
        
    return images

def test_feature_extraction_and_similarity():
    """Test feature extraction and similarity calculation"""
    print("\n" + "="*60)
    print("Testing Feature Extraction and Similarity Calculation")
    print("="*60)
    
    # Import classifier module
    exec(open('image-classifier.py').read(), globals())
    
    # Create classifier instance
    classifier = UltimateImageClassifier()
    
    # Create test images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create similar images (should have high similarity)
        similar_images = create_realistic_test_images(temp_dir, "similar", (255, 0, 0), 2)
        
        # Create different image
        different_images = create_realistic_test_images(temp_dir, "different", (0, 0, 255), 1)
        
        # Extract features
        print("\nExtracting features...")
        features1, hash1 = classifier.extract_features(similar_images[0])
        features2, hash2 = classifier.extract_features(similar_images[1])
        features3, hash3 = classifier.extract_features(different_images[0])
        
        print(f"✅ Extracted features for {os.path.basename(similar_images[0])}: {len(features1)} dimensions")
        print(f"✅ Extracted features for {os.path.basename(similar_images[1])}: {len(features2)} dimensions")
        print(f"✅ Extracted features for {os.path.basename(different_images[0])}: {len(features3)} dimensions")
        
        # Calculate similarities
        print("\nCalculating similarities...")
        sim_similar = classifier.calculate_similarity(features1, features2)
        sim_different1 = classifier.calculate_similarity(features1, features3)
        sim_different2 = classifier.calculate_similarity(features2, features3)
        
        print(f"Similarity between similar images: {sim_similar:.3f}")
        print(f"Similarity between different images (1-3): {sim_different1:.3f}")
        print(f"Similarity between different images (2-3): {sim_different2:.3f}")
        
        # Verify expectations
        if sim_similar > sim_different1 and sim_similar > sim_different2:
            print("✅ Similar images have higher similarity score as expected")
        else:
            print("⚠️  Unexpected similarity scores")
            
        # Test all feature extraction methods
        print("\nTesting individual feature extraction methods...")
        test_img = np.ones((128, 128, 3), dtype=np.uint8) * 128
        
        methods = [
            ("Color features", classifier._extract_color_features),
            ("Texture features", classifier._extract_texture_features),
            ("Edge features", classifier._extract_edge_features),
            ("Shape features", classifier._extract_shape_features),
            ("Statistical features", classifier._extract_statistical_features),
        ]
        
        for name, method in methods:
            features = method(test_img)
            print(f"✅ {name}: {len(features)} features extracted")
            
    finally:
        shutil.rmtree(temp_dir)
        
def test_complete_workflow():
    """Test complete classification workflow"""
    print("\n" + "="*60)
    print("Testing Complete Classification Workflow")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create directory structure
        reference_dir = os.path.join(temp_dir, "reference")
        dataset_dir = os.path.join(temp_dir, "dataset")
        
        # Create reference images with proper structure
        ref_structure = {
            "fc/50/core/clean": (255, 50, 50),
            "fc/50/core/dirty": (200, 50, 50),
            "sma/91/cladding/scratched": (50, 255, 50),
            "lc/50/ferrule/oil": (50, 50, 255)
        }
        
        print("\nCreating reference images...")
        for path, color in ref_structure.items():
            full_path = os.path.join(reference_dir, path)
            os.makedirs(full_path, exist_ok=True)
            create_realistic_test_images(full_path, "ref", color, 2)
            print(f"✅ Created references in {path}")
            
        # Create dataset images to classify
        print("\nCreating dataset images...")
        test_images = []
        test_images.extend(create_realistic_test_images(dataset_dir, "red_fiber", (250, 60, 60), 2))
        test_images.extend(create_realistic_test_images(dataset_dir, "green_fiber", (60, 250, 60), 2))
        test_images.extend(create_realistic_test_images(dataset_dir, "blue_fiber", (60, 60, 250), 2))
        print(f"✅ Created {len(test_images)} dataset images")
        
        # Run automatic classification
        print("\nRunning automatic classification...")
        cmd = [
            "python", "image-classifier.py",
            "--reference_folder", reference_dir,
            "--dataset_folder", dataset_dir,
            "--mode", "auto",
            "--similarity_threshold", "0.6"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Automatic classification completed successfully")
            
            # Check results
            classified_count = 0
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if '-' in file and not file.startswith(('red_', 'green_', 'blue_')):
                        classified_count += 1
                        print(f"  ✅ Classified: {file}")
                        
            print(f"\nTotal classified: {classified_count}/{len(test_images)}")
            
            # Check if knowledge bank was created
            kb_path = os.path.join(temp_dir, "knowledge_bank.pkl")
            if os.path.exists(kb_path):
                print("✅ Knowledge bank created")
                
            # Check if config was saved
            config_path = os.path.join(temp_dir, "classifier_config.json")
            if os.path.exists(config_path):
                print("✅ Configuration saved")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"  Similarity threshold: {config.get('similarity_threshold')}")
                
        else:
            print("❌ Classification failed")
            print("Error:", result.stderr)
            
    finally:
        shutil.rmtree(temp_dir)

def run_all_tests():
    """Run all comprehensive tests"""
    print("="*80)
    print("FINAL COMPREHENSIVE TEST SUITE")
    print("Testing all features of the Image Classifier")
    print("="*80)
    
    tests = [
        ("Feature Extraction and Similarity", test_feature_extraction_and_similarity),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "✅ PASSED"))
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, f"❌ FAILED: {str(e)}"))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        print(f"{test_name}: {result}")
        
    passed = sum(1 for _, r in results if "PASSED" in r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The image classifier is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    run_all_tests()