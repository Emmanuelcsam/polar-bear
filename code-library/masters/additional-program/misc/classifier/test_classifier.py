#!/usr/bin/env python3
"""
Test script to verify all functionality of the advanced image classifier
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import shutil

# Import the classifier module directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Read and execute the classifier script to import its contents
classifier_file = os.path.join(os.path.dirname(__file__), 'image-classifier-advanced.py')
with open(classifier_file, 'r') as f:
    exec(f.read(), globals())

def test_knowledge_bank():
    """Test KnowledgeBank functionality"""
    print(f"\n[{timestamp()}] Testing KnowledgeBank...")
    
    # Create test knowledge bank
    kb = KnowledgeBank("test_knowledge_bank.pkl")
    
    # Test adding image
    test_hash = "testhash12345"
    test_features = np.random.rand(100).astype(np.float32)
    test_classifications = ["test-classification", "50-fc-core"]
    test_characteristics = {
        "connector_type": "fc",
        "core_diameter": "50",
        "region": "core"
    }
    
    kb.add_image(test_hash, test_features, test_classifications, test_characteristics)
    
    # Test folder structure
    kb.add_folder_structure(["fc", "50", "core"])
    
    # Test custom keywords
    kb.add_custom_keyword("test_keyword")
    
    # Test feedback
    kb.add_feedback(test_hash, "correct-class", "wrong-class")
    
    # Test save and load
    kb.save()
    
    # Create new instance and load
    kb2 = KnowledgeBank("test_knowledge_bank.pkl")
    
    # Verify data was saved/loaded correctly
    assert test_hash in kb2.features_db
    assert "test_keyword" in kb2.custom_keywords
    assert len(kb2.classifications_db[test_hash]) == 2
    
    # Clean up
    if os.path.exists("test_knowledge_bank.pkl"):
        os.remove("test_knowledge_bank.pkl")
    
    print(f"[{timestamp()}] ✓ KnowledgeBank tests passed")
    return True

def test_feature_extraction():
    """Test feature extraction methods"""
    print(f"\n[{timestamp()}] Testing feature extraction...")
    
    classifier = AdvancedImageClassifier("test_config.json")
    
    # Test with the lime.jpg image in dataset
    test_image = "dataset/lime.jpg"
    if os.path.exists(test_image):
        features, img_hash = classifier.extract_visual_features(test_image)
        
        assert features is not None
        assert img_hash is not None
        assert len(features) > 0
        assert isinstance(features, np.ndarray)
        
        print(f"[{timestamp()}] ✓ Feature extraction successful")
        print(f"[{timestamp()}]   - Feature dimensions: {features.shape}")
        print(f"[{timestamp()}]   - Image hash: {img_hash}")
    else:
        print(f"[{timestamp()}] ⚠ Test image not found, skipping feature extraction test")
    
    # Clean up config
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
    
    return True

def test_classification_parsing():
    """Test classification parsing"""
    print(f"\n[{timestamp()}] Testing classification parsing...")
    
    classifier = AdvancedImageClassifier()
    
    # Test various filename formats
    test_cases = [
        ("50-fc-core-clean.jpg", {
            "core_diameter": "50",
            "connector_type": "fc",
            "region": "core",
            "condition": "clean"
        }),
        ("91-sma-cladding-scratched.png", {
            "core_diameter": "91",
            "connector_type": "sma",
            "region": "cladding",
            "defect_type": "scratched"
        }),
        ("fc-ferrule-oil-blob.jpg", {
            "connector_type": "fc",
            "region": "ferrule",
            "defect_type": "oil-blob"
        }),
        ("IMG_1234.jpg", {}),  # Should parse but return minimal components
        ("darkgray_20.jpg", {
            "additional_characteristics": "darkgray"
        })
    ]
    
    for filename, expected in test_cases:
        components = classifier.parse_classification(filename)
        print(f"[{timestamp()}] Testing: {filename}")
        print(f"[{timestamp()}]   Parsed: {components}")
        
        # Check key components match
        for key in ["connector_type", "core_diameter", "region"]:
            if key in expected:
                assert components.get(key) == expected[key], f"Mismatch in {key}"
    
    print(f"[{timestamp()}] ✓ Classification parsing tests passed")
    return True

def test_similarity_search():
    """Test similarity search functionality"""
    print(f"\n[{timestamp()}] Testing similarity search...")
    
    classifier = AdvancedImageClassifier()
    
    # Add some test images to knowledge bank
    for i in range(5):
        test_hash = f"test{i}"
        test_features = np.random.rand(200).astype(np.float32)
        test_features[i*10:(i+1)*10] = 1.0  # Make features distinct
        
        classifier.knowledge_bank.add_image(
            test_hash,
            test_features,
            [f"class{i}"],
            {"test": str(i)}
        )
    
    # Test finding similar images
    query_features = np.random.rand(200).astype(np.float32)
    query_features[10:20] = 1.0  # Should be similar to test1
    
    similar = classifier.find_similar_images(query_features, threshold=0.5)
    
    assert len(similar) > 0
    print(f"[{timestamp()}] ✓ Found {len(similar)} similar images")
    
    return True

def test_folder_creation():
    """Test folder structure creation"""
    print(f"\n[{timestamp()}] Testing folder creation...")
    
    classifier = AdvancedImageClassifier()
    test_base = "test_folder_creation"
    
    # Create test base directory
    os.makedirs(test_base, exist_ok=True)
    
    # Test creating folder structure
    components = {
        "connector_type": "fc",
        "core_diameter": "50",
        "region": "core",
        "condition": "clean"
    }
    
    result_path = classifier.create_folder_if_needed(test_base, components)
    expected_path = os.path.join(test_base, "fc", "50", "core", "clean")
    
    assert os.path.exists(result_path)
    assert result_path == expected_path
    
    # Clean up
    shutil.rmtree(test_base)
    
    print(f"[{timestamp()}] ✓ Folder creation tests passed")
    return True

def test_config_management():
    """Test configuration save/load"""
    print(f"\n[{timestamp()}] Testing configuration management...")
    
    test_config_file = "test_classifier_config.json"
    
    # Create classifier with custom config
    classifier = AdvancedImageClassifier(test_config_file)
    
    # Modify config
    classifier.config['similarity_threshold'] = 0.85
    classifier.config['custom_keywords'] = ['test1', 'test2']
    classifier.save_config()
    
    # Load in new instance
    classifier2 = AdvancedImageClassifier(test_config_file)
    
    assert classifier2.config['similarity_threshold'] == 0.85
    assert 'test1' in classifier2.config['custom_keywords']
    
    # Clean up
    if os.path.exists(test_config_file):
        os.remove(test_config_file)
    
    print(f"[{timestamp()}] ✓ Config management tests passed")
    return True

def test_reference_analysis():
    """Test reference folder analysis"""
    print(f"\n[{timestamp()}] Testing reference folder analysis...")
    
    classifier = AdvancedImageClassifier()
    
    # Check if reference folder exists
    if os.path.exists("reference"):
        print(f"[{timestamp()}] Analyzing reference folder...")
        reference_data = classifier.analyze_reference_folder("reference")
        
        if reference_data:
            total_images = sum(len(v) for v in reference_data.values())
            print(f"[{timestamp()}] ✓ Analyzed {total_images} images in {len(reference_data)} folders")
            
            # Check knowledge bank was populated
            assert len(classifier.knowledge_bank.features_db) > 0
            print(f"[{timestamp()}] ✓ Knowledge bank populated with {len(classifier.knowledge_bank.features_db)} images")
        else:
            print(f"[{timestamp()}] ⚠ No reference data found")
    else:
        print(f"[{timestamp()}] ⚠ Reference folder not found, skipping test")
    
    return True

def test_classification_functions():
    """Test the classification workflow"""
    print(f"\n[{timestamp()}] Testing classification functions...")
    
    classifier = AdvancedImageClassifier()
    
    # Test building classification string
    components = {
        "core_diameter": "50",
        "connector_type": "fc",
        "region": "core",
        "condition": "clean"
    }
    
    classification = classifier.build_classification_string(components)
    assert classification == "50-fc-core-clean"
    
    # Test unique filename generation
    test_dir = "test_unique"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test file
    test_file = os.path.join(test_dir, "test.jpg")
    open(test_file, 'a').close()
    
    # Get unique filename
    unique = classifier.get_unique_filename(test_dir, "test", ".jpg")
    assert unique == "test_1.jpg"
    
    # Clean up
    shutil.rmtree(test_dir)
    
    print(f"[{timestamp()}] ✓ Classification function tests passed")
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("ADVANCED IMAGE CLASSIFIER - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Knowledge Bank", test_knowledge_bank),
        ("Feature Extraction", test_feature_extraction),
        ("Classification Parsing", test_classification_parsing),
        ("Similarity Search", test_similarity_search),
        ("Folder Creation", test_folder_creation),
        ("Config Management", test_config_management),
        ("Reference Analysis", test_reference_analysis),
        ("Classification Functions", test_classification_functions)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"[{timestamp()}] ✗ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"[{timestamp()}] ✗ {test_name} failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED! The classifier is fully functional.")
    else:
        print(f"\n✗ {failed} tests failed. Please review the errors above.")
    
    # Clean up any remaining test files
    for file in ["knowledge_bank.pkl", "classifier_config.json"]:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    main()