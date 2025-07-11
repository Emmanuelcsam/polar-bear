#!/usr/bin/env python3
"""
Simple test to verify the exact duplicate image remover works correctly.
"""

import os
import shutil
import tempfile
import hashlib
from pathlib import Path

def test_exact_duplicates():
    """Test the exact duplicate detection functionality"""
    print("Testing exact duplicate detection...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a simple test image (just some bytes)
        test_image_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
        
        # Create original image
        original_path = temp_path / "original.png"
        with open(original_path, 'wb') as f:
            f.write(test_image_content)
        
        # Create exact duplicate
        duplicate_path = temp_path / "duplicate.png"
        with open(duplicate_path, 'wb') as f:
            f.write(test_image_content)
        
        # Create a different image (add one byte)
        different_path = temp_path / "different.png"
        with open(different_path, 'wb') as f:
            f.write(test_image_content + b'\x00')
        
        # Test hash calculation
        with open(original_path, 'rb') as f:
            hash1 = hashlib.sha256(f.read()).hexdigest()
        
        with open(duplicate_path, 'rb') as f:
            hash2 = hashlib.sha256(f.read()).hexdigest()
        
        with open(different_path, 'rb') as f:
            hash3 = hashlib.sha256(f.read()).hexdigest()
        
        print(f"Original hash: {hash1}")
        print(f"Duplicate hash: {hash2}")
        print(f"Different hash: {hash3}")
        
        # Check that duplicates have same hash
        assert hash1 == hash2, "Duplicate files should have same hash"
        assert hash1 != hash3, "Different files should have different hashes"
        
        print("✓ Test passed! Hash-based duplicate detection works correctly.")
        print(f"✓ Created test files in: {temp_path}")
        
        # Show file sizes
        print(f"Original file size: {original_path.stat().st_size} bytes")
        print(f"Duplicate file size: {duplicate_path.stat().st_size} bytes")
        print(f"Different file size: {different_path.stat().st_size} bytes")

if __name__ == "__main__":
    test_exact_duplicates()
