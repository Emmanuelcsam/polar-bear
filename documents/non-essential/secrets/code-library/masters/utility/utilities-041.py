#!/usr/bin/env python3
"""
Numpy JSON Encoder Module
Standalone utility for converting numpy data types to JSON-serializable formats.
Extracted from legacy codebase and optimized for reuse.
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy data types.
    
    Converts:
    - numpy integers to Python int
    - numpy floats to Python float  
    - numpy arrays to Python lists
    
    Usage:
        json.dump(data, file, cls=NumpyEncoder)
        json.dumps(data, cls=NumpyEncoder)
    """
    
    def default(self, obj):
        # Convert numpy integer types to Python int for JSON compatibility
        if isinstance(obj, np.integer):
            return int(obj)
        
        # Convert numpy float types to Python float for JSON compatibility
        if isinstance(obj, np.floating):
            return float(obj)
        
        # Convert numpy arrays to Python lists for JSON compatibility
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Fall back to default JSON encoder for other types
        return super(NumpyEncoder, self).default(obj)


def save_numpy_data(data, filepath, indent=2):
    """
    Convenience function to save data containing numpy types to JSON.
    
    Args:
        data: Dictionary or object containing numpy data
        filepath: Path to save JSON file
        indent: JSON indentation level (default: 2)
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


def dumps_numpy(obj, **kwargs):
    """
    Convenience function to serialize numpy-containing data to JSON string.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dumps
    
    Returns:
        JSON string representation
    """
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)


def main():
    """Test the NumpyEncoder functionality."""
    print("Testing NumpyEncoder...")
    
    # Test data with various numpy types
    test_data = {
        'int_array': np.array([1, 2, 3, 4, 5]),
        'float_array': np.array([1.1, 2.2, 3.3]),
        'matrix': np.array([[1, 2], [3, 4]]),
        'scalar_int': np.int64(42),
        'scalar_float': np.float32(3.14),
        'regular_data': {'name': 'test', 'value': 100}
    }
    
    # Test serialization
    json_string = dumps_numpy(test_data, indent=2)
    print("Serialized data:")
    print(json_string)
    
    # Test deserialization
    reconstructed = json.loads(json_string)
    print("\nReconstructed data:")
    print(reconstructed)
    
    print("\nNumpyEncoder test completed successfully!")


if __name__ == "__main__":
    main()
