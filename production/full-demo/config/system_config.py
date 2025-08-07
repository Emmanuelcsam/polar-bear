"""
System configuration for blob detection compatibility.
"""

class SystemConfig:
    """System configuration for defect detection parameters."""

    # Blob detection parameters
    MIN_BLOB_AREA = 50
    MAX_BLOB_AREA = 5000
    MIN_BLOB_CIRCULARITY = 0.3

    # Additional parameters for compatibility
    SCRATCH_KERNEL_SIZE = (5, 15)
    SCRATCH_BINARY_THRESHOLD = 30
    SSIM_THRESHOLD = 0.95
    MIN_DEFECT_SIZE = 10
    MAX_DEFECT_SIZE = 5000
