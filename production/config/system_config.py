"""
System configuration module containing all tunable parameters.
This makes the system easier to adjust for different products or environments.
"""


class SystemConfig:
    """
    Centralized configuration for all tunable parameters.
    This makes the system easier to adjust for different products or
    environments.
    """
    # --- General ---
    REFERENCE_IMAGE_PATH = "good.bmp"
    LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # --- Preprocessing ---
    GAUSSIAN_BLUR_KERNEL = (7, 7)

    # --- SSIM Difference Thresholding ---
    # Lower this value if minor differences should be ignored.
    SSIM_THRESHOLD = 0.85

    # --- Scratch Detection (Morphological) ---
    # Kernel size should be larger than the scratch width.
    SCRATCH_KERNEL_SIZE = (25, 25)
    # Threshold for the combined Top/Black-hat result.
    SCRATCH_BINARY_THRESHOLD = 50

    # --- Blob Detection (Contours) ---
    MIN_BLOB_AREA = 200  # Minimum pixel area to be considered a blob.
    MAX_BLOB_AREA = 5000  # Maximum pixel area.
    # Circularity: 1.0 is a perfect circle. Lower values are more irregular.
    # We use this to distinguish blobs from scratches.
    MIN_BLOB_CIRCULARITY = 0.3

    # --- Circle Detection (Hough Transform) ---
    HOUGH_DP = 1.2  # Inverse ratio of accumulator resolution.
    HOUGH_MIN_DIST = 100  # Minimum distance between detected circle centers.
    HOUGH_PARAM1 = 100  # Upper threshold for internal Canny edge detector.
    HOUGH_PARAM2 = 60  # Accumulator threshold for center detection.
    HOUGH_MIN_RADIUS = 10  # Minimum circle radius in pixels.
    HOUGH_MAX_RADIUS = 100  # Maximum circle radius in pixels. 