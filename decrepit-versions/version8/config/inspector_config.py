from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from data_structures.zone_definition import ZoneDefinition

@dataclass
class InspectorConfig:
    """Class to hold all configuration parameters for the fiber inspection process."""
    OUTPUT_DIR_NAME: str = "fiber_inspection_output" # Name of the directory to save results.
    MIN_DEFECT_AREA_PX: int = 10  # Minimum area in pixels for a contour to be considered a defect.
    PERFORM_CALIBRATION: bool = False # Whether to attempt system calibration with a target.
    CALIBRATION_IMAGE_PATH: Optional[str] = None # Path to the calibration target image.
    CALIBRATION_DOT_SPACING_UM: float = 10.0 # Known spacing of dots on calibration target in microns.
    CALIBRATION_FILE_JSON: str = "calibration_data.json" # File to save/load calibration data.
    DEFAULT_ZONES: List[ZoneDefinition] = field(default_factory=lambda: [ # Default zone definitions.
        ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=0.4, # e.g. 50um core for 125um cladding
                       color_bgr=(255, 0, 0), max_defect_size_um=5.0), # Blue for core
        ZoneDefinition(name="cladding", r_min_factor_or_um=0.4, r_max_factor_or_um=1.0,
                       color_bgr=(0, 255, 0), max_defect_size_um=10.0), # Green for cladding
        ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=1.0, r_max_factor_or_um=2.0, # e.g. up to 250um for ferrule
                       color_bgr=(0, 0, 255), max_defect_size_um=25.0), # Red for ferrule
        ZoneDefinition(name="adhesive", r_min_factor_or_um=2.0, r_max_factor_or_um=2.2, # Area just beyond ferrule
                       color_bgr=(0, 255, 255), max_defect_size_um=50.0, defects_allowed=False) # Yellow for adhesive
    ])
    GAUSSIAN_BLUR_KERNEL_SIZE: Tuple[int, int] = (7, 7)  # Kernel size for Gaussian blur.
    GAUSSIAN_BLUR_SIGMA: int = 2  # Sigma for Gaussian blur.
    BILATERAL_FILTER_D: int = 9 # Diameter of each pixel neighborhood for bilateral filter.
    BILATERAL_FILTER_SIGMA_COLOR: int = 75 # Filter sigma in the color space.
    BILATERAL_FILTER_SIGMA_SPACE: int = 75 # Filter sigma in the coordinate space.
    CLAHE_CLIP_LIMIT: float = 2.0 # Clip limit for CLAHE.
    CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8) # Tile grid size for CLAHE.
    HOUGH_DP_VALUES: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5]) # Inverse ratio of accumulator resolution.
    HOUGH_MIN_DIST_FACTOR: float = 0.25 # Minimum distance between centers of detected circles, as a factor of image smaller dimension.
    HOUGH_PARAM1_VALUES: List[int] = field(default_factory=lambda: [70, 100, 130])  # Upper threshold for Canny edge detector in Hough.
    HOUGH_PARAM2_VALUES: List[int] = field(default_factory=lambda: [35, 45, 55])  # Accumulator threshold for circle detection.
    HOUGH_MIN_RADIUS_FACTOR: float = 0.1 # Minimum circle radius as a factor of image smaller dimension.
    HOUGH_MAX_RADIUS_FACTOR: float = 0.45 # Maximum circle radius as a factor of image smaller dimension.
    CIRCLE_CONFIDENCE_THRESHOLD: float = 0.3 # Minimum confidence for a detected circle to be considered valid.
    DO2MR_KERNEL_SIZES: List[Tuple[int, int]] = field(default_factory=lambda: [(5, 5), (9, 9), (13, 13)]) # Structuring element sizes.
    DO2MR_GAMMA_VALUES: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0]) # Sensitivity parameter for thresholding residual.
    DO2MR_MEDIAN_BLUR_KERNEL_SIZE: int = 5 # Kernel size for median blur on residual image.
    DO2MR_MORPH_OPEN_KERNEL_SIZE: Tuple[int, int] = (3, 3) # Kernel for morphological opening post-threshold.
    LEI_KERNEL_LENGTHS: List[int] = field(default_factory=lambda: [11, 17, 23]) # Lengths of the linear detector.
    LEI_ANGLE_STEP: int = 15  # Angular resolution for scratch detection (degrees).
    LEI_THRESHOLD_FACTOR: float = 2.0 # Factor for Otsu or adaptive thresholding on response map.
    LEI_MORPH_CLOSE_KERNEL_SIZE: Tuple[int, int] = (5, 1) # Kernel for morphological closing (elongated for scratches). (length, thickness)
    LEI_MIN_SCRATCH_AREA_PX: int = 15 # Minimum area for a scratch.
    CANNY_LOW_THRESHOLD: int = 50 # Low threshold for Canny edge detection.
    CANNY_HIGH_THRESHOLD: int = 150 # High threshold for Canny edge detection.
    ADAPTIVE_THRESH_BLOCK_SIZE: int = 11 # Block size for adaptive thresholding.
    ADAPTIVE_THRESH_C: int = 2 # Constant subtracted from the mean in adaptive thresholding.
    MIN_METHODS_FOR_CONFIRMED_DEFECT: int = 2 # Min number of methods that must detect a defect.
    CONFIDENCE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: { # Weights for different detection methods.
        "do2mr": 1.0, "lei": 1.0, "canny": 0.6, "adaptive_thresh": 0.7, "otsu_global": 0.5,
    })
    DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: { # BGR colors for defect types.
        "Region": (0, 255, 255), "Scratch": (255, 0, 255), "Contamination": (255, 165, 0),
        "Pit": (0, 128, 255), "Chip": (128, 0, 128), "Linear Region": (255,105,180) # Hot Pink for Linear Region
    })
    FONT_SCALE: float = 0.5 # Font scale for annotations.
    LINE_THICKNESS: int = 1 # Line thickness for drawing.
    BATCH_SUMMARY_FILENAME: str = "batch_inspection_summary.csv" # Filename for the overall batch summary.
    DETAILED_REPORT_PER_IMAGE: bool = True # Whether to generate a detailed CSV for each image.
    SAVE_ANNOTATED_IMAGE: bool = True # Whether to save the annotated image.
    SAVE_DEFECT_MAPS: bool = False # Whether to save intermediate defect maps (for debugging).
    SAVE_HISTOGRAM: bool = True # Whether to save the polar defect distribution histogram.

if __name__ == '__main__':
    # Example of creating an instance of InspectorConfig
    config = InspectorConfig()
    print(f"Created InspectorConfig instance.")
    print(f"Default output directory name: {config.OUTPUT_DIR_NAME}")
    print(f"Default core zone color: {config.DEFAULT_ZONES[0].color_bgr}")
