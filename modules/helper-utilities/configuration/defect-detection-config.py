from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class DefectDetectionConfig:
    """Unified configuration for all detection methods"""
    # General parameters
    min_defect_area_px: int = 5
    max_defect_area_px: int = 10000
    
    # Preprocessing parameters
    gaussian_blur_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(3,3), (5,5), (7,7)])
    bilateral_params: List[Tuple[int, int, int]] = field(default_factory=lambda: [(9,75,75), (7,50,50)])
    clahe_params: List[Tuple[float, Tuple[int, int]]] = field(default_factory=lambda: [(2.0,(8,8)), (3.0,(8,8))])
    
    # Fiber detection parameters
    hough_dp_values: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5])
    hough_param1_values: List[int] = field(default_factory=lambda: [50, 70, 100])
    hough_param2_values: List[int] = field(default_factory=lambda: [30, 40, 50])
    
    # DO2MR parameters
    do2mr_kernel_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(5,5), (11,11), (15,15), (21,21)])
    do2mr_gamma_values: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0, 3.5])
    
    # LEI parameters
    lei_kernel_lengths: List[int] = field(default_factory=lambda: [9, 11, 15, 19, 21])
    lei_angle_steps: List[int] = field(default_factory=lambda: [5, 10, 15])
    lei_threshold_factors: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])
    
    # Advanced detection parameters
    hessian_scales: List[float] = field(default_factory=lambda: [1, 2, 3, 4])
    frangi_scales: List[float] = field(default_factory=lambda: [1, 1.5, 2, 2.5, 3])
    log_scales: List[float] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8])
    phase_congruency_scales: int = 4
    phase_congruency_orientations: int = 6
    
    # Ensemble parameters
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'do2mr': 1.0,
        'lei': 1.0,
        'hessian': 0.9,
        'frangi': 0.9,
        'phase_congruency': 0.85,
        'radon': 0.8,
        'gradient': 0.8,
        'log': 0.9,
        'doh': 0.85,
        'mser': 0.8,
        'watershed': 0.85,
        'canny': 0.7,
        'adaptive': 0.75,
        'lbp': 0.7,
        'otsu': 0.7,
        'morphological': 0.75
    })
    min_methods_for_detection: int = 2
    ensemble_vote_threshold: float = 0.3

if __name__ == '__main__':
    config = DefectDetectionConfig()
    print("Created DefectDetectionConfig instance with default values:")
    print(config)
