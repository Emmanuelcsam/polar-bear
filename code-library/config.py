#!/usr/bin/env python3
"""
Unified Configuration Management System
======================================
Merges all configuration functionality from various config files into a single,
comprehensive configuration management system.
"""

import os
import json
import yaml
import time
import torch
import logging
import pathlib
import argparse
import configparser
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import copy
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays and Path objects"""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
        except ImportError:
            pass
        
        # Handle Path objects
        if isinstance(obj, (Path, pathlib.Path)):
            return str(obj)
        
        # Handle callable defaults (like time.time)
        if callable(obj):
            return obj()
        
        return super().default(obj)


@dataclass
class ProcessingConfig:
    """Configuration for image processing"""
    # Basic settings
    variations_enabled: bool = True
    num_variations: int = 49
    ram_only_mode: bool = True
    parallel_processing: bool = True
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    cache_enabled: bool = True
    cache_dir: str = "cache"
    
    # ML Configuration
    ml_enabled: bool = True
    pytorch_enabled: bool = True
    tensorflow_enabled: bool = False
    model_device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    
    # Real-time configuration
    realtime_enabled: bool = False
    camera_index: int = 0
    max_fps: int = 30
    display_results: bool = True
    
    # Preprocessing filters
    preprocessing_filters: List[str] = field(default_factory=lambda: [
        "gaussian_blur", "bilateral_filter", "median_blur",
        "morphological_open", "morphological_close", "morphological_gradient",
        "adaptive_threshold", "otsu_threshold", "canny_edge",
        "sobel_gradient", "laplacian", "histogram_equalization"
    ])
    
    # Preprocessing parameters (from fiber optic config)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: List[int] = field(default_factory=lambda: [8, 8])
    gaussian_blur_kernel_size: List[int] = field(default_factory=lambda: [5, 5])
    enable_illumination_correction: bool = True
    illumination_method: str = "rolling_ball"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_dir: str = "logs"
    detailed_logging: bool = True
    
    # Performance
    image_resize_factor: float = 1.0
    jpeg_quality: int = 95
    png_compression: int = 1
    
    # Image analysis parameters (from 0_config.py)
    analysis_duration_seconds: int = 60
    learning_mode: str = 'auto'  # 'auto' or 'manual'


@dataclass
class LocalizationConfig:
    """Configuration for fiber localization"""
    hough_dp: float = 1.2
    hough_min_dist_factor: float = 0.15
    hough_param1: int = 70
    hough_param2: int = 35
    hough_min_radius_factor: float = 0.08
    hough_max_radius_factor: float = 0.45
    enable_contour_fallback: bool = True
    enable_circle_fit_fallback: bool = True


@dataclass
class DefectDetectionConfig:
    """Configuration for defect detection"""
    # Detection algorithms
    detection_algorithms: List[str] = field(default_factory=lambda: [
        "do2mr", "lei", "matrix_variance", "statistical_anomaly", 
        "morphological_defects", "texture_analysis", "edge_discontinuity", 
        "intensity_outliers", "geometric_irregularity"
    ])
    
    # Algorithm weights
    algorithm_weights: Dict[str, float] = field(default_factory=lambda: {
        "do2mr": 1.0,
        "lei": 0.8,
        "matrix_variance": 0.6,
        "morph_gradient": 0.4,
        "black_hat": 0.6,
        "lei_advanced": 0.7,
        "skeletonization": 0.3
    })
    
    # Detection parameters
    confidence_threshold: float = 0.7
    min_defect_size: int = 5
    max_defect_size: int = 1000
    min_defect_area_px: int = 5
    use_ml_detection: bool = True
    anomaly_threshold: float = 3.0
    cluster_eps: float = 5.0
    cluster_min_samples: int = 3
    multi_scale_detection: bool = True
    enable_validation: bool = True
    scratch_aspect_ratio_threshold: float = 3.0


@dataclass
class AlgorithmParameters:
    """Specific algorithm parameters"""
    # DO2MR parameters
    do2mr_kernel_size: int = 5
    do2mr_gamma_core: float = 1.2
    do2mr_gamma_cladding: float = 1.5
    do2mr_multi_scale_kernels: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    
    # LEI parameters
    lei_kernel_lengths: List[int] = field(default_factory=lambda: [7, 11, 15, 21, 31])
    lei_angle_step_deg: int = 10
    lei_multi_scale_enabled: bool = True
    lei_scales: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    
    # Matrix variance parameters
    variance_threshold: float = 15.0
    local_window_size: int = 3
    segment_grid: List[int] = field(default_factory=lambda: [3, 3])
    
    # Morphological parameters
    morph_gradient_kernel_size: List[int] = field(default_factory=lambda: [5, 5])
    black_hat_kernel_size: List[int] = field(default_factory=lambda: [11, 11])
    
    # Other parameters
    flat_field_image_path: Optional[str] = None
    sobel_scharr_ksize: int = 3
    skeletonization_dilation_kernel_size: List[int] = field(default_factory=lambda: [3, 3])


@dataclass
class SeparationConfig:
    """Configuration for zone separation"""
    methods_enabled: List[str] = field(default_factory=lambda: [
        "adaptive_intensity", "bright_core_extractor", "computational_separation",
        "geometric_approach", "gradient_approach", "hough_separation",
        "intelligent_segmenter", "segmentation", "threshold_separation",
        "unified_core_cladding_detector", "guess_approach"
    ])
    consensus_threshold: float = 0.6
    learning_rate: float = 0.1
    use_inpainting: bool = True
    parallel_execution: bool = True
    timeout_seconds: int = 30
    visualize_consensus: bool = True
    save_intermediate_masks: bool = False


@dataclass
class VisualizationConfig:
    """Configuration for output visualization"""
    generate_overlays: bool = True
    generate_heatmaps: bool = True
    generate_3d_plots: bool = False
    generate_annotated_images: bool = True
    generate_csv_reports: bool = True
    generate_polar_histograms: bool = True
    save_individual_zones: bool = True
    
    # Colors
    defect_colors: Dict[str, tuple] = field(default_factory=lambda: {
        'pit': (255, 0, 0),
        'scratch': (0, 255, 0),
        'contamination': (0, 0, 255),
        'fiber_damage': (255, 255, 0),
        'crack': (255, 0, 255),
        'other': (128, 128, 128)
    })
    
    # Visual parameters
    overlay_opacity: float = 0.3
    text_size: float = 0.6
    line_thickness: int = 2
    image_dpi: int = 150
    font_scale: float = 0.5
    zone_outline_thickness: int = 2
    defect_outline_thickness: int = 2
    pass_fail_stamp_font_scale: float = 1.5
    pass_fail_stamp_thickness: int = 2
    display_timestamp_on_image: bool = True
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class ModuleConfig:
    """Configuration for a single module"""
    name: str
    enabled: bool = True
    auto_run: bool = False
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: str = "utility"
    description: str = ""


@dataclass
class FiberTypeDefinition:
    """Fiber type configuration"""
    description: str
    typical_core_diameter_um: float
    typical_cladding_diameter_um: float
    pass_fail_rules: Dict[str, Dict[str, Any]]
    zones: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SystemConfig:
    """Main system configuration"""
    # Project info
    project_name: str = "Polar Bear"
    version: str = "1.0"
    
    # Directories
    input_dir: str = "input"
    output_dir: str = "output"
    model_dir: str = "models"
    data_dir: str = "data"
    reference_dir: str = "reference_images"
    
    # Base paths (from config-36.py)
    base_path: Optional[Path] = None
    
    # Device configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cores: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    use_cpu: bool = field(default_factory=lambda: os.getenv("USE_CPU") is not None)
    
    # Configuration submodules
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    detection: DefectDetectionConfig = field(default_factory=DefectDetectionConfig)
    algorithm_params: AlgorithmParameters = field(default_factory=AlgorithmParameters)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # System settings
    debug_mode: bool = False
    verbose_output: bool = True
    save_intermediate_results: bool = False
    cleanup_temp_files: bool = True
    auto_save: bool = True
    cleanup_old_files: bool = False
    cleanup_days: int = 7
    max_file_size_mb: int = 100
    
    # Quality control
    min_image_size: int = 100
    max_image_size: int = 4096
    supported_formats: List[str] = field(default_factory=lambda: [
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'
    ])
    supported_image_extensions: List[str] = field(default_factory=lambda: [
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff"
    ])
    
    # Processing profiles
    processing_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Fiber type definitions
    fiber_type_definitions: Dict[str, FiberTypeDefinition] = field(default_factory=dict)
    
    # Module configurations
    modules: Dict[str, ModuleConfig] = field(default_factory=dict)
    
    # Calibration
    calibration: Dict[str, Any] = field(default_factory=lambda: {
        "default_um_per_px": 0.5,
        "auto_calibration_enabled": True,
        "calibration_method": "cladding_diameter"
    })
    
    # Advanced configuration from config-wizard
    pixels_per_image: int = 100
    batch_size: int = 10
    comparison_samples: int = 100
    enable_logging: bool = True
    auto_optimize: bool = False
    save_snapshots: bool = False
    prune_threshold: float = 0.3
    
    # File paths for data exchange
    generator_params_path: str = field(default_factory=lambda: "data/generator_params.json")
    analysis_results_path: str = field(default_factory=lambda: "data/analysis_results.json")
    anomalies_path: str = field(default_factory=lambda: "data/anomalies.json")
    intensity_data_path: str = field(default_factory=lambda: "data/intensities.npy")
    stats_data_path: str = field(default_factory=lambda: "data/image_stats.csv")
    model_path: str = field(default_factory=lambda: "data/pixel_model.pth")
    avg_image_path: str = field(default_factory=lambda: "data/average_image.npy")
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=lambda: time.time() + 60)


class UnifiedConfigurationManager:
    """
    Unified configuration management system combining all configuration functionality.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "unified_config.json"
        self.config = SystemConfig()
        self._initialize_default_profiles()
        self._initialize_fiber_types()
        self._initialize_modules()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            self._setup_base_path()
    
    def _setup_base_path(self):
        """Setup base path from current file location"""
        if self.config.base_path is None:
            self.config.base_path = pathlib.Path(__file__).resolve().parent.parent
            self.config.data_dir = str(self.config.base_path / "data")
            Path(self.config.data_dir).mkdir(exist_ok=True)
    
    def _initialize_default_profiles(self):
        """Initialize default processing profiles"""
        self.config.processing_profiles = {
            "fast_scan": {
                "description": "Quick scan with minimal processing",
                "preprocessing": asdict(ProcessingConfig(
                    clahe_clip_limit=1.5,
                    clahe_tile_grid_size=[4, 4],
                    gaussian_blur_kernel_size=[3, 3],
                    enable_illumination_correction=False
                )),
                "localization": asdict(LocalizationConfig(
                    hough_dp=1.5,
                    hough_min_dist_factor=0.2,
                    hough_param1=50,
                    hough_param2=25,
                    hough_min_radius_factor=0.1,
                    hough_max_radius_factor=0.4,
                    enable_contour_fallback=True,
                    enable_circle_fit_fallback=False
                )),
                "defect_detection": asdict(DefectDetectionConfig(
                    detection_algorithms=["do2mr"],
                    algorithm_weights={"do2mr": 1.0},
                    multi_scale_detection=False,
                    min_defect_area_px=10,
                    confidence_threshold=0.7
                ))
            },
            "deep_inspection": {
                "description": "Comprehensive inspection with all algorithms",
                "preprocessing": asdict(ProcessingConfig(
                    clahe_clip_limit=2.0,
                    clahe_tile_grid_size=[8, 8],
                    gaussian_blur_kernel_size=[5, 5],
                    enable_illumination_correction=True,
                    illumination_method="rolling_ball"
                )),
                "localization": asdict(self.config.localization),
                "defect_detection": asdict(self.config.detection)
            }
        }
    
    def _initialize_fiber_types(self):
        """Initialize fiber type definitions"""
        self.config.fiber_type_definitions = {
            "single_mode_pc": FiberTypeDefinition(
                description="Single-mode PC connector",
                typical_core_diameter_um=9.0,
                typical_cladding_diameter_um=125.0,
                pass_fail_rules={
                    "Core": {
                        "max_scratches": 0,
                        "max_defects": 0,
                        "max_defect_size_um": 3,
                        "critical_zone": True
                    },
                    "Cladding": {
                        "max_scratches": 5,
                        "max_scratches_gt_5um": 0,
                        "max_defects": 5,
                        "max_defect_size_um": 10,
                        "critical_zone": False
                    }
                },
                zones=[
                    {
                        "name": "Core",
                        "r_min_factor": 0.0,
                        "r_max_factor_core_relative": 1.0,
                        "color_bgr": [255, 0, 0]
                    },
                    {
                        "name": "Cladding",
                        "r_min_factor_cladding_relative": 0.0,
                        "r_max_factor_cladding_relative": 1.0,
                        "color_bgr": [0, 255, 0]
                    }
                ]
            ),
            "multi_mode_pc": FiberTypeDefinition(
                description="Multi-mode PC connector",
                typical_core_diameter_um=50.0,
                typical_cladding_diameter_um=125.0,
                pass_fail_rules={
                    "Core": {
                        "max_scratches": 1,
                        "max_scratch_length_um": 10,
                        "max_defects": 3,
                        "max_defect_size_um": 5,
                        "critical_zone": False
                    },
                    "Cladding": {
                        "max_scratches": "unlimited",
                        "max_defects": "unlimited",
                        "max_defect_size_um": 20,
                        "critical_zone": False
                    }
                },
                zones=[
                    {
                        "name": "Core",
                        "r_min_factor": 0.0,
                        "r_max_factor_core_relative": 1.0,
                        "color_bgr": [255, 100, 100]
                    },
                    {
                        "name": "Cladding",
                        "r_min_factor_cladding_relative": 0.0,
                        "r_max_factor_cladding_relative": 1.0,
                        "color_bgr": [100, 255, 100]
                    }
                ]
            )
        }
    
    def _initialize_modules(self):
        """Initialize module configurations"""
        module_definitions = {
            # Core modules
            'pixel_reader': ('core', 'Reads pixel data from images'),
            'random_generator': ('core', 'Generates random pixel values'),
            'correlator': ('core', 'Finds correlations between data streams'),
            
            # Analysis modules
            'pattern_recognizer': ('analysis', 'Detects patterns in pixel data'),
            'anomaly_detector': ('analysis', 'Identifies anomalies'),
            'intensity_analyzer': ('analysis', 'Analyzes intensity distributions'),
            'geometry_analyzer': ('analysis', 'Finds geometric patterns'),
            'trend_analyzer': ('analysis', 'Analyzes trends over time'),
            'data_calculator': ('analysis', 'Performs advanced calculations'),
            
            # AI modules
            'neural_learner': ('ai', 'Neural network learning'),
            'neural_generator': ('ai', 'Generates images using neural networks'),
            'vision_processor': ('ai', 'Computer vision processing'),
            'hybrid_analyzer': ('ai', 'Combines AI insights'),
            'ml_classifier': ('ai', 'Machine learning classification'),
            
            # HPC modules
            'gpu_accelerator': ('hpc', 'GPU-accelerated processing'),
            'gpu_image_generator': ('hpc', 'GPU-based image generation'),
            'parallel_processor': ('hpc', 'Multi-core parallel processing'),
            'distributed_analyzer': ('hpc', 'Distributed computing'),
            'hpc_optimizer': ('hpc', 'HPC optimization'),
            
            # Real-time modules
            'realtime_processor': ('realtime', 'Real-time monitoring'),
            'live_capture': ('realtime', 'Live video capture'),
            'stream_analyzer': ('realtime', 'Stream analysis'),
            
            # Utility modules
            'batch_processor': ('utility', 'Batch image processing'),
            'image_generator': ('utility', 'Basic image generation'),
            'image_categorizer': ('utility', 'Categorizes images'),
            'learning_engine': ('utility', 'Learning system'),
            'data_store': ('utility', 'Data storage management'),
            'continuous_analyzer': ('utility', 'Continuous analysis'),
            'logger': ('utility', 'Logging system'),
            'visualizer': ('utility', 'Basic visualization'),
            'advanced_visualizer': ('utility', 'Advanced visualization'),
            'network_api': ('utility', 'Network API server'),
            'data_exporter': ('utility', 'Data export/import')
        }
        
        for module_name, (category, description) in module_definitions.items():
            self.config.modules[module_name] = ModuleConfig(
                name=module_name,
                enabled=True,
                auto_run=False,
                category=category,
                description=description
            )
    
    def load_config(self, config_file: str):
        """Load configuration from file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file '{config_file}' not found.")
            return
        
        try:
            # Determine format by extension
            if config_path.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_file, 'r') as f:
                    config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() in ['.ini', '.cfg']:
                config_dict = self._load_ini_config(config_file)
            else:
                logger.error(f"Unsupported configuration format: {config_path.suffix}")
                return
            
            self._update_config_from_dict(config_dict)
            logger.info(f"Configuration loaded from '{config_file}'")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _load_ini_config(self, config_file: str) -> Dict[str, Any]:
        """Load INI configuration and convert to dict"""
        config = configparser.ConfigParser()
        config.read(config_file)
        
        config_dict = {}
        for section_name in config.sections():
            section = config[section_name]
            config_dict[section_name] = {}
            
            for key, value in section.items():
                config_dict[section_name][key] = self._parse_ini_value(value)
        
        return config_dict
    
    def _parse_ini_value(self, value: str) -> Union[str, int, float, bool, List]:
        """Parse INI value to appropriate Python type"""
        if value.lower() in ['true', 'yes', '1', 'on']:
            return True
        elif value.lower() in ['false', 'no', '0', 'off']:
            return False
        
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        return value
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                if isinstance(value, dict) and hasattr(getattr(self.config, key), '__dict__'):
                    # Handle nested configuration objects
                    nested_config = getattr(self.config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self.config, key, value)
    
    def save_config(self, config_file: Optional[str] = None, format_type: str = 'auto'):
        """Save configuration to file"""
        save_path = config_file or self.config_file
        config_path = Path(save_path)
        
        # Auto-detect format
        if format_type == 'auto':
            if config_path.suffix.lower() == '.json':
                format_type = 'json'
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                format_type = 'yaml'
            elif config_path.suffix.lower() in ['.ini', '.cfg']:
                format_type = 'ini'
            else:
                format_type = 'json'
        
        # Create directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format_type == 'json':
                with open(save_path, 'w') as f:
                    json.dump(asdict(self.config), f, indent=4, cls=NumpyEncoder)
            elif format_type == 'yaml':
                with open(save_path, 'w') as f:
                    yaml.dump(asdict(self.config), f, default_flow_style=False, indent=2)
            elif format_type == 'ini':
                self._save_ini_config(save_path)
            
            logger.info(f"Configuration saved to '{save_path}'")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _save_ini_config(self, config_file: str):
        """Save configuration in INI format"""
        config = configparser.ConfigParser()
        config_dict = asdict(self.config)
        
        for section_name, section_data in config_dict.items():
            if isinstance(section_data, dict):
                config[section_name] = {}
                for key, value in section_data.items():
                    if isinstance(value, list):
                        config[section_name][key] = ', '.join(map(str, value))
                    else:
                        config[section_name][key] = str(value)
            else:
                if 'DEFAULT' not in config:
                    config['DEFAULT'] = {}
                config['DEFAULT'][section_name] = str(section_data)
        
        with open(config_file, 'w') as f:
            config.write(f)
    
    def get_processing_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific processing profile"""
        return self.config.processing_profiles.get(profile_name)
    
    def get_fiber_type_definition(self, fiber_type: str) -> Optional[FiberTypeDefinition]:
        """Get fiber type definition"""
        return self.config.fiber_type_definitions.get(fiber_type)
    
    def get_algorithm_parameters(self, algorithm_name: str) -> Dict[str, Any]:
        """Get algorithm-specific parameters"""
        params = {}
        algo_params = asdict(self.config.algorithm_params)
        
        # Filter parameters for specific algorithm
        for key, value in algo_params.items():
            if key.startswith(algorithm_name.lower()):
                param_name = key.replace(f"{algorithm_name.lower()}_", "")
                params[param_name] = value
        
        return params
    
    def update_parameter(self, parameter_path: str, value: Any) -> None:
        """Update a specific parameter using dot notation"""
        keys = parameter_path.split('.')
        current = self.config
        
        # Navigate to the parent of the target parameter
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                logger.error(f"Parameter path '{parameter_path}' not found")
                return
        
        # Set the value
        if hasattr(current, keys[-1]):
            setattr(current, keys[-1], value)
            logger.info(f"Updated parameter '{parameter_path}' to '{value}'")
        else:
            logger.error(f"Parameter '{keys[-1]}' not found in {type(current).__name__}")
    
    def set_module_enabled(self, module_name: str, enabled: bool):
        """Enable or disable a module"""
        if module_name in self.config.modules:
            self.config.modules[module_name].enabled = enabled
            logger.info(f"Module '{module_name}' {'enabled' if enabled else 'disabled'}")
            
            if self.config.auto_save:
                self.save_config()
        else:
            logger.error(f"Module '{module_name}' not found")
    
    def get_enabled_modules(self, category: Optional[str] = None) -> List[str]:
        """Get list of enabled modules"""
        enabled = []
        for name, module in self.config.modules.items():
            if module.enabled:
                if category is None or module.category == category:
                    enabled.append(name)
        return enabled
    
    def create_custom_profile(self, profile_name: str, 
                            base_profile: str = "deep_inspection",
                            modifications: Optional[Dict[str, Any]] = None) -> None:
        """Create a custom processing profile"""
        if base_profile not in self.config.processing_profiles:
            logger.error(f"Base profile '{base_profile}' not found")
            return
        
        # Copy base profile
        new_profile = copy.deepcopy(self.config.processing_profiles[base_profile])
        
        # Apply modifications
        if modifications:
            new_profile = self._merge_dicts(new_profile, modifications)
        
        # Add description
        new_profile["description"] = f"Custom profile based on {base_profile}"
        
        # Save new profile
        self.config.processing_profiles[profile_name] = new_profile
        logger.info(f"Created custom profile '{profile_name}' based on '{base_profile}'")
    
    def _merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries"""
        result = copy.deepcopy(dict1)
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def validate_configuration(self) -> List[str]:
        """Validate the current configuration"""
        errors = []
        
        # Check directories
        if not self.config.input_dir:
            errors.append("Input directory not specified")
        
        if not self.config.output_dir:
            errors.append("Output directory not specified")
        
        # Check processing configuration
        if self.config.processing.max_workers < 1:
            errors.append("Max workers must be at least 1")
        
        if self.config.processing.batch_size < 1:
            errors.append("Batch size must be at least 1")
        
        # Check detection configuration
        if not (0.0 <= self.config.detection.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        if self.config.detection.min_defect_size < 1:
            errors.append("Minimum defect size must be at least 1")
        
        # Check visualization configuration
        if not (0.0 <= self.config.visualization.overlay_opacity <= 1.0):
            errors.append("Overlay opacity must be between 0.0 and 1.0")
        
        # Check profiles
        if not self.config.processing_profiles:
            errors.append("No processing profiles defined")
        
        return errors
    
    def interactive_setup(self):
        """Interactive configuration setup"""
        print("=== Unified Configuration Setup ===\n")
        
        # Basic settings
        print("1. Basic Settings")
        self.config.input_dir = input(f"Input directory [{self.config.input_dir}]: ").strip() or self.config.input_dir
        self.config.output_dir = input(f"Output directory [{self.config.output_dir}]: ").strip() or self.config.output_dir
        
        debug_choice = input(f"Enable debug mode? [{'y' if self.config.debug_mode else 'n'}]: ").strip().lower()
        if debug_choice in ['y', 'yes']:
            self.config.debug_mode = True
        elif debug_choice in ['n', 'no']:
            self.config.debug_mode = False
        
        # Processing settings
        print("\n2. Processing Settings")
        ml_choice = input(f"Enable ML processing? [{'y' if self.config.processing.ml_enabled else 'n'}]: ").strip().lower()
        if ml_choice in ['y', 'yes']:
            self.config.processing.ml_enabled = True
        elif ml_choice in ['n', 'no']:
            self.config.processing.ml_enabled = False
        
        # Advanced settings
        print("\n3. Advanced Settings")
        advanced = input("Configure advanced settings? (y/n): ")
        
        if advanced.lower() == 'y':
            self.config.enable_logging = input("Enable detailed logging? (y/n): ") == 'y'
            self.config.auto_optimize = input("Auto-optimize after batch processing? (y/n): ") == 'y'
            self.config.save_snapshots = input("Save analysis snapshots? (y/n): ") == 'y'
            
            confidence_input = input(f"Confidence threshold [{self.config.detection.confidence_threshold}]: ").strip()
            if confidence_input:
                try:
                    self.config.detection.confidence_threshold = float(confidence_input)
                except ValueError:
                    pass
        
        print("\nConfiguration setup complete!")
    
    def print_status(self):
        """Print current configuration status"""
        print("\n=== UNIFIED CONFIGURATION STATUS ===")
        print(f"\nProject: {self.config.project_name} v{self.config.version}")
        print(f"Device: {self.config.device} (Cores: {self.config.cores})")
        
        print(f"\nDirectories:")
        print(f"  Input: {self.config.input_dir}")
        print(f"  Output: {self.config.output_dir}")
        print(f"  Data: {self.config.data_dir}")
        print(f"  Model: {self.config.model_dir}")
        
        print(f"\nProcessing:")
        print(f"  ML Enabled: {self.config.processing.ml_enabled}")
        print(f"  Parallel Processing: {self.config.processing.parallel_processing}")
        print(f"  Max Workers: {self.config.processing.max_workers}")
        print(f"  Real-time Enabled: {self.config.processing.realtime_enabled}")
        
        print(f"\nDetection:")
        print(f"  Algorithms: {len(self.config.detection.detection_algorithms)}")
        print(f"  Confidence Threshold: {self.config.detection.confidence_threshold}")
        print(f"  Min Defect Size: {self.config.detection.min_defect_size}px")
        
        print(f"\nProfiles: {list(self.config.processing_profiles.keys())}")
        print(f"Fiber Types: {list(self.config.fiber_type_definitions.keys())}")
        
        enabled_modules = self.get_enabled_modules()
        print(f"\nModules: {len(self.config.modules)} total, {len(enabled_modules)} enabled")
        
        # Group by category
        categories = {}
        for name, module in self.config.modules.items():
            if module.category not in categories:
                categories[module.category] = []
            categories[module.category].append((name, module.enabled))
        
        for category, modules in sorted(categories.items()):
            print(f"\n  {category.upper()}:")
            for name, enabled in sorted(modules):
                status = "✓" if enabled else "✗"
                print(f"    {status} {name}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate configuration report"""
        report = {
            'timestamp': time.time(),
            'project_name': self.config.project_name,
            'version': self.config.version,
            'device': self.config.device,
            'summary': {
                'profiles': len(self.config.processing_profiles),
                'fiber_types': len(self.config.fiber_type_definitions),
                'modules_total': len(self.config.modules),
                'modules_enabled': len(self.get_enabled_modules()),
                'detection_algorithms': len(self.config.detection.detection_algorithms)
            },
            'validation': {
                'errors': self.validate_configuration()
            }
        }
        
        # Save report
        report_path = Path(self.config.output_dir) / 'config_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Configuration report saved to '{report_path}'")
        return report


# Convenience functions for backward compatibility
_global_config_manager = None

def get_global_config_manager(config_path: Optional[str] = None) -> UnifiedConfigurationManager:
    """Get or create the global configuration manager instance"""
    global _global_config_manager
    if _global_config_manager is None or config_path is not None:
        _global_config_manager = UnifiedConfigurationManager(config_path)
    return _global_config_manager

def load_config(config_path: str = "unified_config.json") -> SystemConfig:
    """Load configuration from file"""
    manager = get_global_config_manager(config_path)
    return manager.config

def get_processing_profile(profile_name: str) -> Optional[Dict[str, Any]]:
    """Get processing profile"""
    manager = get_global_config_manager()
    return manager.get_processing_profile(profile_name)

def get_fiber_type_definition(fiber_type: str) -> Optional[FiberTypeDefinition]:
    """Get fiber type definition"""
    manager = get_global_config_manager()
    return manager.get_fiber_type_definition(fiber_type)

def get_config():
    """Get current configuration (for compatibility with shared_config.py)"""
    manager = get_global_config_manager()
    return asdict(manager.config)

def set_config_value(key: str, value: Any):
    """Set a configuration value (for compatibility with shared_config.py)"""
    manager = get_global_config_manager()
    manager.update_parameter(key, value)
    return True

def update_config(new_config_dict: Dict[str, Any]):
    """Update multiple configuration values"""
    manager = get_global_config_manager()
    for key, value in new_config_dict.items():
        manager.update_parameter(key, value)
    return True


def main():
    """Main entry point for configuration management"""
    parser = argparse.ArgumentParser(description='Unified Configuration Management System')
    parser.add_argument('--config', help='Configuration file to load')
    parser.add_argument('--save', help='Save configuration to file')
    parser.add_argument('--format', choices=['json', 'yaml', 'ini'], default='json',
                       help='Configuration file format (default: json)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive configuration setup')
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration')
    parser.add_argument('--status', action='store_true',
                       help='Print current configuration status')
    parser.add_argument('--report', action='store_true',
                       help='Generate configuration report')
    parser.add_argument('--preset', choices=['basic', 'ai_powered', 'high_performance', 'real_time', 'full_system'],
                       help='Apply a preset configuration')
    
    args = parser.parse_args()
    
    # Create configuration manager
    config_manager = UnifiedConfigurationManager(args.config)
    
    # Apply preset if specified
    if args.preset:
        presets = {
            'basic': ['pixel_reader', 'pattern_recognizer', 'anomaly_detector', 'visualizer'],
            'ai_powered': ['pixel_reader', 'vision_processor', 'neural_learner', 'neural_generator', 
                          'ml_classifier', 'hybrid_analyzer'],
            'high_performance': ['pixel_reader', 'gpu_accelerator', 'parallel_processor', 
                               'distributed_analyzer', 'hpc_optimizer'],
            'real_time': ['live_capture', 'realtime_processor', 'stream_analyzer'],
            'full_system': 'all'
        }
        
        # Disable all modules first
        for module_name in config_manager.config.modules:
            config_manager.set_module_enabled(module_name, False)
        
        # Enable preset modules
        preset_modules = presets[args.preset]
        if preset_modules == 'all':
            for module_name in config_manager.config.modules:
                config_manager.set_module_enabled(module_name, True)
        else:
            for module_name in preset_modules:
                if module_name in config_manager.config.modules:
                    config_manager.set_module_enabled(module_name, True)
        
        print(f"Applied preset '{args.preset}'")
    
    # Interactive setup
    if args.interactive:
        config_manager.interactive_setup()
    
    # Validate configuration
    if args.validate:
        errors = config_manager.validate_configuration()
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("Configuration is valid")
    
    # Print status
    if args.status:
        config_manager.print_status()
    
    # Generate report
    if args.report:
        report = config_manager.generate_report()
        print(f"\nConfiguration Report:")
        print(f"  Profiles: {report['summary']['profiles']}")
        print(f"  Fiber Types: {report['summary']['fiber_types']}")
        print(f"  Modules: {report['summary']['modules_total']} total, {report['summary']['modules_enabled']} enabled")
        print(f"  Detection Algorithms: {report['summary']['detection_algorithms']}")
        if report['validation']['errors']:
            print(f"  Validation Errors: {len(report['validation']['errors'])}")
    
    # Save configuration
    if args.save:
        config_manager.save_config(args.save, args.format)
        print(f"Configuration saved to: {args.save}")
    
    return 0


if __name__ == "__main__":
    exit(main())