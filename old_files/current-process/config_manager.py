"""
Enhanced Configuration Manager
No argparse - uses interactive string inputs and environment variables
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for image processing"""
    variations_enabled: bool = True
    num_variations: int = 49  # Based on old-processes best practice
    ram_only_mode: bool = True
    parallel_processing: bool = True
    max_workers: int = os.cpu_count() or 4
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    
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
    
    # Preprocessing
    preprocessing_filters: List[str] = field(default_factory=lambda: [
        "gaussian_blur", "bilateral_filter", "median_blur",
        "morphological_open", "morphological_close", "morphological_gradient",
        "adaptive_threshold", "otsu_threshold", "canny_edge",
        "sobel_gradient", "laplacian", "histogram_equalization"
    ])
    
    # Logging
    log_level: str = "DEBUG"
    log_to_file: bool = True
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    detailed_logging: bool = True
    
    # Performance
    image_resize_factor: float = 1.0
    jpeg_quality: int = 95
    png_compression: int = 1


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
class DetectionConfig:
    """Configuration for defect detection"""
    detection_algorithms: List[str] = field(default_factory=lambda: [
        "statistical_anomaly", "morphological_defects", "texture_analysis",
        "edge_discontinuity", "intensity_outliers", "geometric_irregularity"
    ])
    confidence_threshold: float = 0.7
    min_defect_size: int = 5
    max_defect_size: int = 1000
    use_ml_detection: bool = True
    anomaly_threshold: float = 3.0  # Mahalanobis distance threshold
    cluster_eps: float = 5.0  # DBSCAN clustering epsilon
    cluster_min_samples: int = 3


@dataclass
class VisualizationConfig:
    """Configuration for output visualization"""
    generate_overlays: bool = True
    generate_heatmaps: bool = True
    generate_3d_plots: bool = False
    save_format: str = "png"
    overlay_alpha: float = 0.5
    defect_colors: Dict[str, tuple] = field(default_factory=lambda: {
        "scratch": (255, 0, 0),
        "pit": (0, 255, 0),
        "contamination": (0, 0, 255),
        "fiber_damage": (255, 255, 0),
        "unknown": (255, 0, 255)
    })


@dataclass
class SystemConfig:
    """Complete system configuration"""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Paths
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    model_dir: Path = Path("models")
    knowledge_base_path: Path = Path("knowledge_base.json")
    
    # System
    debug_mode: bool = True
    dry_run: bool = False
    interactive_mode: bool = True


class ConfigManager:
    """Manages system configuration with interactive input"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("fiber_config.yaml")
        self.config = SystemConfig()
        self._load_config()
        self._setup_logging()
    
    def _load_config(self):
        """Load configuration from file, environment, or create default"""
        # Try loading from file
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data:
                        self._update_config_from_dict(data)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Override with environment variables
        self._load_from_environment()
        
        # Interactive configuration if enabled
        if self.config.interactive_mode and not os.getenv("FIBER_NO_INTERACTIVE"):
            self._interactive_setup()
    
    def _update_config_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary"""
        if "processing" in data:
            self.config.processing = ProcessingConfig(**data["processing"])
        if "separation" in data:
            self.config.separation = SeparationConfig(**data["separation"])
        if "detection" in data:
            self.config.detection = DetectionConfig(**data["detection"])
        if "visualization" in data:
            self.config.visualization = VisualizationConfig(**data["visualization"])
        
        # Update top-level attributes
        for key in ["input_dir", "output_dir", "model_dir", "knowledge_base_path",
                    "debug_mode", "dry_run", "interactive_mode"]:
            if key in data:
                if key.endswith("_dir") or key.endswith("_path"):
                    setattr(self.config, key, Path(data[key]))
                else:
                    setattr(self.config, key, data[key])
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            "FIBER_DEBUG": ("debug_mode", bool),
            "FIBER_RAM_ONLY": ("processing.ram_only_mode", bool),
            "FIBER_ML_ENABLED": ("processing.ml_enabled", bool),
            "FIBER_REALTIME": ("processing.realtime_enabled", bool),
            "FIBER_LOG_LEVEL": ("processing.log_level", str),
            "FIBER_PARALLEL": ("processing.parallel_processing", bool),
            "FIBER_INPUT_DIR": ("input_dir", Path),
            "FIBER_OUTPUT_DIR": ("output_dir", Path),
        }
        
        for env_var, (config_path, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if type_func == bool:
                        value = value.lower() in ("true", "1", "yes", "on")
                    else:
                        value = type_func(value)
                    
                    # Handle nested attributes
                    if "." in config_path:
                        parent, attr = config_path.split(".", 1)
                        setattr(getattr(self.config, parent), attr, value)
                    else:
                        setattr(self.config, config_path, value)
                except Exception as e:
                    logger.warning(f"Failed to parse {env_var}: {e}")
    
    def _interactive_setup(self):
        """Interactive configuration setup"""
        print("\n=== Fiber Optic Defect Detection Configuration ===\n")
        
        # Quick setup or detailed?
        mode = self._ask_question(
            "Configuration mode? (quick/detailed/skip)",
            default="quick",
            choices=["quick", "detailed", "skip"]
        )
        
        if mode == "skip":
            return
        
        if mode == "quick":
            self._quick_setup()
        else:
            self._detailed_setup()
        
        # Save configuration
        if self._ask_yes_no("Save this configuration for future use?", default=True):
            self.save_config()
    
    def _quick_setup(self):
        """Quick configuration with essential options"""
        print("\n--- Quick Setup ---")
        
        # Processing mode
        self.config.processing.ram_only_mode = self._ask_yes_no(
            "Use RAM-only mode (faster, more memory)?", 
            default=True
        )
        
        # ML Integration
        self.config.processing.ml_enabled = self._ask_yes_no(
            "Enable machine learning features?",
            default=True
        )
        
        if self.config.processing.ml_enabled:
            ml_framework = self._ask_question(
                "ML Framework? (pytorch/tensorflow/both)",
                default="pytorch",
                choices=["pytorch", "tensorflow", "both"]
            )
            self.config.processing.pytorch_enabled = ml_framework in ["pytorch", "both"]
            self.config.processing.tensorflow_enabled = ml_framework in ["tensorflow", "both"]
        
        # Real-time processing
        self.config.processing.realtime_enabled = self._ask_yes_no(
            "Enable real-time video processing?",
            default=False
        )
        
        # Parallel processing
        self.config.processing.parallel_processing = self._ask_yes_no(
            "Enable parallel processing?",
            default=True
        )
        
        if self.config.processing.parallel_processing:
            max_workers = self._ask_number(
                f"Number of worker processes? (1-{os.cpu_count()})",
                default=os.cpu_count(),
                min_val=1,
                max_val=os.cpu_count()
            )
            self.config.processing.max_workers = max_workers
    
    def _detailed_setup(self):
        """Detailed configuration with all options"""
        print("\n--- Detailed Setup ---")
        
        # First do quick setup basics
        self._quick_setup()
        
        print("\n--- Processing Configuration ---")
        self.config.processing.num_variations = self._ask_number(
            "Number of preprocessing variations? (1-100)",
            default=49,
            min_val=1,
            max_val=100
        )
        
        self.config.processing.cache_enabled = self._ask_yes_no(
            "Enable caching for processed images?",
            default=True
        )
        
        print("\n--- Separation Configuration ---")
        self.config.separation.consensus_threshold = self._ask_number(
            "Consensus threshold for segmentation? (0.0-1.0)",
            default=0.6,
            min_val=0.0,
            max_val=1.0,
            is_float=True
        )
        
        self.config.separation.use_inpainting = self._ask_yes_no(
            "Use inpainting for defect removal before segmentation?",
            default=True
        )
        
        print("\n--- Detection Configuration ---")
        self.config.detection.confidence_threshold = self._ask_number(
            "Confidence threshold for defect detection? (0.0-1.0)",
            default=0.7,
            min_val=0.0,
            max_val=1.0,
            is_float=True
        )
        
        self.config.detection.use_ml_detection = self._ask_yes_no(
            "Use ML-based defect detection?",
            default=True
        )
        
        print("\n--- Visualization Configuration ---")
        self.config.visualization.generate_overlays = self._ask_yes_no(
            "Generate defect overlay images?",
            default=True
        )
        
        self.config.visualization.generate_heatmaps = self._ask_yes_no(
            "Generate defect heatmaps?",
            default=True
        )
    
    def _ask_question(self, prompt: str, default: str = "", choices: List[str] = None) -> str:
        """Ask user a question and return response"""
        if choices:
            prompt += f" [{'/'.join(choices)}]"
        if default:
            prompt += f" (default: {default})"
        prompt += ": "
        
        while True:
            response = input(prompt).strip() or default
            if not choices or response in choices:
                return response
            print(f"Please choose from: {', '.join(choices)}")
    
    def _ask_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Ask user a yes/no question"""
        default_str = "yes" if default else "no"
        response = self._ask_question(
            f"{prompt} [yes/no]",
            default=default_str,
            choices=["yes", "no", "y", "n"]
        )
        return response.lower() in ["yes", "y"]
    
    def _ask_number(self, prompt: str, default: float = 0, min_val: float = None, 
                    max_val: float = None, is_float: bool = False) -> float:
        """Ask user for a number"""
        prompt += f" (default: {default}): "
        
        while True:
            response = input(prompt).strip()
            if not response:
                return default
            
            try:
                value = float(response) if is_float else int(response)
                if min_val is not None and value < min_val:
                    print(f"Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must be <= {max_val}")
                    continue
                return value
            except ValueError:
                print("Please enter a valid number")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.processing.log_level.upper())
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            detailed_formatter if self.config.processing.detailed_logging else simple_formatter
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        
        # File handler if enabled
        if self.config.processing.log_to_file:
            self.config.processing.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.config.processing.log_dir / "fiber_detection.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
    
    def save_config(self, path: Optional[Path] = None):
        """Save configuration to file"""
        path = path or self.config_file
        
        # Convert config to dictionary
        config_dict = {
            "processing": asdict(self.config.processing),
            "separation": asdict(self.config.separation),
            "detection": asdict(self.config.detection),
            "visualization": asdict(self.config.visualization),
            "input_dir": str(self.config.input_dir),
            "output_dir": str(self.config.output_dir),
            "model_dir": str(self.config.model_dir),
            "knowledge_base_path": str(self.config.knowledge_base_path),
            "debug_mode": self.config.debug_mode,
            "dry_run": self.config.dry_run,
            "interactive_mode": self.config.interactive_mode
        }
        
        # Convert Path objects to strings
        for section in ["processing", "separation", "detection", "visualization"]:
            for key, value in config_dict[section].items():
                if isinstance(value, Path):
                    config_dict[section][key] = str(value)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {path}")
    
    def get_config(self) -> SystemConfig:
        """Get the current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")


# Singleton instance
_config_manager = None


def get_config_manager(config_file: Optional[Path] = None) -> ConfigManager:
    """Get or create the configuration manager singleton"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager


def get_config() -> SystemConfig:
    """Get the current system configuration"""
    return get_config_manager().get_config()