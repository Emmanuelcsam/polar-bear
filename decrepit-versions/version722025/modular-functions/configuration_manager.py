#!/usr/bin/env python3
"""
Configuration Management - Standalone Module
Extracted from fiber optic defect detection system
Provides flexible configuration management with interactive setup
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import configparser


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays"""
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
        return super().default(obj)


@dataclass
class ProcessingConfig:
    """Configuration for image processing"""
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
    
    # Logging
    log_level: str = "DEBUG"
    log_to_file: bool = True
    log_dir: str = "logs"
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
    anomaly_threshold: float = 3.0
    cluster_eps: float = 5.0
    cluster_min_samples: int = 3


@dataclass
class VisualizationConfig:
    """Configuration for output visualization"""
    generate_overlays: bool = True
    generate_heatmaps: bool = True
    generate_3d_plots: bool = False
    save_individual_zones: bool = True
    defect_colors: Dict[str, tuple] = field(default_factory=lambda: {
        'pit': (255, 0, 0),
        'scratch': (0, 255, 0),
        'contamination': (0, 0, 255),
        'fiber_damage': (255, 255, 0),
        'crack': (255, 0, 255),
        'other': (128, 128, 128)
    })
    overlay_opacity: float = 0.3
    text_size: float = 0.6
    line_thickness: int = 2


@dataclass
class SystemConfig:
    """Main system configuration"""
    # Directories
    input_dir: str = "input"
    output_dir: str = "output"
    model_dir: str = "models"
    
    # Processing configuration
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # System settings
    debug_mode: bool = False
    verbose_output: bool = True
    save_intermediate_results: bool = False
    cleanup_temp_files: bool = True
    
    # Quality control
    min_image_size: int = 100
    max_image_size: int = 4096
    supported_formats: List[str] = field(default_factory=lambda: [
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'
    ])


class ConfigurationManager:
    """Manages system configuration with multiple format support"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = SystemConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """
        Load configuration from file.
        
        Args:
            config_file (str): Path to configuration file
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Determine format by extension
        if config_path.suffix.lower() == '.json':
            self._load_json_config(config_file)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            self._load_yaml_config(config_file)
        elif config_path.suffix.lower() in ['.ini', '.cfg']:
            self._load_ini_config(config_file)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    def _load_json_config(self, config_file: str):
        """Load JSON configuration"""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        self._update_config_from_dict(config_dict)
    
    def _load_yaml_config(self, config_file: str):
        """Load YAML configuration"""
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            self._update_config_from_dict(config_dict)
        except ImportError:
            raise ImportError("PyYAML is required for YAML configuration files")
    
    def _load_ini_config(self, config_file: str):
        """Load INI configuration"""
        config = configparser.ConfigParser()
        config.read(config_file)
        
        # Convert INI to nested dictionary
        config_dict = {}
        for section_name in config.sections():
            section = config[section_name]
            config_dict[section_name] = {}
            
            for key, value in section.items():
                # Try to parse value as appropriate type
                parsed_value = self._parse_ini_value(value)
                config_dict[section_name][key] = parsed_value
        
        self._update_config_from_dict(config_dict)
    
    def _parse_ini_value(self, value: str) -> Union[str, int, float, bool, List]:
        """Parse INI value to appropriate Python type"""
        # Boolean values
        if value.lower() in ['true', 'yes', '1', 'on']:
            return True
        elif value.lower() in ['false', 'no', '0', 'off']:
            return False
        
        # List values (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String value
        return value
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                if isinstance(value, dict):
                    # Handle nested configuration
                    nested_config = getattr(self.config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self.config, key, value)
    
    def save_config(self, config_file: str, format_type: str = 'auto'):
        """
        Save current configuration to file.
        
        Args:
            config_file (str): Path to save configuration
            format_type (str): Format type ('json', 'yaml', 'ini', or 'auto')
        """
        config_path = Path(config_file)
        
        # Auto-detect format if not specified
        if format_type == 'auto':
            if config_path.suffix.lower() == '.json':
                format_type = 'json'
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                format_type = 'yaml'
            elif config_path.suffix.lower() in ['.ini', '.cfg']:
                format_type = 'ini'
            else:
                format_type = 'json'  # Default to JSON
        
        # Create directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in appropriate format
        if format_type == 'json':
            self._save_json_config(config_file)
        elif format_type == 'yaml':
            self._save_yaml_config(config_file)
        elif format_type == 'ini':
            self._save_ini_config(config_file)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _save_json_config(self, config_file: str):
        """Save JSON configuration"""
        config_dict = asdict(self.config)
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=4, cls=NumpyEncoder)
    
    def _save_yaml_config(self, config_file: str):
        """Save YAML configuration"""
        try:
            config_dict = asdict(self.config)
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except ImportError:
            raise ImportError("PyYAML is required for YAML configuration files")
    
    def _save_ini_config(self, config_file: str):
        """Save INI configuration"""
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
                # Top-level scalar values go in a DEFAULT section
                if 'DEFAULT' not in config:
                    config['DEFAULT'] = {}
                config['DEFAULT'][section_name] = str(section_data)
        
        with open(config_file, 'w') as f:
            config.write(f)
    
    def interactive_setup(self):
        """Interactive configuration setup"""
        print("=== Interactive Configuration Setup ===\n")
        
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
        
        parallel_choice = input(f"Enable parallel processing? [{'y' if self.config.processing.parallel_processing else 'n'}]: ").strip().lower()
        if parallel_choice in ['y', 'yes']:
            self.config.processing.parallel_processing = True
        elif parallel_choice in ['n', 'no']:
            self.config.processing.parallel_processing = False
        
        # Real-time settings
        print("\n3. Real-time Settings")
        realtime_choice = input(f"Enable real-time processing? [{'y' if self.config.processing.realtime_enabled else 'n'}]: ").strip().lower()
        if realtime_choice in ['y', 'yes']:
            self.config.processing.realtime_enabled = True
            
            camera_input = input(f"Camera index [{self.config.processing.camera_index}]: ").strip()
            if camera_input:
                try:
                    self.config.processing.camera_index = int(camera_input)
                except ValueError:
                    pass
        elif realtime_choice in ['n', 'no']:
            self.config.processing.realtime_enabled = False
        
        # Detection settings
        print("\n4. Detection Settings")
        confidence_input = input(f"Confidence threshold [{self.config.detection.confidence_threshold}]: ").strip()
        if confidence_input:
            try:
                self.config.detection.confidence_threshold = float(confidence_input)
            except ValueError:
                pass
        
        # Visualization settings
        print("\n5. Visualization Settings")
        overlay_choice = input(f"Generate overlay visualizations? [{'y' if self.config.visualization.generate_overlays else 'n'}]: ").strip().lower()
        if overlay_choice in ['y', 'yes']:
            self.config.visualization.generate_overlays = True
        elif overlay_choice in ['n', 'no']:
            self.config.visualization.generate_overlays = False
        
        print("\nConfiguration setup complete!")
    
    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Check directories
        if not self.config.input_dir:
            errors.append("Input directory not specified")
        
        if not self.config.output_dir:
            errors.append("Output directory not specified")
        
        # Check processing settings
        if self.config.processing.max_workers < 1:
            errors.append("Max workers must be at least 1")
        
        if self.config.processing.batch_size < 1:
            errors.append("Batch size must be at least 1")
        
        if not (0 < self.config.processing.image_resize_factor <= 2.0):
            errors.append("Image resize factor must be between 0 and 2.0")
        
        # Check detection settings
        if not (0.0 <= self.config.detection.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        if self.config.detection.min_defect_size < 1:
            errors.append("Minimum defect size must be at least 1")
        
        if self.config.detection.max_defect_size <= self.config.detection.min_defect_size:
            errors.append("Maximum defect size must be greater than minimum defect size")
        
        # Check visualization settings
        if not (0.0 <= self.config.visualization.overlay_opacity <= 1.0):
            errors.append("Overlay opacity must be between 0.0 and 1.0")
        
        return errors
    
    def get_config(self) -> SystemConfig:
        """Get the current configuration"""
        return self.config
    
    def print_config(self):
        """Print the current configuration in a readable format"""
        print("=== Current Configuration ===")
        print(f"Input Directory: {self.config.input_dir}")
        print(f"Output Directory: {self.config.output_dir}")
        print(f"Model Directory: {self.config.model_dir}")
        print(f"Debug Mode: {self.config.debug_mode}")
        
        print("\nProcessing:")
        print(f"  ML Enabled: {self.config.processing.ml_enabled}")
        print(f"  Parallel Processing: {self.config.processing.parallel_processing}")
        print(f"  Max Workers: {self.config.processing.max_workers}")
        print(f"  Real-time Enabled: {self.config.processing.realtime_enabled}")
        
        print("\nDetection:")
        print(f"  Confidence Threshold: {self.config.detection.confidence_threshold}")
        print(f"  Min Defect Size: {self.config.detection.min_defect_size}")
        print(f"  Max Defect Size: {self.config.detection.max_defect_size}")
        
        print("\nVisualization:")
        print(f"  Generate Overlays: {self.config.visualization.generate_overlays}")
        print(f"  Generate Heatmaps: {self.config.visualization.generate_heatmaps}")
        print(f"  Overlay Opacity: {self.config.visualization.overlay_opacity}")


def main():
    """Command line interface for configuration management"""
    parser = argparse.ArgumentParser(description='Configuration Management for Fiber Optic Analysis')
    parser.add_argument('--config', help='Configuration file to load')
    parser.add_argument('--save', help='Save configuration to file')
    parser.add_argument('--format', choices=['json', 'yaml', 'ini'], default='json',
                       help='Configuration file format (default: json)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive configuration setup')
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration')
    parser.add_argument('--print', action='store_true',
                       help='Print current configuration')
    
    args = parser.parse_args()
    
    # Create configuration manager
    config_manager = ConfigurationManager(args.config)
    
    # Interactive setup
    if args.interactive:
        config_manager.interactive_setup()
    
    # Validate configuration
    if args.validate:
        errors = config_manager.validate_config()
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("Configuration is valid")
    
    # Print configuration
    if args.print:
        config_manager.print_config()
    
    # Save configuration
    if args.save:
        config_manager.save_config(args.save, args.format)
        print(f"Configuration saved to: {args.save}")
    
    return 0


if __name__ == "__main__":
    exit(main())
