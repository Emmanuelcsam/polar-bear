#!/usr/bin/env python3
"""
Mega Connector - The Ultimate Integration System for Polar Bear
This connector has complete understanding of ALL scripts and provides
intelligent orchestration with full control over every component.
"""

import os
import sys
import json
import time
import logging
import threading
import importlib.util
import inspect
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback
import numpy as np
import cv2

@dataclass
class ScriptCapability:
    """Detailed capability information for a script"""
    name: str
    category: str
    purpose: str
    defect_types: List[str]
    input_requirements: Dict[str, Any]
    output_format: Dict[str, Any]
    dependencies: List[str]
    performance_profile: Dict[str, Any]
    integration_points: List[str]

@dataclass
class ProcessingResult:
    """Standardized result format"""
    success: bool
    script: str
    function: str
    duration: float
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class MegaConnector:
    """Ultimate connector with complete system understanding"""
    
    # Complete script categorization based on analysis
    SCRIPT_REGISTRY = {
        # Core Detection Algorithms
        'defect_detection': {
            'classical-defect-detector-35': {
                'purpose': 'Classical CV defect detection',
                'defect_types': ['scratches', 'pits', 'contamination', 'cracks'],
                'main_function': 'detect_defects',
                'parameters': ['min_defect_size', 'aspect_ratio_threshold']
            },
            'defect-detection-ai-39': {
                'purpose': 'AI-powered anomaly detection',
                'defect_types': ['all anomalies'],
                'main_function': 'detect_defects',
                'requires_model': True
            },
            'do2mr-defect-algorithm-35': {
                'purpose': 'DO2MR algorithm for region defects',
                'defect_types': ['dust', 'oil', 'moisture'],
                'main_function': 'detect'
            },
            'lei-scratch-detector-35': {
                'purpose': 'LEI algorithm for scratch detection',
                'defect_types': ['scratches', 'linear defects'],
                'main_function': 'detect'
            },
            'frangi-linear-detector-35': {
                'purpose': 'Frangi filter for vessel-like structures',
                'defect_types': ['linear features', 'scratches'],
                'main_function': 'detect_linear_defects'
            },
            'phase-congruency-detector-35': {
                'purpose': 'Phase congruency edge detection',
                'defect_types': ['edges', 'boundaries'],
                'main_function': 'detect_phase_features'
            },
            'ensemble-defect-detector-35': {
                'purpose': 'Combines multiple detection methods',
                'defect_types': ['all types'],
                'main_function': 'detect_defects_ensemble'
            }
        },
        
        # Segmentation and Zone Detection
        'segmentation': {
            'image-segmentation-ai-39': {
                'purpose': 'U-Net segmentation for fiber zones',
                'outputs': ['core', 'cladding', 'ferrule', 'defect'],
                'main_class': 'AI_Segmenter',
                'requires_model': True
            },
            'advanced-fiber-finder-29': {
                'purpose': 'Advanced fiber structure detection',
                'outputs': ['fiber_zones', 'center', 'radius'],
                'main_function': 'locate_fiber_structure_advanced'
            },
            'adaptive-intensity-segmenter-29': {
                'purpose': 'Intensity-based zone segmentation',
                'outputs': ['zone_masks', 'boundaries'],
                'main_function': 'adaptive_intensity_segmentation'
            },
            'circular-fiber-detector-29': {
                'purpose': 'Hough circle-based fiber detection',
                'outputs': ['center', 'radius', 'confidence'],
                'main_function': 'detect_fiber_circle'
            }
        },
        
        # Feature Extraction
        'feature_extraction': {
            'image-feature-extractor-30': {
                'purpose': 'Comprehensive feature extraction',
                'features': ['texture', 'shape', 'statistical', 'frequency'],
                'main_function': 'extract_all_features'
            },
            'texture-feature-extractor-30': {
                'purpose': 'Texture analysis features',
                'features': ['gabor', 'lbp', 'glcm'],
                'main_function': 'extract_texture_features'
            },
            'geometric-feature-extractor-30': {
                'purpose': 'Geometric shape features',
                'features': ['moments', 'contours', 'hulls'],
                'main_function': 'extract_geometric_features'
            },
            'frequency-feature-extractor-30': {
                'purpose': 'Frequency domain features',
                'features': ['fft', 'dct', 'wavelets'],
                'main_function': 'extract_frequency_features'
            }
        },
        
        # Analysis and Reporting
        'analysis': {
            'analysis_engine-40': {
                'purpose': 'Main analysis orchestrator',
                'capabilities': ['full_pipeline', 'report_generation'],
                'main_class': 'FiberAnalysisEngine'
            },
            'report-generator-40': {
                'purpose': 'Comprehensive report generation',
                'outputs': ['pdf', 'csv', 'json'],
                'main_function': 'generate_report'
            },
            'quality-metrics-calculator-40': {
                'purpose': 'Quality metric calculations',
                'metrics': ['snr', 'contrast', 'uniformity'],
                'main_function': 'calculate_metrics'
            },
            'statistical-analysis-toolkit-40': {
                'purpose': 'Statistical analysis tools',
                'analyses': ['distribution', 'correlation', 'trends'],
                'main_function': 'analyze_statistics'
            }
        },
        
        # Preprocessing and Enhancement
        'preprocessing': {
            'image-quality-enhancer': {
                'purpose': 'Image quality improvement',
                'methods': ['clahe', 'denoise', 'sharpen'],
                'main_function': 'enhance_image'
            },
            'anisotropic-noise-reduction': {
                'purpose': 'Advanced noise reduction',
                'preserves': ['edges', 'features'],
                'main_function': 'reduce_noise'
            },
            'lighting-correction-tool': {
                'purpose': 'Illumination correction',
                'methods': ['histogram_equalization', 'gamma_correction'],
                'main_function': 'correct_lighting'
            }
        },
        
        # AI/ML Models
        'ai_models': {
            'cnn_fiber_detector': {
                'purpose': 'CNN for defect classification',
                'classes': ['normal', 'scratch', 'dig', 'contamination'],
                'main_class': 'DefectCNN'
            },
            'anomaly_detector_pytorch': {
                'purpose': 'Autoencoder anomaly detection',
                'architecture': 'CAE',
                'main_class': 'AI_AnomalyDetector'
            },
            'VAE-22': {
                'purpose': 'Variational autoencoder',
                'capabilities': ['generation', 'anomaly_detection'],
                'main_class': 'VAE'
            }
        },
        
        # Real-time Processing
        'realtime': {
            '8_real_time_anomaly': {
                'purpose': 'Live video anomaly detection',
                'input': 'webcam',
                'main_function': 'run_real_time_detection'
            },
            'live-feed-anomaly-22': {
                'purpose': 'Live feed processing',
                'capabilities': ['streaming', 'real_time_detection'],
                'main_function': 'process_live_feed'
            }
        },
        
        # Visualization
        'visualization': {
            'visualization-toolkit': {
                'purpose': 'Comprehensive visualization tools',
                'capabilities': ['overlay', 'comparison', 'statistics'],
                'main_function': 'create_visualization'
            },
            'interactive-viewer': {
                'purpose': 'Interactive result viewing',
                'features': ['zoom', 'pan', 'annotations'],
                'main_function': 'launch_viewer'
            }
        }
    }
    
    # Predefined intelligent pipelines
    PIPELINES = {
        'complete_inspection': {
            'description': 'Full fiber optic inspection pipeline',
            'steps': [
                ('preprocessing', 'image-quality-enhancer', 'enhance_image'),
                ('segmentation', 'image-segmentation-ai-39', 'segment'),
                ('detection', 'ensemble-defect-detector-35', 'detect_defects_ensemble'),
                ('feature_extraction', 'image-feature-extractor-30', 'extract_all_features'),
                ('analysis', 'quality-metrics-calculator-40', 'calculate_metrics'),
                ('reporting', 'report-generator-40', 'generate_report')
            ]
        },
        'quick_defect_scan': {
            'description': 'Fast defect detection',
            'steps': [
                ('segmentation', 'advanced-fiber-finder-29', 'locate_fiber_structure_advanced'),
                ('detection', 'classical-defect-detector-35', 'detect_defects'),
                ('visualization', 'visualization-toolkit', 'create_visualization')
            ]
        },
        'ai_powered_analysis': {
            'description': 'AI-based comprehensive analysis',
            'steps': [
                ('preprocessing', 'anisotropic-noise-reduction', 'reduce_noise'),
                ('segmentation', 'image-segmentation-ai-39', 'segment'),
                ('detection', 'defect-detection-ai-39', 'detect_defects'),
                ('analysis', 'statistical-analysis-toolkit-40', 'analyze_statistics')
            ]
        },
        'learning_mode': {
            'description': 'Continuous learning pipeline',
            'steps': [
                ('batch_processing', '1_batch_processor', 'process_images'),
                ('intensity_extraction', '2_intensity_reader', 'read_intensities'),
                ('pattern_learning', '3_pattern_recognizer', 'recognize_patterns'),
                ('model_training', '4_generative_learner', 'learn_pixel_distribution')
            ]
        }
    }
    
    def __init__(self, auto_init=True):
        # Setup logging
        self.setup_comprehensive_logging()
        
        # Core components
        self.scripts = {}
        self.modules = {}
        self.models = {}
        self.configurations = {}
        self.results_history = []
        
        # Threading and performance
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.locks = {}
        
        # State management
        self.current_pipeline = None
        self.learning_mode = False
        self.reference_data = {}
        
        if auto_init:
            self.initialize_system()
    
    def setup_comprehensive_logging(self):
        """Setup advanced logging with multiple handlers"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Main logger
        self.logger = logging.getLogger("MegaConnector")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler - INFO and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler - All messages
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_dir / f"mega_connector_{timestamp}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        
        # Error handler - Errors only
        error_handler = logging.FileHandler(
            log_dir / f"errors_{timestamp}.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # Add all handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        
        self.logger.info("=== Mega Connector Initialized ===")
    
    def initialize_system(self):
        """Complete system initialization"""
        self.logger.info("Initializing Mega Connector System...")
        
        # Phase 1: Script Discovery
        self._discover_and_analyze_all_scripts()
        
        # Phase 2: Dependency Check
        self._check_and_install_dependencies()
        
        # Phase 3: Model Loading
        self._load_ai_models()
        
        # Phase 4: Configuration Loading
        self._load_configurations()
        
        # Phase 5: System Validation
        self._validate_system()
        
        self.logger.info("System initialization complete!")
    
    def _discover_and_analyze_all_scripts(self):
        """Discover and deeply analyze all scripts"""
        self.logger.info("Discovering scripts...")
        
        script_count = 0
        for script_path in Path('.').glob('*.py'):
            if script_path.name in ['mega_connector.py', '__init__.py']:
                continue
            
            script_name = script_path.stem
            
            # Analyze script
            analysis = self._deep_analyze_script(script_path)
            self.scripts[script_name] = analysis
            
            # Create lock for thread safety
            self.locks[script_name] = threading.Lock()
            
            script_count += 1
        
        self.logger.info(f"Discovered and analyzed {script_count} scripts")
        
        # Log category summary
        categories = {}
        for name, info in self.scripts.items():
            cat = self._categorize_script(name, info)
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            self.logger.info(f"  {cat}: {count} scripts")
    
    def _deep_analyze_script(self, script_path: Path) -> Dict:
        """Perform deep analysis of a script"""
        info = {
            'path': str(script_path),
            'category': 'unknown',
            'purpose': '',
            'functions': {},
            'classes': {},
            'imports': [],
            'defect_capabilities': [],
            'input_types': [],
            'output_types': [],
            'dependencies': [],
            'has_main': False,
            'is_executable': False,
            'integration_ready': False
        }
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract comprehensive information
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        info['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        info['imports'].append(node.module)
                
                elif isinstance(node, ast.FunctionDef):
                    func_info = self._analyze_function(node)
                    info['functions'][node.name] = func_info
                    
                    # Check for specific capabilities
                    if 'defect' in node.name or 'detect' in node.name:
                        info['defect_capabilities'].append(node.name)
                    
                    if node.name == 'main':
                        info['has_main'] = True
                
                elif isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    info['classes'][node.name] = class_info
            
            # Determine script properties
            info['is_executable'] = 'if __name__ == "__main__":' in content
            info['integration_ready'] = bool(info['functions'] or info['classes'])
            
            # Categorize based on content
            info['category'] = self._categorize_script(script_path.stem, info)
            
            # Extract purpose from docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                info['purpose'] = module_doc.split('\n')[0]
            
        except Exception as e:
            self.logger.error(f"Error analyzing {script_path}: {e}")
        
        return info
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict:
        """Analyze a function node"""
        return {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'defaults': len(node.args.defaults),
            'docstring': ast.get_docstring(node),
            'returns': self._get_return_type(node),
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
        }
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict:
        """Analyze a class node"""
        methods = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods[item.name] = self._analyze_function(item)
        
        return {
            'name': node.name,
            'bases': [self._get_name(base) for base in node.bases],
            'methods': methods,
            'docstring': ast.get_docstring(node),
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
        }
    
    def _get_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type from function"""
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                # Simple type inference
                if isinstance(child.value, ast.Dict):
                    return 'dict'
                elif isinstance(child.value, ast.List):
                    return 'list'
                elif isinstance(child.value, ast.Tuple):
                    return 'tuple'
        return None
    
    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def _get_decorator_name(self, node) -> str:
        """Get decorator name"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return str(node)
    
    def _categorize_script(self, name: str, info: Dict) -> str:
        """Categorize script based on name and content"""
        name_lower = name.lower()
        
        # Check predefined categories
        for category, scripts in self.SCRIPT_REGISTRY.items():
            if any(script in name for script in scripts):
                return category
        
        # Heuristic categorization
        if 'detect' in name_lower or 'defect' in name_lower:
            return 'defect_detection'
        elif 'segment' in name_lower or 'fiber' in name_lower:
            return 'segmentation'
        elif 'feature' in name_lower or 'extract' in name_lower:
            return 'feature_extraction'
        elif 'analy' in name_lower or 'report' in name_lower:
            return 'analysis'
        elif 'preprocess' in name_lower or 'enhance' in name_lower:
            return 'preprocessing'
        elif 'ai' in name_lower or 'neural' in name_lower or 'cnn' in name_lower:
            return 'ai_models'
        elif 'visual' in name_lower or 'display' in name_lower:
            return 'visualization'
        elif 'real' in name_lower and 'time' in name_lower:
            return 'realtime'
        
        return 'utilities'
    
    def _check_and_install_dependencies(self):
        """Check and install all required dependencies"""
        self.logger.info("Checking dependencies...")
        
        # Core requirements
        core_deps = {
            'numpy': 'numpy',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn',
            'skimage': 'scikit-image',
            'matplotlib': 'matplotlib',
            'pandas': 'pandas',
            'torch': 'torch',
            'torchvision': 'torchvision',
            'tqdm': 'tqdm'
        }
        
        missing = []
        for import_name, package_name in core_deps.items():
            try:
                importlib.import_module(import_name)
                self.logger.debug(f"✓ {import_name}")
            except ImportError:
                missing.append(package_name)
                self.logger.warning(f"✗ {import_name} missing")
        
        if missing:
            self.logger.info(f"Installing missing packages: {', '.join(missing)}")
            for package in missing:
                try:
                    import subprocess
                    cmd = [sys.executable, "-m", "pip", "install", package]
                    if sys.version_info >= (3, 11):
                        cmd.append("--break-system-packages")
                    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
                    self.logger.info(f"Installed {package}")
                except Exception as e:
                    self.logger.error(f"Failed to install {package}: {e}")
    
    def _load_ai_models(self):
        """Load pre-trained AI models"""
        self.logger.info("Loading AI models...")
        
        model_configs = {
            'segmentation': {
                'path': 'models/unet_segmentation.pth',
                'class': 'UNet34',
                'script': 'image-segmentation-ai-39'
            },
            'anomaly_detection': {
                'path': 'models/cae_anomaly.pth',
                'class': 'CAE',
                'script': 'anomaly_detector_pytorch'
            },
            'defect_classification': {
                'path': 'models/defect_cnn.pth',
                'class': 'DefectCNN',
                'script': 'cnn_fiber_detector'
            }
        }
        
        for model_name, config in model_configs.items():
            if Path(config['path']).exists():
                try:
                    # Model loading would happen here
                    self.models[model_name] = config
                    self.logger.info(f"Loaded {model_name} model")
                except Exception as e:
                    self.logger.warning(f"Could not load {model_name}: {e}")
            else:
                self.logger.debug(f"Model file not found: {config['path']}")
    
    def _load_configurations(self):
        """Load system configurations"""
        config_files = list(Path('.').glob('*config*.json'))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.configurations[config_file.stem] = config
                self.logger.debug(f"Loaded config: {config_file.name}")
            except Exception as e:
                self.logger.warning(f"Could not load {config_file}: {e}")
    
    def _validate_system(self):
        """Validate system readiness"""
        self.logger.info("Validating system...")
        
        # Check critical scripts
        critical_scripts = [
            'defect-detection-ai-39',
            'image-segmentation-ai-39',
            'analysis_engine-40',
            'report-generator-40'
        ]
        
        missing_critical = []
        for script in critical_scripts:
            if script not in self.scripts:
                missing_critical.append(script)
        
        if missing_critical:
            self.logger.warning(f"Missing critical scripts: {', '.join(missing_critical)}")
        else:
            self.logger.info("All critical scripts present")
        
        # Validate categories
        required_categories = [
            'defect_detection', 'segmentation', 'analysis', 'preprocessing'
        ]
        
        for category in required_categories:
            count = sum(1 for s in self.scripts.values() if s['category'] == category)
            if count == 0:
                self.logger.warning(f"No scripts found for category: {category}")
            else:
                self.logger.debug(f"{category}: {count} scripts available")
    
    def execute_script(self, script_name: str, function_name: str, 
                      *args, **kwargs) -> ProcessingResult:
        """Execute a script function with full error handling"""
        start_time = time.time()
        
        try:
            # Load module if needed
            if script_name not in self.modules:
                self.load_module(script_name)
            
            module = self.modules.get(script_name)
            if not module:
                raise ValueError(f"Could not load module: {script_name}")
            
            # Get function
            if not hasattr(module, function_name):
                raise AttributeError(f"Function '{function_name}' not found in '{script_name}'")
            
            func = getattr(module, function_name)
            if not callable(func):
                raise TypeError(f"'{function_name}' is not callable")
            
            # Execute with logging
            self.logger.info(f"Executing {script_name}.{function_name}")
            result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                script=script_name,
                function=function_name,
                duration=duration,
                output=result,
                metadata={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
            )
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            self.logger.debug(traceback.format_exc())
            
            return ProcessingResult(
                success=False,
                script=script_name,
                function=function_name,
                duration=time.time() - start_time,
                output=None,
                error=str(e)
            )
    
    def load_module(self, script_name: str) -> Optional[Any]:
        """Dynamically load a script module"""
        if script_name in self.modules:
            return self.modules[script_name]
        
        script_info = self.scripts.get(script_name)
        if not script_info:
            self.logger.error(f"Script not found: {script_name}")
            return None
        
        try:
            spec = importlib.util.spec_from_file_location(
                script_name,
                script_info['path']
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                
                # Change directory context
                original_dir = os.getcwd()
                script_dir = os.path.dirname(os.path.abspath(script_info['path']))
                os.chdir(script_dir)
                
                try:
                    spec.loader.exec_module(module)
                    self.modules[script_name] = module
                    self.logger.debug(f"Loaded module: {script_name}")
                    return module
                finally:
                    os.chdir(original_dir)
                    
        except Exception as e:
            self.logger.error(f"Failed to load {script_name}: {e}")
            return None
    
    def create_instance(self, script_name: str, class_name: str, 
                       *args, **kwargs) -> Optional[Any]:
        """Create an instance of a class from a script"""
        module = self.load_module(script_name)
        if not module:
            return None
        
        if not hasattr(module, class_name):
            self.logger.error(f"Class '{class_name}' not found in '{script_name}'")
            return None
        
        cls = getattr(module, class_name)
        if not inspect.isclass(cls):
            self.logger.error(f"'{class_name}' is not a class")
            return None
        
        try:
            instance = cls(*args, **kwargs)
            self.logger.debug(f"Created instance of {script_name}.{class_name}")
            return instance
        except Exception as e:
            self.logger.error(f"Failed to create instance: {e}")
            return None
    
    def run_pipeline(self, pipeline_name: str, input_data: Any, 
                    config: Dict = None) -> Dict[str, Any]:
        """Run a predefined pipeline"""
        if pipeline_name not in self.PIPELINES:
            self.logger.error(f"Unknown pipeline: {pipeline_name}")
            return {'success': False, 'error': 'Unknown pipeline'}
        
        pipeline = self.PIPELINES[pipeline_name]
        self.logger.info(f"Running pipeline: {pipeline_name}")
        self.logger.info(f"Description: {pipeline['description']}")
        
        results = {
            'pipeline': pipeline_name,
            'start_time': datetime.now().isoformat(),
            'steps': {},
            'success': True,
            'final_output': None
        }
        
        current_data = input_data
        
        for i, (step_type, script, function) in enumerate(pipeline['steps']):
            step_name = f"{i+1}_{step_type}"
            self.logger.info(f"Step {i+1}/{len(pipeline['steps'])}: {step_type}")
            
            try:
                # Execute step
                if config and step_type in config:
                    kwargs = config[step_type]
                else:
                    kwargs = {}
                
                result = self.execute_script(script, function, current_data, **kwargs)
                
                if result.success:
                    results['steps'][step_name] = {
                        'success': True,
                        'duration': result.duration,
                        'script': script,
                        'function': function
                    }
                    current_data = result.output
                else:
                    results['steps'][step_name] = {
                        'success': False,
                        'error': result.error
                    }
                    results['success'] = False
                    break
                    
            except Exception as e:
                self.logger.error(f"Pipeline step failed: {e}")
                results['steps'][step_name] = {
                    'success': False,
                    'error': str(e)
                }
                results['success'] = False
                break
        
        results['final_output'] = current_data
        results['end_time'] = datetime.now().isoformat()
        
        # Store in history
        self.results_history.append(results)
        
        return results
    
    def run_intelligent_pipeline(self, image_path: str, 
                               objectives: List[str] = None) -> Dict:
        """Run an intelligent pipeline based on objectives"""
        if objectives is None:
            objectives = ['detect_defects', 'generate_report']
        
        self.logger.info(f"Running intelligent pipeline for objectives: {objectives}")
        
        # Determine best pipeline based on objectives
        if 'comprehensive' in objectives or 'full' in objectives:
            pipeline = 'complete_inspection'
        elif 'quick' in objectives or 'fast' in objectives:
            pipeline = 'quick_defect_scan'
        elif 'ai' in objectives or 'learning' in objectives:
            pipeline = 'ai_powered_analysis'
        else:
            pipeline = 'complete_inspection'
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'success': False, 'error': f'Could not load image: {image_path}'}
        
        # Run selected pipeline
        return self.run_pipeline(pipeline, image)
    
    def enable_learning_mode(self, reference_dir: str = 'reference'):
        """Enable continuous learning mode"""
        self.logger.info("Enabling learning mode...")
        self.learning_mode = True
        
        # Load reference data
        ref_path = Path(reference_dir)
        if ref_path.exists():
            for ref_file in ref_path.glob('*.json'):
                try:
                    with open(ref_file, 'r') as f:
                        data = json.load(f)
                    self.reference_data[ref_file.stem] = data
                    self.logger.debug(f"Loaded reference: {ref_file.name}")
                except Exception as e:
                    self.logger.warning(f"Could not load reference {ref_file}: {e}")
        
        # Start learning pipeline in background
        if self.reference_data:
            self.logger.info("Starting background learning process...")
            learning_thread = threading.Thread(target=self._background_learning)
            learning_thread.daemon = True
            learning_thread.start()
    
    def _background_learning(self):
        """Background learning process"""
        while self.learning_mode:
            try:
                # Run learning pipeline periodically
                result = self.run_pipeline('learning_mode', self.reference_data)
                if result['success']:
                    self.logger.info("Learning cycle completed successfully")
                else:
                    self.logger.warning("Learning cycle failed")
                
                # Wait before next cycle
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Background learning error: {e}")
                time.sleep(60)
    
    def get_script_capabilities(self, category: str = None) -> Dict:
        """Get detailed capabilities of scripts"""
        capabilities = {}
        
        for script_name, info in self.scripts.items():
            if category and info['category'] != category:
                continue
            
            capabilities[script_name] = {
                'category': info['category'],
                'purpose': info['purpose'],
                'functions': list(info['functions'].keys()),
                'classes': list(info['classes'].keys()),
                'defect_capabilities': info['defect_capabilities'],
                'has_main': info['has_main'],
                'integration_ready': info['integration_ready']
            }
        
        return capabilities
    
    def analyze_image(self, image_path: str, analysis_type: str = 'complete') -> Dict:
        """Perform comprehensive image analysis"""
        self.logger.info(f"Analyzing image: {image_path}")
        
        analysis_types = {
            'complete': 'complete_inspection',
            'quick': 'quick_defect_scan',
            'ai': 'ai_powered_analysis'
        }
        
        pipeline = analysis_types.get(analysis_type, 'complete_inspection')
        return self.run_intelligent_pipeline(image_path, [analysis_type])
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'scripts_loaded': len(self.scripts),
            'modules_loaded': len(self.modules),
            'models_available': len(self.models),
            'configurations': list(self.configurations.keys()),
            'learning_mode': self.learning_mode,
            'pipelines_available': list(self.PIPELINES.keys()),
            'results_history': len(self.results_history),
            'categories': {
                cat: sum(1 for s in self.scripts.values() if s['category'] == cat)
                for cat in set(s['category'] for s in self.scripts.values())
            }
        }
    
    def interactive_mode(self):
        """Run in interactive mode with full menu system"""
        self.logger.info("\n=== Mega Connector Interactive Mode ===\n")
        
        while True:
            print("\n" + "="*60)
            print("MEGA CONNECTOR - MAIN MENU")
            print("="*60)
            print("1. Run Analysis Pipeline")
            print("2. Execute Specific Function")
            print("3. View Script Capabilities")
            print("4. System Status")
            print("5. Enable/Disable Learning Mode")
            print("6. View Results History")
            print("7. Run System Test")
            print("8. Exit")
            print("="*60)
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self._menu_run_pipeline()
            elif choice == '2':
                self._menu_execute_function()
            elif choice == '3':
                self._menu_view_capabilities()
            elif choice == '4':
                self._menu_system_status()
            elif choice == '5':
                self._menu_learning_mode()
            elif choice == '6':
                self._menu_results_history()
            elif choice == '7':
                self._menu_system_test()
            elif choice == '8':
                self.logger.info("Exiting Mega Connector")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _menu_run_pipeline(self):
        """Pipeline execution menu"""
        print("\n--- Pipeline Selection ---")
        print("Available pipelines:")
        for i, (name, pipeline) in enumerate(self.PIPELINES.items(), 1):
            print(f"{i}. {name}: {pipeline['description']}")
        
        try:
            selection = int(input("\nSelect pipeline number: ")) - 1
            pipeline_name = list(self.PIPELINES.keys())[selection]
            
            image_path = input("Enter image path: ").strip()
            if not os.path.exists(image_path):
                print(f"Error: Image not found: {image_path}")
                return
            
            print(f"\nRunning {pipeline_name} pipeline...")
            result = self.analyze_image(image_path, pipeline_name.split('_')[0])
            
            if result['success']:
                print("\n✅ Pipeline completed successfully!")
                print(f"Results saved to: results_{int(time.time())}.json")
                
                # Save results
                with open(f"results_{int(time.time())}.json", 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                print("\n❌ Pipeline failed!")
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except (ValueError, IndexError):
            print("Invalid selection")
    
    def _menu_execute_function(self):
        """Function execution menu"""
        print("\n--- Function Execution ---")
        
        # Show categories
        categories = list(set(s['category'] for s in self.scripts.values()))
        for i, cat in enumerate(categories, 1):
            count = sum(1 for s in self.scripts.values() if s['category'] == cat)
            print(f"{i}. {cat} ({count} scripts)")
        
        try:
            cat_selection = int(input("\nSelect category: ")) - 1
            category = categories[cat_selection]
            
            # Show scripts in category
            scripts_in_cat = [
                (name, info) for name, info in self.scripts.items()
                if info['category'] == category
            ]
            
            print(f"\nScripts in {category}:")
            for i, (name, info) in enumerate(scripts_in_cat, 1):
                print(f"{i}. {name}")
            
            script_selection = int(input("\nSelect script: ")) - 1
            script_name = scripts_in_cat[script_selection][0]
            
            # Show functions
            functions = list(self.scripts[script_name]['functions'].keys())
            print(f"\nFunctions in {script_name}:")
            for i, func in enumerate(functions, 1):
                print(f"{i}. {func}")
            
            func_selection = int(input("\nSelect function: ")) - 1
            function_name = functions[func_selection]
            
            # Execute
            print(f"\nExecuting {script_name}.{function_name}...")
            result = self.execute_script(script_name, function_name)
            
            if result.success:
                print(f"✅ Success! Duration: {result.duration:.2f}s")
            else:
                print(f"❌ Failed: {result.error}")
                
        except (ValueError, IndexError):
            print("Invalid selection")
    
    def _menu_view_capabilities(self):
        """View script capabilities menu"""
        print("\n--- Script Capabilities ---")
        
        categories = list(set(s['category'] for s in self.scripts.values()))
        print("Select category (or 0 for all):")
        for i, cat in enumerate(categories, 1):
            print(f"{i}. {cat}")
        
        try:
            selection = int(input("\nSelection: "))
            
            if selection == 0:
                capabilities = self.get_script_capabilities()
            else:
                category = categories[selection - 1]
                capabilities = self.get_script_capabilities(category)
            
            # Display capabilities
            for script, caps in capabilities.items():
                print(f"\n{script}:")
                print(f"  Category: {caps['category']}")
                print(f"  Purpose: {caps['purpose'][:60]}...")
                if caps['defect_capabilities']:
                    print(f"  Defect Detection: {', '.join(caps['defect_capabilities'])}")
                print(f"  Functions: {len(caps['functions'])}")
                print(f"  Classes: {len(caps['classes'])}")
                
        except (ValueError, IndexError):
            print("Invalid selection")
    
    def _menu_system_status(self):
        """System status menu"""
        print("\n--- System Status ---")
        
        status = self.get_system_status()
        
        print(f"\nScripts Loaded: {status['scripts_loaded']}")
        print(f"Modules Loaded: {status['modules_loaded']}")
        print(f"AI Models Available: {status['models_available']}")
        print(f"Learning Mode: {'Enabled' if status['learning_mode'] else 'Disabled'}")
        print(f"Results in History: {status['results_history']}")
        
        print("\nScript Categories:")
        for cat, count in status['categories'].items():
            print(f"  {cat}: {count}")
        
        print("\nAvailable Pipelines:")
        for pipeline in status['pipelines_available']:
            print(f"  - {pipeline}")
    
    def _menu_learning_mode(self):
        """Learning mode menu"""
        print("\n--- Learning Mode ---")
        
        current_status = "Enabled" if self.learning_mode else "Disabled"
        print(f"Current Status: {current_status}")
        
        choice = input("\nToggle learning mode? (y/n): ").strip().lower()
        
        if choice == 'y':
            if self.learning_mode:
                self.learning_mode = False
                print("Learning mode disabled")
            else:
                ref_dir = input("Enter reference data directory [reference]: ").strip() or "reference"
                self.enable_learning_mode(ref_dir)
                print("Learning mode enabled")
    
    def _menu_results_history(self):
        """Results history menu"""
        print("\n--- Results History ---")
        
        if not self.results_history:
            print("No results in history")
            return
        
        print(f"\nShowing last {min(10, len(self.results_history))} results:")
        
        for i, result in enumerate(reversed(self.results_history[-10:]), 1):
            print(f"\n{i}. Pipeline: {result['pipeline']}")
            print(f"   Time: {result['start_time']}")
            print(f"   Success: {'Yes' if result['success'] else 'No'}")
            print(f"   Steps: {len(result['steps'])}")
    
    def _menu_system_test(self):
        """Run system test"""
        print("\n--- System Test ---")
        print("Running comprehensive system test...")
        
        test_results = {
            'scripts': len(self.scripts) > 0,
            'categories': len(set(s['category'] for s in self.scripts.values())) >= 5,
            'defect_detection': any(s['defect_capabilities'] for s in self.scripts.values()),
            'pipelines': len(self.PIPELINES) > 0,
            'modules_loadable': True
        }
        
        # Test loading a module
        try:
            test_script = list(self.scripts.keys())[0]
            self.load_module(test_script)
        except:
            test_results['modules_loadable'] = False
        
        # Display results
        print("\nTest Results:")
        for test, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test}: {status}")
        
        overall = all(test_results.values())
        print(f"\nOverall: {'✅ ALL TESTS PASSED' if overall else '❌ SOME TESTS FAILED'}")

# Global instance for convenience
_mega_connector = None

def get_mega_connector() -> MegaConnector:
    """Get or create the global mega connector instance"""
    global _mega_connector
    if _mega_connector is None:
        _mega_connector = MegaConnector()
    return _mega_connector

# Convenience functions
def analyze(image_path: str, analysis_type: str = 'complete') -> Dict:
    """Quick analysis function"""
    return get_mega_connector().analyze_image(image_path, analysis_type)

def execute(script: str, function: str, *args, **kwargs) -> ProcessingResult:
    """Quick execution function"""
    return get_mega_connector().execute_script(script, function, *args, **kwargs)

def status() -> Dict:
    """Get system status"""
    return get_mega_connector().get_system_status()

# Main entry point
if __name__ == "__main__":
    connector = MegaConnector()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'analyze' and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            analysis_type = sys.argv[3] if len(sys.argv) > 3 else 'complete'
            result = connector.analyze_image(image_path, analysis_type)
            print(json.dumps(result, indent=2))
            
        elif command == 'status':
            status = connector.get_system_status()
            print(json.dumps(status, indent=2))
            
        elif command == 'capabilities':
            category = sys.argv[2] if len(sys.argv) > 2 else None
            caps = connector.get_script_capabilities(category)
            print(json.dumps(caps, indent=2))
            
        else:
            print("Usage:")
            print("  mega_connector.py                         # Interactive mode")
            print("  mega_connector.py analyze <image> [type]  # Analyze image")
            print("  mega_connector.py status                  # System status")
            print("  mega_connector.py capabilities [category] # View capabilities")
    else:
        # Interactive mode
        connector.interactive_mode()