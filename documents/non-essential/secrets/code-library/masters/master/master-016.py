#!/usr/bin/env python3
"""
Polar Bear Brain - Unified Intelligence System for Defect Detection
This is the main brain that orchestrates all scripts and provides intelligent
decision-making for image analysis and defect detection.
"""

import os
import sys
import json
import time
import logging
import threading
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from datetime import datetime
import hashlib

# Setup comprehensive logging
class BrainLogger:
    """Advanced logging system with file and console output"""
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamp-based log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"polar_bear_brain_{timestamp}.log"
        
        # Setup main logger
        self.logger = logging.getLogger("PolarBearBrain")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with INFO level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with DEBUG level
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S.%f"
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {self.log_file}")
    
    def get_logger(self):
        return self.logger

class DependencyManager:
    """Automatic dependency detection and installation"""
    def __init__(self, logger):
        self.logger = logger
        self.installed_packages = set()
        self._check_pip_version()
    
    def _check_pip_version(self):
        """Ensure pip is up to date"""
        try:
            self.logger.info("Checking pip version...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True, text=True
            )
            self.logger.debug(f"Pip version: {result.stdout.strip()}")
            
            # Update pip if needed
            self.logger.info("Updating pip to latest version...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True
            )
        except Exception as e:
            self.logger.warning(f"Could not update pip: {e}")
    
    def check_and_install(self, package_name: str, import_name: str = None):
        """Check if package is installed, install if not"""
        if import_name is None:
            import_name = package_name.split('[')[0].split('==')[0].split('>=')[0]
        
        if import_name in self.installed_packages:
            return True
        
        try:
            importlib.import_module(import_name)
            self.installed_packages.add(import_name)
            self.logger.debug(f"Package {import_name} already installed")
            return True
        except ImportError:
            self.logger.info(f"Installing {package_name}...")
            try:
                cmd = [sys.executable, "-m", "pip", "install", package_name]
                # Add --break-system-packages for newer pip versions
                if sys.version_info >= (3, 11):
                    cmd.append("--break-system-packages")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info(f"Successfully installed {package_name}")
                    self.installed_packages.add(import_name)
                    return True
                else:
                    self.logger.error(f"Failed to install {package_name}: {result.stderr}")
                    return False
            except Exception as e:
                self.logger.error(f"Error installing {package_name}: {e}")
                return False
    
    def ensure_core_dependencies(self):
        """Ensure all core dependencies are installed"""
        core_deps = [
            ("numpy", "numpy"),
            ("opencv-python", "cv2"),
            ("Pillow", "PIL"),
            ("scipy", "scipy"),
            ("scikit-image", "skimage"),
            ("scikit-learn", "sklearn"),
            ("matplotlib", "matplotlib"),
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("tqdm", "tqdm"),
            ("pandas", "pandas"),
            ("psutil", "psutil"),
            ("requests", "requests")
        ]
        
        self.logger.info("Checking core dependencies...")
        for package, import_name in core_deps:
            self.check_and_install(package, import_name)

class ConfigurationManager:
    """Interactive configuration system"""
    def __init__(self, logger, config_file="polar_bear_brain_config.json"):
        self.logger = logger
        self.config_file = Path(config_file)
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self):
        """Load existing config or create new one interactively"""
        if self.config_file.exists():
            self.logger.info(f"Loading configuration from {self.config_file}")
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            self.logger.info("No configuration found. Starting interactive setup...")
            return self._interactive_setup()
    
    def _interactive_setup(self):
        """Interactive configuration setup"""
        config = {}
        
        print("\n=== Polar Bear Brain Configuration ===\n")
        
        # Processing mode
        print("Select processing mode:")
        print("1. Real-time video processing")
        print("2. Batch image processing")
        print("3. Both (hybrid mode)")
        mode_choice = input("Enter choice [1-3] (default: 3): ").strip() or "3"
        
        mode_map = {"1": "realtime", "2": "batch", "3": "hybrid"}
        config['processing_mode'] = mode_map.get(mode_choice, "hybrid")
        
        # Input source
        if config['processing_mode'] in ['realtime', 'hybrid']:
            config['video_source'] = input("Enter video source (0 for webcam, or file path) [0]: ").strip() or "0"
            try:
                config['video_source'] = int(config['video_source'])
            except ValueError:
                pass
        
        # Image directories
        if config['processing_mode'] in ['batch', 'hybrid']:
            config['input_dir'] = input("Enter input image directory [./input]: ").strip() or "./input"
            config['output_dir'] = input("Enter output directory [./output]: ").strip() or "./output"
            config['reference_dir'] = input("Enter reference data directory [./reference]: ").strip() or "./reference"
        
        # Processing options
        config['enable_ai'] = input("Enable AI-powered detection? [Y/n]: ").strip().lower() != 'n'
        config['enable_visualization'] = input("Enable result visualization? [Y/n]: ").strip().lower() != 'n'
        config['save_intermediate'] = input("Save intermediate results? [y/N]: ").strip().lower() == 'y'
        
        # Detection thresholds
        print("\nDetection sensitivity (1-10, where 10 is most sensitive)")
        sensitivity = input("Enter sensitivity [7]: ").strip() or "7"
        try:
            config['sensitivity'] = min(10, max(1, int(sensitivity)))
        except ValueError:
            config['sensitivity'] = 7
        
        # Performance options
        config['max_workers'] = input("Maximum parallel workers [4]: ").strip() or "4"
        try:
            config['max_workers'] = int(config['max_workers'])
        except ValueError:
            config['max_workers'] = 4
        
        config['use_gpu'] = input("Use GPU acceleration if available? [Y/n]: ").strip().lower() != 'n'
        
        # Learning mode
        config['learning_mode'] = input("Enable continuous learning? [Y/n]: ").strip().lower() != 'n'
        
        # Save configuration
        self._save_config(config)
        return config
    
    def _save_config(self, config):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        self.logger.info(f"Configuration saved to {self.config_file}")
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def update(self, key, value):
        """Update configuration value"""
        self.config[key] = value
        self._save_config(self.config)

class ScriptRegistry:
    """Registry of all available scripts and their capabilities"""
    def __init__(self, logger):
        self.logger = logger
        self.scripts = {}
        self.categories = {
            'segmentation': [],
            'detection': [],
            'analysis': [],
            'preprocessing': [],
            'feature_extraction': [],
            'ai_models': [],
            'utilities': [],
            'visualization': []
        }
        self._discover_scripts()
    
    def _discover_scripts(self):
        """Discover and categorize all scripts"""
        self.logger.info("Discovering scripts...")
        
        # Define script patterns for categorization
        patterns = {
            'segmentation': ['segment', 'fiber-analysis', 'fiber-finder', 'circle-detector'],
            'detection': ['defect', 'anomaly', 'scratch', 'detector'],
            'analysis': ['analyzer', 'analysis', 'report', 'metric'],
            'preprocessing': ['preprocess', 'enhance', 'filter', 'noise', 'prepare'],
            'feature_extraction': ['feature', 'extract', 'descriptor'],
            'ai_models': ['ai-', 'neural', 'cnn', 'vae', 'learning'],
            'utilities': ['util', 'helper', 'tool', 'manager'],
            'visualization': ['visual', 'display', 'viewer', 'plot']
        }
        
        # Scan current directory for Python files
        for script_path in Path('.').glob('*.py'):
            if script_path.name in ['polar_bear_brain.py', '__init__.py']:
                continue
            
            script_name = script_path.stem
            categorized = False
            
            # Try to categorize based on name patterns
            for category, keywords in patterns.items():
                if any(keyword in script_name.lower() for keyword in keywords):
                    self.categories[category].append(script_name)
                    categorized = True
                    break
            
            if not categorized:
                self.categories['utilities'].append(script_name)
            
            # Store script info
            self.scripts[script_name] = {
                'path': str(script_path),
                'module': None,
                'capabilities': self._analyze_script_capabilities(script_path)
            }
        
        # Log discovery results
        total_scripts = len(self.scripts)
        self.logger.info(f"Discovered {total_scripts} scripts")
        for category, scripts in self.categories.items():
            if scripts:
                self.logger.debug(f"  {category}: {len(scripts)} scripts")
    
    def _analyze_script_capabilities(self, script_path):
        """Analyze what a script can do"""
        capabilities = {
            'has_main': False,
            'classes': [],
            'functions': [],
            'imports_cv2': False,
            'imports_torch': False,
            'imports_numpy': False
        }
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Quick analysis without full AST parsing
            capabilities['has_main'] = 'if __name__ == "__main__":' in content
            capabilities['imports_cv2'] = 'import cv2' in content or 'from cv2' in content
            capabilities['imports_torch'] = 'import torch' in content or 'from torch' in content
            capabilities['imports_numpy'] = 'import numpy' in content or 'from numpy' in content
            
            # Extract function names (simple regex)
            import re
            functions = re.findall(r'def\s+(\w+)\s*\(', content)
            capabilities['functions'] = functions[:10]  # Limit to first 10
            
            # Extract class names
            classes = re.findall(r'class\s+(\w+)\s*[\(:]', content)
            capabilities['classes'] = classes[:10]  # Limit to first 10
            
        except Exception as e:
            self.logger.debug(f"Error analyzing {script_path}: {e}")
        
        return capabilities
    
    def get_scripts_for_task(self, task_type):
        """Get relevant scripts for a specific task"""
        task_mapping = {
            'segmentation': ['segmentation', 'preprocessing'],
            'defect_detection': ['detection', 'ai_models'],
            'analysis': ['analysis', 'feature_extraction'],
            'full_pipeline': ['preprocessing', 'segmentation', 'detection', 'analysis']
        }
        
        relevant_categories = task_mapping.get(task_type, [])
        scripts = []
        
        for category in relevant_categories:
            scripts.extend(self.categories.get(category, []))
        
        return scripts

class ProcessingPipeline:
    """Main processing pipeline that orchestrates script execution"""
    def __init__(self, logger, config, script_registry, connector_interface):
        self.logger = logger
        self.config = config
        self.registry = script_registry
        self.connector = connector_interface
        self.results_cache = {}
        
    def process_image(self, image_path: str, reference_data: Dict = None) -> Dict:
        """Process a single image through the full pipeline"""
        self.logger.info(f"Processing image: {image_path}")
        start_time = time.time()
        
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'preprocessing': {},
            'segmentation': {},
            'detection': {},
            'analysis': {},
            'errors': []
        }
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Step 1: Preprocessing
            self.logger.debug("Running preprocessing...")
            preprocessed = self._run_preprocessing(image, results)
            
            # Step 2: Segmentation
            self.logger.debug("Running segmentation...")
            masks = self._run_segmentation(preprocessed, results)
            
            # Step 3: Defect Detection
            self.logger.debug("Running defect detection...")
            defects = self._run_detection(preprocessed, masks, results)
            
            # Step 4: Analysis and Reporting
            self.logger.debug("Running analysis...")
            analysis = self._run_analysis(preprocessed, masks, defects, results)
            
            # Step 5: Visualization (if enabled)
            if self.config.get('enable_visualization', True):
                self._create_visualization(image, results)
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            results['errors'].append(str(e))
        
        results['processing_time'] = time.time() - start_time
        self.logger.info(f"Image processing completed in {results['processing_time']:.2f} seconds")
        
        return results
    
    def _run_preprocessing(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Run preprocessing scripts"""
        preprocessed = image.copy()
        
        # Try to use advanced preprocessing if available
        preprocessing_scripts = self.registry.get_scripts_for_task('preprocessing')
        
        for script_name in preprocessing_scripts:
            if 'enhance' in script_name or 'quality' in script_name:
                try:
                    # Use connector to execute preprocessing
                    response = self.connector.send_command(
                        'root', 'execute_function',
                        script=script_name,
                        function='enhance_image',
                        args=[preprocessed]
                    )
                    if response and response.get('status') == 'ok':
                        # Assume the function returns enhanced image
                        preprocessed = response.get('result', preprocessed)
                        results['preprocessing'][script_name] = 'success'
                except Exception as e:
                    self.logger.debug(f"Preprocessing with {script_name} failed: {e}")
        
        return preprocessed
    
    def _run_segmentation(self, image: np.ndarray, results: Dict) -> Dict:
        """Run segmentation to find fiber regions"""
        masks = {
            'core': None,
            'cladding': None,
            'ferrule': None
        }
        
        # Try AI segmentation first if available
        if self.config.get('enable_ai', True):
            try:
                # Check for AI segmentation script
                if 'image-segmentation-ai-39' in self.registry.scripts:
                    response = self.connector.send_command(
                        'root', 'execute_function',
                        script='image-segmentation-ai-39',
                        function='segment',
                        args=[image]
                    )
                    if response and response.get('status') == 'ok':
                        masks = response.get('result', masks)
                        results['segmentation']['method'] = 'ai'
                        return masks
            except Exception as e:
                self.logger.debug(f"AI segmentation failed: {e}")
        
        # Fallback to classical methods
        segmentation_scripts = self.registry.get_scripts_for_task('segmentation')
        
        for script_name in segmentation_scripts:
            if 'fiber-finder' in script_name or 'circle-detector' in script_name:
                try:
                    response = self.connector.send_command(
                        'root', 'execute_function',
                        script=script_name,
                        function='find_fiber',
                        args=[image]
                    )
                    if response and response.get('status') == 'ok':
                        masks = response.get('result', masks)
                        results['segmentation']['method'] = script_name
                        break
                except Exception as e:
                    self.logger.debug(f"Segmentation with {script_name} failed: {e}")
        
        return masks
    
    def _run_detection(self, image: np.ndarray, masks: Dict, results: Dict) -> List[Dict]:
        """Run defect detection algorithms"""
        all_defects = []
        
        # Get detection scripts
        detection_scripts = self.registry.get_scripts_for_task('defect_detection')
        
        # Use ensemble approach - run multiple detectors
        for script_name in detection_scripts[:3]:  # Limit to top 3 for performance
            try:
                response = self.connector.send_command(
                    'root', 'execute_function',
                    script=script_name,
                    function='detect_defects',
                    args=[image],
                    kwargs={'masks': masks}
                )
                if response and response.get('status') == 'ok':
                    defects = response.get('result', [])
                    all_defects.extend(defects)
                    results['detection'][script_name] = len(defects)
            except Exception as e:
                self.logger.debug(f"Detection with {script_name} failed: {e}")
        
        # Merge and deduplicate defects
        merged_defects = self._merge_defects(all_defects)
        results['detection']['total_defects'] = len(merged_defects)
        
        return merged_defects
    
    def _merge_defects(self, defects: List[Dict]) -> List[Dict]:
        """Merge overlapping defects from multiple detectors"""
        if not defects:
            return []
        
        # Simple merging based on location proximity
        merged = []
        used = set()
        
        for i, defect1 in enumerate(defects):
            if i in used:
                continue
            
            # Start with current defect
            merged_defect = defect1.copy()
            confidence_sum = defect1.get('confidence', 0.5)
            count = 1
            
            # Check for overlapping defects
            for j, defect2 in enumerate(defects[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if defects overlap (simple distance check)
                if self._defects_overlap(defect1, defect2):
                    used.add(j)
                    confidence_sum += defect2.get('confidence', 0.5)
                    count += 1
            
            # Update confidence as average
            merged_defect['confidence'] = confidence_sum / count
            merged_defect['detector_count'] = count
            merged.append(merged_defect)
        
        return merged
    
    def _defects_overlap(self, d1: Dict, d2: Dict) -> bool:
        """Check if two defects overlap"""
        # Simple implementation - can be improved
        if 'bbox' in d1 and 'bbox' in d2:
            x1, y1, w1, h1 = d1['bbox']
            x2, y2, w2, h2 = d2['bbox']
            
            # Check for intersection
            return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
        
        return False
    
    def _run_analysis(self, image: np.ndarray, masks: Dict, defects: List[Dict], results: Dict) -> Dict:
        """Run analysis and generate metrics"""
        analysis = {
            'quality_score': 0.0,
            'defect_summary': {},
            'recommendations': []
        }
        
        # Calculate quality score
        if masks['core'] is not None:
            core_area = np.sum(masks['core'])
            defect_area = sum(d.get('area', 0) for d in defects)
            
            if core_area > 0:
                defect_ratio = defect_area / core_area
                analysis['quality_score'] = max(0, 1 - defect_ratio) * 100
        
        # Defect summary
        analysis['defect_summary'] = {
            'total_count': len(defects),
            'high_confidence': sum(1 for d in defects if d.get('confidence', 0) > 0.8),
            'types': {}
        }
        
        # Generate recommendations
        if len(defects) == 0:
            analysis['recommendations'].append("No defects detected. Fiber appears to be in good condition.")
        elif len(defects) < 3:
            analysis['recommendations'].append("Minor defects detected. Regular monitoring recommended.")
        else:
            analysis['recommendations'].append("Multiple defects detected. Immediate inspection recommended.")
        
        results['analysis'] = analysis
        return analysis
    
    def _create_visualization(self, image: np.ndarray, results: Dict):
        """Create visualization of results"""
        vis = image.copy()
        
        # Draw defects
        for defect in results.get('detection', {}).get('defects', []):
            if 'bbox' in defect:
                x, y, w, h = defect['bbox']
                color = (0, 0, 255) if defect.get('confidence', 0) > 0.8 else (0, 255, 255)
                cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        
        # Save visualization
        if self.config.get('output_dir'):
            output_path = Path(self.config['output_dir']) / 'visualizations'
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_path / f"result_{timestamp}.jpg"
            cv2.imwrite(str(filename), vis)
            
            results['visualization_path'] = str(filename)

class ConnectorInterface:
    """Enhanced connector interface for script communication"""
    def __init__(self, logger):
        self.logger = logger
        self.hivemind = self._initialize_hivemind()
    
    def _initialize_hivemind(self):
        """Initialize the hivemind connector system"""
        try:
            # Import the existing hivemind connector
            spec = importlib.util.spec_from_file_location(
                "hivemind_connector",
                "hivemind_connector.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get the ScriptInterface class
                if hasattr(module, 'ScriptInterface'):
                    return module.ScriptInterface()
        except Exception as e:
            self.logger.warning(f"Could not initialize hivemind connector: {e}")
        
        return None
    
    def send_command(self, target: str, command: str, **kwargs) -> Dict:
        """Send command to a script through the connector system"""
        if self.hivemind:
            try:
                if command == 'execute_function':
                    script = kwargs.get('script', '')
                    function = kwargs.get('function', '')
                    args = kwargs.get('args', [])
                    kw = kwargs.get('kwargs', {})
                    
                    result = self.hivemind.execute_function(script, function, *args, **kw)
                    return {'status': 'ok', 'result': result}
                elif command == 'get_parameter':
                    script = kwargs.get('script', '')
                    param = kwargs.get('parameter', '')
                    value = self.hivemind.get_parameter(script, param)
                    return {'status': 'ok', 'value': value}
                elif command == 'set_parameter':
                    script = kwargs.get('script', '')
                    param = kwargs.get('parameter', '')
                    value = kwargs.get('value')
                    success = self.hivemind.set_parameter(script, param, value)
                    return {'status': 'ok' if success else 'error'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        
        return {'status': 'error', 'message': 'Hivemind not available'}

class PolarBearBrain:
    """Main brain controller for the Polar Bear system"""
    def __init__(self):
        # Initialize logging
        self.brain_logger = BrainLogger()
        self.logger = self.brain_logger.get_logger()
        
        self.logger.info("Initializing Polar Bear Brain...")
        
        # Initialize dependency manager
        self.deps = DependencyManager(self.logger)
        self.deps.ensure_core_dependencies()
        
        # Initialize configuration
        self.config = ConfigurationManager(self.logger)
        
        # Initialize script registry
        self.registry = ScriptRegistry(self.logger)
        
        # Initialize connector interface
        self.connector = ConnectorInterface(self.logger)
        
        # Initialize processing pipeline
        self.pipeline = ProcessingPipeline(
            self.logger, self.config, self.registry, self.connector
        )
        
        # State tracking
        self.running = False
        self.processed_count = 0
        self.start_time = None
        
        self.logger.info("Polar Bear Brain initialized successfully!")
    
    def run(self):
        """Main execution loop"""
        self.running = True
        self.start_time = time.time()
        
        mode = self.config.get('processing_mode', 'hybrid')
        
        try:
            if mode == 'realtime':
                self._run_realtime_mode()
            elif mode == 'batch':
                self._run_batch_mode()
            else:  # hybrid
                self._run_hybrid_mode()
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def _run_realtime_mode(self):
        """Run real-time video processing"""
        self.logger.info("Starting real-time processing mode...")
        
        video_source = self.config.get('video_source', 0)
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {video_source}")
            return
        
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every Nth frame to maintain performance
            if frame_count % 5 == 0:  # Process every 5th frame
                # Save frame temporarily
                temp_path = f"/tmp/frame_{frame_count}.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Process frame
                results = self.pipeline.process_image(temp_path)
                self.processed_count += 1
                
                # Display results
                if self.config.get('enable_visualization', True):
                    self._display_realtime_results(frame, results)
            
            # Show frame
            cv2.imshow('Polar Bear Brain - Real-time Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _run_batch_mode(self):
        """Run batch image processing"""
        self.logger.info("Starting batch processing mode...")
        
        input_dir = Path(self.config.get('input_dir', './input'))
        output_dir = Path(self.config.get('output_dir', './output'))
        
        # Create directories
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            self.logger.warning(f"No images found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process images in parallel
        max_workers = self.config.get('max_workers', 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for image_path in image_files:
                future = executor.submit(self._process_single_image, image_path, output_dir)
                futures.append((future, image_path))
            
            # Wait for completion with progress
            for i, (future, image_path) in enumerate(futures):
                try:
                    result = future.result()
                    self.processed_count += 1
                    self.logger.info(f"Processed {i+1}/{len(image_files)}: {image_path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
    
    def _process_single_image(self, image_path: Path, output_dir: Path) -> Dict:
        """Process a single image and save results"""
        results = self.pipeline.process_image(str(image_path))
        
        # Save results
        result_file = output_dir / f"{image_path.stem}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _run_hybrid_mode(self):
        """Run both batch and real-time modes"""
        self.logger.info("Starting hybrid mode...")
        
        # Start batch processing in background
        batch_thread = threading.Thread(target=self._run_batch_mode)
        batch_thread.daemon = True
        batch_thread.start()
        
        # Run real-time in foreground
        self._run_realtime_mode()
    
    def _display_realtime_results(self, frame: np.ndarray, results: Dict):
        """Overlay results on frame for real-time display"""
        # Add quality score
        quality = results.get('analysis', {}).get('quality_score', 0)
        color = (0, 255, 0) if quality > 80 else (0, 255, 255) if quality > 50 else (0, 0, 255)
        cv2.putText(frame, f"Quality: {quality:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add defect count
        defect_count = len(results.get('detection', {}).get('defects', []))
        cv2.putText(frame, f"Defects: {defect_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        
        # Calculate statistics
        if self.start_time:
            runtime = time.time() - self.start_time
            self.logger.info(f"\n=== Polar Bear Brain Session Summary ===")
            self.logger.info(f"Total runtime: {runtime:.2f} seconds")
            self.logger.info(f"Images processed: {self.processed_count}")
            if self.processed_count > 0:
                self.logger.info(f"Average processing time: {runtime/self.processed_count:.2f} seconds/image")
            self.logger.info("========================================\n")
        
        self.logger.info("Polar Bear Brain shutdown complete")

def main():
    """Main entry point"""
    brain = PolarBearBrain()
    brain.run()

if __name__ == "__main__":
    main()