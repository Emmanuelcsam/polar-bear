import os
import sys
import json
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Any
import shutil
import warnings
import logging
import traceback
import ast
import inspect

warnings.filterwarnings('ignore')

# ==============================================================================
# SETUP LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# ==============================================================================
# CHECK OPTIONAL DEPENDENCIES
# ==============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("matplotlib not available, some visualizations will be skipped")

try:
    from scipy.ndimage import binary_opening, binary_closing
    HAS_SCIPY_FULL = True
except ImportError:
    HAS_SCIPY_FULL = False
    logging.warning("Some scipy components not available, using basic post-processing")

# ==============================================================================
# HELPER CLASSES & FUNCTIONS
# ==============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class BasicFallbackSegmenter:
    """Fallback segmentation method using basic image processing (Hough Circles)."""
    @staticmethod
    def segment(image_path: str, output_dir: str) -> dict:
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'success': False, 'error': 'Could not load image'}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Find circles using HoughCircles
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                param1=50, param2=30, minRadius=10, maxRadius=min(gray.shape) // 2
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                sorted_circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
                
                if len(sorted_circles) >= 2:
                    cladding, core = sorted_circles[0], sorted_circles[1]
                elif sorted_circles:
                    cladding = sorted_circles[0]
                    core = [cladding[0], cladding[1], int(cladding[2] * 0.2)]
                else:
                    return {'success': False, 'error': 'HoughCircles found no circles'}

                return {
                    'success': True,
                    'center': (float(cladding[0]), float(cladding[1])),
                    'core_radius': float(core[2]),
                    'cladding_radius': float(cladding[2]),
                    'confidence': 0.3,
                    'method': 'basic_fallback'
                }
            else:
                h, w = gray.shape
                center = (w // 2, h // 2)
                cladding_radius = min(h, w) // 3
                core_radius = cladding_radius // 5
                
                return {
                    'success': True,
                    'center': center,
                    'core_radius': float(core_radius),
                    'cladding_radius': float(cladding_radius),
                    'confidence': 0.1,
                    'method': 'basic_fallback_estimate'
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Fallback segmentation failed: {e}'}

class MethodDiscoverySystem:
    """
    An advanced system to dynamically discover entry point functions
    in external segmentation method scripts.
    """
    def __init__(self, methods_dir: Path, logger: logging.Logger):
        self.methods_dir = methods_dir
        self.logger = logger
        self.known_patterns = {
            'adaptive_intensity': 'adaptive_segment_image',
            'bright_core_extractor': 'analyze_core',
            'computational_separation': 'process_fiber_image_veridian',
            'geometric_approach': 'segment_with_geometric',
            'gradient_approach': 'segment_with_gradient',
            'guess_approach': 'segment_fiber_with_multimodal_analysis',
            'hough_separation': 'segment_with_hough',
            'segmentation': 'run_segmentation_pipeline',
            'threshold_separation': 'segment_with_threshold',
            'unified_core_cladding_detector': 'detect_core_cladding',
            'intelligent_segmenter': 'run_intelligent_segmentation',
        }

    def discover(self, method_path: Path) -> Optional[Tuple[str, str]]:
        """
        Attempts to find the entry point using a series of strategies.
        Returns a tuple of (function_name, import_statement) or None.
        """
        module_name = method_path.stem
        self.logger.info(f"  -> Discovering entry point for '{module_name}'...")

        # Strategy 1: Check known hardcoded patterns (fastest)
        if module_name in self.known_patterns:
            func_name = self.known_patterns[module_name]
            self.logger.info(f"     Found using known pattern: '{func_name}'")
            return func_name, f"from {module_name} import {func_name}"

        # Strategy 2: Abstract Syntax Tree (AST) Parsing (reliable, no execution)
        try:
            func_name = self._parse_module_ast(method_path)
            if func_name:
                self.logger.info(f"     Found using AST analysis: '{func_name}'")
                return func_name, f"from {module_name} import {func_name}"
        except Exception as e:
            self.logger.warning(f"     AST parsing failed for {module_name}: {e}")

        # Strategy 3: Subprocess Inspection (slower, but comprehensive)
        try:
            func_name = self._subprocess_discovery(module_name)
            if func_name:
                self.logger.info(f"     Found using subprocess inspection: '{func_name}'")
                return func_name, f"from {module_name} import {func_name}"
        except Exception as e:
            self.logger.error(f"     Subprocess discovery failed for {module_name}: {e}")

        self.logger.warning(f"  -> Failed to discover entry point for '{module_name}'.")
        return None

    def _parse_module_ast(self, module_path: Path) -> Optional[str]:
        """Parses the module's AST to find a likely entry function."""
        with open(module_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        tree = ast.parse(source_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Look for functions with a (path, output_dir) signature
                args = [arg.arg for arg in node.args.args]
                if len(args) >= 2 and ('path' in args[0] or 'image' in args[0]) and 'dir' in args[1]:
                    return node.name
                # Look for common entry point names
                if node.name in ['run', 'main', 'process', 'segment', 'analyze']:
                     return node.name
        return None

    def _subprocess_discovery(self, module_name: str) -> Optional[str]:
        """Runs a discovery script in a subprocess to inspect the module."""
        discover_script = f"""
import sys, inspect
sys.path.insert(0, r"{self.methods_dir.resolve()}")
try:
    module = __import__("{module_name}")
    
    # Priority names
    standard_names = [
        'run_segmentation', 'segment_image', 'process_image', 'main', 
        'segment', 'analyze', 'process', 'detect', 'extract'
    ]
    for name in standard_names:
        if hasattr(module, name) and callable(getattr(module, name)):
            print(f"{{name}}")
            exit(0)
            
    # Fallback to any function with relevant keywords in name
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            if any(key in name.lower() for key in ['segment', 'process', 'analyze', 'detect']):
                print(f"{{name}}")
                exit(0)
except Exception:
    pass
"""
        result = subprocess.run(
            [sys.executable, "-c", discover_script],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
        return None


class SegmentationResult:
    """Standardized result format for all segmentation methods."""
    def __init__(self, method_name: str, image_path: str):
        self.method_name = method_name
        self.image_path = image_path
        self.center: Optional[Tuple[float, float]] = None
        self.core_radius: Optional[float] = None
        self.cladding_radius: Optional[float] = None
        self.masks: Optional[Dict[str, np.ndarray]] = None
        self.confidence: float = 0.5
        self.execution_time: float = 0.0
        self.error: Optional[str] = None
        
    def to_dict(self):
        return {
            'method_name': self.method_name,
            'center': self.center,
            'core_radius': self.core_radius,
            'cladding_radius': self.cladding_radius,
            'confidence': self.confidence,
            'execution_time': self.execution_time,
            'error': self.error,
            'has_masks': self.masks is not None
        }

class EnhancedConsensusSystem:
    """Model-aware voting system with fallback support."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _calculate_iou(self, mask1: Optional[np.ndarray], mask2: Optional[np.ndarray]) -> float:
        """Calculates Intersection over Union for two binary masks."""
        if mask1 is None or mask2 is None:
            return 0.0
        try:
            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)
            iou_score = np.sum(intersection) / (np.sum(union) + 1e-6)
            return float(iou_score)
        except Exception as e:
            self.logger.warning(f"IoU calculation failed: {e}")
            return 0.0

    def generate_consensus(self,
                           results: List[SegmentationResult],
                           method_scores: Dict[str, float],
                           image_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Generates a final consensus model by first running a preliminary pixel
        vote to identify high-agreement methods, then calculating a weighted
        average of their geometric parameters to create an ideal final mask.
        """
        valid_results = [r for r in results if r.error is None and r.masks is not None]
        
        if not valid_results:
            self.logger.warning("No valid results with masks. Attempting fallback to parameters.")
            param_results = [r for r in results if r.error is None and r.center is not None 
                           and r.core_radius is not None and r.cladding_radius is not None]
            if not param_results:
                self.logger.error("No valid results at all. Consensus failed.")
                return None
            
            self.logger.info("Using best result based on parameters as fallback consensus.")
            best_result = max(param_results, key=lambda r: r.confidence)
            masks = self.create_masks_from_params(
                best_result.center, best_result.core_radius, best_result.cladding_radius, image_shape
            )
            return {
                'masks': masks, 'center': best_result.center, 'core_radius': best_result.core_radius,
                'cladding_radius': best_result.cladding_radius, 'contributing_methods': [best_result.method_name],
                'num_valid_results': 0, 'fallback_used': True, 'all_results': [r.to_dict() for r in results]
            }
            
        if len(valid_results) == 1:
            self.logger.info("Only one valid result, using it directly.")
            r = valid_results[0]
            return {
                'masks': r.masks, 'center': r.center, 'core_radius': r.core_radius,
                'cladding_radius': r.cladding_radius, 'contributing_methods': [r.method_name],
                'num_valid_results': 1, 'all_results': [r.to_dict() for r in results]
            }

        self.logger.info(f"Generating consensus from {len(valid_results)} valid results...")
        h, w = image_shape

        # Stage 1: Preliminary Weighted Pixel Vote
        weighted_votes = np.zeros((h, w, 3), dtype=np.float32)
        total_weight = 0
        for r in valid_results:
            weight = method_scores.get(r.method_name, 1.0) * r.confidence
            total_weight += weight
            if r.masks.get('core') is not None: weighted_votes[:, :, 0] += r.masks['core'] * weight
            if r.masks.get('cladding') is not None: weighted_votes[:, :, 1] += r.masks['cladding'] * weight
            if r.masks.get('ferrule') is not None: weighted_votes[:, :, 2] += r.masks['ferrule'] * weight

        if total_weight > 0: weighted_votes /= total_weight
        
        preliminary_classification = np.argmax(weighted_votes, axis=2)
        prelim_core_mask = (preliminary_classification == 0)
        prelim_cladding_mask = (preliminary_classification == 1)

        # Stage 2: Identify High-Agreement Methods
        high_agreement_results = []
        for r in valid_results:
            core_iou = self._calculate_iou(r.masks.get('core'), prelim_core_mask)
            cladding_iou = self._calculate_iou(r.masks.get('cladding'), prelim_cladding_mask)
            if core_iou > 0.6 and cladding_iou > 0.6: high_agreement_results.append(r)

        if not high_agreement_results:
            self.logger.warning("No methods passed high-agreement threshold. Using all valid results.")
            high_agreement_results = valid_results
        
        # Stage 3: Parameter-Space Consensus
        params = {'cx': [], 'cy': [], 'core_r': [], 'clad_r': []}
        weights = []
        for r in high_agreement_results:
            if all([r.center, r.core_radius is not None, r.cladding_radius is not None]):
                weight = method_scores.get(r.method_name, 1.0) * r.confidence
                params['cx'].append(r.center[0]); params['cy'].append(r.center[1])
                params['core_r'].append(r.core_radius); params['clad_r'].append(r.cladding_radius)
                weights.append(weight)

        if not weights:
             self.logger.error("No valid parameters to average. Consensus failed.")
             return None

        final_center = (np.average(params['cx'], weights=weights), np.average(params['cy'], weights=weights))
        final_core_radius = np.average(params['core_r'], weights=weights)
        final_cladding_radius = np.average(params['clad_r'], weights=weights)

        # Stage 4: Generate Final Ideal Masks
        final_masks = self.create_masks_from_params(final_center, final_core_radius, final_cladding_radius, image_shape)
        final_masks['core'], final_masks['cladding'], final_masks['ferrule'] = self.ensure_mask_consistency(
             final_masks['core'], final_masks['cladding'], final_masks['ferrule']
        )
        
        return {
            'masks': final_masks, 'center': final_center, 'core_radius': final_core_radius,
            'cladding_radius': final_cladding_radius,
            'contributing_methods': [r.method_name for r in high_agreement_results],
            'num_valid_results': len(valid_results), 'all_results': [r.to_dict() for r in results]
        }

    def create_masks_from_params(self, center: Tuple[float, float], core_radius: float, 
                               cladding_radius: float, image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        h, w = image_shape
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - center[0])**2 + (y_grid - center[1])**2)
        
        core_mask = (dist_from_center <= core_radius).astype(np.uint8)
        cladding_mask = ((dist_from_center > core_radius) & (dist_from_center <= cladding_radius)).astype(np.uint8)
        ferrule_mask = (dist_from_center > cladding_radius).astype(np.uint8)
        
        return {'core': core_mask, 'cladding': cladding_mask, 'ferrule': ferrule_mask}

    def ensure_mask_consistency(self, core_mask, cladding_mask, ferrule_mask):
        if HAS_SCIPY_FULL:
            kernel = np.ones((5, 5), dtype=np.uint8)
            core_mask = binary_closing(binary_opening(core_mask, kernel), kernel).astype(np.uint8)
            cladding_mask = binary_closing(binary_opening(cladding_mask, kernel), kernel).astype(np.uint8)
        
        cladding_mask[core_mask == 1] = 0
        ferrule_mask[np.logical_or(core_mask == 1, cladding_mask == 1)] = 0
        return core_mask, cladding_mask, ferrule_mask

# ==============================================================================
# MAIN ORCHESTRATION SYSTEM
# ==============================================================================

class UnifiedSegmentationSystem:
    """Main unifier system that orchestrates all segmentation methods."""
    
    def __init__(self, methods_dir: str = "zone_methods"):
        self.logger = logging.getLogger(__name__)
        self.methods_dir = Path(methods_dir)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.dataset_stats = {'method_scores': {}, 'method_accuracy': {}}
        self.knowledge_file = self.output_dir / "segmentation_knowledge.json"
        self.load_knowledge()
        
        self.methods: Dict[str, Dict] = {}
        self.method_info: Dict[str, Dict] = {}
        
        self.discovery_system = MethodDiscoverySystem(self.methods_dir, self.logger)
        self.load_methods()
        
        self.consensus_system = EnhancedConsensusSystem()
        self.vulnerable_methods = [
            'adaptive_intensity', 'gradient_approach', 'guess_approach', 
            'threshold_separation', 'intelligent_segmenter'
        ]

    def load_knowledge(self):
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r') as f:
                    self.dataset_stats.update(json.load(f))
                self.logger.info(f"Loaded knowledge from {self.knowledge_file}")
            except Exception as e:
                self.logger.warning(f"Could not load knowledge: {e}")
    
    def save_knowledge(self):
        try:
            with open(self.knowledge_file, 'w') as f:
                json.dump(self.dataset_stats, f, indent=4, cls=NumpyEncoder)
            self.logger.info(f"Saved updated knowledge to {self.knowledge_file}")
        except Exception as e:
            self.logger.error(f"Failed to save knowledge: {e}")
    
    def load_methods(self):
        """Load methods using the adaptive discovery system."""
        if not self.methods_dir.is_dir():
            self.logger.warning(f"Methods directory '{self.methods_dir}' does not exist. Creating it...")
            self.methods_dir.mkdir(parents=True, exist_ok=True)
            return

        self.logger.info(f"Discovering methods in {self.methods_dir}...")
        method_files = [f for f in self.methods_dir.glob('*.py') if not f.name.startswith('_')]
        
        for method_path in method_files:
            entry_info = self.discovery_system.discover(method_path)
            if entry_info:
                method_name = method_path.stem
                func_name, import_stmt = entry_info
                self.methods[method_name] = {
                    'path': method_path,
                    'score': self.dataset_stats['method_scores'].get(method_name, 1.0),
                    'function_name': func_name,
                    'import_statement': import_stmt,
                }
                self.method_info[method_name] = {'function_name': func_name, 'import_statement': import_stmt}
                self.logger.info(f"Loaded method: '{method_name}' (entry point: '{func_name}')")
        
        # Add fallback method
        self.methods['fallback_basic'] = {'score': 0.5}


    def detect_and_inpaint_anomalies(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect and inpaint anomalies with error handling."""
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
            _, defect_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
            
            if HAS_SCIPY_FULL:
                defect_mask = binary_opening(defect_mask, structure=np.ones((3,3)), iterations=2).astype(np.uint8)
                    
            inpainted_image = cv2.inpaint(image, defect_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            return inpainted_image, defect_mask
        except Exception as e:
            self.logger.error(f"Error in detect_and_inpaint_anomalies: {e}")
            return image, np.zeros(image.shape[:2], dtype=np.uint8)

    def run_method_isolated(self, method_name: str, image_path: Path, temp_output: Path) -> dict:
        """Run a method in isolation with improved error handling."""
        if method_name == 'fallback_basic':
            return BasicFallbackSegmenter.segment(str(image_path), str(temp_output))
        
        result_file = temp_output / f"{method_name}_result.json"
        runner_script_path = temp_output / "runner.py"
        
        method_details = self.method_info.get(method_name)
        if not method_details:
            return {'success': False, 'error': f'Method {method_name} not loaded'}
        
        script_content = f"""
import sys, json, os, traceback
from pathlib import Path
import numpy as np
# Suppress warnings and backend errors in subprocess
import warnings
warnings.filterwarnings('ignore')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, r"{self.methods_dir.resolve()}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)): return int(obj)
        if isinstance(obj, (np.floating, np.float_)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    try:
        {method_details['import_statement']}
        # Call the dynamically discovered function
        result = {method_details['function_name']}(r"{image_path.resolve()}", r"{temp_output.resolve()}")
        if not isinstance(result, dict):
             result = {{'success': False, 'error': f'Method returned invalid type: {{type(result)}}'}}
    except Exception as e:
        result = {{'success': False, 'error': f'Exception: {{e}}', 'traceback': traceback.format_exc()}}
    
    with open(r"{result_file.resolve()}", 'w') as f:
        json.dump(result, f, indent=4, cls=NpEncoder)

if __name__ == "__main__":
    main()
"""
        try:
            with open(runner_script_path, 'w') as f: f.write(script_content)
            proc = subprocess.run(
                [sys.executable, str(runner_script_path)],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, 'MPLBACKEND': 'Agg'}
            )
            
            if result_file.exists():
                with open(result_file, 'r') as f: return json.load(f)
            else:
                return {'success': False, 'error': proc.stderr or "No result file produced"}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Method timed out after 60 seconds'}
        except Exception as e:
            return {'success': False, 'error': f'Subprocess execution failed: {e}'}

    def run_method(self, method_name: str, image_path: Path, image_shape: Tuple[int, int]) -> SegmentationResult:
        """Run a single method with comprehensive error handling."""
        result = SegmentationResult(method_name, str(image_path))
        start_time = time.time()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                method_output = self.run_method_isolated(method_name, image_path, Path(temp_dir))
                
                if method_output and method_output.get('success'):
                    result.center = tuple(method_output['center']) if 'center' in method_output else None
                    result.core_radius = method_output.get('core_radius')
                    result.cladding_radius = method_output.get('cladding_radius')
                    result.confidence = method_output.get('confidence', 0.5)

                    if all([result.center, result.core_radius, result.cladding_radius]):
                        h, w = image_shape
                        if not (0 <= result.center[0] <= w and 0 <= result.center[1] <= h and 
                                0 < result.core_radius < result.cladding_radius < min(h,w)/2):
                            result.error = "Parameters out of reasonable bounds"
                        else:
                            result.masks = self.consensus_system.create_masks_from_params(
                                result.center, result.core_radius, result.cladding_radius, image_shape
                            )
                    else:
                        result.error = "Method returned invalid/missing parameters"
                else:
                    result.error = method_output.get('error', 'Unknown failure') if method_output else "Empty method output"
        except Exception as e:
            result.error = f"Exception during method execution: {e}"
            self.logger.error(f"Error running {method_name}: {e}\n{traceback.format_exc()}")
            
        result.execution_time = time.time() - start_time
        return result

    def update_learning(self, consensus: Dict, all_results: List[SegmentationResult]):
        """Update learning model based on method agreement with consensus."""
        if not consensus or 'masks' not in consensus: return
        self.logger.info("Updating learning model...")
        consensus_masks = consensus['masks']
        
        for result in all_results:
            if result.error or not result.masks: continue
            try:
                core_iou = self.consensus_system._calculate_iou(result.masks['core'], consensus_masks['core'])
                cladding_iou = self.consensus_system._calculate_iou(result.masks['cladding'], consensus_masks['cladding'])
                avg_iou = (core_iou + cladding_iou) / 2
                
                current_score = self.dataset_stats['method_scores'].get(result.method_name, 1.0)
                # Nudge score towards a value based on its IoU agreement
                target_score = 0.1 + (1.9 * avg_iou)
                new_score = current_score * 0.9 + target_score * 0.1
                
                self.dataset_stats['method_scores'][result.method_name] = new_score
                self.dataset_stats['method_accuracy'][result.method_name] = avg_iou
                if result.method_name in self.methods:
                    self.methods[result.method_name]['score'] = new_score
            except Exception as e:
                self.logger.warning(f"Error updating score for {result.method_name}: {e}")
        
        self.save_knowledge()

    def process_image(self, image_path: Path, output_dir_str: str) -> Optional[Dict]:
        """Process a single image with the full pipeline."""
        self.logger.info(f"\n{'='*20} Processing: {image_path.name} {'='*20}")
        if not image_path.exists():
            self.logger.error(f"Image file does not exist: {image_path}"); return None
            
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            self.logger.error(f"Failed to load image: {image_path}"); return None
            
        image_shape = original_img.shape[:2]

        self.logger.info("Running pre-processing: Anomaly detection and inpainting...")
        inpainted_img, defect_mask = self.detect_and_inpaint_anomalies(original_img)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
            inpainted_image_path = Path(tmp_f.name)
            cv2.imwrite(str(inpainted_image_path), inpainted_img)
        self.logger.info("Inpainting complete.")

        all_results = []
        for method_name in self.methods:
            use_inpainted = method_name in self.vulnerable_methods
            current_image_path = inpainted_image_path if use_inpainted else image_path
            self.logger.info(f"-> Running {method_name} (image: {'inpainted' if use_inpainted else 'original'})...")
            result = self.run_method(method_name, current_image_path, image_shape)
            all_results.append(result)
            if result.error: self.logger.warning(f"   Failed: {result.error}")
            else: self.logger.info(f"   Success - Confidence: {result.confidence:.2f}, Time: {result.execution_time:.2f}s")
        
        consensus = self.consensus_system.generate_consensus(
            all_results, {name: info['score'] for name, info in self.methods.items()}, image_shape
        )
        
        if consensus:
            self.logger.info(f"Consensus achieved. Contributing methods: {', '.join(consensus['contributing_methods'])}")
            if not consensus.get('fallback_used', False): self.update_learning(consensus, all_results)
            saved_paths = self.save_results(image_path, consensus, original_img, output_dir_str, defect_mask)
            consensus['saved_regions'] = saved_paths
        else:
            self.logger.error("No consensus could be reached. No output generated.")
            
        os.remove(inpainted_image_path)
        return consensus

    def save_results(self, image_path: Path, consensus: Dict, image: np.ndarray, output_dir: str, defect_mask: np.ndarray) -> List[str]:
        """Saves consensus results, separated region images, and visualizations."""
        result_dir = Path(output_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        try:
            report = {k: v for k, v in consensus.items() if k not in ['masks', 'all_results']}
            report['method_accuracies'] = self.dataset_stats.get('method_accuracy', {})
            with open(result_dir / "consensus_report.json", 'w') as f:
                json.dump(report, f, indent=4, cls=NumpyEncoder)
            
            masks = consensus.get('masks', {})
            if masks:
                for name, mask in masks.items():
                    cv2.imwrite(str(result_dir / f"mask_{name}.png"), mask * 255)
                    region_img = cv2.bitwise_and(image, image, mask=mask)
                    region_path = str(result_dir / f"region_{name}.png")
                    cv2.imwrite(region_path, region_img)
                    saved_paths.append(region_path)
                
                cv2.imwrite(str(result_dir / "detected_defects.png"), defect_mask)
                
                if HAS_MATPLOTLIB:
                    self.create_summary_visualization(result_dir, image, masks, consensus)
            
            self.logger.info(f"All results saved to: {result_dir.resolve()}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}\n{traceback.format_exc()}")
        return saved_paths

    def create_summary_visualization(self, result_dir: Path, original_image: np.ndarray, masks: Dict, consensus: Dict):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
            fig.suptitle(f'Unified Segmentation Analysis: {result_dir.name}', fontsize=16)
            img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Plot 1: Original Image with Final Boundaries
            axes[0, 0].imshow(img_rgb)
            axes[0, 0].set_title('Original with Final Boundaries')
            if 'center' in consensus:
                theta = np.linspace(0, 2 * np.pi, 100)
                cx, cy = consensus['center']
                axes[0, 0].plot(cx + consensus['core_radius'] * np.cos(theta), cy + consensus['core_radius'] * np.sin(theta), 'lime', lw=2, label='Core')
                axes[0, 0].plot(cx + consensus['cladding_radius'] * np.cos(theta), cy + consensus['cladding_radius'] * np.sin(theta), 'cyan', lw=2, label='Cladding')
                axes[0, 0].legend()

            # Plot 2: Final Segmentation Mask
            final_mask_viz = np.zeros_like(img_rgb)
            final_mask_viz[masks.get('core', 0) > 0] = [255, 0, 0]
            final_mask_viz[masks.get('cladding', 0) > 0] = [0, 255, 0]
            final_mask_viz[masks.get('ferrule', 0) > 0] = [0, 0, 255]
            axes[0, 1].imshow(final_mask_viz)
            axes[0, 1].set_title('Final Segmentation Masks')
            
            # Plot 3: Method Performance Text
            axes[1, 0].axis('off'); axes[1, 0].set_title('Method Performance')
            accuracies = self.dataset_stats.get('method_accuracy', {})
            text_content = "Method Performance (IoU):\n" + "-"*25
            if accuracies:
                sorted_methods = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
                text_content += "\n" + "\n".join([f"{m[:20].ljust(22)}: {a:.3f}" for m, a in sorted_methods])
            else: text_content += "\n(No accuracy data available)"
            axes[1, 0].text(0.05, 0.95, text_content, family='monospace', va='top', fontsize=10)

            # Plot 4: Combined Separated Regions
            composite = np.zeros_like(img_rgb)
            for name, mask in masks.items():
                composite += cv2.cvtColor(cv2.bitwise_and(original_image, original_image, mask=mask), cv2.COLOR_BGR2RGB)
            axes[1, 1].imshow(composite); axes[1, 1].set_title('Final Separated Regions')

            for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
            plt.savefig(result_dir / "summary_analysis.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}\n{traceback.format_exc()}")
            if 'fig' in locals(): plt.close(fig)

    def run(self):
        """Main execution loop for the system."""
        print("\n" + "="*80)
        print("UNIFIED FIBER OPTIC SEGMENTATION SYSTEM".center(80))
        print("Dynamic Method Discovery Edition".center(80))
        print("="*80)
        
        print(f"\nMethods directory: {self.methods_dir.resolve()}")
        print(f"Loaded methods: {list(m for m in self.methods if m != 'fallback_basic')}")
        
        if len(self.methods) <= 1: # Only fallback exists
            print("\nWarning: No segmentation methods found. Only the basic fallback will be used.")
            if input("Continue with fallback only? (y/n): ").lower() != 'y': return
        
        folder_path = input("\nEnter the folder path containing images: ").strip().strip('"\'')
        folder_path = Path(folder_path)
        
        if not folder_path.is_dir():
            print(f"Error: Folder not found: {folder_path}"); return
            
        image_files = [p for p in folder_path.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']]
        if not image_files:
            print(f"No image files found in {folder_path}"); return
            
        print(f"\nFound {len(image_files)} images to process.")
        for i, img_path in enumerate(sorted(image_files), 1):
            try:
                output_subdir = self.output_dir / img_path.stem
                self.process_image(img_path, str(output_subdir))
            except Exception as e:
                self.logger.error(f"FATAL: Failed to process {img_path.name}: {e}")
                traceback.print_exc()
                
        print("\n" + "="*80 + "\n" + "Processing complete.".center(80) + "\n" + "="*80)

def main():
    """Main entry point for the script."""
    try:
        methods_dir = sys.argv[1] if len(sys.argv) > 1 else "zone_methods"
        system = UnifiedSegmentationSystem(methods_dir)
        system.run()
    except Exception as e:
        logging.critical(f"A fatal error occurred in the main execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
