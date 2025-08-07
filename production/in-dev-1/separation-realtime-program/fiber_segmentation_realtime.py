"""
Real-time fiber optic segmentation system with live camera feed.
Integrates Pylon camera with multiple segmentation methods for continuous analysis.
"""

import time
import threading
import queue
import cv2
import numpy as np
from pathlib import Path
import tempfile
import json
import os
import sys
import subprocess
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import matplotlib for visualizations
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualizations will be skipped")

# Import scipy components for enhanced processing
try:
    from scipy.ndimage import median_filter, gaussian_filter
    from scipy.ndimage import binary_opening, binary_closing
    HAS_SCIPY_FULL = True
except ImportError:
    HAS_SCIPY_FULL = False
    print("Warning: Some scipy components not available, using basic processing")

# Import Pylon camera functionality
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
    print("INFO: Pylon SDK found. Basler camera support is enabled.")
    try:
        from genicam import GenericException
    except ImportError:
        class GenericException(Exception):
            pass
except ImportError:
    print("WARNING: Pylon SDK not found. Cannot use Basler camera.")
    print("Please install pypylon: pip install pypylon")


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class SegmentationResult:
    """Standardized result format for all segmentation methods"""
    def __init__(self, method_name: str, image_path: str):
        self.method_name = method_name
        self.image_path = image_path
        self.center = None
        self.core_radius = None
        self.cladding_radius = None
        self.masks = None
        self.confidence = 0.5
        self.execution_time = 0.0
        self.error = None
        
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
    """Model aware voting system for combining segmentation results"""
    def __init__(self, min_agreement_ratio=0.3):
        self.min_agreement_ratio = min_agreement_ratio

    def _calculate_iou(self, mask1, mask2):
        """Calculates Intersection over Union for two binary masks."""
        if mask1 is None or mask2 is None:
            return 0.0
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou_score = np.sum(intersection) / (np.sum(union) + 1e-6)
        return iou_score

    def generate_consensus(self, results: List[SegmentationResult], 
                          method_scores: Dict[str, float], 
                          image_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Generates a final consensus model using weighted voting."""
        valid_results = [r for r in results if r.error is None and r.masks is not None]
        
        if len(valid_results) < 2:
            print("! Not enough valid results to form a consensus.")
            return None

        print(f"\nGenerating consensus from {len(valid_results)} valid results...")
        h, w = image_shape

        # Stage 1: Preliminary Weighted Pixel Vote
        weighted_votes = np.zeros((h, w, 3), dtype=np.float32)
        for r in valid_results:
            weight = method_scores.get(r.method_name, 1.0) * r.confidence
            if r.masks.get('core') is not None:
                weighted_votes[:, :, 0] += (r.masks['core'] > 0).astype(np.float32) * weight
            if r.masks.get('cladding') is not None:
                weighted_votes[:, :, 1] += (r.masks['cladding'] > 0).astype(np.float32) * weight
            if r.masks.get('ferrule') is not None:
                weighted_votes[:, :, 2] += (r.masks['ferrule'] > 0).astype(np.float32) * weight

        preliminary_classification = np.argmax(weighted_votes, axis=2)
        prelim_core_mask = (preliminary_classification == 0)
        prelim_cladding_mask = (preliminary_classification == 1)

        # Stage 2: Identify High-Agreement Methods
        high_agreement_results = []
        for r in valid_results:
            core_iou = self._calculate_iou(r.masks['core'], prelim_core_mask)
            cladding_iou = self._calculate_iou(r.masks['cladding'], prelim_cladding_mask)
            if core_iou > 0.6 and cladding_iou > 0.6:
                high_agreement_results.append(r)

        if not high_agreement_results:
            print("! No methods passed the high-agreement threshold. Using all valid results.")
            high_agreement_results = valid_results
        
        print(f"  Found {len(high_agreement_results)} methods for parameter averaging.")

        # Stage 3: Parameter-Space Consensus
        consensus_params = {'cx': [], 'cy': [], 'core_r': [], 'clad_r': []}
        weights = []
        for r in high_agreement_results:
            weight = method_scores.get(r.method_name, 1.0) * r.confidence
            if r.center and r.core_radius is not None and r.cladding_radius is not None:
                consensus_params['cx'].append(r.center[0])
                consensus_params['cy'].append(r.center[1])
                consensus_params['core_r'].append(r.core_radius)
                consensus_params['clad_r'].append(r.cladding_radius)
                weights.append(weight)

        if not weights:
            print("! No valid parameters to average. Consensus failed.")
            return None

        final_center = (
            np.average(consensus_params['cx'], weights=weights),
            np.average(consensus_params['cy'], weights=weights)
        )
        final_core_radius = np.average(consensus_params['core_r'], weights=weights)
        final_cladding_radius = np.average(consensus_params['clad_r'], weights=weights)

        # Stage 4: Generate Final Ideal Masks
        final_masks = self.create_masks_from_params(
            final_center, final_core_radius, final_cladding_radius, image_shape
        )
        
        final_masks['core'], final_masks['cladding'], final_masks['ferrule'] = self.ensure_mask_consistency(
            final_masks['core'], final_masks['cladding'], final_masks['ferrule']
        )
        
        return {
            'masks': final_masks,
            'center': final_center,
            'core_radius': final_core_radius,
            'cladding_radius': final_cladding_radius,
            'contributing_methods': [r.method_name for r in high_agreement_results],
            'num_valid_results': len(valid_results),
            'all_results': [r.to_dict() for r in results]
        }

    def create_masks_from_params(self, center: Tuple[float, float], core_radius: float, 
                               cladding_radius: float, image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Creates binary masks from geometric parameters."""
        h, w = image_shape
        cx, cy = center
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        
        core_mask = (dist_from_center <= core_radius).astype(np.uint8)
        cladding_mask = ((dist_from_center > core_radius) & 
                        (dist_from_center <= cladding_radius)).astype(np.uint8)
        ferrule_mask = (dist_from_center > cladding_radius).astype(np.uint8)
        
        return {'core': core_mask, 'cladding': cladding_mask, 'ferrule': ferrule_mask}

    def ensure_mask_consistency(self, core_mask, cladding_mask, ferrule_mask):
        """Ensure masks are mutually exclusive and spatially clean."""
        if not HAS_SCIPY_FULL:
            return core_mask, cladding_mask, ferrule_mask
            
        kernel = np.ones((5, 5), dtype=np.uint8)
        
        core_mask = binary_closing(binary_opening(core_mask, kernel), kernel).astype(np.uint8)
        cladding_mask = binary_closing(binary_opening(cladding_mask, kernel), kernel).astype(np.uint8)
        
        cladding_mask[core_mask == 1] = 0
        ferrule_mask[core_mask == 1] = 0
        ferrule_mask[cladding_mask == 1] = 0
        
        return core_mask, cladding_mask, ferrule_mask


class PylonFrameGrabber(threading.Thread):
    """A dedicated thread to continuously grab frames from a Basler pylon camera."""
    
    def __init__(self):
        super().__init__(name="PylonGrabber")
        self.daemon = True
        self.camera = None
        self.latest_frame = None
        self.is_running = threading.Event()
        self.lock = threading.Lock()
        
        if PYLON_AVAILABLE:
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def run(self):
        """The main loop of the grabbing thread."""
        logging.info("PylonFrameGrabber thread started.")
        
        if not PYLON_AVAILABLE:
            logging.critical("ERROR: Pylon SDK not available. Cannot use Basler camera.")
            return
            
        try:
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.camera.Open()
            logging.info(f"Using device: {self.camera.GetDeviceInfo().GetModelName()}")

            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_running.set()
            logging.info("Camera started grabbing frames.")

            while self.is_running.is_set():
                if not self.camera.IsGrabbing():
                    logging.warning("Camera stopped grabbing unexpectedly.")
                    break
                
                try:
                    grabResult = self.camera.RetrieveResult(
                        5000, pylon.TimeoutHandling_ThrowException
                    )
                    if grabResult.GrabSucceeded():
                        image = self.converter.Convert(grabResult)
                        frame = image.GetArray()
                        with self.lock:
                            self.latest_frame = frame.copy()
                    else:
                        logging.error(f"Grab failed: {grabResult.ErrorCode} "
                                   f"{grabResult.ErrorDescription}")
                    grabResult.Release()
                except GenericException as e:
                    logging.error(f"An error occurred while grabbing a frame: {e}")
                    time.sleep(0.1)
            
        except pylon.RuntimeException as e:
            logging.critical(f"Pylon runtime exception: {e}. Is a camera connected?")
        except Exception as e:
            logging.critical(f"An unexpected error occurred in PylonFrameGrabber: {e}", exc_info=True)
        finally:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
                logging.info("Camera stopped grabbing.")
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
                logging.info("Camera closed.")
            self.is_running.clear()
            logging.info("PylonFrameGrabber thread finished.")

    def read(self):
        """Returns the most recent frame."""
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        """Signals the thread to stop."""
        logging.info("Stopping PylonFrameGrabber thread.")
        self.is_running.clear()


class UnifiedSegmentationSystem:
    """Main unifier system that orchestrates all segmentation methods"""
    
    def __init__(self, methods_dir: str = "zones_methods"):
        self.methods_dir = Path(methods_dir)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.dataset_stats = {'method_scores': {}, 'method_accuracy': {}}
        
        self.knowledge_file = self.output_dir / "segmentation_knowledge.json"
        self.load_knowledge()
        
        self.methods = {}
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
                    print(f"✓ Loaded knowledge from {self.knowledge_file}")
            except Exception as e:
                print(f"! Could not load knowledge ({e}), starting fresh")
    
    def save_knowledge(self):
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.dataset_stats, f, indent=4, cls=NumpyEncoder)
        print(f"✓ Saved updated knowledge to {self.knowledge_file}")
    
    def load_methods(self):
        method_files = [
            'adaptive_intensity.py', 'bright_core_extractor.py', 'computational_separation.py',
            'geometric_approach.py', 'gradient_approach.py', 'guess_approach.py',
            'hough_separation.py', 'segmentation.py', 'threshold_separation.py',
            'unified_core_cladding_detector.py'
        ]
        
        for method_file in method_files:
            method_name = Path(method_file).stem
            method_path = self.methods_dir / method_file
            if method_path.exists():
                self.methods[method_name] = {
                    'path': method_path,
                    'score': self.dataset_stats['method_scores'].get(method_name, 1.0)
                }
                print(f"✓ Loaded method: {method_name} (score: {self.methods[method_name]['score']:.2f})")

    def detect_and_inpaint_anomalies(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect and inpaint anomalies in the image."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        _, defect_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
        if HAS_SCIPY_FULL:
            defect_mask = binary_opening(defect_mask, structure=np.ones((3,3)), iterations=2).astype(np.uint8)
        inpainted_image = cv2.inpaint(image, defect_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return inpainted_image, defect_mask

    def run_method_isolated(self, method_name: str, image_path: Path, temp_output: Path) -> dict:
        """Run a method in isolated subprocess and capture JSON output."""
        result_file = temp_output / f"{method_name}_result.json"
        runner_script_path = temp_output / "runner.py"
        
        script_content = f"""
import sys, json, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, r"{self.methods_dir.resolve()}")

def main():
    image_path_str = r"{image_path.resolve()}"
    output_dir_str = r"{temp_output.resolve()}"
    result = {{'success': False, 'error': 'Unknown execution error'}}
    try:
"""
        
        method_map = {
            'adaptive_intensity': "from adaptive_intensity import adaptive_segment_image\n        result = adaptive_segment_image(image_path_str, output_dir=output_dir_str)",
            'bright_core_extractor': "from bright_core_extractor import analyze_core\n        result = analyze_core(image_path_str, output_dir_str)",
            'computational_separation': "from computational_separation import process_fiber_image_veridian\n        result = process_fiber_image_veridian(image_path_str, output_dir_str)",
            'geometric_approach': "from geometric_approach import segment_with_geometric\n        result = segment_with_geometric(image_path_str, output_dir_str)",
            'gradient_approach': "from gradient_approach import segment_with_gradient\n        result = segment_with_gradient(image_path_str, output_dir_str)",
            'guess_approach': "from guess_approach import segment_fiber_with_multimodal_analysis\n        result = segment_fiber_with_multimodal_analysis(image_path_str, output_dir_str)",
            'hough_separation': "from hough_separation import segment_with_hough\n        result = segment_with_hough(image_path_str, output_dir_str)",
            'segmentation': "from segmentation import run_segmentation_pipeline, DEFAULT_CONFIG\n        pipeline_result = run_segmentation_pipeline(Path(image_path_str), {{}}, DEFAULT_CONFIG, Path(output_dir_str))\n        result = pipeline_result['result'] if pipeline_result and 'result' in pipeline_result else {{'success': False, 'error': 'Pipeline failed'}}",
            'threshold_separation': "from threshold_separation import segment_with_threshold\n        result = segment_with_threshold(image_path_str, output_dir_str)",
            'unified_core_cladding_detector': "from unified_core_cladding_detector import detect_core_cladding\n        result = detect_core_cladding(image_path_str, output_dir_str)",
        }
        
        call_logic = method_map.get(method_name)
        if not call_logic:
            return {'success': False, 'error': f'Runner for method {method_name} not implemented.'}
        script_content += f"        {call_logic}\n"

        script_content += f"""
    except Exception as e:
        import traceback
        result['error'] = f"Exception in {{method_name}}: {{e}}\\n{{traceback.format_exc()}}"
    
    with open(r"{result_file.resolve()}", 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
"""
        with open(runner_script_path, 'w') as f:
            f.write(script_content)

        try:
            subprocess.run(
                [sys.executable, str(runner_script_path)],
                capture_output=True, text=True, timeout=120, check=False,
                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen', 'MPLBACKEND': 'Agg'}
            )
            if result_file.exists():
                with open(result_file, 'r') as f:
                    return json.load(f)
            return {'success': False, 'error': 'No result file produced.'}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Method timed out'}
        except Exception as e:
            return {'success': False, 'error': f'Subprocess execution failed: {e}'}

    def run_method(self, method_name: str, image_path: Path, image_shape: Tuple[int, int]) -> SegmentationResult:
        result = SegmentationResult(method_name, str(image_path))
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            method_output = self.run_method_isolated(method_name, image_path, Path(temp_dir))
            
            if method_output and method_output.get('success'):
                result.center = tuple(method_output.get('center')) if method_output.get('center') else None
                result.core_radius = method_output.get('core_radius')
                result.cladding_radius = method_output.get('cladding_radius')
                result.confidence = method_output.get('confidence', 0.5)

                if all([result.center, result.core_radius, result.cladding_radius]):
                    result.masks = self.consensus_system.create_masks_from_params(
                        result.center, result.core_radius, result.cladding_radius, image_shape
                    )
                    if result.masks and result.masks.get('core') is not None:
                        contours, _ = cv2.findContours(result.masks['core'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cnt = max(contours, key=cv2.contourArea)
                            area = cv2.contourArea(cnt)
                            perimeter = cv2.arcLength(cnt, True)
                            if perimeter > 0:
                                circularity = (4 * np.pi * area) / (perimeter**2)
                                if circularity < 0.85:
                                    result.confidence *= 0.5
                                    print(f"  ! Penalizing {method_name} for low circularity ({circularity:.2f})")
                else:
                    result.error = "Method returned invalid/missing parameters."
            else:
                result.error = method_output.get('error', 'Unknown failure') if method_output else "Empty method output"
        
        result.execution_time = time.time() - start_time
        return result

    def update_learning(self, consensus: Dict, all_results: List[SegmentationResult]):
        print("\nUpdating learning model...")
        consensus_masks = consensus['masks']
        
        for result in all_results:
            if result.error or not result.masks:
                continue
            core_iou = self.consensus_system._calculate_iou(result.masks.get('core'), consensus_masks.get('core'))
            cladding_iou = self.consensus_system._calculate_iou(result.masks.get('cladding'), consensus_masks.get('cladding'))
            avg_iou = (core_iou + cladding_iou) / 2
            
            current_score = self.dataset_stats['method_scores'].get(result.method_name, 1.0)
            learning_rate = 0.1
            target_score = 0.1 + (1.9 * avg_iou)
            new_score = current_score * (1 - learning_rate) + target_score * learning_rate
            
            self.dataset_stats['method_scores'][result.method_name] = new_score
            self.dataset_stats['method_accuracy'][result.method_name] = avg_iou
            self.methods[result.method_name]['score'] = new_score
        
        print("  ✓ Method scores updated.")
        self.save_knowledge()

    def process_frame_realtime(self, frame: np.ndarray, frame_id: int = None) -> Optional[Dict]:
        """Process a single frame in real-time mode."""
        try:
            # Create temporary file for frame
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
                cv2.imwrite(str(temp_path), frame)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_output:
                output_dir = Path(temp_output)
                
                # Process using existing pipeline but skip heavy I/O
                consensus = self._process_image_lightweight(temp_path, output_dir, frame.shape[:2])
                
                # Cleanup
                temp_path.unlink()
                
                return consensus
                
        except Exception as e:
            print(f"ERROR in real-time frame processing: {e}")
            return None

    def _process_image_lightweight(self, image_path: Path, output_dir: Path, image_shape: Tuple[int, int]) -> Optional[Dict]:
        """Lightweight version of process_image for real-time processing."""
        print(f"🔄 Processing frame: {image_path.name}")
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            return None
        
        # Run subset of most reliable methods for speed
        priority_methods = self._get_priority_methods_for_realtime()
        
        all_results = []
        for method_name in priority_methods:
            print(f"Running {method_name}...")
            result = self.run_method(method_name, image_path, image_shape)
            all_results.append(result)
            
            if result.error:
                print(f"  ✗ Failed: {result.error}")
            else:
                print(f"  ✓ Success - Confidence: {result.confidence:.2f}")
        
        # Generate consensus
        consensus = self.consensus_system.generate_consensus(
            all_results, 
            {name: info['score'] for name, info in self.methods.items()}, 
            image_shape
        )
        
        if consensus:
            print("✓ Real-time consensus achieved.")
            self.update_learning(consensus, all_results)
        
        return consensus

    def _get_priority_methods_for_realtime(self) -> List[str]:
        """Get priority methods for real-time processing."""
        method_priorities = {
            'geometric_approach': 0.9,
            'threshold_separation': 0.8,
            'hough_separation': 0.7,
            'unified_core_cladding_detector': 0.6,
            'adaptive_intensity': 0.5,
        }
        
        available_methods = []
        for method_name in self.methods.keys():
            if method_name in method_priorities:
                priority = method_priorities[method_name] * self.methods[method_name]['score']
                available_methods.append((method_name, priority))
        
        available_methods.sort(key=lambda x: x[1], reverse=True)
        return [method[0] for method in available_methods[:4]]


class RealtimeSegmentationProcessor:
    """Real-time processor that combines Pylon camera with segmentation system."""
    
    def __init__(self, methods_dir: str = "zones_methods", buffer_size: int = 10):
        # Core components
        self.segmentation_system = UnifiedSegmentationSystem(methods_dir)
        self.frame_grabber = PylonFrameGrabber() if PYLON_AVAILABLE else None
        
        # Threading and synchronization
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.display_thread = None
        
        # Control flags
        self.is_running = threading.Event()
        self.should_process = threading.Event()
        self.frame_counter = 0
        self.processed_counter = 0
        
        # Configuration
        self.process_every_n_frames = 30
        self.adaptive_processing = True
        self.min_processing_interval = 10
        self.max_processing_interval = 120
        
        # Performance tracking
        self.last_process_time = 0
        self.avg_processing_time = 2.0
        self.target_fps = 30
        
        # Results storage
        self.latest_consensus = None
        self.latest_masks = None
        self.performance_history = []
        
        print(f"✓ Real-time processor initialized")
        if not PYLON_AVAILABLE:
            print("WARNING: Pylon not available - running in simulation mode")

    def start_camera(self) -> bool:
        """Start the camera and frame grabbing."""
        if not PYLON_AVAILABLE:
            print("ERROR: Pylon SDK not available")
            return False
            
        try:
            self.frame_grabber.start()
            
            # Wait for camera to start grabbing
            timeout = 10
            start_time = time.time()
            while not self.frame_grabber.is_running.is_set():
                if time.time() - start_time > timeout:
                    print("ERROR: Camera startup timeout")
                    return False
                time.sleep(0.1)
            
            print("✓ Camera started successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start camera: {e}")
            return False

    def stop_camera(self):
        """Stop the camera and cleanup."""
        if self.frame_grabber:
            self.frame_grabber.stop()
            if self.frame_grabber.is_alive():
                self.frame_grabber.join(timeout=5)
            print("✓ Camera stopped")

    def frame_producer_thread(self):
        """Thread function that continuously grabs frames from camera."""
        print("📹 Frame producer thread started")
        
        while self.is_running.is_set():
            if not self.frame_grabber or not self.frame_grabber.is_running.is_set():
                time.sleep(0.1)
                continue
                
            # Get latest frame from grabber
            frame = self.frame_grabber.read()
            if frame is None:
                time.sleep(0.01)
                continue
                
            self.frame_counter += 1
            
            # Decide whether to queue this frame for processing
            should_queue = False
            
            if self.adaptive_processing:
                frames_since_last = self.frame_counter - self.processed_counter
                if frames_since_last >= self.process_every_n_frames:
                    should_queue = True
            else:
                if self.frame_counter % self.process_every_n_frames == 0:
                    should_queue = True
            
            if should_queue and self.should_process.is_set():
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame.copy(),
                        'frame_number': self.frame_counter,
                        'timestamp': time.time()
                    })
                except queue.Full:
                    pass
            
            time.sleep(0.001)
        
        print("📹 Frame producer thread finished")

    def frame_processor_thread(self):
        """Thread function that processes queued frames."""
        print("🔧 Frame processor thread started")
        
        while self.is_running.is_set() or not self.frame_queue.empty():
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                
                if not self.should_process.is_set():
                    continue
                
                print(f"\n🔄 Processing frame {frame_data['frame_number']} "
                      f"(queue size: {self.frame_queue.qsize()})")
                
                # Process the frame
                start_time = time.time()
                consensus = self.segmentation_system.process_frame_realtime(
                    frame_data['frame'], 
                    frame_data['frame_number']
                )
                processing_time = time.time() - start_time
                
                # Update performance tracking
                self.avg_processing_time = (
                    0.8 * self.avg_processing_time + 0.2 * processing_time
                )
                
                # Adaptive processing interval adjustment
                if self.adaptive_processing:
                    self._adjust_processing_interval(processing_time)
                
                # Store results
                if consensus:
                    self.latest_consensus = consensus
                    self.latest_masks = consensus['masks']
                    
                    # Put result in result queue for display
                    self.result_queue.put({
                        'frame_number': frame_data['frame_number'],
                        'consensus': consensus,
                        'processing_time': processing_time,
                        'timestamp': time.time()
                    })
                
                self.processed_counter = frame_data['frame_number']
                self.performance_history.append({
                    'frame_number': frame_data['frame_number'],
                    'processing_time': processing_time,
                    'timestamp': time.time()
                })
                
                # Keep only recent history
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-50:]
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"ERROR in frame processor: {e}")
                continue
        
        print("🔧 Frame processor thread finished")

    def _adjust_processing_interval(self, processing_time: float):
        """Dynamically adjust processing interval based on performance."""
        target_interval = max(
            int(processing_time * self.target_fps * 2),
            self.min_processing_interval
        )
        target_interval = min(target_interval, self.max_processing_interval)
        
        self.process_every_n_frames = int(
            0.7 * self.process_every_n_frames + 0.3 * target_interval
        )
        
        print(f"📊 Adjusted processing interval to {self.process_every_n_frames} frames "
              f"(processing time: {processing_time:.2f}s)")

    def display_thread_func(self):
        """Thread function for displaying live feed with overlay."""
        print("🖥️ Display thread started")
        
        cv2.namedWindow('Real-time Fiber Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Fiber Segmentation', 1200, 800)
        
        last_display_frame = None
        last_result = None
        
        while self.is_running.is_set():
            # Get latest frame for display
            if self.frame_grabber:
                current_frame = self.frame_grabber.read()
                if current_frame is not None:
                    last_display_frame = current_frame.copy()
            
            # Check for new processing results
            try:
                result = self.result_queue.get_nowait()
                last_result = result
            except queue.Empty:
                pass
            
            # Create display frame
            if last_display_frame is not None:
                display_frame = self._create_display_frame(last_display_frame, last_result)
                cv2.imshow('Real-time Fiber Segmentation', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quit requested by user")
                self.stop()
                break
            elif key == ord('p'):
                if self.should_process.is_set():
                    self.should_process.clear()
                    print("⏸️ Processing paused")
                else:
                    self.should_process.set()
                    print("▶️ Processing resumed")
            elif key == ord('s'):
                if last_result:
                    self._save_current_results(last_display_frame, last_result)
            
            time.sleep(0.033)
        
        cv2.destroyAllWindows()
        print("🖥️ Display thread finished")

    def _create_display_frame(self, frame: np.ndarray, result: Optional[Dict]) -> np.ndarray:
        """Create display frame with overlays."""
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Add status overlay
        status_text = f"Frame: {self.frame_counter} | Processed: {self.processed_counter}"
        status_text += f" | Queue: {self.frame_queue.qsize()}"
        status_text += f" | Interval: {self.process_every_n_frames}"
        
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add processing status
        if self.should_process.is_set():
            cv2.putText(display_frame, "PROCESSING: ON", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "PROCESSING: OFF", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add segmentation overlay if available
        if result and result.get('consensus'):
            consensus = result['consensus']
            
            # Draw center and radii
            if consensus.get('center') and consensus.get('core_radius'):
                center = (int(consensus['center'][0]), int(consensus['center'][1]))
                core_radius = int(consensus['core_radius'])
                cladding_radius = int(consensus['cladding_radius'])
                
                # Draw circles
                cv2.circle(display_frame, center, core_radius, (0, 255, 0), 2)
                cv2.circle(display_frame, center, cladding_radius, (255, 0, 0), 2)
                cv2.circle(display_frame, center, 3, (0, 0, 255), -1)
                
                # Add text info
                info_text = f"Core: {core_radius}px | Cladding: {cladding_radius}px"
                cv2.putText(display_frame, info_text, (10, height - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                methods_text = f"Methods: {', '.join(consensus.get('contributing_methods', []))}"
                cv2.putText(display_frame, methods_text, (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add controls help
        help_text = "Controls: Q=Quit, P=Pause/Resume, S=Save"
        cv2.putText(display_frame, help_text, (width - 400, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame

    def _save_current_results(self, frame: np.ndarray, result: Dict):
        """Save current frame and segmentation results."""
        timestamp = int(time.time())
        save_dir = Path(f"realtime_results_{timestamp}")
        save_dir.mkdir(exist_ok=True)
        
        # Save original frame
        cv2.imwrite(str(save_dir / "frame.png"), frame)
        
        # Save consensus results if available
        if result.get('consensus'):
            consensus = result['consensus']
            
            # Save JSON report
            with open(save_dir / "consensus_report.json", 'w') as f:
                json.dump(consensus, f, indent=4, cls=NumpyEncoder)
            
            # Save masks if available
            masks = consensus.get('masks')
            if masks:
                cv2.imwrite(str(save_dir / "mask_core.png"), masks['core'] * 255)
                cv2.imwrite(str(save_dir / "mask_cladding.png"), masks['cladding'] * 255)
                cv2.imwrite(str(save_dir / "mask_ferrule.png"), masks['ferrule'] * 255)
        
        print(f"💾 Results saved to {save_dir}")

    def start(self):
        """Start the real-time processing system."""
        if not PYLON_AVAILABLE:
            print("ERROR: Cannot start - Pylon SDK not available")
            return False
        
        print("🚀 Starting real-time segmentation system...")
        
        # Start camera
        if not self.start_camera():
            return False
        
        # Set running flag
        self.is_running.set()
        self.should_process.set()
        
        # Start threads
        self.processing_thread = threading.Thread(
            target=self.frame_processor_thread, 
            name="FrameProcessor"
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start frame producer thread
        producer_thread = threading.Thread(
            target=self.frame_producer_thread,
            name="FrameProducer"
        )
        producer_thread.daemon = True
        producer_thread.start()
        
        # Start display thread
        self.display_thread = threading.Thread(
            target=self.display_thread_func,
            name="DisplayThread"
        )
        self.display_thread.daemon = True
        self.display_thread.start()
        
        print("✅ Real-time system started successfully")
        print("\nControls:")
        print("  Q - Quit application")
        print("  P - Pause/Resume processing")
        print("  S - Save current results")
        print("\nWaiting for threads to finish...")
        
        # Wait for threads to complete
        try:
            self.display_thread.join()
            self.processing_thread.join()
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.stop()
        
        return True

    def stop(self):
        """Stop the real-time processing system."""
        print("🛑 Stopping real-time system...")
        
        # Clear running flags
        self.is_running.clear()
        self.should_process.clear()
        
        # Stop camera
        self.stop_camera()
        
        print("✅ Real-time system stopped")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_history:
            return {}
        
        recent_times = [p['processing_time'] for p in self.performance_history[-10:]]
        
        return {
            'total_frames': self.frame_counter,
            'processed_frames': self.processed_counter,
            'processing_rate': self.processed_counter / max(self.frame_counter, 1),
            'avg_processing_time': self.avg_processing_time,
            'recent_avg_time': sum(recent_times) / len(recent_times) if recent_times else 0,
            'current_interval': self.process_every_n_frames,
            'queue_size': self.frame_queue.qsize()
        }


def main():
    """Main function to run the real-time segmentation system."""
    import sys
    
    methods_dir = sys.argv[1] if len(sys.argv) > 1 else "zones_methods"
    
    # Create and start the real-time processor
    processor = RealtimeSegmentationProcessor(methods_dir)
    
    try:
        success = processor.start()
        if not success:
            print("Failed to start real-time processor")
            return
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        processor.stop()
        
        # Print final statistics
        stats = processor.get_performance_stats()
        if stats:
            print("\n📊 Final Performance Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 