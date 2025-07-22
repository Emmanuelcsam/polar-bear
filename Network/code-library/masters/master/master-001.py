#!/usr/bin/env python3
"""
Automated Image Processing Studio
=================================
An intelligent image processing system that automatically finds the right combination
of processing scripts to match a target image using reinforcement learning and
similarity analysis.

Key Features:
- Automatic script combination discovery
- Reinforcement learning for optimization
- Anomaly and similarity detection
- RAM-based processing
- Comprehensive logging
- Auto-dependency installation
"""

import os
import sys
import json
import pickle
import hashlib
import time
import shutil
import subprocess
import importlib.util
import importlib.metadata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')


class DependencyManager:
    """Manages automatic dependency detection and installation"""
    
    REQUIRED_PACKAGES = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'pillow': 'PIL',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'psutil': 'psutil'
    }
    
    @classmethod
    def check_and_install_dependencies(cls):
        """Check and install missing dependencies"""
        missing_packages = []
        
        print("🔍 Checking dependencies...")
        
        for package_name, import_name in cls.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(import_name)
                print(f"✓ {package_name} is installed")
            except ImportError:
                print(f"✗ {package_name} is missing")
                missing_packages.append(package_name)
        
        if missing_packages:
            print("\n📦 Installing missing packages...")
            for package in missing_packages:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ])
                    print(f"✓ Successfully installed {package}")
                except subprocess.CalledProcessError:
                    print(f"✗ Failed to install {package}")
                    raise RuntimeError(f"Failed to install required package: {package}")
        
        print("\n✅ All dependencies are satisfied!\n")


# Import after dependency check
try:
    import numpy as np
    import cv2
    from sklearn.metrics import mean_squared_error
    from sklearn.decomposition import PCA
except ImportError as e:
    print(f"❌ Failed to import required module: {e}")
    print("Please run DependencyManager.check_and_install_dependencies() first")
    sys.exit(1)


class ImageProcessor:
    """Handles individual image processing operations"""
    
    def __init__(self):
        self.cache = {}
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        if image.dtype != np.uint8:
            # Handle floating point images
            if image.min() < 0 or image.max() > 1:
                image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)
        return image
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features from an image for comparison"""
        gray = self.to_grayscale(image)
        
        features = {
            'shape': image.shape,
            'mean': np.mean(gray),
            'std': np.std(gray),
            'histogram': cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten(),
            'edges': cv2.Canny(gray, 50, 150).mean(),
            'corners': cv2.cornerHarris(gray, 2, 3, 0.04).mean()
        }
        
        # Add texture features using Gabor filters
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0)
            kernels.append(kernel)
        
        gabor_features = []
        for kernel in kernels:
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_features.append(filtered.mean())
        
        features['gabor'] = np.array(gabor_features)
        
        return features
    
    def calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity between two images"""
        # Ensure same size
        if img1.shape != img2.shape:
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        # Convert to grayscale for comparison
        gray1 = self.to_grayscale(img1)
        gray2 = self.to_grayscale(img2)
        
        # Multiple similarity metrics
        mse = mean_squared_error(gray1.flatten(), gray2.flatten())
        ssim = self._calculate_ssim(gray1, gray2)
        hist_similarity = cv2.compareHist(
            cv2.calcHist([gray1], [0], None, [256], [0, 256]),
            cv2.calcHist([gray2], [0], None, [256], [0, 256]),
            cv2.HISTCMP_CORREL
        )
        
        # Combine metrics (lower is better)
        similarity = (1 - ssim) + (mse / 255**2) + (1 - hist_similarity)
        return similarity / 3
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate structural similarity index"""
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


class ScriptManager:
    """Manages loading and execution of processing scripts"""
    
    def __init__(self, scripts_dir: str = "scripts"):
        self.scripts_dir = Path(scripts_dir)
        self.functions = {}
        self.function_info = {}
        self._load_scripts()
    
    def _load_scripts(self):
        """Load all processing scripts"""
        if not self.scripts_dir.exists():
            print(f"Scripts directory '{self.scripts_dir}' not found")
            return
        
        # Look for scripts in main directory and subdirectories
        script_files = list(self.scripts_dir.rglob("*.py"))
        
        for script_path in script_files:
            if script_path.name.startswith("_"):
                continue
            
            try:
                # Load module
                spec = importlib.util.spec_from_file_location(
                    script_path.stem, script_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check for process_image function
                if hasattr(module, 'process_image'):
                    rel_path = script_path.relative_to(self.scripts_dir)
                    func_name = str(rel_path)
                    
                    self.functions[func_name] = module.process_image
                    self.function_info[func_name] = {
                        'path': script_path,
                        'category': self._categorize_script(script_path.name)
                    }
            except Exception as e:
                continue
    
    def _categorize_script(self, filename: str) -> str:
        """Categorize script based on filename"""
        filename_lower = filename.lower()
        
        categories = {
            'blur': 'filtering',
            'filter': 'filtering',
            'gaussian': 'filtering',
            'median': 'filtering',
            'edge': 'edge_detection',
            'canny': 'edge_detection',
            'sobel': 'edge_detection',
            'threshold': 'thresholding',
            'otsu': 'thresholding',
            'binary': 'thresholding',
            'morph': 'morphology',
            'erode': 'morphology',
            'dilate': 'morphology',
            'enhance': 'enhancement',
            'histogram': 'enhancement',
            'clahe': 'enhancement',
            'circle': 'detection',
            'detect': 'detection',
            'contour': 'detection',
            'color': 'color',
            'gray': 'color',
            'hsv': 'color'
        }
        
        for keyword, category in categories.items():
            if keyword in filename_lower:
                return category
        
        return 'other'
    
    def get_scripts_by_category(self) -> Dict[str, List[str]]:
        """Get scripts organized by category"""
        categorized = defaultdict(list)
        for func_name, info in self.function_info.items():
            categorized[info['category']].append(func_name)
        return dict(categorized)


class ReinforcementLearner:
    """Implements reinforcement learning for script combination optimization"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.memory = deque(maxlen=2000)
    
    def get_state_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key"""
        return hashlib.md5(state.tobytes()).hexdigest()
    
    def choose_action(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """Update Q-values based on experience"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] = current_q + \
            self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def replay(self, batch_size: int = 32):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            self.learn(state, action, reward, next_state, done)


class AnomalyDetector:
    """Detects anomalies and similarities in processed images"""
    
    def __init__(self):
        self.anomaly_library = []
        self.similarity_library = []
        self.pca = PCA(n_components=50)
        self.is_fitted = False
    
    def add_to_library(self, image: np.ndarray, features: Dict[str, Any], 
                      is_anomaly: bool):
        """Add image features to appropriate library"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'shape': image.shape,
            'hash': hashlib.md5(image.tobytes()).hexdigest()
        }
        
        if is_anomaly:
            self.anomaly_library.append(entry)
        else:
            self.similarity_library.append(entry)
    
    def detect_anomalies(self, image: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if image contains anomalies"""
        # Flatten features for analysis
        feature_vector = self._flatten_features(features)
        
        if len(self.similarity_library) > 10 and not self.is_fitted:
            # Fit PCA on normal samples
            normal_features = [self._flatten_features(entry['features']) 
                             for entry in self.similarity_library]
            self.pca.fit(normal_features)
            self.is_fitted = True
        
        anomaly_score = 0
        anomaly_reasons = []
        
        # Check against known anomalies
        if self.anomaly_library:
            for anomaly in self.anomaly_library:
                similarity = self._compare_features(features, anomaly['features'])
                if similarity > 0.8:
                    anomaly_score += similarity
                    anomaly_reasons.append(f"Similar to known anomaly (score: {similarity:.2f})")
        
        # Statistical anomaly detection
        if self.is_fitted:
            try:
                reconstructed = self.pca.inverse_transform(
                    self.pca.transform([feature_vector])
                )[0]
                reconstruction_error = np.mean((feature_vector - reconstructed) ** 2)
                
                if reconstruction_error > 0.1:
                    anomaly_score += reconstruction_error
                    anomaly_reasons.append(f"High reconstruction error: {reconstruction_error:.3f}")
            except:
                pass
        
        return {
            'is_anomaly': anomaly_score > 0.5,
            'anomaly_score': anomaly_score,
            'reasons': anomaly_reasons
        }
    
    def find_similar(self, features: Dict[str, Any], top_k: int = 5) -> List[Dict]:
        """Find similar images from library"""
        similarities = []
        
        for entry in self.similarity_library:
            similarity = self._compare_features(features, entry['features'])
            similarities.append({
                'entry': entry,
                'similarity': similarity
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def _flatten_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Flatten features dictionary to vector"""
        flat_features = []
        
        for key, value in features.items():
            if key == 'histogram':
                flat_features.extend(value[:50])  # Use first 50 bins
            elif key == 'gabor':
                flat_features.extend(value)
            elif key in ['mean', 'std', 'edges', 'corners']:
                flat_features.append(value)
        
        return np.array(flat_features)
    
    def _compare_features(self, feat1: Dict, feat2: Dict) -> float:
        """Compare two feature dictionaries"""
        similarities = []
        
        # Compare histograms
        if 'histogram' in feat1 and 'histogram' in feat2:
            hist_sim = cv2.compareHist(
                feat1['histogram'].astype(np.float32),
                feat2['histogram'].astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            similarities.append(hist_sim)
        
        # Compare scalar features
        for key in ['mean', 'std', 'edges', 'corners']:
            if key in feat1 and key in feat2:
                diff = abs(feat1[key] - feat2[key]) / (max(feat1[key], feat2[key]) + 1e-10)
                similarities.append(1 - diff)
        
        return np.mean(similarities) if similarities else 0


class AutomatedProcessingStudio:
    """Main class for the automated image processing studio"""
    
    def __init__(self, scripts_dir: str = "scripts", cache_dir: str = ".studio_cache"):
        self.scripts_dir = scripts_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.processor = ImageProcessor()
        self.script_manager = ScriptManager(scripts_dir)
        self.anomaly_detector = AnomalyDetector()
        
        # Initialize reinforcement learner
        self.num_scripts = len(self.script_manager.functions)
        self.learner = ReinforcementLearner(
            state_size=100,  # Feature vector size
            action_size=self.num_scripts
        )
        
        # Processing history
        self.processing_history = []
        self.successful_combinations = []
        self.failed_combinations = []
        
        # Load saved state if exists
        self._load_state()
    
    def _load_state(self):
        """Load saved state from disk"""
        state_file = self.cache_dir / "studio_state.pkl"
        if state_file.exists():
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.learner.q_table = state.get('q_table', self.learner.q_table)
                    self.learner.epsilon = state.get('epsilon', self.learner.epsilon)
                    self.successful_combinations = state.get('successful_combinations', [])
                    self.failed_combinations = state.get('failed_combinations', [])
                    self.anomaly_detector.anomaly_library = state.get('anomaly_library', [])
                    self.anomaly_detector.similarity_library = state.get('similarity_library', [])
                print("📂 Loaded previous state from cache")
            except Exception as e:
                print(f"⚠️ Failed to load state: {e}")
    
    def _save_state(self):
        """Save current state to disk"""
        state_file = self.cache_dir / "studio_state.pkl"
        try:
            state = {
                'q_table': dict(self.learner.q_table),
                'epsilon': self.learner.epsilon,
                'successful_combinations': self.successful_combinations,
                'failed_combinations': self.failed_combinations,
                'anomaly_library': self.anomaly_detector.anomaly_library,
                'similarity_library': self.anomaly_detector.similarity_library
            }
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"⚠️ Failed to save state: {e}")
    
    def process_to_match(self, input_image: np.ndarray, target_image: np.ndarray,
                        max_iterations: int = 100, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """Find script combination to transform input image to match target"""
        print("\n🎯 Starting automated processing to match target image...")
        
        start_time = time.time()
        results = {
            'success': False,
            'iterations': 0,
            'final_similarity': 1.0,
            'pipeline': [],
            'processing_log': [],
            'anomalies_detected': [],
            'similarities_found': []
        }
        
        # Extract target features
        target_features = self.processor.extract_features(target_image)
        target_state = self.anomaly_detector._flatten_features(target_features)
        
        # Initialize
        current_image = input_image.copy()
        best_image = current_image.copy()
        best_similarity = 1.0
        best_pipeline = []
        
        script_names = list(self.script_manager.functions.keys())
        
        for iteration in range(max_iterations):
            # Get current state
            current_features = self.processor.extract_features(current_image)
            current_state = self.anomaly_detector._flatten_features(current_features)
            
            # Calculate similarity
            similarity = self.processor.calculate_similarity(current_image, target_image)
            
            # Log progress
            log_entry = {
                'iteration': iteration,
                'similarity': similarity,
                'timestamp': datetime.now().isoformat()
            }
            results['processing_log'].append(log_entry)
            
            # Check if we've reached the target
            if similarity < similarity_threshold:
                results['success'] = True
                results['final_similarity'] = similarity
                results['pipeline'] = best_pipeline
                print(f"\n✅ Success! Matched target with similarity: {similarity:.4f}")
                break
            
            # Choose action (script to apply)
            action = self.learner.choose_action(current_state)
            script_name = script_names[action]
            
            try:
                # Apply script
                script_func = self.script_manager.functions[script_name]
                processed = script_func(current_image)
                processed = self.processor.normalize_image(processed)
                
                # Calculate reward
                new_similarity = self.processor.calculate_similarity(processed, target_image)
                reward = (similarity - new_similarity) * 10  # Positive if improvement
                
                # Update learner
                new_features = self.processor.extract_features(processed)
                new_state = self.anomaly_detector._flatten_features(new_features)
                self.learner.learn(current_state, action, reward, new_state, False)
                
                # Check for anomalies
                anomaly_result = self.anomaly_detector.detect_anomalies(processed, new_features)
                if anomaly_result['is_anomaly']:
                    results['anomalies_detected'].append({
                        'iteration': iteration,
                        'script': script_name,
                        'details': anomaly_result
                    })
                
                # Update if better
                if new_similarity < best_similarity:
                    best_similarity = new_similarity
                    best_image = processed.copy()
                    best_pipeline = results['pipeline'].copy() + [script_name]
                    current_image = processed
                    
                    # Add to similarity library
                    self.anomaly_detector.add_to_library(
                        processed, new_features, is_anomaly=False
                    )
                    
                    print(f"  ✓ Iteration {iteration}: Applied {script_name} "
                          f"(similarity: {new_similarity:.4f})")
                else:
                    # Occasionally accept worse results for exploration
                    if np.random.random() < 0.1:
                        current_image = processed
                
                results['pipeline'].append(script_name)
                
            except Exception as e:
                print(f"  ✗ Error applying {script_name}: {e}")
                reward = -1
                self.learner.learn(current_state, action, reward, current_state, True)
            
            # Periodic status update
            if iteration % 10 == 0:
                print(f"  → Progress: {iteration}/{max_iterations} iterations, "
                      f"best similarity: {best_similarity:.4f}")
        
        # Final results
        results['iterations'] = iteration + 1
        results['final_similarity'] = best_similarity
        results['final_image'] = best_image
        results['processing_time'] = time.time() - start_time
        
        # Update successful/failed combinations
        if results['success']:
            self.successful_combinations.append({
                'pipeline': best_pipeline,
                'similarity': best_similarity,
                'timestamp': datetime.now().isoformat()
            })
        else:
            self.failed_combinations.append({
                'pipeline': results['pipeline'][-10:],  # Last 10 attempts
                'best_similarity': best_similarity,
                'timestamp': datetime.now().isoformat()
            })
        
        # Save state
        self._save_state()
        
        # Generate detailed report
        self._generate_report(results, input_image, target_image, best_image)
        
        return results
    
    def find_anomalies_and_similarities(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image for anomalies and find similar processed images"""
        print("\n🔍 Analyzing image for anomalies and similarities...")
        
        features = self.processor.extract_features(image)
        
        # Detect anomalies
        anomaly_result = self.anomaly_detector.detect_anomalies(image, features)
        
        # Find similar images
        similar_images = self.anomaly_detector.find_similar(features)
        
        results = {
            'anomaly_analysis': anomaly_result,
            'similar_images': similar_images,
            'features': features
        }
        
        print(f"\n📊 Analysis Results:")
        print(f"  - Anomaly detected: {anomaly_result['is_anomaly']}")
        print(f"  - Anomaly score: {anomaly_result['anomaly_score']:.3f}")
        print(f"  - Similar images found: {len(similar_images)}")
        
        return results
    
    def _generate_report(self, results: Dict, input_image: np.ndarray,
                        target_image: np.ndarray, output_image: np.ndarray):
        """Generate detailed processing report"""
        report_dir = self.cache_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir.mkdir(exist_ok=True)
        
        # Save images
        cv2.imwrite(str(report_dir / "input.png"), input_image)
        cv2.imwrite(str(report_dir / "target.png"), target_image)
        cv2.imwrite(str(report_dir / "output.png"), output_image)
        
        # Create comparison image
        h = max(input_image.shape[0], target_image.shape[0], output_image.shape[0])
        w = input_image.shape[1] + target_image.shape[1] + output_image.shape[1] + 20
        
        comparison = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Place images
        comparison[:input_image.shape[0], :input_image.shape[1]] = \
            cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR) if len(input_image.shape) == 2 else input_image
        
        x_offset = input_image.shape[1] + 10
        comparison[:target_image.shape[0], x_offset:x_offset+target_image.shape[1]] = \
            cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR) if len(target_image.shape) == 2 else target_image
        
        x_offset += target_image.shape[1] + 10
        comparison[:output_image.shape[0], x_offset:x_offset+output_image.shape[1]] = \
            cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR) if len(output_image.shape) == 2 else output_image
        
        cv2.imwrite(str(report_dir / "comparison.png"), comparison)
        
        # Generate text report
        report_text = f"""
Automated Processing Studio Report
=================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Processing Summary:
------------------
Success: {results['success']}
Iterations: {results['iterations']}
Processing Time: {results['processing_time']:.2f} seconds
Final Similarity: {results['final_similarity']:.4f}

Pipeline:
---------
"""
        for i, script in enumerate(results['pipeline'], 1):
            report_text += f"{i}. {script}\n"
        
        if results['anomalies_detected']:
            report_text += "\nAnomalies Detected:\n-------------------\n"
            for anomaly in results['anomalies_detected']:
                report_text += f"- Iteration {anomaly['iteration']}: {anomaly['script']}\n"
                report_text += f"  Score: {anomaly['details']['anomaly_score']:.3f}\n"
        
        report_text += f"\nLearning Statistics:\n-------------------\n"
        report_text += f"Epsilon (exploration rate): {self.learner.epsilon:.4f}\n"
        report_text += f"Successful combinations: {len(self.successful_combinations)}\n"
        report_text += f"Failed combinations: {len(self.failed_combinations)}\n"
        
        # Write report
        with open(report_dir / "report.txt", 'w') as f:
            f.write(report_text)
        
        # Save processing log as JSON
        with open(report_dir / "processing_log.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Report saved to: {report_dir}")
    
    def interactive_setup(self):
        """Interactive configuration without argparse"""
        print("\n🎨 Automated Image Processing Studio")
        print("=" * 40)
        
        config = {}
        
        # Get input image
        while True:
            config['input_path'] = input("Enter path to input image: ").strip()
            if os.path.exists(config['input_path']):
                break
            print("❌ File not found. Please try again.")
        
        # Get target image
        while True:
            config['target_path'] = input("Enter path to target image: ").strip()
            if os.path.exists(config['target_path']):
                break
            print("❌ File not found. Please try again.")
        
        # Get parameters
        try:
            max_iter = input("Maximum iterations (default: 100): ").strip()
            config['max_iterations'] = int(max_iter) if max_iter else 100
        except:
            config['max_iterations'] = 100
        
        try:
            threshold = input("Similarity threshold (default: 0.1): ").strip()
            config['similarity_threshold'] = float(threshold) if threshold else 0.1
        except:
            config['similarity_threshold'] = 0.1
        
        # Ask about anomaly detection
        anomaly_check = input("Perform anomaly detection? (y/n, default: y): ").strip().lower()
        config['check_anomalies'] = anomaly_check != 'n'
        
        return config


def main():
    """Main entry point"""
    # Check dependencies first
    DependencyManager.check_and_install_dependencies()
    
    # Create studio instance
    studio = AutomatedProcessingStudio()
    
    print(f"\n📚 Loaded {len(studio.script_manager.functions)} processing scripts")
    
    # Get configuration
    config = studio.interactive_setup()
    
    # Load images
    input_image = cv2.imread(config['input_path'], cv2.IMREAD_UNCHANGED)
    target_image = cv2.imread(config['target_path'], cv2.IMREAD_UNCHANGED)
    
    if input_image is None or target_image is None:
        print("❌ Failed to load images")
        return
    
    # Process to match target
    results = studio.process_to_match(
        input_image,
        target_image,
        max_iterations=config['max_iterations'],
        similarity_threshold=config['similarity_threshold']
    )
    
    # Perform anomaly detection if requested
    if config['check_anomalies'] and 'final_image' in results:
        anomaly_results = studio.find_anomalies_and_similarities(results['final_image'])
        
        # Save anomaly results
        anomaly_dir = studio.cache_dir / "anomalies"
        anomaly_dir.mkdir(exist_ok=True)
        
        with open(anomaly_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(anomaly_results, f, indent=2, default=str)
    
    print("\n✨ Processing complete!")
    print(f"📊 Final similarity: {results['final_similarity']:.4f}")
    print(f"⏱️ Processing time: {results['processing_time']:.2f} seconds")
    print(f"🔄 Total iterations: {results['iterations']}")


if __name__ == "__main__":
    main()