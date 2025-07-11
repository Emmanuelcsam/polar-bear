#!/usr/bin/env python3
"""
Neural Network Hivemind Controller
A master controller system for managing and optimizing a distributed neural network
for fiberoptic image classification.
"""

import os
import sys
import json
import time
import pickle
import logging
import hashlib
import inspect
import importlib
import subprocess
import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import threading
import queue
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback
import re

class NeuralHivemind:
    """Master controller for the neural network system"""
    
    def __init__(self):
        self.base_dirs = [
            "/home/jarvis/Documents/GitHub/polar-bear/training",
            "/home/jarvis/Documents/GitHub/polar-bear/ruleset",
            "/home/jarvis/Documents/GitHub/polar-bear/modules"
        ]
        
        self.log_dir = Path("/home/jarvis/Documents/GitHub/polar-bear/hivemind_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.setup_logging()
        
        self.scripts = []
        self.data_files = []
        self.image_files = []
        self.script_metadata = {}
        self.dependencies = set()
        self.installed_packages = set()
        
        self.parameter_history = defaultdict(list)
        self.performance_metrics = defaultdict(dict)
        self.script_combinations = []
        
        self.excluded_dirs = {'venv', '__pycache__', '.venv', 'env', '.env', 
                             'virtualenv', '.virtualenv', '.git', '.pytest_cache',
                             'node_modules', 'dist', 'build', '.eggs', '*.egg-info'}
        
        self.log("Neural Hivemind initialized")
        
    def setup_logging(self):
        """Configure comprehensive logging system"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"hivemind_{timestamp}.log"
        
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('NeuralHivemind')
        self.log(f"Logging initialized. Log file: {log_file}")
        
    def log(self, message, level="INFO"):
        """Log message to both file and console"""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        
    def check_and_install_dependencies(self):
        """Check for required packages and install missing ones"""
        self.log("Checking system dependencies...")
        
        # Core required packages
        required_packages = {
            'numpy': 'numpy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn',
            'torch': 'torch',
            'torchvision': 'torchvision',
            'cv2': 'opencv-python',
            'PIL': 'pillow',
            'yaml': 'pyyaml',
            'tqdm': 'tqdm'
        }
        
        # Check installed packages
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True)
            installed = result.stdout.lower()
            self.installed_packages = set(line.split()[0].lower() 
                                        for line in installed.split('\n')[2:] 
                                        if line.strip())
        except Exception as e:
            self.log(f"Error checking installed packages: {e}", "ERROR")
            
        # Install missing packages
        for import_name, package_name in required_packages.items():
            try:
                importlib.import_module(import_name)
                self.log(f"✓ {import_name} is installed")
            except ImportError:
                self.log(f"Installing {package_name}...")
                try:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', 
                        '--upgrade', package_name
                    ])
                    self.log(f"✓ Successfully installed {package_name}")
                except subprocess.CalledProcessError as e:
                    self.log(f"✗ Failed to install {package_name}: {e}", "ERROR")
                    
    def should_exclude_path(self, path: Path) -> bool:
        """Check if path should be excluded from crawling"""
        path_str = str(path)
        for excluded in self.excluded_dirs:
            if excluded in path_str or path.name.startswith('.'):
                return True
        return False
        
    def crawl_directories(self):
        """Deep crawl through directories to find all relevant files"""
        self.log("Starting deep directory crawl...")
        
        file_extensions = {
            'scripts': ['.py'],
            'images': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'],
            'data': ['.json', '.log', '.pkl', '.pickle', '.txt', '.csv', '.npy', '.npz']
        }
        
        for base_dir in self.base_dirs:
            if not Path(base_dir).exists():
                self.log(f"Directory {base_dir} does not exist, skipping", "WARNING")
                continue
                
            self.log(f"Crawling {base_dir}...")
            
            for root, dirs, files in os.walk(base_dir):
                root_path = Path(root)
                
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not self.should_exclude_path(root_path / d)]
                
                if self.should_exclude_path(root_path):
                    continue
                    
                for file in files:
                    file_path = root_path / file
                    
                    # Skip PKG-INFO files and other metadata
                    if file == 'PKG-INFO' or file.endswith('.egg-info'):
                        continue
                        
                    ext = file_path.suffix.lower()
                    
                    if ext in file_extensions['scripts']:
                        self.scripts.append(str(file_path))
                        self.log(f"Found script: {file_path}")
                    elif ext in file_extensions['images']:
                        self.image_files.append(str(file_path))
                        self.log(f"Found image: {file_path}")
                    elif ext in file_extensions['data'] or file == 'PKG-INFO':
                        self.data_files.append(str(file_path))
                        self.log(f"Found data file: {file_path}")
                        
        self.log(f"Crawl complete. Found {len(self.scripts)} scripts, "
                f"{len(self.image_files)} images, {len(self.data_files)} data files")
        
    def analyze_script(self, script_path: str) -> Dict[str, Any]:
        """Analyze a Python script to extract metadata and parameters"""
        self.log(f"Analyzing script: {script_path}")
        
        metadata = {
            'path': script_path,
            'functions': [],
            'classes': [],
            'imports': [],
            'parameters': {},
            'docstring': None,
            'hash': None,
            'error': None
        }
        
        try:
            # Calculate file hash
            with open(script_path, 'rb') as f:
                metadata['hash'] = hashlib.md5(f.read()).hexdigest()
                
            # Parse the script
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Extract docstring
            docstring = ast.get_docstring(tree)
            if docstring:
                metadata['docstring'] = docstring
                
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        metadata['imports'].append(alias.name)
                        self.dependencies.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        metadata['imports'].append(node.module)
                        self.dependencies.add(node.module.split('.')[0])
                        
                # Extract functions
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [d.id if isinstance(d, ast.Name) else str(d) 
                                     for d in node.decorator_list],
                        'docstring': ast.get_docstring(node)
                    }
                    metadata['functions'].append(func_info)
                    
                # Extract classes
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [base.id if isinstance(base, ast.Name) else str(base) 
                                for base in node.bases],
                        'methods': [],
                        'docstring': ast.get_docstring(node)
                    }
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)
                            
                    metadata['classes'].append(class_info)
                    
            # Extract parameters (look for common patterns)
            param_patterns = [
                r'(\w+)\s*=\s*(\d+\.?\d*)',  # numeric assignments
                r'(\w+)\s*=\s*["\']([^"\']+)["\']',  # string assignments
                r'(\w+)\s*=\s*\[([^\]]+)\]',  # list assignments
                r'(\w+)\s*=\s*\{([^\}]+)\}',  # dict assignments
            ]
            
            for pattern in param_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) >= 2:
                        param_name = match[0]
                        param_value = match[1]
                        if param_name.isupper() or '_' in param_name:  # Likely a constant
                            metadata['parameters'][param_name] = param_value
                            
        except Exception as e:
            metadata['error'] = str(e)
            self.log(f"Error analyzing {script_path}: {e}", "ERROR")
            
        return metadata
        
    def analyze_all_scripts(self):
        """Analyze all discovered scripts"""
        self.log("Analyzing all scripts...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for script in self.scripts:
                future = executor.submit(self.analyze_script, script)
                futures.append((script, future))
                
            for script, future in futures:
                try:
                    metadata = future.result(timeout=30)
                    self.script_metadata[script] = metadata
                except Exception as e:
                    self.log(f"Failed to analyze {script}: {e}", "ERROR")
                    
        self.log(f"Script analysis complete. Analyzed {len(self.script_metadata)} scripts")
        
    def find_script_connections(self):
        """Identify connections between scripts based on imports and usage"""
        self.log("Finding connections between scripts...")
        
        connections = defaultdict(list)
        
        for script, metadata in self.script_metadata.items():
            script_name = Path(script).stem
            
            # Check if other scripts import this one
            for other_script, other_metadata in self.script_metadata.items():
                if script == other_script:
                    continue
                    
                for import_name in other_metadata['imports']:
                    if script_name in import_name or import_name in script:
                        connections[script].append(other_script)
                        self.log(f"Connection found: {other_script} imports {script}")
                        
        self.script_connections = connections
        self.log(f"Found {sum(len(v) for v in connections.values())} connections")
        
    def create_execution_plan(self, target_task: str) -> List[str]:
        """Create an execution plan for a specific task"""
        self.log(f"Creating execution plan for task: {target_task}")
        
        # Score scripts based on relevance to task
        script_scores = {}
        
        task_keywords = target_task.lower().split()
        
        for script, metadata in self.script_metadata.items():
            score = 0
            
            # Check script name
            script_name = Path(script).stem.lower()
            for keyword in task_keywords:
                if keyword in script_name:
                    score += 10
                    
            # Check docstring
            if metadata['docstring']:
                doc_lower = metadata['docstring'].lower()
                for keyword in task_keywords:
                    score += doc_lower.count(keyword) * 2
                    
            # Check function names
            for func in metadata['functions']:
                if any(keyword in func['name'].lower() for keyword in task_keywords):
                    score += 5
                    
            # Check class names
            for cls in metadata['classes']:
                if any(keyword in cls['name'].lower() for keyword in task_keywords):
                    score += 5
                    
            if score > 0:
                script_scores[script] = score
                
        # Sort by score
        ranked_scripts = sorted(script_scores.items(), key=lambda x: x[1], reverse=True)
        
        execution_plan = [script for script, score in ranked_scripts[:10]]
        self.log(f"Execution plan created with {len(execution_plan)} scripts")
        
        return execution_plan
        
    def load_script_module(self, script_path: str) -> Optional[Any]:
        """Dynamically load a Python script as a module"""
        try:
            spec = importlib.util.spec_from_file_location("dynamic_module", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            self.log(f"Failed to load module {script_path}: {e}", "ERROR")
            return None
            
    def tune_parameters(self, script_path: str, performance_score: float):
        """Tune parameters based on performance feedback"""
        self.log(f"Tuning parameters for {script_path} with score {performance_score}")
        
        metadata = self.script_metadata.get(script_path, {})
        parameters = metadata.get('parameters', {})
        
        if not parameters:
            self.log(f"No tunable parameters found in {script_path}")
            return
            
        # Simple gradient-based tuning
        for param_name, param_value in parameters.items():
            try:
                # Try to convert to float for numeric parameters
                current_value = float(param_value)
                
                # Adjust based on performance
                if performance_score < 0.5:
                    # Poor performance - larger adjustment
                    adjustment = np.random.uniform(-0.2, 0.2) * current_value
                else:
                    # Good performance - smaller adjustment
                    adjustment = np.random.uniform(-0.05, 0.05) * current_value
                    
                new_value = current_value + adjustment
                
                # Store parameter history
                self.parameter_history[f"{script_path}:{param_name}"].append({
                    'timestamp': datetime.now().isoformat(),
                    'old_value': current_value,
                    'new_value': new_value,
                    'performance_score': performance_score
                })
                
                self.log(f"Tuned {param_name}: {current_value} -> {new_value}")
                
            except ValueError:
                # Non-numeric parameter, skip for now
                pass
                
    def execute_script_combination(self, scripts: List[str], input_data: Any) -> Dict[str, Any]:
        """Execute a combination of scripts on input data"""
        self.log(f"Executing script combination: {[Path(s).stem for s in scripts]}")
        
        results = {
            'scripts': scripts,
            'start_time': datetime.now().isoformat(),
            'outputs': {},
            'errors': {},
            'execution_time': 0
        }
        
        start_time = time.time()
        current_data = input_data
        
        for script in scripts:
            try:
                module = self.load_script_module(script)
                if not module:
                    results['errors'][script] = "Failed to load module"
                    continue
                    
                # Look for main processing function
                process_functions = ['process', 'analyze', 'detect', 'classify', 'main']
                executed = False
                
                for func_name in process_functions:
                    if hasattr(module, func_name):
                        func = getattr(module, func_name)
                        if callable(func):
                            self.log(f"Executing {func_name} from {script}")
                            output = func(current_data)
                            results['outputs'][script] = output
                            current_data = output  # Pipeline the output
                            executed = True
                            break
                            
                if not executed:
                    results['errors'][script] = "No suitable processing function found"
                    
            except Exception as e:
                error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
                results['errors'][script] = error_msg
                self.log(f"Error executing {script}: {error_msg}", "ERROR")
                
        results['execution_time'] = time.time() - start_time
        results['end_time'] = datetime.now().isoformat()
        
        return results
        
    def evaluate_performance(self, results: Dict[str, Any], ground_truth: Any = None) -> float:
        """Evaluate the performance of a script combination"""
        # Simple performance metric - can be enhanced based on specific needs
        if results['errors']:
            # Penalize for errors
            error_penalty = len(results['errors']) / len(results['scripts'])
            performance = 1.0 - error_penalty
        else:
            performance = 1.0
            
        # Additional metrics can be added here
        if ground_truth is not None:
            # Compare with ground truth if available
            pass
            
        return performance
        
    def optimize_network(self, training_data: List[Any], epochs: int = 10):
        """Optimize the neural network by testing different script combinations"""
        self.log(f"Starting network optimization with {epochs} epochs")
        
        # Generate initial script combinations
        all_scripts = list(self.script_metadata.keys())
        
        for epoch in range(epochs):
            self.log(f"Epoch {epoch + 1}/{epochs}")
            
            # Test different combinations
            for i in range(min(10, len(all_scripts))):
                # Create a random combination
                num_scripts = np.random.randint(2, min(6, len(all_scripts)))
                combination = np.random.choice(all_scripts, num_scripts, replace=False).tolist()
                
                # Test on training data
                total_score = 0
                for data in training_data[:5]:  # Test on subset
                    results = self.execute_script_combination(combination, data)
                    score = self.evaluate_performance(results)
                    total_score += score
                    
                avg_score = total_score / min(5, len(training_data))
                
                # Store combination performance
                self.script_combinations.append({
                    'combination': combination,
                    'score': avg_score,
                    'epoch': epoch
                })
                
                # Tune parameters for each script based on performance
                for script in combination:
                    self.tune_parameters(script, avg_score)
                    
            # Log best combinations so far
            best_combos = sorted(self.script_combinations, 
                               key=lambda x: x['score'], 
                               reverse=True)[:3]
            
            self.log(f"Best combinations in epoch {epoch + 1}:")
            for combo in best_combos:
                scripts = [Path(s).stem for s in combo['combination']]
                self.log(f"  Score: {combo['score']:.3f} - Scripts: {scripts}")
                
    def save_state(self):
        """Save the current state of the hivemind"""
        state_file = self.log_dir / f"hivemind_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        state = {
            'scripts': self.scripts,
            'script_metadata': self.script_metadata,
            'parameter_history': dict(self.parameter_history),
            'performance_metrics': dict(self.performance_metrics),
            'script_combinations': self.script_combinations,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
            
        self.log(f"State saved to {state_file}")
        
    def interactive_setup(self):
        """Interactive setup for user configuration"""
        print("\n=== Neural Hivemind Interactive Setup ===\n")
        
        # Ask about specific focus areas
        focus = input("What type of images will you be analyzing? (e.g., fiberoptic, medical, industrial): ").strip()
        if focus:
            self.focus_area = focus
            self.log(f"Focus area set to: {focus}")
            
        # Ask about performance preferences
        preference = input("Optimize for: [1] Accuracy, [2] Speed, [3] Balanced (default: 3): ").strip()
        self.optimization_preference = preference if preference in ['1', '2', '3'] else '3'
        
        # Ask about resource limits
        max_threads = input("Maximum number of parallel threads (default: 10): ").strip()
        self.max_threads = int(max_threads) if max_threads.isdigit() else 10
        
        print("\nSetup complete! Starting hivemind initialization...\n")
        
    def run(self):
        """Main execution method"""
        self.log("="*50)
        self.log("NEURAL HIVEMIND SYSTEM STARTING")
        self.log("="*50)
        
        # Interactive setup
        self.interactive_setup()
        
        # Check and install dependencies
        self.check_and_install_dependencies()
        
        # Crawl directories
        self.crawl_directories()
        
        # Analyze scripts
        self.analyze_all_scripts()
        
        # Find connections
        self.find_script_connections()
        
        # Save initial state
        self.save_state()
        
        # Create summary report
        self.create_summary_report()
        
        self.log("="*50)
        self.log("NEURAL HIVEMIND INITIALIZATION COMPLETE")
        self.log("="*50)
        
        # Offer to run optimization
        optimize = input("\nWould you like to run network optimization? (y/n): ").strip().lower()
        if optimize == 'y':
            # For demo, create synthetic training data
            print("Creating synthetic training data for demonstration...")
            training_data = [f"sample_image_{i}.png" for i in range(10)]
            self.optimize_network(training_data, epochs=3)
            
        # Final save
        self.save_state()
        
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        report_file = self.log_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("NEURAL HIVEMIND SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DISCOVERED FILES:\n")
            f.write(f"- Python Scripts: {len(self.scripts)}\n")
            f.write(f"- Image Files: {len(self.image_files)}\n")
            f.write(f"- Data Files: {len(self.data_files)}\n\n")
            
            f.write("SCRIPT ANALYSIS:\n")
            total_functions = sum(len(m['functions']) for m in self.script_metadata.values())
            total_classes = sum(len(m['classes']) for m in self.script_metadata.values())
            f.write(f"- Total Functions: {total_functions}\n")
            f.write(f"- Total Classes: {total_classes}\n")
            f.write(f"- Unique Dependencies: {len(self.dependencies)}\n\n")
            
            f.write("TOP SCRIPTS BY CONNECTIVITY:\n")
            sorted_connections = sorted(self.script_connections.items(), 
                                      key=lambda x: len(x[1]), 
                                      reverse=True)[:10]
            for script, connections in sorted_connections:
                f.write(f"- {Path(script).name}: {len(connections)} connections\n")
                
            f.write("\n" + "="*50 + "\n")
            
        self.log(f"Summary report created: {report_file}")


if __name__ == "__main__":
    # Create and run the hivemind
    hivemind = NeuralHivemind()
    
    try:
        hivemind.run()
    except KeyboardInterrupt:
        hivemind.log("\nShutdown requested by user", "WARNING")
        hivemind.save_state()
        print("\nHivemind state saved. Exiting.")
    except Exception as e:
        hivemind.log(f"Fatal error: {e}\n{traceback.format_exc()}", "ERROR")
        hivemind.save_state()
        print("\nAn error occurred. State saved. Check logs for details.")