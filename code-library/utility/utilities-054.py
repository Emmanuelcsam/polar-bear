#!/usr/bin/env python3
"""
Script Wrappers for Legacy Scripts
Provides wrapper classes to integrate existing scripts with the connector system
"""

import sys
import os
import importlib.util
import threading
import time
import json
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import inspect
import ast

# Import the script interface
from script_interface import ScriptInterface, ConnectorClient

class GenericScriptWrapper(ScriptInterface):
    """Generic wrapper for scripts that don't implement ScriptInterface"""
    
    def __init__(self, script_path: Path):
        self.script_path = script_path
        script_name = script_path.stem
        
        # Initialize parent class
        super().__init__(script_name, f"Wrapped script: {script_name}")
        
        # Load the script module
        self.module = self._load_module()
        
        # Analyze script to find parameters and functions
        self._analyze_script()
        
        # Register discovered parameters
        self._register_discovered_parameters()
        
    def _load_module(self):
        """Load the script as a module"""
        try:
            spec = importlib.util.spec_from_file_location(
                self.script_path.stem, 
                self.script_path
            )
            module = importlib.util.module_from_spec(spec)
            
            # Redirect stdout to capture prints
            original_stdout = sys.stdout
            
            # Execute module
            spec.loader.exec_module(module)
            
            sys.stdout = original_stdout
            
            return module
        except Exception as e:
            self.logger.error(f"Failed to load module: {e}")
            return None
            
    def _analyze_script(self):
        """Analyze script to find configurable parameters and main functions"""
        self.discovered_params = {}
        self.main_functions = []
        
        if not self.module:
            return
            
        # Find global variables that look like parameters
        for name, value in vars(self.module).items():
            if not name.startswith('_') and isinstance(value, (int, float, str, bool, list, dict)):
                if name.isupper() or name.endswith('_CONFIG') or name.endswith('_PARAMS'):
                    self.discovered_params[name] = value
                    
        # Find main functions
        for name, obj in vars(self.module).items():
            if callable(obj) and not name.startswith('_'):
                if name in ['main', 'run', 'execute', 'start', 'process']:
                    self.main_functions.append((name, obj))
                elif 'main' in name.lower() or 'run' in name.lower():
                    self.main_functions.append((name, obj))
                    
        # Also analyze the source code for more insights
        try:
            with open(self.script_path, 'r') as f:
                tree = ast.parse(f.read())
                
            # Find assignments that look like config
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id.isupper() or '_CONFIG' in target.id:
                                # Try to evaluate the value
                                try:
                                    value = ast.literal_eval(node.value)
                                    self.discovered_params[target.id] = value
                                except:
                                    pass
        except Exception as e:
            self.logger.warning(f"Could not analyze source: {e}")
            
    def _register_discovered_parameters(self):
        """Register discovered parameters"""
        for param_name, value in self.discovered_params.items():
            if isinstance(value, bool):
                self.register_parameter(param_name, value, [True, False])
            elif isinstance(value, int):
                # Guess reasonable ranges
                if 0 <= value <= 1:
                    self.register_parameter(param_name, value, [0, 1])
                elif 0 <= value <= 100:
                    self.register_parameter(param_name, value, range(0, 101))
                else:
                    self.register_parameter(param_name, value)
            elif isinstance(value, float):
                if 0 <= value <= 1:
                    self.register_parameter(param_name, value)
                else:
                    self.register_parameter(param_name, value)
            else:
                self.register_parameter(param_name, value)
                
        # Register some standard parameters
        self.register_parameter("auto_run", True, [True, False])
        self.register_parameter("verbose", False, [True, False])
        
    def run(self):
        """Run the wrapped script"""
        if not self.module:
            self.logger.error("Module not loaded")
            return
            
        self.logger.info(f"Running wrapped script: {self.script_name}")
        
        # Update module parameters with our controlled values
        for param_name, param_info in self.parameters.items():
            if param_name in self.discovered_params and hasattr(self.module, param_name):
                setattr(self.module, param_name, param_info["value"])
                
        # Try to run main functions
        if self.main_functions:
            for func_name, func in self.main_functions:
                try:
                    self.logger.info(f"Executing function: {func_name}")
                    
                    # Check function signature
                    sig = inspect.signature(func)
                    
                    if len(sig.parameters) == 0:
                        # No parameters
                        result = func()
                    else:
                        # Try to call with common parameter patterns
                        try:
                            result = func()
                        except TypeError:
                            # Try with sys.argv
                            result = func(sys.argv[1:])
                            
                    if result is not None:
                        self.update_results(func_name, result)
                        
                except Exception as e:
                    self.logger.error(f"Error running {func_name}: {e}")
                    
        # If no main functions found, check if script runs on import
        elif hasattr(self.module, '__name__') and self.module.__name__ == '__main__':
            self.logger.info("Script appears to run on import")
            
            
class AnomalyDetectionWrapper(ScriptInterface):
    """Specialized wrapper for anomaly detection scripts"""
    
    def __init__(self, script_path: Path):
        super().__init__("anomaly_detection", "Anomaly Detection System")
        self.script_path = script_path
        
        # Register specific parameters
        self.register_parameter("threshold", 3.0, [1.0, 2.0, 2.5, 3.0, 3.5, 4.0])
        self.register_parameter("method", "zscore", ["zscore", "isolation_forest", "lof"])
        self.register_parameter("window_size", 100, range(10, 1000))
        self.register_parameter("contamination", 0.1, [0.01, 0.05, 0.1, 0.2])
        
        # Register variables
        self.register_variable("anomalies_detected", 0)
        self.register_variable("data_points_processed", 0)
        self.register_variable("last_anomaly_time", None)
        
        self.module = None
        
    def run(self):
        """Run anomaly detection"""
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("anomaly_detection", self.script_path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
            
            # Configure parameters
            if hasattr(self.module, 'ANOMALY_THRESHOLD'):
                self.module.ANOMALY_THRESHOLD = self.get_parameter("threshold")
                
            # Run the detection
            if hasattr(self.module, 'main'):
                result = self.module.main()
                if result:
                    self.update_results("detection_results", result)
                    
        except Exception as e:
            self.logger.error(f"Error running anomaly detection: {e}")
            

class CNNScriptWrapper(ScriptInterface):
    """Wrapper for CNN-based scripts"""
    
    def __init__(self, script_path: Path, model_type: str = "generic"):
        script_name = f"cnn_{model_type}"
        super().__init__(script_name, f"CNN {model_type} Model")
        self.script_path = script_path
        self.model_type = model_type
        
        # Register CNN-specific parameters
        self.register_parameter("epochs", 10, range(1, 101))
        self.register_parameter("batch_size", 32, [16, 32, 64, 128])
        self.register_parameter("learning_rate", 0.001, [0.0001, 0.001, 0.01, 0.1])
        self.register_parameter("optimizer", "adam", ["adam", "sgd", "rmsprop"])
        self.register_parameter("dropout_rate", 0.5, [0.0, 0.25, 0.5, 0.75])
        
        # Model architecture parameters
        self.register_parameter("num_filters", 32, [16, 32, 64, 128])
        self.register_parameter("kernel_size", 3, [3, 5, 7])
        
        # Register variables
        self.register_variable("current_epoch", 0)
        self.register_variable("training_loss", 0.0)
        self.register_variable("validation_accuracy", 0.0)
        self.register_variable("model_saved", False)
        
    def run(self):
        """Run CNN training/inference"""
        try:
            # Load module
            spec = importlib.util.spec_from_file_location(self.script_name, self.script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Inject our parameters
            module.EPOCHS = self.get_parameter("epochs")
            module.BATCH_SIZE = self.get_parameter("batch_size")
            module.LEARNING_RATE = self.get_parameter("learning_rate")
            
            spec.loader.exec_module(module)
            
            # Find and run main function
            if hasattr(module, 'main'):
                result = module.main()
                self.update_results("training_complete", True)
                if result:
                    self.update_results("final_results", result)
                    
        except Exception as e:
            self.logger.error(f"Error running CNN script: {e}")
            

class CollaborativeWrapper(ScriptInterface):
    """Wrapper that enables collaboration between scripts"""
    
    def __init__(self, script_path: Path, script_name: str):
        super().__init__(script_name, f"Collaborative {script_name}")
        self.script_path = script_path
        self.client = ConnectorClient(self)
        self.collaboration_data = {}
        
        # Register collaboration parameters
        self.register_parameter("enable_collaboration", True, [True, False])
        self.register_parameter("collaboration_interval", 5, range(1, 60))
        self.register_parameter("share_results", True, [True, False])
        
        # Start collaboration thread
        self.collab_thread = threading.Thread(target=self._collaboration_loop, daemon=True)
        self.collab_thread.start()
        
    def _collaboration_loop(self):
        """Handle collaboration with other scripts"""
        while self.running:
            if self.get_parameter("enable_collaboration"):
                # Check for collaboration requests
                response = self.client.script.send_to_connector({
                    "command": "get_collaborations",
                    "script_name": self.script_name
                })
                
                if response and response.get("requests"):
                    for request in response["requests"]:
                        self._handle_collaboration_request(request)
                        
                # Share our results if enabled
                if self.get_parameter("share_results") and self.results:
                    self.client.broadcast_data({
                        "type": "results_update",
                        "results": self.results
                    })
                    
            time.sleep(self.get_parameter("collaboration_interval"))
            
    def _handle_collaboration_request(self, request):
        """Handle incoming collaboration request"""
        source = request.get("source")
        data = request.get("data")
        
        self.logger.info(f"Received collaboration request from {source}")
        
        # Store collaboration data
        self.collaboration_data[source] = data
        
        # Process based on data type
        if data.get("type") == "parameter_sync":
            # Sync parameters with other script
            for param, value in data.get("parameters", {}).items():
                if param in self.parameters:
                    self.set_parameter(param, value)
                    
        elif data.get("type") == "results_update":
            # Process results from other script
            self.register_variable(f"{source}_results", data.get("results"))
            
    def run(self):
        """Run the wrapped script with collaboration"""
        # Register with connector
        self.client.register_script()
        
        # Run the actual script
        super().run()


def create_wrapper_for_script(script_path: Path) -> Optional[ScriptInterface]:
    """Factory function to create appropriate wrapper for a script"""
    script_name = script_path.stem
    
    # Check script type and create appropriate wrapper
    if "anomaly" in script_name.lower():
        return AnomalyDetectionWrapper(script_path)
    elif "cnn" in script_name.lower():
        if "fiber" in script_name:
            return CNNScriptWrapper(script_path, "fiber_detector")
        elif "cifar" in script_name:
            return CNNScriptWrapper(script_path, "cifar10")
        else:
            return CNNScriptWrapper(script_path, "generic")
    elif any(keyword in script_name.lower() for keyword in ["collab", "integrated"]):
        return CollaborativeWrapper(script_path, script_name)
    else:
        # Use generic wrapper
        return GenericScriptWrapper(script_path)


def wrap_and_run_script(script_path: str):
    """Wrap and run a script with connector integration"""
    path = Path(script_path)
    
    if not path.exists():
        print(f"Script not found: {script_path}")
        return
        
    # Create wrapper
    wrapper = create_wrapper_for_script(path)
    
    if wrapper:
        # Run with connector integration if requested
        if "--with-connector" in sys.argv:
            wrapper.run_with_connector()
        else:
            wrapper.start()
            wrapper.run()
            wrapper.stop()
    else:
        print(f"Could not create wrapper for: {script_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
        wrap_and_run_script(script_path)
    else:
        print("Usage: python script_wrappers.py <script_path> [--with-connector]")