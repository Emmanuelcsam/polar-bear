#!/usr/bin/env python3
"""
Script Interface Module
Provides a unified interface for connectors to control all analysis/reporting scripts
"""

import json
import os
import sys
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
import threading
import queue
import time
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ScriptInfo:
    """Information about a script"""
    name: str
    path: str
    module_name: str
    main_function: Optional[str] = None
    parameters: Dict[str, Any] = None
    description: str = ""
    dependencies: List[str] = None
    status: str = "idle"  # idle, running, completed, failed
    last_run: Optional[datetime] = None
    last_result: Optional[Dict] = None


@dataclass
class ParameterInfo:
    """Information about a script parameter"""
    name: str
    type: str
    default: Any
    description: str
    required: bool = True
    validation: Optional[Callable] = None


class ScriptManager:
    """Manages script execution and parameter control"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.scripts: Dict[str, ScriptInfo] = {}
        self.modules: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self.parameter_registry: Dict[str, Dict[str, ParameterInfo]] = {}
        self.execution_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = True
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger("ScriptManager")
        self.logger.setLevel(logging.INFO)
        
        # Load configuration
        self.load_config()
        
        # Discover scripts
        self.discover_scripts()
        
        # Start execution thread
        self.execution_thread = threading.Thread(target=self._execution_worker)
        self.execution_thread.daemon = True
        self.execution_thread.start()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def discover_scripts(self):
        """Discover all Python scripts in the directory"""
        script_dir = Path(__file__).parent
        
        # Define script metadata for all scripts in the directory
        script_metadata = {
            "analysis_engine.py": {
                "main_function": "run_full_pipeline",
                "description": "Main analysis orchestration engine",
                "parameters": {
                    "image_path": {"type": "str", "description": "Path to input image", "required": True},
                    "output_dir": {"type": "str", "description": "Output directory", "default": "."},
                    "config": {"type": "dict", "description": "Configuration overrides", "default": {}},
                    "verbose": {"type": "bool", "description": "Verbose output", "default": False}
                }
            },
            "defect-analyzer.py": {
                "main_function": "analyze_defects",
                "description": "Analyzes defects in refined masks",
                "parameters": {
                    "refined_masks": {"type": "dict", "description": "Dictionary of refined masks", "required": True},
                    "pixels_per_micron": {"type": "float", "description": "Calibration factor", "default": 0.5}
                }
            },
            "report-generator.py": {
                "main_function": "generate_annotated_image",
                "description": "Generates visual analysis reports",
                "parameters": {
                    "image": {"type": "numpy.ndarray", "description": "Input image (BGR)", "required": True},
                    "results": {"type": "dict", "description": "Analysis results", "required": True},
                    "localization": {"type": "dict", "description": "Fiber localization data", "required": True},
                    "zone_masks": {"type": "dict", "description": "Zone segmentation masks", "required": True},
                    "output_path": {"type": "str", "description": "Output image path", "required": True}
                }
            },
            "quality-metrics-calculator.py": {
                "main_function": "calculate_comprehensive_quality_metrics",
                "description": "Calculates comprehensive quality metrics",
                "parameters": {
                    "image": {"type": "numpy.ndarray", "description": "Grayscale image", "required": True},
                    "defect_mask": {"type": "numpy.ndarray", "description": "Binary defect mask", "default": None},
                    "roi_mask": {"type": "numpy.ndarray", "description": "Region of interest mask", "default": None}
                }
            },
            "batch-summary-reporter.py": {
                "main_function": "generate_batch_summary",
                "description": "Generates summary reports for batch processing",
                "parameters": {
                    "results_list": {"type": "list", "description": "List of analysis results", "required": True},
                    "output_path": {"type": "str", "description": "Output file path", "default": "batch_summary.json"}
                }
            },
            "contour-analyzer.py": {
                "main_function": "analyze_contours",
                "description": "Analyzes contours in images",
                "parameters": {
                    "image": {"type": "numpy.ndarray", "description": "Input image", "required": True},
                    "threshold": {"type": "float", "description": "Contour detection threshold", "default": 127}
                }
            },
            "csv-report-creator.py": {
                "main_function": "create_csv_report",
                "description": "Creates CSV reports from analysis results",
                "parameters": {
                    "results": {"type": "dict", "description": "Analysis results", "required": True},
                    "output_path": {"type": "str", "description": "CSV output path", "required": True}
                }
            },
            "data-aggregation-reporter.py": {
                "main_function": "aggregate_data",
                "description": "Aggregates data from multiple sources",
                "parameters": {
                    "data_sources": {"type": "list", "description": "List of data source paths", "required": True},
                    "output_format": {"type": "str", "description": "Output format (json, csv)", "default": "json"}
                }
            },
            "defect-characterizer.py": {
                "main_function": "characterize_defect_geometry",
                "description": "Characterizes defect geometry and properties",
                "parameters": {
                    "contour": {"type": "numpy.ndarray", "description": "Defect contour", "required": True},
                    "mask": {"type": "numpy.ndarray", "description": "Binary defect mask", "required": True},
                    "um_per_px": {"type": "float", "description": "Microns per pixel", "default": None}
                }
            },
            "defect-characterizer-v4.py": {
                "main_function": "characterize_defects_v4",
                "description": "Advanced defect characterization version 4",
                "parameters": {
                    "image": {"type": "numpy.ndarray", "description": "Input image", "required": True},
                    "defects": {"type": "list", "description": "List of defect regions", "required": True}
                }
            },
            "defect-cluster-analyzer.py": {
                "main_function": "analyze_defect_clusters",
                "description": "Analyzes clustering patterns in defects",
                "parameters": {
                    "defect_locations": {"type": "list", "description": "List of defect coordinates", "required": True},
                    "clustering_threshold": {"type": "float", "description": "Distance threshold for clustering", "default": 10.0}
                }
            },
            "defect-information-extractor.py": {
                "main_function": "extract_defect_information",
                "description": "Extracts detailed information from defects",
                "parameters": {
                    "image": {"type": "numpy.ndarray", "description": "Input image", "required": True},
                    "defect_mask": {"type": "numpy.ndarray", "description": "Binary defect mask", "required": True}
                }
            },
            "defect-measurement-tool.py": {
                "main_function": "measure_defects",
                "description": "Measures defect dimensions and properties",
                "parameters": {
                    "defects": {"type": "list", "description": "List of defect objects", "required": True},
                    "calibration": {"type": "float", "description": "Calibration factor", "default": 1.0}
                }
            },
            "detailed-report-generator.py": {
                "main_function": "generate_detailed_report",
                "description": "Generates detailed analysis reports",
                "parameters": {
                    "analysis_results": {"type": "dict", "description": "Complete analysis results", "required": True},
                    "output_path": {"type": "str", "description": "Report output path", "required": True},
                    "include_plots": {"type": "bool", "description": "Include visualization plots", "default": True}
                }
            },
            "exhaustive-comparator.py": {
                "main_function": "compare_exhaustively",
                "description": "Performs exhaustive comparison between samples",
                "parameters": {
                    "sample1": {"type": "dict", "description": "First sample data", "required": True},
                    "sample2": {"type": "dict", "description": "Second sample data", "required": True},
                    "comparison_metrics": {"type": "list", "description": "Metrics to compare", "default": []}
                }
            },
            "image-result-handler.py": {
                "main_function": "handle_image_results",
                "description": "Handles and processes image analysis results",
                "parameters": {
                    "results": {"type": "dict", "description": "Image analysis results", "required": True},
                    "output_dir": {"type": "str", "description": "Output directory", "default": "."}
                }
            },
            "image-statistics-calculator.py": {
                "main_function": "calculate_image_statistics",
                "description": "Calculates statistical properties of images",
                "parameters": {
                    "image": {"type": "numpy.ndarray", "description": "Input image", "required": True},
                    "roi": {"type": "tuple", "description": "Region of interest (x, y, w, h)", "default": None}
                }
            },
            "individual-report-saver.py": {
                "main_function": "save_individual_report",
                "description": "Saves individual analysis reports",
                "parameters": {
                    "report_data": {"type": "dict", "description": "Report data to save", "required": True},
                    "filename": {"type": "str", "description": "Output filename", "required": True},
                    "format": {"type": "str", "description": "Output format (json, pdf, html)", "default": "json"}
                }
            },
            "integrated-analysis-tool.py": {
                "main_function": "run_integrated_analysis",
                "description": "Runs integrated analysis combining multiple tools",
                "parameters": {
                    "image_path": {"type": "str", "description": "Path to input image", "required": True},
                    "analysis_modules": {"type": "list", "description": "List of analysis modules to run", "default": []}
                }
            },
            "morphological-analyzer.py": {
                "main_function": "analyze_morphology",
                "description": "Analyzes morphological properties",
                "parameters": {
                    "binary_image": {"type": "numpy.ndarray", "description": "Binary input image", "required": True},
                    "operations": {"type": "list", "description": "Morphological operations to apply", "default": []}
                }
            },
            "pass-fail-criteria-applier.py": {
                "main_function": "apply_pass_fail_criteria",
                "description": "Applies pass/fail criteria to results",
                "parameters": {
                    "results": {"type": "dict", "description": "Analysis results", "required": True},
                    "criteria": {"type": "dict", "description": "Pass/fail criteria", "required": True}
                }
            },
            "pass-fail-criteria-applier-v3.py": {
                "main_function": "apply_criteria_v3",
                "description": "Advanced pass/fail criteria application version 3",
                "parameters": {
                    "results": {"type": "dict", "description": "Analysis results", "required": True},
                    "criteria_config": {"type": "dict", "description": "Criteria configuration", "required": True}
                }
            },
            "pass-fail-evaluator.py": {
                "main_function": "evaluate_pass_fail",
                "description": "Evaluates samples against pass/fail criteria",
                "parameters": {
                    "results": {"type": "dict", "description": "Sample results", "required": True},
                    "criteria": {"type": "dict", "description": "Evaluation criteria", "default": {}}
                }
            },
            "radial-profile-analyzer.py": {
                "main_function": "analyze_radial_profile",
                "description": "Analyzes radial intensity profiles",
                "parameters": {
                    "image": {"type": "numpy.ndarray", "description": "Input image", "required": True},
                    "center": {"type": "tuple", "description": "Center point (x, y)", "required": True},
                    "max_radius": {"type": "int", "description": "Maximum radius to analyze", "default": None}
                }
            },
            "reporting-module.py": {
                "main_function": "generate_report",
                "description": "Main reporting module",
                "parameters": {
                    "data": {"type": "dict", "description": "Data to report", "required": True},
                    "template": {"type": "str", "description": "Report template", "default": "default"}
                }
            },
            "similarity-analyzer.py": {
                "main_function": "analyze_similarity",
                "description": "Analyzes similarity between samples",
                "parameters": {
                    "sample1": {"type": "numpy.ndarray", "description": "First sample", "required": True},
                    "sample2": {"type": "numpy.ndarray", "description": "Second sample", "required": True},
                    "method": {"type": "str", "description": "Similarity metric", "default": "ssim"}
                }
            },
            "statistical-analysis-toolkit.py": {
                "main_function": "perform_statistical_analysis",
                "description": "Comprehensive statistical analysis toolkit",
                "parameters": {
                    "data": {"type": "numpy.ndarray", "description": "Data to analyze", "required": True},
                    "tests": {"type": "list", "description": "Statistical tests to perform", "default": []}
                }
            },
            "structural-comparator.py": {
                "main_function": "compare_structures",
                "description": "Compares structural properties between samples",
                "parameters": {
                    "structure1": {"type": "dict", "description": "First structure data", "required": True},
                    "structure2": {"type": "dict", "description": "Second structure data", "required": True}
                }
            }
        }
        
        # Discover scripts
        for script_path in script_dir.glob("*.py"):
            if script_path.name in ["__init__.py", "script_interface.py", "connector.py", "hivemind_connector.py"]:
                continue
            
            script_name = script_path.name
            module_name = script_path.stem.replace("-", "_")
            
            script_info = ScriptInfo(
                name=script_name,
                path=str(script_path),
                module_name=module_name,
                dependencies=[]
            )
            
            # Add metadata if available
            if script_name in script_metadata:
                metadata = script_metadata[script_name]
                script_info.main_function = metadata.get("main_function")
                script_info.description = metadata.get("description", "")
                script_info.parameters = metadata.get("parameters", {})
                
                # Register parameters
                self._register_parameters(script_name, metadata.get("parameters", {}))
            
            self.scripts[script_name] = script_info
        
        self.logger.info(f"Discovered {len(self.scripts)} scripts")
    
    def _register_parameters(self, script_name: str, parameters: Dict[str, Dict]):
        """Register script parameters"""
        param_info = {}
        for param_name, param_data in parameters.items():
            param_info[param_name] = ParameterInfo(
                name=param_name,
                type=param_data.get("type", "any"),
                default=param_data.get("default", None),
                description=param_data.get("description", ""),
                required=param_data.get("required", False)
            )
        self.parameter_registry[script_name] = param_info
    
    def load_module(self, script_name: str) -> Any:
        """Dynamically load a script module"""
        if script_name in self.modules:
            return self.modules[script_name]
        
        script_info = self.scripts.get(script_name)
        if not script_info:
            raise ValueError(f"Unknown script: {script_name}")
        
        try:
            spec = importlib.util.spec_from_file_location(
                script_info.module_name, 
                script_info.path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.modules[script_name] = module
            return module
        except Exception as e:
            self.logger.error(f"Failed to load module {script_name}: {e}")
            raise
    
    def get_script_info(self, script_name: str) -> Optional[ScriptInfo]:
        """Get information about a script"""
        return self.scripts.get(script_name)
    
    def get_all_scripts(self) -> List[ScriptInfo]:
        """Get information about all scripts"""
        return list(self.scripts.values())
    
    def update_parameter(self, script_name: str, param_name: str, value: Any):
        """Update a script parameter"""
        with self.lock:
            if script_name not in self.config:
                self.config[script_name] = {}
            
            self.config[script_name][param_name] = value
            self.save_config()
            
            # If it's a shared parameter, update it globally
            if param_name in self.config.get("shared_parameters", {}):
                self.config["shared_parameters"][param_name] = value
    
    def get_parameter(self, script_name: str, param_name: str) -> Any:
        """Get a script parameter value"""
        # Check script-specific first
        if script_name in self.config and param_name in self.config[script_name]:
            return self.config[script_name][param_name]
        
        # Check shared parameters
        if param_name in self.config.get("shared_parameters", {}):
            return self.config["shared_parameters"][param_name]
        
        # Return default from registry
        if script_name in self.parameter_registry:
            param_info = self.parameter_registry[script_name].get(param_name)
            if param_info:
                return param_info.default
        
        return None
    
    def execute_script(self, script_name: str, parameters: Dict[str, Any] = None, 
                      async_mode: bool = True) -> Dict[str, Any]:
        """Execute a script with given parameters"""
        if script_name not in self.scripts:
            return {"error": f"Unknown script: {script_name}"}
        
        execution_info = {
            "script_name": script_name,
            "parameters": parameters or {},
            "timestamp": datetime.now()
        }
        
        if async_mode:
            self.execution_queue.put(execution_info)
            return {"status": "queued", "script": script_name}
        else:
            return self._execute_script_sync(execution_info)
    
    def _execute_script_sync(self, execution_info: Dict) -> Dict[str, Any]:
        """Execute a script synchronously"""
        script_name = execution_info["script_name"]
        parameters = execution_info["parameters"]
        
        try:
            # Update script status
            with self.lock:
                self.scripts[script_name].status = "running"
                self.scripts[script_name].last_run = execution_info["timestamp"]
            
            # Try to use script wrappers first for better control
            try:
                from script_wrappers import execute_script as execute_wrapped
                
                # Merge parameters with config
                merged_params = {}
                if script_name in self.config:
                    merged_params.update(self.config[script_name])
                merged_params.update(parameters)
                
                # Execute using wrapper
                result = execute_wrapped(script_name, **merged_params)
                
                # Update status based on wrapper result
                with self.lock:
                    if result.get("success", False):
                        self.scripts[script_name].status = "completed"
                        self.scripts[script_name].last_result = {
                            "success": True,
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        self.scripts[script_name].status = "failed"
                        self.scripts[script_name].last_result = {
                            "success": False,
                            "error": result.get("error", "Unknown error"),
                            "timestamp": datetime.now().isoformat()
                        }
                
                return result
                
            except ImportError:
                # Fallback to direct module loading
                pass
            
            # Original direct module loading approach
            module = self.load_module(script_name)
            
            # Get main function
            script_info = self.scripts[script_name]
            if script_info.main_function:
                main_func = getattr(module, script_info.main_function, None)
                if not main_func:
                    raise AttributeError(f"Function {script_info.main_function} not found")
            else:
                # Try to find a main function
                main_func = getattr(module, "main", None)
                if not main_func:
                    # Look for analyze_* or generate_* functions
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj) and (
                            name.startswith("analyze_") or 
                            name.startswith("generate_") or
                            name.startswith("calculate_")
                        ):
                            main_func = obj
                            break
            
            if not main_func:
                raise ValueError(f"No main function found in {script_name}")
            
            # Merge parameters with config
            merged_params = {}
            if script_name in self.config:
                merged_params.update(self.config[script_name])
            merged_params.update(parameters)
            
            # Execute function
            result = main_func(**merged_params)
            
            # Update status
            with self.lock:
                self.scripts[script_name].status = "completed"
                self.scripts[script_name].last_result = {
                    "success": True,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {"success": True, "result": result}
            
        except Exception as e:
            # Update status
            with self.lock:
                self.scripts[script_name].status = "failed"
                self.scripts[script_name].last_result = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            
            self.logger.error(f"Failed to execute {script_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def _execution_worker(self):
        """Worker thread for executing scripts"""
        while self.running:
            try:
                execution_info = self.execution_queue.get(timeout=1)
                result = self._execute_script_sync(execution_info)
                self.result_queue.put({
                    "script_name": execution_info["script_name"],
                    "result": result
                })
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Execution worker error: {e}")
    
    def get_execution_results(self) -> List[Dict]:
        """Get pending execution results"""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results
    
    def get_script_status(self, script_name: str) -> Dict[str, Any]:
        """Get current status of a script"""
        script_info = self.scripts.get(script_name)
        if not script_info:
            return {"error": "Script not found"}
        
        return {
            "name": script_info.name,
            "status": script_info.status,
            "last_run": script_info.last_run.isoformat() if script_info.last_run else None,
            "last_result": script_info.last_result
        }
    
    def stop(self):
        """Stop the script manager"""
        self.running = False
        if self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)


class ConnectorInterface:
    """Interface for connectors to interact with scripts"""
    
    def __init__(self, script_manager: ScriptManager):
        self.script_manager = script_manager
        self.logger = logging.getLogger("ConnectorInterface")
    
    def handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a command from a connector"""
        cmd_type = command.get("command")
        
        if cmd_type == "list_scripts":
            scripts = self.script_manager.get_all_scripts()
            return {
                "scripts": [asdict(s) for s in scripts]
            }
        
        elif cmd_type == "get_script_info":
            script_name = command.get("script")
            info = self.script_manager.get_script_info(script_name)
            if info:
                return asdict(info)
            return {"error": "Script not found"}
        
        elif cmd_type == "execute_script":
            script_name = command.get("script")
            parameters = command.get("parameters", {})
            async_mode = command.get("async", True)
            return self.script_manager.execute_script(script_name, parameters, async_mode)
        
        elif cmd_type == "get_parameter":
            script_name = command.get("script")
            param_name = command.get("parameter")
            value = self.script_manager.get_parameter(script_name, param_name)
            return {"parameter": param_name, "value": value}
        
        elif cmd_type == "update_parameter":
            script_name = command.get("script")
            param_name = command.get("parameter")
            value = command.get("value")
            self.script_manager.update_parameter(script_name, param_name, value)
            return {"status": "updated", "parameter": param_name}
        
        elif cmd_type == "get_status":
            script_name = command.get("script")
            return self.script_manager.get_script_status(script_name)
        
        elif cmd_type == "get_results":
            results = self.script_manager.get_execution_results()
            return {"results": results}
        
        elif cmd_type == "reload_config":
            self.script_manager.load_config()
            return {"status": "config_reloaded"}
        
        else:
            return {"error": f"Unknown command: {cmd_type}"}


# Global instance for easy access
_script_manager = None
_connector_interface = None


def get_script_manager() -> ScriptManager:
    """Get the global script manager instance"""
    global _script_manager
    if _script_manager is None:
        _script_manager = ScriptManager()
    return _script_manager


def get_connector_interface() -> ConnectorInterface:
    """Get the global connector interface instance"""
    global _connector_interface
    if _connector_interface is None:
        _connector_interface = ConnectorInterface(get_script_manager())
    return _connector_interface


if __name__ == "__main__":
    # Test the script manager
    manager = get_script_manager()
    interface = get_connector_interface()
    
    # List scripts
    result = interface.handle_command({"command": "list_scripts"})
    print(f"Found {len(result['scripts'])} scripts")
    
    # Get script info
    result = interface.handle_command({
        "command": "get_script_info",
        "script": "analysis_engine.py"
    })
    print(f"Script info: {result}")
    
    # Keep running for testing
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
        print("\nScript manager stopped")