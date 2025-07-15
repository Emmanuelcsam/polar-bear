#!/usr/bin/env python3
"""
Script Wrappers Module
Provides wrapper functions for all analysis/reporting scripts to enable connector control
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.util
import numpy as np
from datetime import datetime

# Setup logging
logger = logging.getLogger("ScriptWrappers")
logger.setLevel(logging.INFO)


class ScriptWrapper:
    """Base wrapper class for script execution"""
    
    def __init__(self, script_name: str, module_path: str):
        self.script_name = script_name
        self.module_path = module_path
        self.module = None
        self.logger = logging.getLogger(f"Wrapper_{script_name}")
    
    def load_module(self):
        """Load the script module dynamically"""
        if self.module is None:
            spec = importlib.util.spec_from_file_location(
                self.script_name.replace(".py", ""),
                self.module_path
            )
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        return self.module
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the script with given parameters"""
        raise NotImplementedError("Subclasses must implement execute method")


class AnalysisEngineWrapper(ScriptWrapper):
    """Wrapper for analysis_engine.py"""
    
    def __init__(self):
        super().__init__("analysis_engine.py", 
                        Path(__file__).parent / "analysis_engine.py")
    
    def execute(self, image_path: str, output_dir: str = ".", 
                config_overrides: Dict = None, verbose: bool = False) -> Dict[str, Any]:
        """Execute the analysis engine"""
        try:
            module = self.load_module()
            
            # Create engine instance
            engine = module.AnalysisEngine(verbose=verbose)
            
            # Apply config overrides if provided
            if config_overrides:
                engine.update_parameters(config_overrides)
            
            # Run the pipeline
            results = engine.run_full_pipeline(image_path, output_dir)
            
            return {
                "success": True,
                "results": results,
                "output_files": {
                    "annotated_image": results.get("annotated_image_path"),
                    "csv_report": results.get("csv_report_path"),
                    "json_summary": results.get("json_summary_path")
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to execute analysis engine: {e}")
            return {"success": False, "error": str(e)}


class DefectAnalyzerWrapper(ScriptWrapper):
    """Wrapper for defect-analyzer.py"""
    
    def __init__(self):
        super().__init__("defect-analyzer.py",
                        Path(__file__).parent / "defect-analyzer.py")
    
    def execute(self, refined_masks: Dict[str, np.ndarray], 
                pixels_per_micron: float = 0.5) -> Dict[str, Any]:
        """Execute defect analysis"""
        try:
            module = self.load_module()
            
            # Call the analyze_defects function
            defect_infos = module.analyze_defects(refined_masks, pixels_per_micron)
            
            # Convert DefectInfo objects to dictionaries
            results = []
            for info in defect_infos:
                results.append({
                    "defect_id": info.defect_id,
                    "mask_type": info.mask_type,
                    "area_px": info.area_px,
                    "area_um2": info.area_um2,
                    "centroid": info.centroid,
                    "bbox": info.bbox,
                    "major_axis": info.major_axis_length,
                    "minor_axis": info.minor_axis_length,
                    "circularity": info.circularity,
                    "aspect_ratio": info.aspect_ratio,
                    "eccentricity": info.eccentricity,
                    "classification": info.classification,
                    "confidence": info.confidence,
                    "zone": info.zone
                })
            
            return {
                "success": True,
                "defects": results,
                "total_defects": len(results)
            }
        except Exception as e:
            self.logger.error(f"Failed to execute defect analyzer: {e}")
            return {"success": False, "error": str(e)}


class ReportGeneratorWrapper(ScriptWrapper):
    """Wrapper for report-generator.py"""
    
    def __init__(self):
        super().__init__("report-generator.py",
                        Path(__file__).parent / "report-generator.py")
    
    def execute(self, image: np.ndarray, results: Dict[str, Any],
                localization: Dict[str, Any], zone_masks: Dict[str, np.ndarray],
                output_path: str, fiber_type: str = "single_mode") -> Dict[str, Any]:
        """Execute report generation"""
        try:
            module = self.load_module()
            
            # Create ReportGenerator instance
            generator = module.ReportGenerator()
            
            # Generate annotated image
            annotated_path = generator.generate_annotated_image(
                image, results, localization, zone_masks, 
                output_path, fiber_type
            )
            
            return {
                "success": True,
                "annotated_image_path": annotated_path,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to execute report generator: {e}")
            return {"success": False, "error": str(e)}


class QualityMetricsWrapper(ScriptWrapper):
    """Wrapper for quality-metrics-calculator.py"""
    
    def __init__(self):
        super().__init__("quality-metrics-calculator.py",
                        Path(__file__).parent / "quality-metrics-calculator.py")
    
    def execute(self, image: np.ndarray, defect_mask: np.ndarray = None,
                roi_mask: np.ndarray = None, save_report: bool = False,
                report_path: str = None) -> Dict[str, Any]:
        """Execute quality metrics calculation"""
        try:
            module = self.load_module()
            
            # Calculate comprehensive metrics
            metrics = module.calculate_comprehensive_quality_metrics(
                image, defect_mask, roi_mask
            )
            
            # Generate report if requested
            if save_report and report_path:
                module.generate_quality_report(
                    image, metrics, defect_mask, roi_mask, 
                    save_path=report_path
                )
            
            return {
                "success": True,
                "metrics": metrics,
                "quality_score": metrics.get("quality_score"),
                "report_saved": save_report
            }
        except Exception as e:
            self.logger.error(f"Failed to execute quality metrics: {e}")
            return {"success": False, "error": str(e)}


class UniversalScriptWrapper(ScriptWrapper):
    """Universal wrapper for any script with standard interface"""
    
    def __init__(self, script_name: str):
        script_path = Path(__file__).parent / script_name
        super().__init__(script_name, script_path)
    
    def execute(self, function_name: str = None, **kwargs) -> Dict[str, Any]:
        """Execute a specific function from the script"""
        try:
            module = self.load_module()
            
            # If no function specified, try to find main function
            if not function_name:
                # Look for common function patterns
                for name in ["main", "run", "execute", "analyze", "process"]:
                    if hasattr(module, name):
                        function_name = name
                        break
                
                # Look for functions matching script name pattern
                if not function_name:
                    script_base = self.script_name.replace(".py", "").replace("-", "_")
                    for attr_name in dir(module):
                        if attr_name.startswith(script_base) or \
                           attr_name.startswith("analyze_") or \
                           attr_name.startswith("generate_") or \
                           attr_name.startswith("calculate_"):
                            function_name = attr_name
                            break
            
            if not function_name:
                return {"success": False, "error": "No executable function found"}
            
            # Get the function
            func = getattr(module, function_name, None)
            if not callable(func):
                return {"success": False, "error": f"{function_name} is not callable"}
            
            # Execute the function
            result = func(**kwargs)
            
            return {
                "success": True,
                "function": function_name,
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute {self.script_name}: {e}")
            return {"success": False, "error": str(e)}


# Additional specialized wrappers for other scripts
class BatchSummaryReporterWrapper(ScriptWrapper):
    """Wrapper for batch-summary-reporter.py"""
    
    def __init__(self):
        super().__init__("batch-summary-reporter.py",
                        Path(__file__).parent / "batch-summary-reporter.py")
    
    def execute(self, results_list: List[Dict], output_path: str = None) -> Dict[str, Any]:
        """Execute batch summary reporting"""
        try:
            module = self.load_module()
            
            # Try to find the main function
            if hasattr(module, 'generate_batch_summary'):
                summary = module.generate_batch_summary(results_list, output_path)
                return {"success": True, "summary": summary}
            else:
                # Use universal approach
                return UniversalScriptWrapper(self.script_name).execute(**{"results_list": results_list, "output_path": output_path})
        except Exception as e:
            self.logger.error(f"Failed to execute batch summary reporter: {e}")
            return {"success": False, "error": str(e)}


class DefectCharacterizerWrapper(ScriptWrapper):
    """Wrapper for defect-characterizer.py"""
    
    def __init__(self):
        super().__init__("defect-characterizer.py",
                        Path(__file__).parent / "defect-characterizer.py")
    
    def execute(self, contour: np.ndarray = None, mask: np.ndarray = None,
                um_per_px: float = None, **kwargs) -> Dict[str, Any]:
        """Execute defect characterization"""
        try:
            module = self.load_module()
            
            # If specific parameters provided, use characterize_defect_geometry
            if contour is not None and mask is not None:
                if hasattr(module, 'characterize_defect_geometry'):
                    result = module.characterize_defect_geometry(contour, mask, um_per_px)
                    return {"success": True, "geometry": result}
            
            # Otherwise use universal approach
            return UniversalScriptWrapper(self.script_name).execute(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to execute defect characterizer: {e}")
            return {"success": False, "error": str(e)}


class DataAggregationReporterWrapper(ScriptWrapper):
    """Wrapper for data-aggregation-reporter.py"""
    
    def __init__(self):
        super().__init__("data-aggregation-reporter.py",
                        Path(__file__).parent / "data-aggregation-reporter.py")
    
    def execute(self, data_sources: List[str] = None, output_format: str = "json", **kwargs) -> Dict[str, Any]:
        """Execute data aggregation reporting"""
        try:
            module = self.load_module()
            
            # Look for main aggregation function
            if hasattr(module, 'aggregate_data'):
                result = module.aggregate_data(data_sources, output_format)
                return {"success": True, "aggregated_data": result}
            else:
                return UniversalScriptWrapper(self.script_name).execute(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to execute data aggregation reporter: {e}")
            return {"success": False, "error": str(e)}


class PassFailEvaluatorWrapper(ScriptWrapper):
    """Wrapper for pass-fail-evaluator.py"""
    
    def __init__(self):
        super().__init__("pass-fail-evaluator.py",
                        Path(__file__).parent / "pass-fail-evaluator.py")
    
    def execute(self, results: Dict[str, Any], criteria: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Execute pass/fail evaluation"""
        try:
            module = self.load_module()
            
            # Look for evaluation function
            if hasattr(module, 'evaluate_pass_fail'):
                evaluation = module.evaluate_pass_fail(results, criteria)
                return {"success": True, "evaluation": evaluation}
            else:
                return UniversalScriptWrapper(self.script_name).execute(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to execute pass-fail evaluator: {e}")
            return {"success": False, "error": str(e)}


# Registry of all available wrappers
SCRIPT_WRAPPERS = {
    "analysis_engine.py": AnalysisEngineWrapper,
    "defect-analyzer.py": DefectAnalyzerWrapper,
    "report-generator.py": ReportGeneratorWrapper,
    "quality-metrics-calculator.py": QualityMetricsWrapper,
    "batch-summary-reporter.py": BatchSummaryReporterWrapper,
    "defect-characterizer.py": DefectCharacterizerWrapper,
    "data-aggregation-reporter.py": DataAggregationReporterWrapper,
    "pass-fail-evaluator.py": PassFailEvaluatorWrapper,
}


def get_wrapper(script_name: str) -> ScriptWrapper:
    """Get appropriate wrapper for a script"""
    if script_name in SCRIPT_WRAPPERS:
        return SCRIPT_WRAPPERS[script_name]()
    else:
        # Return universal wrapper for unknown scripts
        return UniversalScriptWrapper(script_name)


def execute_script(script_name: str, **parameters) -> Dict[str, Any]:
    """Main entry point for executing any script with parameters"""
    wrapper = get_wrapper(script_name)
    return wrapper.execute(**parameters)


def list_available_scripts() -> List[Dict[str, str]]:
    """List all available scripts with their wrapper types"""
    scripts = []
    
    # Add known scripts
    for script_name, wrapper_class in SCRIPT_WRAPPERS.items():
        scripts.append({
            "name": script_name,
            "wrapper": wrapper_class.__name__,
            "type": "specialized"
        })
    
    # Add other Python scripts in directory
    script_dir = Path(__file__).parent
    for script_path in script_dir.glob("*.py"):
        if script_path.name not in SCRIPT_WRAPPERS and \
           script_path.name not in ["__init__.py", "script_interface.py", 
                                   "script_wrappers.py", "connector.py", 
                                   "hivemind_connector.py"]:
            scripts.append({
                "name": script_path.name,
                "wrapper": "UniversalScriptWrapper",
                "type": "universal"
            })
    
    return scripts


if __name__ == "__main__":
    # Test the wrappers
    print("Available scripts:")
    for script in list_available_scripts():
        print(f"  - {script['name']} ({script['wrapper']})")
    
    # Example usage
    if len(sys.argv) > 1:
        script_name = sys.argv[1]
        # Parse remaining arguments as key=value pairs
        params = {}
        for arg in sys.argv[2:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                params[key] = value
        
        print(f"\nExecuting {script_name} with parameters: {params}")
        result = execute_script(script_name, **params)
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print("\nUsage: python script_wrappers.py <script_name> [param1=value1 param2=value2 ...]")