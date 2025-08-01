

import sys
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import importlib.util
import numpy as np

# --- Project Integrator ---

class ProjectIntegrator:
    """
    Scans the project structure to understand its components and enables inter-module imports.
    """
    def __init__(self):
        # Assuming this script is at /path/to/project/modules/analysis-reporting/analysis_engine.py
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.known_modules = {
            "artificial-intelligence": "AI & ML Models",
            "preprocessing": "Image Preprocessing Tools",
            "defect-detection": "Defect Detection Algorithms",
            "feature-engineering": "Feature Engineering Toolkit",
            "visualization": "Visualization Tools",
            "real-time-monitoring": "Real-Time Monitoring System",
            "iteration6-lab-framework": "Main Lab Framework",
        }
        self.found_modules = {}
        self.imported_modules = {}

    def scan_and_log_modules(self):
        """
        Scans for known modules and logs their presence.
        """
        log_message("Scanning for connected modules...", "INFO")
        for dir_name, description in self.known_modules.items():
            module_path = self.project_root / dir_name
            if module_path.exists() and module_path.is_dir():
                self.found_modules[dir_name] = str(module_path)
                log_message(f"Found module '{description}' at: {module_path}", "INFO")
            else:
                log_message(f"Module '{description}' not found.", "WARNING")
        
        if self.found_modules:
            log_message("Analysis Engine is connected to the larger project structure.", "INFO")
        else:
            log_message("Analysis Engine is running in a standalone mode.", "WARNING")

    def add_project_root_to_path(self):
        """
        Adds the project root to sys.path to allow for absolute imports.
        """
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
            print(f"[INFO] (Integrator) Added {self.project_root} to system path.")

    def import_module(self, module_name: str, file_path: str):
        """
        Dynamically imports a Python module from a given file path.
        Handles module names with hyphens by replacing them with underscores for the import.
        """
        safe_module_name = module_name.replace('-', '_')
        if safe_module_name in self.imported_modules:
            return self.imported_modules[safe_module_name]
        
        try:
            spec = importlib.util.spec_from_file_location(safe_module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.imported_modules[safe_module_name] = module
            log_message(f"Successfully imported module '{module_name}'", "DEBUG")
            return module
        except Exception as e:
            log_message(f"Failed to import module '{module_name}' from {file_path}: {e}", "ERROR")
            return None


# --- Import all analysis modules ---
# Note: This assumes all scripts are in the same directory and are importable.
# We will handle them as modules. For now, let's list them.
# In a real-world scenario, these would be proper Python modules.

from batch_summary_reporter import save_batch_summary_report_csv
from contour_analyzer import analyze_defect_contours
from csv_report_creator import CSVReportGenerator
from data_aggregation_reporter import DataAggregator
from defect_analysis_tool import DefectAnalyzer
from defect_characterizer import characterize_defects_comprehensive as characterize_defects_standard
from defect_characterizer_v4 import DefectCharacterizer as DefectCharacterizerV4
from defect_cluster_analyzer import DefectClusterAnalyzer
from defect_information_extractor import DefectInfo
from detailed_report_generator import generate_detailed_report
from exhaustive_comparator import compute_exhaustive_comparison
from image_result_handler import ImageResult
from image_statistics_calculator import ImageAnalysisStats
from individual_report_saver import save_individual_image_report_csv
from integrated_analysis_tool import IntegratedFiberAnalyzer
from morphological_analyzer import comprehensive_morphological_analysis
from pass_fail_criteria_applier import apply_pass_fail_criteria as pass_fail_standard
from pass_fail_criteria_applier_v3 import apply_pass_fail_criteria as pass_fail_v3
from quality_metrics_calculator import calculate_comprehensive_quality_metrics
from radial_profile_analyzer import analyze_multi_scale_profiles
from report_generator import ReportGenerator
from similarity_analyzer import ImageSimilarityAnalyzer
from statistical_analysis_toolkit import StatisticalAnalysisToolkit
from structural_comparator import compute_image_structural_comparison


# --- Dependency Management ---

def check_and_install_dependencies():
    """
    Checks for required libraries and installs them if they are missing.
    """
    required_libraries = {
        "pandas": "pandas",
        "numpy": "numpy",
        "cv2": "opencv-python",
        "skimage": "scikit-image",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "sklearn": "scikit-learn"
    }
    
    missing_libraries = []
    for lib_import, lib_install in required_libraries.items():
        try:
            __import__(lib_import)
        except ImportError:
            missing_libraries.append(lib_install)
            
    if missing_libraries:
        log_message(f"Missing libraries: {', '.join(missing_libraries)}. Attempting to install...", "WARNING")
        for lib in missing_libraries:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", lib])
                log_message(f"Successfully installed {lib}", "INFO")
            except subprocess.CalledProcessError as e:
                log_message(f"Failed to install {lib}. Please install it manually.", "ERROR")
                sys.exit(1)

# --- Logging ---

def setup_logging():
    """
    Sets up logging to both the console and a file.
    """
    log_file = "analysis_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def log_message(message, level="INFO"):
    """
    Logs a message at the specified level.
    """
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    elif level == "DEBUG":
        logging.debug(message)

# --- Configuration Management ---

def load_config(config_path="config.json") -> Dict[str, Any]:
    """
    Loads the JSON configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        log_message("Configuration file loaded successfully.", "INFO")
        return config
    except FileNotFoundError:
        log_message("config.json not found. Using default parameters.", "ERROR")
        return {}
    except json.JSONDecodeError:
        log_message("Error decoding config.json. Check for syntax errors.", "ERROR")
        return {}

# --- Main Engine Class ---

class AnalysisEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}

    def get_user_input(self, question: str) -> str:
        """
        Asks the user a question and returns the answer.
        """
        log_message(f"USER_INPUT_REQUIRED: {question}", "INFO")
        return input(f"{question} ")

    def run_full_pipeline(self, image_path: str, pipeline_options: Dict[str, str] = None):
        """
        Runs the full analysis pipeline on a single image.
        The neural network can control the flow by setting pipeline_options.
        """
        log_message(f"Starting full analysis pipeline for: {image_path}", "INFO")
        
        if not Path(image_path).exists():
            log_message(f"Image file not found: {image_path}", "ERROR")
            return

        if pipeline_options is None:
            pipeline_options = {
                "characterizer": "standard",  # "standard" or "v4"
                "pass_fail": "standard"  # "standard" or "v3"
            }

        # 1. Load Image
        try:
            import cv2
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError("Image could not be read.")
            log_message(f"Image loaded successfully. Shape: {image.shape}", "INFO")
        except Exception as e:
            log_message(f"Error loading image: {e}", "ERROR")
            return

        # 2. Run Integrated Analysis (for center, masks, etc.)
        integrated_analyzer = IntegratedFiberAnalyzer()
        # This tool is highly integrated, so we run it to get fundamental geometry.
        # We can make its internal steps and parameters tunable later if needed.
        geo_results = integrated_analyzer.analyze_image(image_path)
        if not geo_results.get('success'):
            log_message("Failed to get geometric data from integrated analyzer. Aborting.", "ERROR")
            return
        
        log_message("Geometric analysis complete.", "INFO")
        self.results['geometry'] = geo_results

        # 3. Morphological Analysis for defect detection
        log_message("Starting morphological analysis for defect detection...", "INFO")
        morph_config = self.config.get('morphological_analyzer', {})
        # We can make the mask more specific later, e.g., using zone masks from geo_results
        defect_mask = comprehensive_morphological_analysis(
            image,
            mask=None, # Analyze the whole image for now
            **morph_config
        ).get('combined_defects')
        
        if defect_mask is None or np.sum(defect_mask) == 0:
            log_message("No defects found by morphological analysis.", "INFO")
            # Even if no defects, we can still generate a report
            self.results['defects'] = []
        else:
            log_message(f"Detected {np.sum(defect_mask > 0)} defect pixels.", "INFO")
            self.results['defect_mask'] = defect_mask

            # 4. Defect Characterization
            characterizer_choice = pipeline_options.get("characterizer", "standard")
            log_message(f"Using defect characterizer: {characterizer_choice}", "INFO")

            if characterizer_choice == "standard":
                char_config = self.config.get('defect_characterizer', {})
                # The standard characterizer needs a list of dicts with masks
                # For now, we pass the combined mask as a single defect region
                detected_defects_info = [{"defect_mask": defect_mask, "zone": "Unknown"}]
                characterized_defects = characterize_defects_standard(
                    detected_defects_info,
                    image,
                    um_per_px=self.config.get('shared_parameters', {}).get('pixels_per_micron'),
                    min_defect_area=char_config.get('min_defect_area_px', 5)
                )
            else: # v4
                char_v4 = DefectCharacterizerV4(**self.config.get('defect_characterizer_v4', {}))
                characterized_defects, _ = char_v4.characterize_defects(
                    defect_mask,
                    image_filename=Path(image_path).name,
                    um_per_px=self.config.get('shared_parameters', {}).get('pixels_per_micron')
                )
            
            log_message(f"Characterized {len(characterized_defects)} defects.", "INFO")
            self.results['defects'] = characterized_defects

            # 5. Pass/Fail Evaluation
            pass_fail_choice = pipeline_options.get("pass_fail", "standard")
            log_message(f"Using pass/fail evaluator: {pass_fail_choice}", "INFO")

            if pass_fail_choice == "standard":
                # This requires DefectInfo objects, so we'd need a conversion step.
                # For now, let's assume a simplified dict structure is compatible.
                # This highlights a future integration task.
                log_message("Standard pass/fail requires data conversion. Skipping for now.", "WARNING")
                pass_fail_results = {"status": "INCONCLUSIVE", "reason": "Data structure mismatch"}
            else: # v3
                import pandas as pd
                defects_df = pd.DataFrame(self.results['defects'])
                status, reasons = pass_fail_v3(defects_df)
                pass_fail_results = {"status": status, "reasons": reasons}

            log_message(f"Pass/Fail result: {pass_fail_results.get('status')}", "INFO")
            self.results['pass_fail'] = pass_fail_results

        # 6. Generate Reports
        log_message("Generating reports...", "INFO")
        self.generate_reports(image_path, image, geo_results)

        log_message("Full analysis pipeline finished.", "INFO")
        return self.results

    def generate_reports(self, image_path, image, geo_results):
        """
        Generates all the output reports for the analysis.
        """
        output_dir = Path("./analysis_output")
        output_dir.mkdir(exist_ok=True)
        base_filename = Path(image_path).stem

        # 1. Annotated Image
        report_generator = ReportGenerator()
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        report_data = {
            "characterized_defects": self.results.get('defects', []),
            "overall_status": self.results.get('pass_fail', {}).get('status', 'UNKNOWN')
        }
        annotated_image_path = output_dir / f"{base_filename}_annotated.png"
        report_generator.generate_annotated_image(
            original_bgr_image=bgr_image,
            analysis_results=report_data,
            zone_masks=geo_results.get('masks', {}).get('cleaned_masks'),
            output_path=annotated_image_path
        )

        # 2. CSV Report
        csv_reporter = CSVReportGenerator()
        csv_report_path = output_dir / f"{base_filename}_report.csv"
        
        # The CSV reporter expects a specific dict structure. We assemble it here.
        csv_data = {
            "characterized_defects": self.results.get('defects', []),
            "overall_status": self.results.get('pass_fail', {}).get('status', 'UNKNOWN'),
            "failure_reasons": self.results.get('pass_fail', {}).get('reasons', [])
        }
        csv_reporter.generate_defect_csv_report(csv_data, csv_report_path)
        
        # 3. Summary Report File
        summary_path = output_dir / f"{base_filename}_summary.json"
        with open(summary_path, 'w') as f:
            # Create a serializable version of the results
            serializable_results = self.results.copy()
            # Remove non-serializable items like numpy arrays
            serializable_results.pop('defect_mask', None)
            serializable_results.pop('geometry', None) # This can be large
            json.dump(serializable_results, f, indent=4, default=str)
        log_message(f"Summary JSON report saved to {summary_path}", "INFO")

    def update_parameters(self, new_params: Dict[str, Any]):
        """
        Allows the neural network to update the parameters in the config.
        """
        log_message("Updating parameters from external controller.", "INFO")
        for key, value in new_params.items():
            if key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value
        log_message(f"Updated parameters: {json.dumps(new_params, indent=2)}", "DEBUG")


if __name__ == '__main__':
    # 0. Setup Project Integration and Path
    integrator = ProjectIntegrator()
    integrator.add_project_root_to_path()
    
    # 1. Setup Logging
    setup_logging()
    
    # Now that logging is configured, scan and log modules
    integrator.scan_and_log_modules()
    
    # 2. Check and Install Dependencies
    check_and_install_dependencies()
    
    # 3. Load Configuration
    config_path = Path(__file__).parent / "config.json"
    config = load_config(config_path)
    
    # 4. Initialize Engine
    engine = AnalysisEngine(config, integrator)
    
    # 5. Start Interactive Loop
    log_message("Analysis Engine Initialized.", "INFO")
    
    if len(sys.argv) > 1:
        image_to_process = sys.argv[1]
        engine.run_full_pipeline(image_to_process)
    else:
        while True:
            print("\n--- Analysis Engine Control ---")
            print("1. Run analysis on an image")
            print("2. Update parameters")
            print("3. Exit")
            choice = input("Enter your choice: ")
            
            if choice == '1':
                img_path = engine.get_user_input("Enter the path to the image:")
                engine.run_full_pipeline(img_path)
            elif choice == '2':
                try:
                    param_update_str = engine.get_user_input("Enter parameter updates as a JSON string:")
                    param_update = json.loads(param_update_str)
                    engine.update_parameters(param_update)
                    log_message("Parameters updated successfully.", "INFO")
                except json.JSONDecodeError:
                    log_message("Invalid JSON string for parameter update.", "ERROR")
            elif choice == '3':
                log_message("Exiting Analysis Engine.", "INFO")
                break
            else:
                print("Invalid choice. Please try again.")

