import os
import sys
import json
import shutil
import time
from pathlib import Path
import logging
import shlex

# --- Setup Logging ---
# Configures a logger to print detailed, timestamped messages to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Add script directories to Python path ---
# This ensures that we can import our custom modules (process, separation, detection)
# as long as they are in the same directory as this script.
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# --- Import from your custom scripts ---
# This block attempts to import the necessary components from your other scripts.
# If any import fails, it will log a fatal error and exit.
try:
    from process import reimagine_image
    from separation import UnifiedSegmentationSystem
    from detection import OmniFiberAnalyzer, OmniConfig
    from data_acquisition import integrate_with_pipeline
    logging.info("Successfully imported custom processing & analysis modules.")
except ImportError as e:
    logging.error(f"Fatal Error: Failed to import a required module: {e}")
    logging.error("Please ensure process.py, separation.py, detection.py, and data_acquisition.py are in the same directory as app.py.")
    sys.exit(1)


class PipelineOrchestrator:
    """
    This class manages the entire multi-stage defect analysis pipeline.
    It controls the flow from processing to separation to detection to final analysis.
    """
    def __init__(self, config_path):
        """Initializes the orchestrator with configuration."""
        logging.info("Initializing Pipeline Orchestrator...")
        self.config_path = Path(config_path).resolve()  # Store absolute config path
        self.config = self.load_config(config_path)
        self.config = self.resolve_config_paths(self.config)  # Resolve relative paths
        self.results_base_dir = Path(self.config['paths']['results_dir'])
        self.results_base_dir.mkdir(parents=True, exist_ok=True)  # Create with parents
        logging.info(f"Results will be saved in: {self.results_base_dir}")

    def load_config(self, config_path):
        """Loads the JSON configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Successfully loaded configuration from {config_path}")
            return config
        except Exception as e:
            logging.error(f"Fatal Error: Could not load or parse config.json: {e}")
            sys.exit(1)

    def resolve_config_paths(self, config):
        """Convert relative paths in config to absolute paths based on config file location"""
        config_dir = self.config_path.parent
        
        # Update paths to be absolute
        for key in ['results_dir', 'zones_methods_dir', 'detection_knowledge_base']:
            if key in config['paths']:
                path = Path(config['paths'][key])
                if not path.is_absolute():
                    # Make it absolute relative to the config directory
                    config['paths'][key] = str(config_dir / path)
        
        return config

    def run_full_pipeline(self, input_image_path: Path):
        """
        Runs the entire analysis pipeline for a single image, creating a
        dedicated folder for its results.
        """
        start_time = time.time()
        logging.info(f"--- Starting full pipeline for: {input_image_path.name} ---")

        # Create a unique directory for this image's results
        run_dir = self.results_base_dir / input_image_path.stem
        run_dir.mkdir(exist_ok=True)

        # === STAGE 1: PROCESSING (REIMAGINE) ===
        reimagined_dir, all_images_to_separate = self.run_processing_stage(input_image_path, run_dir)

        # === STAGE 2: SEPARATION (ZONING) ===
        separated_dir, all_images_to_detect = self.run_separation_stage(all_images_to_separate, run_dir, input_image_path)

        # === STAGE 3: DETECTION (ANALYSIS) ===
        self.run_detection_stage(all_images_to_detect, run_dir)

        # === STAGE 4: DATA ACQUISITION (FINAL ANALYSIS) ===
        self.run_data_acquisition_stage(input_image_path.stem, run_dir)

        end_time = time.time()
        logging.info(f"--- Pipeline for {input_image_path.name} completed in {end_time - start_time:.2f} seconds ---")

    def run_processing_stage(self, input_image_path, run_dir):
        """Runs the process.py script to generate multiple image versions."""
        logging.info(">>> STAGE 1: PROCESSING - Reimagining images...")
        process_cfg = self.config['process_settings']
        reimagined_dir = run_dir / process_cfg['output_folder_name']
        
        try:
            # We call the imported function directly.
            reimagine_image(str(input_image_path), str(reimagined_dir))
        except Exception as e:
            logging.error(f"Error during reimagine_image for {input_image_path.name}: {e}")
            # Even if it fails, we continue with the original image
        
        # Gather all images for the next stage. This includes the original image
        # and all its reimagined versions.
        all_images_to_separate = [input_image_path]
        reimagined_files = list(reimagined_dir.glob('*.jpg')) if reimagined_dir.exists() else []
        all_images_to_separate.extend(reimagined_files)
        
        logging.info(f"Processing stage complete. Found {len(reimagined_files)} reimagined images.")
        return reimagined_dir, all_images_to_separate

    def run_separation_stage(self, image_paths, run_dir, original_image_path):
        """Runs separation.py to create zoned regions for each image."""
        logging.info(">>> STAGE 2: SEPARATION - Generating zoned regions...")
        separation_cfg = self.config['separation_settings']
        zones_methods_dir = self.config['paths']['zones_methods_dir']
        separated_dir = run_dir / separation_cfg['output_folder_name']
        separated_dir.mkdir(exist_ok=True)
        
        all_separated_regions = []

        try:
            # Initialize the segmentation system with the correct methods directory
            separator = UnifiedSegmentationSystem(methods_dir=zones_methods_dir)
            
            for image_path in image_paths:
                logging.info(f"Separating image: {image_path.name}")
                # Define the output directory for this specific image's separation results
                image_separation_output_dir = separated_dir / image_path.stem
                
                # Run separation and get consensus masks
                consensus = separator.process_image(image_path, str(image_separation_output_dir))
                
                if consensus and consensus.get('saved_regions'):
                    all_separated_regions.extend([Path(p) for p in consensus['saved_regions']])
        
        except Exception as e:
            logging.error(f"A critical error occurred in the separation stage: {e}", exc_info=True)

        # The final list for detection must include all separated regions PLUS the un-separated original image.
        all_images_to_detect = all_separated_regions + [original_image_path]

        logging.info(f"Separation stage complete. Generated {len(all_separated_regions)} separated regions.")
        logging.info(f"Total inputs for detection stage: {len(all_images_to_detect)}")
        return separated_dir, all_images_to_detect
    
    def run_detection_stage(self, image_paths, run_dir):
        """Runs detection.py to perform defect analysis on all provided images."""
        logging.info(">>> STAGE 3: DETECTION - Analyzing for defects...")
        detection_cfg = self.config['detection_settings']
        detection_output_dir = run_dir / detection_cfg['output_folder_name']
        detection_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create the config object for the detector from our main config file
            omni_config_dict = detection_cfg['config'].copy()
            
            # Handle the knowledge base path
            kb_path = self.config['paths'].get('detection_knowledge_base')
            if kb_path:
                omni_config_dict['knowledge_base_path'] = kb_path
            
            # Pass the dictionary to the OmniConfig dataclass to create an instance
            omni_config = OmniConfig(**omni_config_dict)

            # Initialize the analyzer once with the full configuration
            analyzer = OmniFiberAnalyzer(omni_config)

            for image_path in image_paths:
                logging.info(f"Detecting defects in: {image_path.name}")
                # Define the output dir for this specific image's detection report
                image_detection_output_dir = detection_output_dir / image_path.stem
                image_detection_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Run analysis. The modified detection.py accepts an output directory.
                analyzer.analyze_end_face(str(image_path), str(image_detection_output_dir))
        
        except Exception as e:
            logging.error(f"A critical error occurred in the detection stage: {e}", exc_info=True)
        
        logging.info("Detection stage complete.")

    def run_data_acquisition_stage(self, image_name, run_dir):
        """Run final data acquisition and comprehensive analysis"""
        logging.info(">>> STAGE 4: DATA ACQUISITION - Aggregating results and generating final analysis...")
        
        try:
            # Call the data acquisition integration function
            report = integrate_with_pipeline(str(run_dir), image_name)
            
            # Log summary results
            logging.info(f"Final Analysis Complete:")
            logging.info(f"  Status: {report['pass_fail_status']}")
            logging.info(f"  Quality Score: {report['quality_score']}/100")
            logging.info(f"  Total Unique Defects: {report['total_defects_found']}")
            
            # Log statistics
            if 'statistics' in report:
                stats = report['statistics']
                logging.info(f"  Defect Breakdown:")
                for defect_type, count in stats.get('by_type', {}).items():
                    logging.info(f"    {defect_type}: {count}")
                    
                logging.info(f"  Severity Distribution:")
                for severity, count in stats.get('by_severity', {}).items():
                    logging.info(f"    {severity}: {count}")
                    
            # Log processing efficiency
            if 'processing_info' in report:
                proc_info = report['processing_info']
                logging.info(f"  Processing Efficiency:")
                logging.info(f"    Raw defects: {proc_info.get('total_raw_defects', 0)}")
                logging.info(f"    After deduplication: {proc_info.get('defects_after_merging', 0)}")
                logging.info(f"    Reduction ratio: {proc_info.get('reduction_ratio', 0):.1%}")
            
        except Exception as e:
            logging.error(f"Data acquisition stage failed: {e}", exc_info=True)
            logging.error("The pipeline will continue, but final aggregated results are not available.")

# --- New Interactive Functions ---

def ask_for_images() -> list[Path]:
    """
    Prompts the user to enter one or more image paths and validates them.
    Handles paths with spaces correctly if they are quoted.
    """
    print("\nEnter one or more full image paths. Separate paths with spaces.")
    print("Example: C:\\Users\\Test\\img1.png \"C:\\My Images\\test.png\"")
    paths_input = input("> ").strip()
    
    if not paths_input:
        return []
        
    # Use shlex to correctly parse command-line style input, handling quotes
    path_strings = shlex.split(paths_input)
    
    valid_paths = []
    invalid_paths = []
    for path_str in path_strings:
        path = Path(path_str)
        if path.is_file():
            valid_paths.append(path)
        else:
            invalid_paths.append(str(path))
            
    if invalid_paths:
        logging.warning(f"The following paths were not found and will be skipped: {', '.join(invalid_paths)}")
        
    return valid_paths

def ask_for_folder() -> Path | None:
    """Prompts the user for a single folder path and validates it."""
    folder_path_str = input("\nEnter the full path to the folder containing images: ").strip()
    
    # Handle quoted paths
    if folder_path_str:
        folder_path_str = shlex.split(folder_path_str)[0] if folder_path_str else ""
    
    if not folder_path_str:
        return None
        
    folder_path = Path(folder_path_str)
    
    if folder_path.is_dir():
        return folder_path
    else:
        logging.error(f"Directory not found: {folder_path}")
        return None

def main():
    """
    The main entry point of the application, featuring an interactive menu.
    """
    print("\n" + "="*80)
    print("UNIFIED FIBER OPTIC DEFECT DETECTION PIPELINE".center(80))
    print("WITH COMPREHENSIVE DATA ACQUISITION".center(80))
    print("Interactive Mode".center(80))
    print("="*80)
    
    # Get config path from user
    config_path_str = input("Enter path to config.json (or press Enter for default 'config.json'): ").strip()
    if not config_path_str:
        config_path_str = "config.json"

    # Remove leading/trailing quotes that might be pasted from file explorers
    if config_path_str.startswith('"') and config_path_str.endswith('"'):
        config_path_str = config_path_str[1:-1]
    
    config_path = Path(config_path_str)
    if not config_path.exists():
        logging.error(f"Fatal Error: Configuration file not found at: {config_path}")
        print("\nPlease run setup.py first to create the necessary files and directories.")
        sys.exit(1)
        
    # Initialize the orchestrator
    try:
        orchestrator = PipelineOrchestrator(str(config_path))
    except Exception as e:
        logging.error(f"Failed to initialize pipeline: {e}")
        print("\nPlease check your configuration file and ensure all required directories exist.")
        print("You may need to run setup.py first.")
        sys.exit(1)
    
    # Main menu loop
    while True:
        print("\n--- MAIN MENU ---")
        print("1. Process a list of specific images")
        print("2. Process all images in a folder")
        print("3. Exit")
        
        choice = input("Please select an option (1-3): ").strip()
        
        if choice == '1':
            image_paths = ask_for_images()
            if not image_paths:
                logging.warning("No valid image paths provided.")
                continue
            
            logging.info(f"Starting processing for {len(image_paths)} image(s).")
            for image_path in image_paths:
                try:
                    orchestrator.run_full_pipeline(image_path)
                except Exception as e:
                    logging.error(f"Failed to process {image_path}: {e}")
                    continue
            logging.info("Finished processing all specified images.")
            
        elif choice == '2':
            folder_path = ask_for_folder()
            if not folder_path:
                continue
                
            logging.info(f"Searching for images in directory: {folder_path}")
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
            # Remove duplicates
            image_files = list(set(image_files))
            
            if not image_files:
                logging.warning(f"No images with extensions ({', '.join(image_extensions)}) found in {folder_path}")
                continue
            
            logging.info(f"Found {len(image_files)} images to process. Starting batch.")
            for image_file in sorted(image_files):
                try:
                    orchestrator.run_full_pipeline(image_file)
                except Exception as e:
                    logging.error(f"Failed to process {image_file}: {e}")
                    continue
            logging.info("Finished processing all images in the folder.")
            
        elif choice == '3':
            print("\nExiting the application. Goodbye!")
            break
            
        else:
            logging.warning("Invalid choice. Please enter a number from 1 to 3.")

if __name__ == "__main__":
    main()