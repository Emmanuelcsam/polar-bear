

import os
import sys
import subprocess

# Ensure the core modules can be imported
# This adds the 'neural_framework' directory to the Python path
framework_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(framework_dir))

from neural_framework.core.logger import log
from neural_framework.core.dependency_manager import DependencyManager
from neural_framework.core.orchestrator import Orchestrator

def main():
    """
    Main entrypoint for the Neural Framework.
    """
    log.info("=================================================")
    log.info("Initializing Neural Network Integration Framework")
    log.info("=================================================")

    project_root = os.path.dirname(framework_dir)
    
    # --- Phase 1: Dependency Management (Skipped for now) ---
    # The dependency situation is complex. We will proceed to the orchestration
    # and deal with ImportErrors as they arise during execution.
    # To run the dependency check, uncomment the following lines:
    #
    # log.info("Phase 1: Starting Dependency Management...")
    # manager = DependencyManager(project_root)
    # missing_packages = manager.check_and_install()
    # if isinstance(missing_packages, list) and missing_packages:
    #     command = f"pip install --upgrade {' '.join(missing_packages)}"
    #     log.critical("The framework cannot continue without its dependencies.")
    #     log.info("To install them, please run the following command in your terminal:")
    #     print(f"\n    {command}\n")
    #     sys.exit(1)
    # log.info("Dependency check passed. All required packages are installed.")
    # log.info("--- Dependency Management Complete ---")

    # --- Phase 2: Module Analysis & Orchestration ---
    log.info("\nPhase 2: Starting Module Analysis & Orchestration...")
    orchestrator = Orchestrator(project_root)
    orchestrator.discover_modules()
    log.info("--- Module Analysis Complete ---")

    # --- Phase 3: Demonstration ---
    log.info("\n--- Running Demonstration ---")
    log.info("The orchestrator will now attempt to run the 'main' function from the 'blob_defect_finder' module.")
    log.info("This demonstrates the ability to dynamically load and execute a specific, unambiguous function.")

    orchestrator.run_function('main', module_name='blob_defect_finder')
    log.info("Demonstration finished. Check for output files like 'blob_detection_test.png' if it was successful.")
    
    log.info("\nFramework initialization and demonstration complete.")



if __name__ == "__main__":
    main()

