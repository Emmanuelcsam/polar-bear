#!/usr/bin/env python3
"""Script to organize files into subdirectories based on functionality"""

import os
import shutil
import re
from pathlib import Path

# Get the parent directory
parent_dir = Path("..")

# Define the mapping of files to directories
file_mappings = {
    "core/base": [
        "__init__*.py",
        "main.py", "main-*.py", "main-application.py", "main-controller.py",
        "polar_bear_brain.py", "polar_bear_master.py",
        "orchestrator*.py", "master_orchestrator.py",
        "shared_state.py", "common_data_and_utils.py"
    ],
    
    "core/connectors": [
        "connector.py", "connector-*.py",
        "hivemind_connector.py", "hivemind_connector-*.py",
        "connector_interface*.py", "connector_enhanced.py",
        "mega_connector.py", "ultimate_mega_connector.py", "universal_connector.py"
    ],
    
    "core/config": [
        "0_config.py", "config*.py",
        "configuration-manager*.py", "config-management*.py",
        "config-wizard.py", "setup-configuration-tool*.py",
        "shared_config*.py", "polar_bear_config.json"
    ],
    
    "core/setup": [
        "setup.py", "setup-*.py",
        "1_setup_directories.py",
        "auto_installer_refactored.py"
    ],
    
    "ai_ml": [
        "cnn*.py", "neural*.py", "ml*.py", "nn*.py",
        "VAE*.py", "transformer*.py", "pytorch*.py", "tensorflow*.py",
        "machine-learning*.py", "deep*.py", "ai_demo.py",
        "anomaly-model-trainer*.py", "defect-detection-ai*.py",
        "quality-classification-ai*.py", "segmentation-model-trainer*.py",
        "training*.py", "learner*.py"
    ],
    
    "image_processing": [
        "image-processor*.py", "image_processor*.py", "image-*.py",
        "clahe*.py", "histogram*.py", "gaussian*.py", "bilateral*.py",
        "median*.py", "morphological*.py", "erosion*.py", "dilation*.py",
        "canny*.py", "sobel*.py", "laplacian*.py", "edge*.py",
        "resize*.py", "rotate*.py", "flip*.py", "crop*.py",
        "filter*.py", "blur*.py", "threshold*.py", "adaptive*.py",
        "enhancement*.py", "preprocessing*.py", "transform*.py"
    ],
    
    "data_processing": [
        "batch*.py", "data*.py", "intensity*.py", "pixel*.py",
        "matrix*.py", "csv*.py", "exporter*.py", "loader*.py",
        "reader*.py", "processor*.py", "acquisition*.py"
    ],
    
    "defect_detection": [
        "defect*.py", "anomaly*.py", "scratch*.py", "detection*.py",
        "do2mr*.py", "lei*.py", "blob*.py", "mser*.py",
        "outlier*.py", "deviation*.py", "fault*.py"
    ],
    
    "utilities/helpers": [
        "helper*.py", "utility*.py", "utils*.py", "tool*.py",
        "logger*.py", "logging*.py", "debug*.py", "timer*.py",
        "performance*.py", "optimizer*.py", "numpy-json-encoder*.py"
    ],
    
    "utilities/monitoring": [
        "monitor*.py", "realtime*.py", "live*.py", "streaming*.py",
        "dashboard*.py", "continuous*.py", "advanced_monitoring*.py",
        "start_monitoring*.sh", "stop_monitoring*.sh"
    ],
    
    "utilities/reporting": [
        "report*.py", "reporting*.py", "summary*.py", "analysis_reporting*",
        "result*.py", "visualiz*.py", "display*.py", "viewer*.py",
        "output*.py", "detailed-report*.py"
    ],
    
    "tests": [
        "test*.py", "conftest.py", "run_test*.py", "validate*.py",
        "comprehensive_testing*.py", "*_test.py", "check*.py"
    ],
    
    "integration": [
        "integration*.py", "api*.py", "workflow*.py", "interface*.py",
        "gateway*.py", "microservice*.py", "wrapper*.py"
    ],
    
    "fiber_analysis": [
        "fiber*.py", "circular-fiber*.py", "fiber_analysis*",
        "find_fiber*.py", "fiber-optic*.py"
    ],
    
    "zone_analysis": [
        "zone*.py", "zone_analysis*", "region*.py", "area*.py"
    ],
    
    "docs": [
        "*.md", "*.txt", "*.pdf", "requirements*.txt",
        "LICENSE*", "README*", "CHANGELOG*", "*.json", "*.html"
    ],
    
    "experimental": [
        "experimental*.py", "proto*.py", "demo*.py", "example*.py",
        "tutorial*.py", "sample*.py", "poc*.py"
    ],
    
    "workflows": [
        "workflow*.py", "pipeline*.py", "process*.py", "run*.sh", "run*.bat"
    ],
    
    "models": [
        "*.pt", "*.pkl", "*.pth", "*.onnx", "*.pb", "*.h5"
    ],
    
    "web": [
        "*.js", "*.ts", "*.tsx", "*.jsx", "*.html", "*.css"
    ],
    
    "misc": [
        "*.exe", "*.lnk", ".gitignore", ".gitattributes", "hpc-*", "realtime-py-*"
    ]
}

# Additional egg-info directories mapping
egg_info_mappings = {
    "utilities/helpers": ["helper_utilities.egg-info*"],
    "ai_ml": ["artificial_intelligence.egg-info*", "ml_models.egg-info*"],
    "image_processing": ["computer_vision.egg-info*", "preprocessing.egg-info*"],
    "data_processing": ["data_store.egg-info*"],
    "defect_detection": ["defect_detection.egg-info*", "detection_models.egg-info*"],
    "utilities/monitoring": ["monitoring_systems.egg-info*", "real_time_monitoring.egg-info*"],
    "utilities/reporting": ["analysis_reporting.egg-info*", "visualization.egg-info*"],
    "core/config": ["configuration.egg-info*"],
    "integration": ["communication_interfaces.egg-info*", "deployment_configs.egg-info*"],
    "fiber_analysis": ["fiber_analysis.egg-info*"],
    "zone_analysis": ["zone_analysis.egg-info*"],
    "experimental": ["experimental_features.egg-info*", "feature_engineering.egg-info*"]
}

def move_files():
    """Move files to their appropriate directories"""
    moved_count = 0
    
    # First, process regular file mappings
    for target_dir, patterns in file_mappings.items():
        target_path = parent_dir / target_dir
        
        for pattern in patterns:
            # Convert shell-style pattern to regex
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            regex_pattern = f"^{regex_pattern}$"
            
            for file_path in parent_dir.iterdir():
                if file_path.is_file() and re.match(regex_pattern, file_path.name):
                    try:
                        dest_path = target_path / file_path.name
                        print(f"Moving {file_path.name} -> {target_dir}/")
                        shutil.move(str(file_path), str(dest_path))
                        moved_count += 1
                    except Exception as e:
                        print(f"Error moving {file_path.name}: {e}")
    
    # Then process egg-info directories
    for target_dir, patterns in egg_info_mappings.items():
        target_path = parent_dir / target_dir
        
        for pattern in patterns:
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            regex_pattern = f"^{regex_pattern}$"
            
            for dir_path in parent_dir.iterdir():
                if dir_path.is_dir() and re.match(regex_pattern, dir_path.name):
                    try:
                        dest_path = target_path / dir_path.name
                        print(f"Moving directory {dir_path.name} -> {target_dir}/")
                        shutil.move(str(dir_path), str(dest_path))
                        moved_count += 1
                    except Exception as e:
                        print(f"Error moving directory {dir_path.name}: {e}")
    
    print(f"\nTotal items moved: {moved_count}")

if __name__ == "__main__":
    print("Starting file organization...")
    move_files()
    print("File organization complete!")