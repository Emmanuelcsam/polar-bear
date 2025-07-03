# src/main.py
import argparse
import logging
from pathlib import Path
import yaml
import cv2
import numpy as np
import pandas as pd

from .zone_segmentation import segment_zones
from .defect_detection import detect_defects
from .feature_extraction import extract_features
from .clustering import cluster_defects
from .dataset_builder import build_datasets
from .utils import list_images, load_image

def setup_logging(log_level="INFO"):
    """Sets up basic logging."""
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Fiber Optic Defect Detection Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file.")
    parser.add_argument("--image_folder", type=str,
                        help="Path to the folder containing images to process.")
    args = parser.parse_args()

    setup_logging()
    config_path = Path('fiber_defect_inspection') / args.config
    config = load_config(config_path)

    if args.image_folder:
        config['input_folder'] = args.image_folder

    logging.info("Starting the defect detection pipeline.")
    
    input_path = Path('fiber_defect_inspection') / config['input_folder']
    output_path = Path('fiber_defect_inspection') / config['output_folder']
    output_path.mkdir(exist_ok=True)
    
    image_paths = list_images(input_path)
    all_features = []
    
    for image_path in image_paths:
        logging.info(f"Processing image: {image_path.name}")
        image = load_image(image_path)
        
        if image is None:
            logging.warning(f"Could not load image {image_path.name}, skipping.")
            continue
            
        segmentation_results = segment_zones(image, config)
        
        defect_mask, defect_regions = detect_defects(image, segmentation_results.get("masks"), config)
        
        features = extract_features(image, defect_regions, segmentation_results.get("masks"), segmentation_results.get("metrics"))
        
        for f in features:
            f['image_name'] = image_path.name
        all_features.extend(features)
        
        # Visualization and other steps...
        # (Code from previous steps remains here)

    # Cluster the defects and build datasets
    if all_features:
        features_df = pd.DataFrame(all_features)
        try:
            clustered_df = cluster_defects(features_df, config)
            build_datasets(clustered_df, config)
            
            features_output_path = output_path / "defect_features_clustered.csv"
            clustered_df.to_csv(features_output_path, index=False)
            logging.info(f"Saved clustered defect features to {features_output_path}")

        except ValueError as e:
            logging.error(f"Clustering or dataset building failed: {e}")
            features_output_path = output_path / "defect_features_unclustered.csv"
            features_df.to_csv(features_output_path, index=False)
            logging.info(f"Saved unclustered defect features to {features_output_path}")

    logging.info("Defect detection pipeline finished.")

if __name__ == "__main__":
    main()