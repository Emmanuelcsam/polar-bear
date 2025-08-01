# src/dataset_builder.py
import pandas as pd
from pathlib import Path
import shutil

def build_datasets(clustered_df, config):
    """
    Builds and saves the final datasets.
    
    Args:
        clustered_df: A pandas DataFrame of clustered defect features.
        config: The configuration dictionary.
    """
    if clustered_df.empty:
        print("No data to build datasets.")
        return

    output_folder = Path(config['output_folder'])
    
    # Image-level dataset
    image_level_df = clustered_df.groupby('image_name').agg(
        num_defects=('defect_id', 'count'),
        num_scratches=('cluster', lambda x: (x == 0).sum()), # Assuming cluster 0 is scratches
        num_pits=('cluster', lambda x: (x == 1).sum()),      # Assuming cluster 1 is pits
        # Add more aggregations as needed
    ).reset_index()
    image_level_df.to_csv(output_folder / "image_level_dataset.csv", index=False)
    
    # Region-level dataset (zone-level)
    region_level_df = clustered_df.groupby(['image_name', 'zone']).agg(
        num_defects=('defect_id', 'count'),
        total_area_px=('area_px', 'sum'),
    ).reset_index()
    region_level_df.to_csv(output_folder / "region_level_dataset.csv", index=False)
    
    # Defect library
    defect_library_folder = Path(config['defect_library_folder'])
    defect_library_folder.mkdir(exist_ok=True)
    
    # This is a simplified version. A full implementation would crop the defect patches.
    clustered_df.to_csv(defect_library_folder / "defect_library_index.csv", index=False)

    print("Datasets built successfully.")