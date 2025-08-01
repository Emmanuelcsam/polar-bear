# src/clustering.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_defects(features_df, config):
    """
    Clusters the detected defects based on their features.
    
    Args:
        features_df: A pandas DataFrame of defect features.
        config: The configuration dictionary.
        
    Returns:
        The input DataFrame with an added 'cluster' column.
    """
    if features_df.empty:
        return features_df
        
    # Select features for clustering
    # This is a basic selection, can be improved based on feature importance analysis
    features_for_clustering = [
        'area_px', 'aspect_ratio', 'solidity', 'mean_intensity',
        'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'
    ]
    
    # Check if all required columns are present
    missing_cols = [col for col in features_for_clustering if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for clustering: {missing_cols}")

    # Handle missing values
    features_df[features_for_clustering] = features_df[features_for_clustering].fillna(0)

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df[features_for_clustering])
    
    # Perform K-Means clustering
    num_clusters = config.get('num_clusters', 4)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    features_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    return features_df