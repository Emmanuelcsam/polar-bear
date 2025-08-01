#!/usr/bin/env python3

import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class OmniConfig:
    """Configuration for OmniFiberAnalyzer - matches expected structure from app.py"""
    # Path to saved knowledge base file containing reference model data (None means use default)
    knowledge_base_path: Optional[str] = None
    # Minimum pixel area for a region to be considered a defect (filters noise)
    min_defect_size: int = 10
    # Maximum pixel area for a defect (larger areas might be image artifacts)
    max_defect_size: int = 5000
    # Dictionary mapping severity levels to confidence thresholds for classification
    severity_thresholds: Optional[Dict[str, float]] = None
    # Minimum confidence score (0-1) to report a detected anomaly
    confidence_threshold: float = 0.3
    # Multiplier for standard deviation to set anomaly detection threshold
    anomaly_threshold_multiplier: float = 2.5
    # Whether to generate and save visualization images
    enable_visualization: bool = True
    
    def __post_init__(self):
        # Initialize default severity thresholds if none provided
        if self.severity_thresholds is None:
            # Create mapping from severity levels to minimum confidence scores
            self.severity_thresholds = {
                'CRITICAL': 0.9,  # 90%+ confidence = critical defect
                'HIGH': 0.7,      # 70-89% = high severity
                'MEDIUM': 0.5,    # 50-69% = medium severity
                'LOW': 0.3,       # 30-49% = low severity
                'NEGLIGIBLE': 0.1 # 10-29% = negligible
            }


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        # Convert numpy integer types to Python int for JSON compatibility
        if isinstance(obj, np.integer):
            return int(obj)
        # Convert numpy float types to Python float for JSON compatibility
        if isinstance(obj, np.floating):
            return float(obj)
        # Convert numpy arrays to Python lists for JSON compatibility
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Fall back to default JSON encoder for other types
        return super(NumpyEncoder, self).default(obj) 