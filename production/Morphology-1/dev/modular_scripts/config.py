#!/usr/bin/env python3
"""
Configuration module containing all system parameters and thresholds.
This module can be imported by other modules or used standalone.
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class SystemConfig:
    """System configuration for defect detection parameters."""
    
    # Blob detection parameters
    MIN_BLOB_AREA: int = 50
    MAX_BLOB_AREA: int = 5000
    MIN_BLOB_CIRCULARITY: float = 0.3
    
    # Scratch detection parameters
    SCRATCH_KERNEL_SIZE: tuple = (5, 15)
    SCRATCH_BINARY_THRESHOLD: int = 30
    
    # SSIM detection parameters
    SSIM_THRESHOLD: float = 0.95
    
    # General defect parameters
    MIN_DEFECT_SIZE: int = 10
    MAX_DEFECT_SIZE: int = 5000


@dataclass
class OmniConfig:
    """Configuration for comprehensive anomaly analyzer."""
    
    # Path to saved knowledge base file
    knowledge_base_path: Optional[str] = None
    
    # Defect size thresholds
    min_defect_size: int = 10
    max_defect_size: int = 5000
    
    # Severity thresholds mapping
    severity_thresholds: Optional[Dict[str, float]] = None
    
    # Detection thresholds
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    
    # Visualization settings
    enable_visualization: bool = True
    
    def __post_init__(self):
        """Initialize default severity thresholds if none provided."""
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'CRITICAL': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'NEGLIGIBLE': 0.1
            }


def get_default_system_config():
    """Get default system configuration instance."""
    return SystemConfig()


def get_default_omni_config():
    """Get default OmniConfig instance."""
    return OmniConfig()


if __name__ == "__main__":
    # Test configuration when run directly
    print("System Configuration:")
    config = get_default_system_config()
    for field in config.__dataclass_fields__:
        print(f"  {field}: {getattr(config, field)}")
    
    print("\nOmni Configuration:")
    omni = get_default_omni_config()
    for field in omni.__dataclass_fields__:
        value = getattr(omni, field)
        if isinstance(value, dict):
            print(f"  {field}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {field}: {value}")
