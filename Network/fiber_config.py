#!/usr/bin/env python3
"""
Configuration module for Fiber Optics Neural Network
NO ARGPARSE OR FLAGS - All configuration is done through this module
"the program will log all of its processes in a log file as soon as they happen"
"""

import torch
from pathlib import Path
from datetime import datetime
import json
import os

class FiberOpticsConfig:
    """Central configuration for the entire fiber optics analysis system"""
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing FiberOpticsConfig")
        
        # Project paths
        self.PROJECT_ROOT = Path(__file__).parent
        self.DATA_PATH = Path("C:/Users/Saem1001/Documents/GitHub/polar-bear/reference")
        self.TENSORIZED_DATA_PATH = self.DATA_PATH / "tensorized-data"
        self.REFERENCE_PATH = self.DATA_PATH / "reference-images"
        self.RESULTS_PATH = self.PROJECT_ROOT / "results"
        self.CHECKPOINTS_PATH = self.PROJECT_ROOT / "checkpoints"
        self.LOGS_PATH = self.PROJECT_ROOT / "logs"
        
        # Create directories if they don't exist
        for path in [self.RESULTS_PATH, self.CHECKPOINTS_PATH, self.LOGS_PATH]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Device configuration - "the program will run entirely in hpc meaning that it has to be able to run on gpu"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_WORKERS = 4 if torch.cuda.is_available() else 0
        self.PIN_MEMORY = torch.cuda.is_available()
        
        # Neural Network Parameters
        # "I=Ax1+Bx2+Cx3... =S(R) where each coefficient of x corresponds to a change in the program"
        self.EQUATION_COEFFICIENTS = {
            'A': 1.0,  # Gradient intensity weight
            'B': 1.0,  # Pixel position weight  
            'C': 0.0,  # Manual circle alignment (commented out as requested)
            'D': 1.0,  # Feature correlation weight
            'E': 1.0,  # Anomaly detection weight
        }
        
        # "the weights of the neural network will be dependent on the average intensity gradient"
        self.GRADIENT_WEIGHT_FACTOR = 1.0
        # "another weight will be dependent on the average pixel position"
        self.POSITION_WEIGHT_FACTOR = 1.0
        
        # Region categories - "core, cladding and ferrule"
        self.REGION_CATEGORIES = {
            'core': ['core-batch-1', 'core-batch-2', 'core-batch-3', 'core-batch-4',
                    'core-batch-5', 'core-batch-6', 'core-batch-7', 'core-batch-8'],
            'cladding': ['cladding-batch-1', 'cladding-batch-3', 'cladding-batch-4',
                        'cladding-batch-5', 'cladding-features-batch-1', '50-cladding', '91-cladding'],
            'ferrule': ['ferrule-batch-1', 'ferrule-batch-2', 'ferrule-batch-3', 'ferrule-batch-4'],
            'defects': ['dirty-image', 'scratch-library-bmp', '91-scratched']
        }
        
        # Model parameters
        self.INPUT_SIZE = (256, 256)  # Standard input size
        self.NUM_REGIONS = 3  # core, cladding, ferrule
        self.BATCH_SIZE = 32 if torch.cuda.is_available() else 8
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 100
        
        # "the program must achieve over .7"
        self.SIMILARITY_THRESHOLD = 0.7
        self.ANOMALY_THRESHOLD = 0.3
        
        # Multi-scale parameters - "Features from different scales are correlated"
        self.SCALES = [1.0, 0.75, 0.5, 0.25]
        self.FEATURE_CHANNELS = [64, 128, 256, 512]
        
        # Logging configuration
        self.LOG_LEVEL = "DEBUG"
        self.LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        self.LOG_FILE = self.LOGS_PATH / f"fiber_optics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Load any existing configuration
        self.config_file = self.PROJECT_ROOT / "fiber_optics_config.json"
        self.load_config()
        
        print(f"[{datetime.now()}] FiberOpticsConfig initialized successfully")
        print(f"[{datetime.now()}] Using device: {self.DEVICE}")
        print(f"[{datetime.now()}] Tensorized data path: {self.TENSORIZED_DATA_PATH}")
    
    def save_config(self):
        """Save current configuration to JSON file"""
        config_dict = {
            'equation_coefficients': self.EQUATION_COEFFICIENTS,
            'gradient_weight_factor': self.GRADIENT_WEIGHT_FACTOR,
            'position_weight_factor': self.POSITION_WEIGHT_FACTOR,
            'similarity_threshold': self.SIMILARITY_THRESHOLD,
            'anomaly_threshold': self.ANOMALY_THRESHOLD,
            'batch_size': self.BATCH_SIZE,
            'learning_rate': self.LEARNING_RATE,
            'num_epochs': self.NUM_EPOCHS
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"[{datetime.now()}] Configuration saved to {self.config_file}")
    
    def load_config(self):
        """Load configuration from JSON file if exists"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Update configuration
            self.EQUATION_COEFFICIENTS.update(config_dict.get('equation_coefficients', {}))
            self.GRADIENT_WEIGHT_FACTOR = config_dict.get('gradient_weight_factor', 1.0)
            self.POSITION_WEIGHT_FACTOR = config_dict.get('position_weight_factor', 1.0)
            self.SIMILARITY_THRESHOLD = config_dict.get('similarity_threshold', 0.7)
            self.ANOMALY_THRESHOLD = config_dict.get('anomaly_threshold', 0.3)
            
            print(f"[{datetime.now()}] Configuration loaded from {self.config_file}")
    
    def update_equation_coefficient(self, coefficient: str, value: float):
        """Update equation coefficient - 'allow me to see and tweak the parameters'"""
        if coefficient in self.EQUATION_COEFFICIENTS:
            old_value = self.EQUATION_COEFFICIENTS[coefficient]
            self.EQUATION_COEFFICIENTS[coefficient] = value
            print(f"[{datetime.now()}] Updated coefficient {coefficient}: {old_value} -> {value}")
            self.save_config()
        else:
            print(f"[{datetime.now()}] Warning: Unknown coefficient {coefficient}")
    
    def get_device(self):
        """Get the computation device"""
        return self.DEVICE
    
    def get_region_folders(self, region: str):
        """Get folder names for a specific region"""
        return self.REGION_CATEGORIES.get(region, [])

# Global configuration instance
_config_instance = None

def get_config():
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = FiberOpticsConfig()
    return _config_instance

# Test the configuration
if __name__ == "__main__":
    config = get_config()
    print(f"[{datetime.now()}] Configuration test completed")
    print(f"[{datetime.now()}] Next script: logger.py")
