import torch
from pathlib import Path


class FiberOpticsConfig:
    """
    Central configuration for the fiber optics neural network.
    "I have a folder with a large database of .pt files of different images used as reference 
    and the files are separated by folder name"
    """
    
    def __init__(self):
        # Paths configuration
        self.REFERENCE_ROOT = Path("C:/Users/Saem1001/Documents/GitHub/polar-bear/reference")
        self.TENSORIZED_DATA_PATH = self.REFERENCE_ROOT / "tensorized-data"
        self.RESULTS_PATH = self.REFERENCE_ROOT / "tensorized-results"
        
        # Region categories mapping from your directory structure
        self.REGION_CATEGORIES = {
            'core': [
                'core-batch-1', 'core-batch-2', 'core-batch-3', 'core-batch-4',
                'core-batch-5', 'core-batch-6', 'core-batch-7', 'core-batch-8',
                'large-core-batch'
            ],
            'cladding': [
                'cladding-batch-1', 'cladding-batch-3', 'cladding-batch-4', 
                'cladding-batch-5', 'cladding-features-batch-1',
                '50-cladding', '91-cladding'
            ],
            'ferrule': [
                'ferrule-batch-1', 'ferrule-batch-2', 'ferrule-batch-3', 
                'ferrule-batch-4'
            ],
            'defects': [
                'dirty-image', '91-scratched', 'scratch-library-bmp'
            ],
            'full_images': [
                '50', '91', 'sma', 'sma-clean'
            ]
        }
        
        # Neural network parameters
        self.INPUT_CHANNELS = 3  # RGB images
        self.HIDDEN_LAYERS = [512, 256, 128, 64]  # Hidden layer dimensions
        self.NUM_REGIONS = 3  # core, cladding, ferrule
        
        # Weight initialization parameters
        # "the weights of the neural network will be dependent on the average intensity gradient"
        self.GRADIENT_WEIGHT_FACTOR = 0.3
        # "another weight will be dependent on the average pixel position"
        self.PIXEL_POSITION_WEIGHT_FACTOR = 0.2
        # "another weight you will comment out completely will be a manual circle alignment"
        # self.CIRCLE_ALIGNMENT_WEIGHT_FACTOR = 0.5  # COMMENTED OUT AS REQUESTED
        
        # Similarity threshold
        # "when classifying and comparing to the reference the program must achieve over .7"
        self.SIMILARITY_THRESHOLD = 0.7
        
        # Training parameters
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 100
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # HPC/Parallel processing
        # "the program will run entirely in hpc meaning that it has to be able to run on gpu and in parallel"
        self.NUM_WORKERS = 4
        self.USE_DISTRIBUTED = torch.cuda.device_count() > 1
        
        # Loss minimization parameters
        # "the program will calculate its losses and try to minimize its losses by small percentile adjustments"
        self.LOSS_ADJUSTMENT_RATE = 0.01  # 1% adjustment
        
        # Anomaly detection parameters
        self.ANOMALY_HEAT_MAP_THRESHOLD = 0.3
        self.STRUCTURAL_SIMILARITY_WEIGHT = 0.5
        self.LOCAL_ANOMALY_WEIGHT = 0.5
        
        # Export configuration
        # "each result matrix will be the smallest files type and size possible"
        self.EXPORT_FORMAT = 'npz'  # Compressed numpy format
        self.EXPORT_COMPRESSION = True
        
        # Data loading parameters
        self.PRELOAD_REFERENCES = True  # Load all reference tensors into memory
        self.CACHE_SIZE = 1000  # Number of images to keep in memory cache
        
    def get_device(self):
        """Get the computation device with proper error handling"""
        if self.DEVICE.type == 'cuda':
            return self.DEVICE
        else:
            print("WARNING: CUDA not available, using CPU. This will be slower.")
            return torch.device('cpu')
    
    def validate_paths(self):
        """Validate that all required paths exist"""
        required_paths = [
            self.REFERENCE_ROOT,
            self.TENSORIZED_DATA_PATH
        ]
        
        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            raise ValueError(f"Missing required paths: {missing_paths}")
        
        return True
    
    def get_equation_parameters(self):
        """
        Get the parameters for the main equation:
        "I=Ax1+Bx2+Cx3... =S(R) where each coefficient of x corresponds to a change in the program"
        """
        return {
            'A': self.GRADIENT_WEIGHT_FACTOR,
            'B': self.PIXEL_POSITION_WEIGHT_FACTOR,
            # 'C': self.CIRCLE_ALIGNMENT_WEIGHT_FACTOR,  # COMMENTED OUT
            'threshold': self.SIMILARITY_THRESHOLD
        }


# Global config instance
_config = None

def get_config():
    """Get or create the global configuration instance"""
    global _config
    if _config is None:
        _config = FiberOpticsConfig()
    return _config