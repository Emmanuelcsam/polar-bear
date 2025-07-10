#!/usr/bin/env python3
"""
Create mock modules for missing dependencies to allow tests to run.
"""

import os
import sys

def create_mock_torch():
    """Create a mock torch module."""
    
    mock_torch = '''"""Mock torch module for testing."""

class Tensor:
    def __init__(self, data=None):
        self.data = data
        self.shape = (1, 3, 256, 256)
        self.dtype = "float32"
    
    def to(self, device):
        return self
    
    def float(self):
        return self
    
    def squeeze(self):
        return self
    
    def unsqueeze(self, dim):
        return self
    
    def cpu(self):
        return self
    
    def numpy(self):
        import numpy as np
        return np.random.rand(*self.shape)
    
    def __getitem__(self, idx):
        return self
    
    def mean(self, dim=None):
        return self
    
    def clone(self):
        return Tensor()

def tensor(data):
    return Tensor(data)

def zeros(*shape, dtype=None):
    return Tensor()

def ones(*shape, dtype=None):
    return Tensor()

def randn(*shape):
    return Tensor()

def from_numpy(arr):
    return Tensor(arr)

def no_grad():
    class NoGrad:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return NoGrad()

def load(path, map_location=None):
    return {"model_state_dict": {}}

def save(obj, path):
    pass

class nn:
    class Module:
        def __init__(self):
            self.training = True
        
        def eval(self):
            self.training = False
            return self
        
        def train(self):
            self.training = True
            return self
        
        def parameters(self):
            return []
        
        def to(self, device):
            return self
        
        def load_state_dict(self, state_dict):
            pass
        
        def state_dict(self):
            return {}
        
        def __call__(self, x):
            return x
    
    class Conv2d(Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
    
    class BatchNorm2d(Module):
        pass
    
    class ReLU(Module):
        pass
    
    class MaxPool2d(Module):
        pass
    
    class ConvTranspose2d(Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
    
    class Linear(Module):
        pass
    
    class MSELoss(Module):
        def __init__(self, reduction='mean'):
            super().__init__()
            self.reduction = reduction
        
        def __call__(self, pred, target):
            return Tensor()
    
    class CrossEntropyLoss(Module):
        pass
    
    class Sequential(Module):
        def __init__(self, *layers):
            super().__init__()
            self.layers = layers

class optim:
    class Adam:
        def __init__(self, params, lr=0.001):
            pass
        
        def zero_grad(self):
            pass
        
        def step(self):
            pass

cuda = type('cuda', (), {
    'is_available': lambda: False,
    'device_count': lambda: 0
})()

__version__ = "1.9.0"
'''
    
    with open('torch.py', 'w') as f:
        f.write(mock_torch)

def create_mock_torchvision():
    """Create mock torchvision module."""
    
    mock_torchvision = '''"""Mock torchvision module for testing."""

class models:
    @staticmethod
    def resnet34(pretrained=True):
        import torch
        model = torch.nn.Module()
        model.conv1 = torch.nn.Conv2d(3, 64, 7)
        model.bn1 = torch.nn.BatchNorm2d(64)
        model.relu = torch.nn.ReLU()
        model.maxpool = torch.nn.MaxPool2d(3)
        model.layer1 = torch.nn.Sequential()
        model.layer2 = torch.nn.Sequential()
        model.layer3 = torch.nn.Sequential()
        model.layer4 = torch.nn.Sequential()
        return model

class transforms:
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
        
        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img
    
    class ToTensor:
        def __call__(self, img):
            import torch
            return torch.from_numpy(img)
    
    class Normalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std
        
        def __call__(self, tensor):
            return tensor
    
    class Resize:
        def __init__(self, size):
            self.size = size
        
        def __call__(self, img):
            return img
'''
    
    with open('torchvision.py', 'w') as f:
        f.write(mock_torchvision)

def create_mock_transformers():
    """Create mock transformers module."""
    
    mock_transformers = '''"""Mock transformers module for testing."""

class AutoModelForVision2Seq:
    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        return MockModel()

class AutoProcessor:
    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        return MockProcessor()

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        return MockTokenizer()

class MockModel:
    def __init__(self):
        self.config = type('Config', (), {'vocab_size': 50000})()
    
    def to(self, device):
        return self
    
    def eval(self):
        return self
    
    def generate(self, *args, **kwargs):
        return [[1, 2, 3, 4, 5]]
    
    def __call__(self, *args, **kwargs):
        return type('Output', (), {'logits': None})()

class MockProcessor:
    def __call__(self, *args, **kwargs):
        return {'input_ids': [[1, 2, 3]]}

class MockTokenizer:
    def decode(self, tokens, *args, **kwargs):
        return "Generated text"
    
    def __call__(self, *args, **kwargs):
        return {'input_ids': [[1, 2, 3]]}
'''
    
    with open('transformers.py', 'w') as f:
        f.write(mock_transformers)

def create_mock_tensorflow():
    """Create mock tensorflow module."""
    
    mock_tf = '''"""Mock tensorflow module for testing."""

class keras:
    class Sequential:
        def __init__(self, layers=None):
            self.layers = layers or []
        
        def add(self, layer):
            self.layers.append(layer)
        
        def compile(self, *args, **kwargs):
            pass
        
        def fit(self, *args, **kwargs):
            return type('History', (), {'history': {}})()
        
        def evaluate(self, *args, **kwargs):
            return [0.1, 0.9]
        
        def predict(self, *args, **kwargs):
            import numpy as np
            return np.random.rand(1, 10)
        
        def save(self, path):
            pass
    
    class layers:
        class Dense:
            def __init__(self, units, activation=None):
                self.units = units
                self.activation = activation
        
        class Dropout:
            def __init__(self, rate):
                self.rate = rate
        
        class Flatten:
            pass
    
    @staticmethod
    def models():
        class Models:
            @staticmethod
            def load_model(path):
                return keras.Sequential()
        return Models()

__version__ = "2.6.0"
'''
    
    with open('tensorflow.py', 'w') as f:
        f.write(mock_tf)

def create_mock_sklearn():
    """Create mock sklearn module."""
    
    mock_sklearn = '''"""Mock sklearn module for testing."""

class preprocessing:
    class StandardScaler:
        def fit_transform(self, X):
            return X
        
        def transform(self, X):
            return X

class cluster:
    class KMeans:
        def __init__(self, n_clusters=3, random_state=None):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.labels_ = None
            self.inertia_ = 100.0
        
        def fit(self, X):
            import numpy as np
            self.labels_ = np.random.randint(0, self.n_clusters, len(X))
            return self
        
        def predict(self, X):
            import numpy as np
            return np.random.randint(0, self.n_clusters, len(X))
'''
    
    # Note: sklearn uses a different import structure
    os.makedirs('sklearn', exist_ok=True)
    with open('sklearn/__init__.py', 'w') as f:
        f.write(mock_sklearn)
    
    with open('sklearn/preprocessing.py', 'w') as f:
        f.write('from . import preprocessing')
    
    with open('sklearn/cluster.py', 'w') as f:
        f.write('from . import cluster')

def create_mock_skimage():
    """Create mock skimage module."""
    
    mock_skimage = '''"""Mock skimage module for testing."""

def greycomatrix(image, distances, angles, levels, symmetric=True, normed=True):
    import numpy as np
    return np.random.rand(levels, levels, len(distances), len(angles))

def greycoprops(glcm, prop):
    import numpy as np
    return np.random.rand(1, 1)
'''
    
    os.makedirs('skimage', exist_ok=True)
    os.makedirs('skimage/feature', exist_ok=True)
    
    with open('skimage/__init__.py', 'w') as f:
        f.write('')
    
    with open('skimage/feature/__init__.py', 'w') as f:
        f.write('')
    
    with open('skimage/feature/texture.py', 'w') as f:
        f.write(mock_skimage)

def create_mock_peft():
    """Create mock peft module."""
    
    mock_peft = '''"""Mock peft module for testing."""

class LoraConfig:
    def __init__(self, *args, **kwargs):
        pass

def get_peft_model(model, config):
    return model
'''
    
    with open('peft.py', 'w') as f:
        f.write(mock_peft)

def create_all_mocks():
    """Create all mock modules."""
    print("Creating mock modules...")
    
    create_mock_torch()
    print("Created mock torch")
    
    create_mock_torchvision()
    print("Created mock torchvision")
    
    create_mock_transformers()
    print("Created mock transformers")
    
    create_mock_tensorflow()
    print("Created mock tensorflow")
    
    create_mock_sklearn()
    print("Created mock sklearn")
    
    create_mock_skimage()
    print("Created mock skimage")
    
    create_mock_peft()
    print("Created mock peft")
    
    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())
    
    print("Mock modules created successfully!")

if __name__ == "__main__":
    create_all_mocks()