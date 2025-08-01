"""Mock torch module for testing."""

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

def randint(low, high, size):
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

class _nn_module:
    pass

nn = _nn_module()

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


# Set nn module attributes
nn.Module = Module
nn.Conv2d = Conv2d
nn.BatchNorm2d = BatchNorm2d  
nn.ReLU = ReLU
nn.MaxPool2d = MaxPool2d
nn.ConvTranspose2d = ConvTranspose2d
nn.Linear = Linear
nn.MSELoss = MSELoss
nn.CrossEntropyLoss = CrossEntropyLoss
nn.Sequential = Sequential

__version__ = "1.9.0"
