"""Mock tensorflow module for testing."""

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
