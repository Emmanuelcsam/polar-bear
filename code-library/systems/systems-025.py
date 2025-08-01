"""Mock transformers module for testing."""

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
