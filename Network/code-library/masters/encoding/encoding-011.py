"""Mock peft module for testing."""

class LoraConfig:
    def __init__(self, *args, **kwargs):
        pass

def get_peft_model(model, config):
    return model
