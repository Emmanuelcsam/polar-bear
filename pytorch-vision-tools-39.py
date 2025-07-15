"""Mock torchvision module for testing."""

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
