import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as T

vgg = models.vgg16(pretrained=True).features[:16].eval()
transform = T.Compose([T.Resize(256), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
img = transform(Image.open('ref.jpg').convert('RGB')).unsqueeze(0)
with torch.no_grad(): features = vgg(img)
torch.save(features, 'features.pt')