import torch
import torchvision.transforms as T
from PIL import Image
import glob

for f in glob.glob('refs/*.jpg'):
    img = T.Compose([T.Resize(256), T.ToTensor()])(Image.open(f).convert('RGB'))
    torch.save(img, f.replace('.jpg','.pt'))