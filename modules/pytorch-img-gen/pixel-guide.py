import torch
from PIL import Image
import torchvision.transforms as T

ref = T.ToTensor()(Image.open('ref.jpg').convert('RGB'))
while 1:
    gen = torch.load('generated.pt').squeeze()
    guided = gen * 0.7 + ref * 0.3
    torch.save(guided.unsqueeze(0), 'generated.pt')