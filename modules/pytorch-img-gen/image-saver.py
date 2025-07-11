import torch
import torchvision.utils as vutils
while 1:
    img = torch.load('generated.pt')
    vutils.save_image((img+1)/2, 'output.png')