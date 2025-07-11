import torch
while 1: 
    noise = torch.randn(1, 3, 256, 256)
    torch.save(noise, 'noise.pt')