import torch
import torch.nn as nn

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,64,3,1,1), nn.ReLU(), nn.Conv2d(64,64,3,1,1), nn.ReLU(), nn.Conv2d(64,3,3,1,1), nn.Tanh())
    def forward(self, x): return self.net(x)

gen = G()
noise = torch.load('noise.pt')
img = gen(noise)
torch.save(img, 'generated.pt')