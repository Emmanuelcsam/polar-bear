import torch
import torch.optim as optim
import torchvision.models as models

vgg = models.vgg16(pretrained=True).features[:16].eval()
target = torch.load('features.pt')
img = torch.randn(1,3,256,256, requires_grad=True)
opt = optim.Adam([img], lr=0.01)
while 1:
    opt.zero_grad()
    loss = ((vgg(img) - target)**2).mean()
    loss.backward()
    opt.step()
    print(loss.item())