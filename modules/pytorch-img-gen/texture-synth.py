import torch
import torch.nn.functional as F

ref = torch.load('features.pt')
canvas = torch.randn_like(ref, requires_grad=True)
while 1:
    loss = F.mse_loss(canvas, ref)
    loss.backward()
    with torch.no_grad():
        canvas -= canvas.grad * 0.01
        canvas.grad.zero_()
        print(loss.item())