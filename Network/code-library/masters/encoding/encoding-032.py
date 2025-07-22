try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    T = None

from pathlib import Path
from PIL import Image
from core.config import DEVICE
from core.datastore import put, get
from core.logger import log

if TORCH_AVAILABLE:
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            f = 32*32
            self.enc = nn.Sequential(nn.Flatten(), nn.Linear(f,128), nn.ReLU(), nn.Linear(128,32))
            self.dec = nn.Sequential(nn.Linear(32,128), nn.ReLU(), nn.Linear(128,f), nn.Sigmoid())
        def forward(self,x): return self.dec(self.enc(x))
    
    model = AE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.MSELoss()
    tfm = T.Compose([T.Grayscale(), T.Resize((32,32)), T.ToTensor()])
else:
    model = None
    opt = None
    loss_fn = None
    tfm = None

def _save():
    if TORCH_AVAILABLE:
        put("ae", model.state_dict())

def _load():
    if TORCH_AVAILABLE:
        sd = get("ae")
        if sd: model.load_state_dict(sd); log("torch","weights loaded")

def train_folder(folder, epochs=1):
    if not TORCH_AVAILABLE:
        log("torch", "PyTorch not available")
        return
    imgs = [tfm(Image.open(p)).to(DEVICE) for p in Path(folder).glob("*")]
    if not imgs: return
    x = torch.stack(imgs)
    for _ in range(epochs):
        opt.zero_grad(); out = model(x); loss = loss_fn(out,x); loss.backward(); opt.step()
    _save(); log("torch","trained loss", loss.item())

def generate(n=1):
    if not TORCH_AVAILABLE:
        log("torch", "PyTorch not available")
        return None
    with torch.no_grad():
        z = torch.randn(n,32, device=DEVICE)
        imgs = model.dec(z).cpu().view(n,1,32,32).mul(255).byte().numpy()
        put("gen", imgs); log("torch","generated", n, "image(s)")
        return imgs

if TORCH_AVAILABLE:
    _load()
    log("torch","module ready on", DEVICE)
else:
    log("torch","module loaded (PyTorch not available)")