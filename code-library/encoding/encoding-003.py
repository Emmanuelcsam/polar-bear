import argparse, random, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import cv2

from anomaly_detector_pytorch import CAE

class CleanOnlyDataset(Dataset):
    def __init__(self, roots):
        self.files = []
        for r in roots:
            self.files += list(Path(r).glob('*.png'))
            self.files += list(Path(r).glob('*.jpg'))
        self.trans = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
        ])

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = cv2.imread(str(self.files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.trans(Image.fromarray(img))

def main(args):
    torch.manual_seed(0); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = CleanOnlyDataset(args.clean_dirs)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    num_workers=4, pin_memory=True)

    model = CAE().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(args.epochs):
        model.train(); run = 0.
        for x in dl:
            x = x.to(device)
            recon = model(x)
            loss = torch.mean((recon - x)**2)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()
        print(f"epoch {epoch+1}/{args.epochs}  MSE={run/len(dl):.5f}")

    torch.save({'model': model.state_dict()}, args.out)
    print("✓ autoencoder weights saved:", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--clean_dirs', nargs='+', required=True)
    p.add_argument('--bs', default=16, type=int)
    p.add_argument('--epochs', default=50, type=int)
    p.add_argument('--out', default='cae_last.pth')
    main(p.parse_args())