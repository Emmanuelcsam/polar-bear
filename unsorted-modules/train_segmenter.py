import argparse, random, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torch.optim import Adam
import torchvision.transforms as T
from PIL import Image

# --- simple seed for reproducibility
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class CutPasteDataset(Dataset):
    def __init__(self, img_paths: list[Path], out_size: int = 512):
        self.img_paths = img_paths
        self.size = out_size
        self.aug = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor()
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # --------------- synthetic defect -----------------
        h, w, _ = rgb.shape
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        box_w = random.randint(20, 80)
        box_h = random.randint(20, 80)

        defect_patch = rgb[y1:y1+box_h, x1:x1+box_w].copy()
        defect_patch = cv2.GaussianBlur(defect_patch, (9, 9), 0)

        # where to paste
        x2 = random.randint(0, w - box_w)
        y2 = random.randint(0, h - box_h)
        rgb_aug = rgb.copy()
        rgb_aug[y2:y2+box_h, x2:x2+box_w] = defect_patch

        # binary mask
        mask = np.zeros((h, w), np.uint8)
        mask[y2:y2+box_h, x2:x2+box_w] = 1

        # torchvision transforms
        x = self.aug(Image.fromarray(rgb_aug))
        mask = self.aug(Image.fromarray(mask*255)).squeeze(0) > 0

        # channel ID for segmentation: background=0, defect=4
        seg = torch.zeros_like(mask, dtype=torch.long)
        seg[mask] = 4
        return x, seg

def main(args):
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Import the model here to avoid circular imports
    from ai_segmenter_pytorch import UNet34
    
    # collect all clean images (user must specify a folder with "known‑good")
    img_paths = list(Path(args.clean_dir).glob('*.png')) + \
                list(Path(args.clean_dir).glob('*.jpg'))
    print(f"✓ Found {len(img_paths)} defect‑free training images")

    # dataset = synthetic cut‑paste (supervised for defect head)
    dataset = CutPasteDataset(img_paths, out_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True,
                        num_workers=4, pin_memory=True)

    model = UNet34(n_classes=5).to(device)  # include defect class 4
    opt = Adam(model.parameters(), 1e-4)
    ce = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train(); running = 0.
        for x, seg in loader:
            x, seg = x.to(device), seg.to(device)
            out = model(x)
            loss = ce(out, seg)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"epoch {epoch+1:02d}/{args.epochs}  CE={running/len(loader):.4f}")

    # save
    ckpt = {'model': model.state_dict(), 'img_size': args.img_size}
    torch.save(ckpt, args.out)
    print(f"✓ saved weights => {args.out}")

if __name__ == '__main__':
    import cv2
    p = argparse.ArgumentParser()
    p.add_argument('--clean_dir', required=True)
    p.add_argument('--img_size', default=512, type=int)
    p.add_argument('--bs', default=8, type=int)
    p.add_argument('--epochs', default=30, type=int)
    p.add_argument('--out', default='segmenter_best.pth')
    main(p.parse_args())