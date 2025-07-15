
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import random

from .base_module import BaseModule
from .vision import CAE, UNet34 # Import model architectures from the vision module

# ---------------------------------------------------------------------------- #
#                      Anomaly Model Training Module                           #
# ---------------------------------------------------------------------------- #

class CleanOnlyDataset(Dataset):
    def __init__(self, roots, image_size=512):
        self.files = []
        for r in roots:
            self.files.extend(Path(r).glob('*.png'))
            self.files.extend(Path(r).glob('*.jpg'))
        self.trans = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = cv2.imread(str(self.files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.trans(Image.fromarray(img))

class AnomalyTrainerModule(BaseModule):
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self._register_tunable_parameter("epochs", 50)
        self._register_tunable_parameter("batch_size", 16)
        self._register_tunable_parameter("learning_rate", 1e-3)
        self._register_tunable_parameter("image_size", 512)
        self._register_tunable_parameter("num_workers", 4)
        self._register_tunable_parameter("seed", 42)

    def execute(self, clean_data_dirs: list, output_model_name: str):
        params = self.get_tunable_parameters()
        torch.manual_seed(params['seed'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Starting anomaly model training on device: {device}")

        ds = CleanOnlyDataset(clean_data_dirs, image_size=params['image_size'])
        dl = DataLoader(ds, batch_size=params['batch_size'], shuffle=True,
                        num_workers=params['num_workers'], pin_memory=True)

        model = CAE().to(device)
        opt = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        for epoch in range(params['epochs']):
            model.train()
            run_loss = 0.
            for x in dl:
                x = x.to(device)
                recon = model(x)
                loss = torch.mean((recon - x)**2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                run_loss += loss.item()
            
            avg_loss = run_loss / len(dl)
            self.logger.info(f"Epoch {epoch+1}/{params['epochs']} - MSE Loss: {avg_loss:.5f}")

        output_path = Path(self.config.get("paths.model_weights")) / output_model_name
        output_path.parent.mkdir(exist_ok=True)
        torch.save({'model': model.state_dict()}, output_path)
        self.logger.info(f"Training complete. Model saved to: {output_path}")

# ---------------------------------------------------------------------------- #
#                    Segmentation Model Training Module                        #
# ---------------------------------------------------------------------------- #

class CutPasteDataset(Dataset):
    def __init__(self, img_paths: list, out_size: int = 512):
        self.img_paths = img_paths
        self.size = out_size
        self.aug = T.Compose([T.Resize((self.size, self.size)), T.ToTensor()])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        h, w, _ = rgb.shape
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        box_w, box_h = random.randint(20, 80), random.randint(20, 80)

        defect_patch = cv2.GaussianBlur(rgb[y1:y1+box_h, x1:x1+box_w].copy(), (9, 9), 0)

        x2, y2 = random.randint(0, w - box_w), random.randint(0, h - box_h)
        rgb_aug = rgb.copy()
        rgb_aug[y2:y2+box_h, x2:x2+box_w] = defect_patch

        mask = np.zeros((h, w), np.uint8)
        mask[y2:y2+box_h, x2:x2+box_w] = 1

        x = self.aug(Image.fromarray(rgb_aug))
        mask = self.aug(Image.fromarray(mask*255)).squeeze(0) > 0
        
        # Class 4 for defects
        seg = torch.zeros_like(mask, dtype=torch.long)
        seg[mask] = 4
        return x, seg

class SegmentationTrainerModule(BaseModule):
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self._register_tunable_parameter("epochs", 30)
        self._register_tunable_parameter("batch_size", 8)
        self._register_tunable_parameter("learning_rate", 1e-4)
        self._register_tunable_parameter("image_size", 512)
        self._register_tunable_parameter("num_workers", 4)
        self._register_tunable_parameter("seed", 42)

    def execute(self, clean_data_dir: str, output_model_name: str):
        params = self.get_tunable_parameters()
        random.seed(params['seed']); np.random.seed(params['seed']); torch.manual_seed(params['seed'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Starting segmentation model training on device: {device}")

        img_paths = list(Path(clean_data_dir).glob('*.png')) + list(Path(clean_data_dir).glob('*.jpg'))
        self.logger.info(f"Found {len(img_paths)} clean images for training.")

        dataset = CutPasteDataset(img_paths, out_size=params['image_size'])
        loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True,
                            num_workers=params['num_workers'], pin_memory=True)

        # n_classes = 5 to include the synthetic defect class
        model = UNet34(n_classes=5).to(device)
        opt = optim.Adam(model.parameters(), lr=params['learning_rate'])
        ce = torch.nn.CrossEntropyLoss()

        for epoch in range(params['epochs']):
            model.train()
            run_loss = 0.
            for x, seg in loader:
                x, seg = x.to(device), seg.to(device)
                out = model(x)
                loss = ce(out, seg)
                opt.zero_grad()
                loss.backward()
                opt.step()
                run_loss += loss.item()
            
            avg_loss = run_loss / len(loader)
            self.logger.info(f"Epoch {epoch+1}/{params['epochs']} - CrossEntropy Loss: {avg_loss:.4f}")

        output_path = Path(self.config.get("paths.model_weights")) / output_model_name
        output_path.parent.mkdir(exist_ok=True)
        ckpt = {'model': model.state_dict(), 'img_size': params['image_size']}
        torch.save(ckpt, output_path)
        self.logger.info(f"Training complete. Model saved to: {output_path}")
