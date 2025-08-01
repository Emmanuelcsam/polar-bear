# train.py
# Fiber Optic End-Face Multi-Stage CNN Training Script
# William & Mary HPC Bora Ready – Torch DDP, Lightning-fast DataLoader, and Statistical Priors

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset import make_dataloader
from model import EndfaceNet, CompositeLoss
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train Endface Defect CNN")
    parser.add_argument('--config', required=True, type=str, help='Path to config (YAML)')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    return parser.parse_args()

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # --- DDP initialization (Torchrun/SLURM ready) ---
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(args.local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1

    # --- Directory setup ---
    os.makedirs(cfg['ckpt_dir'], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}") if rank == 0 else None

    # --- DataLoader ---
    dl_train = make_dataloader(
        root=cfg['data_root'], batch_size=cfg['bs'],
        train=True, num_workers=cfg['w'], lmdb_dir=cfg.get('lmdb'))
    steps_per_epoch = len(dl_train)

    # --- Reference statistics/prior tensors ---
    ref_stats = torch.load(cfg['ref_stats'], map_location="cpu") if 'ref_stats' in cfg else None

    # --- Model and loss ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndfaceNet(num_classes=cfg['num_classes']).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank]) if world_size > 1 else model

    loss_fn = CompositeLoss(prior_stats=ref_stats, class_weights=None)  # Optionally, add class_weight from your stats

    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg.get('min_lr', 1e-5))

    start_epoch = 0
    ckpt_path = os.path.join(cfg['ckpt_dir'], "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        cp = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler'])
        start_epoch = cp.get('epoch', 0) + 1
        print(f"[INFO] Resuming from checkpoint at epoch {start_epoch}")

    # --- Training Loop ---
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        running_loss = 0.0
        if isinstance(dl_train.sampler, DistributedSampler):
            dl_train.sampler.set_epoch(epoch)
        for i, (imgs, _) in enumerate(dl_train):
            imgs = imgs.to(device)
            # For segmentation/classification, masks and labels must be loaded from data
            # Here, we mock placeholders:
            bs = imgs.size(0)
            target_masks = torch.zeros(bs, 3, imgs.shape[2], imgs.shape[3], device=imgs.device)  # Replace by real labels
            target_labels = torch.zeros(bs, cfg['num_classes'], device=imgs.device)  # Replace by real labels
            stat_feats = torch.zeros(bs, 88, device=imgs.device)  # Real features if available

            pred_masks, pred_logits, stat_out = model(imgs)
            loss = loss_fn(pred_masks, target_masks, pred_logits, target_labels, stat_out, ref_stats)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if writer is not None and (i % 20 == 0 or i == steps_per_epoch - 1):
                writer.add_scalar("Loss/train_step", loss.item(), epoch * steps_per_epoch + i)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch * steps_per_epoch + i)

        scheduler.step()
        avg_loss = running_loss / steps_per_epoch
        if writer is not None:
            writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

        # --- Save distributed checkpoints ---
        if (rank == 0) and ((epoch + 1) % 5 == 0 or (epoch + 1) == cfg['epochs']):
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(cfg['ckpt_dir'], f"epoch_{epoch:02d}.pt"))
            torch.save(state, ckpt_path)

        print(f"Epoch {epoch+1}/{cfg['epochs']}: Train loss: {avg_loss:.4f}")

    if writer is not None: writer.close()
    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main() 