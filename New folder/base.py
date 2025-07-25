# main.py
# The single, unified entry point for training, evaluation, and analysis.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.utils.prune as prune
from torch.cuda.amp import GradScaler, autocast
from box import Box
import yaml
import os
from pathlib import Path
import logging

from model import FiberOpticNet
from dataset import FiberOpticsDataset

# --- Focal Loss Implementation ---
# As recommended by your statistical analysis for handling class imbalance.
class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# --- Configuration and Setup ---

def setup_logging():
    """Sets up a simple and clean logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file into a Box object for dot notation access."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return Box(config_data)

def setup_distributed():
    """Initializes the distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False

# --- Core Logic ---

def main():
    """Main function to run the fiber optics analysis system."""
    log = setup_logging()
    config = load_config()
    
    rank, world_size, local_rank, is_distributed = setup_distributed()
    
    log.info("--- Fiber Optics Analysis System ---")
    log.info(f"Mode: {'Distributed' if is_distributed else 'Single GPU'}")
    
    # --- Data Loading ---
    log.info("Setting up datasets and dataloaders...")
    train_dataset = FiberOpticsDataset(config=config, mode='train')
    val_dataset = FiberOpticsDataset(config=config, mode='val')
    
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size // world_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size // world_size,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # --- Model, Optimizer, and Loss ---
    log.info("Initializing model, optimizer, and loss function...")
    model = FiberOpticNet(config.model).cuda(local_rank)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        
    optimizer = optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    scaler = GradScaler(enabled=config.training.use_amp)

    # Loss Functions
    if config.loss.type == 'focal':
        classification_loss_fn = FocalLoss(alpha=config.loss.focal_alpha, gamma=config.loss.focal_gamma)
        log.info("Using Focal Loss for classification.")
    else:
        classification_loss_fn = nn.CrossEntropyLoss()
        log.info("Using Cross-Entropy Loss for classification.")
        
    anomaly_loss_fn = nn.BCEWithLogitsLoss()
    similarity_loss_fn = nn.CosineEmbeddingLoss()

    # --- Training Loop ---
    log.info("Starting training...")
    for epoch in range(config.training.num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            images = batch['image'].cuda(local_rank, non_blocking=True)
            ref_images = batch['reference'].cuda(local_rank, non_blocking=True)
            labels = batch['label'].cuda(local_rank, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=config.training.use_amp):
                output = model(images, ref_images, equation_coeffs=config.equation.coefficients)
                
                loss_cls = classification_loss_fn(output['region_logits'], labels)
                anomaly_target = (labels == config.data.class_map.defects).float().view(-1, 1, 1, 1)
                anomaly_target = anomaly_target.expand_as(output['anomaly_map'])
                loss_anomaly = anomaly_loss_fn(output['anomaly_map'], anomaly_target)
                target_sim = torch.ones(images.size(0)).cuda(local_rank)
                loss_sim = similarity_loss_fn(output['embedding'], output['ref_embedding'], target_sim)
                
                loss = (config.loss.weights.classification * loss_cls +
                        config.loss.weights.anomaly * loss_anomaly +
                        config.loss.weights.similarity * loss_sim)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if rank == 0 and i % config.training.log_interval == 0:
                log.info(f"Epoch {epoch+1}/{config.training.num_epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            log.info(f"--- Epoch {epoch+1} Summary ---")
            log.info(f"Average Training Loss: {avg_loss:.4f}")
            evaluate(model, val_loader, config, local_rank, log)
            
            checkpoint_dir = Path(config.system.checkpoints_path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.module.state_dict(), checkpoint_dir / f"epoch_{epoch+1}.pth")

    log.info("Training complete.")
    
    # --- Optional Post-Training Steps ---
    if rank == 0:
        if config.optimization.get('prune_after_training', False):
            log.info("Pruning model...")
            prune_model(model.module, config.optimization.pruning_ratio)
            torch.save(model.module.state_dict(), checkpoint_dir / "pruned_model.pth")

        if config.optimization.get('distill_after_training', False):
            log.info("Starting knowledge distillation...")
            distill_model(model.module, train_loader, val_loader, config, local_rank, log)


def evaluate(model, val_loader, config, local_rank, log):
    """Evaluation function."""
    model.eval()
    total_similarity = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].cuda(local_rank, non_blocking=True)
            labels = batch['label'].cuda(local_rank, non_blocking=True)
            
            with autocast(enabled=config.training.use_amp):
                output = model(images, ref_image=None, equation_coeffs=config.equation.coefficients)
            
            total_similarity += output['final_similarity_score'].sum().item()
            _, predicted = torch.max(output['region_logits'], 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_similarity = total_similarity / total_samples
    accuracy = correct_predictions / total_samples
    
    log.info(f"Validation Avg. Similarity: {avg_similarity:.4f} (Target: >{config.similarity.threshold})")
    log.info(f"Validation Accuracy: {accuracy:.2%}")

# --- Real-time Optimization Functions ---

def prune_model(model, ratio=0.3):
    """Prunes the model globally."""
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=ratio,
    )
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

def distill_model(teacher_model, train_loader, val_loader, config, local_rank, log):
    """Trains a smaller student model using knowledge distillation."""
    log.info("Creating student model for distillation...")
    student_model = FiberOpticNet(config.student_model).cuda(local_rank)
    
    optimizer = optim.AdamW(student_model.parameters(), lr=config.optimizer.learning_rate)
    distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')
    classification_loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=config.training.use_amp)
    
    teacher_model.eval()
    
    for epoch in range(config.optimization.distillation_epochs):
        student_model.train()
        for batch in train_loader:
            images = batch['image'].cuda(local_rank, non_blocking=True)
            labels = batch['label'].cuda(local_rank, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=config.training.use_amp):
                with torch.no_grad():
                    teacher_output = teacher_model(images)
                
                student_output = student_model(images)
                
                soft_targets = F.log_softmax(teacher_output['region_logits'] / config.optimization.temperature, dim=1)
                soft_predictions = F.log_softmax(student_output['region_logits'] / config.optimization.temperature, dim=1)
                loss_distill = distillation_loss_fn(soft_predictions, soft_targets) * (config.optimization.temperature ** 2)
                loss_student = classification_loss_fn(student_output['region_logits'], labels)
                loss = config.optimization.alpha * loss_distill + (1 - config.optimization.alpha) * loss_student
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        log.info(f"Distillation Epoch {epoch+1}: Loss = {loss.item():.4f}")
        # Simple eval for student
        student_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].cuda(local_rank)
                labels = batch['label'].cuda(local_rank)
                with autocast(enabled=config.training.use_amp):
                    outputs = student_model(images)
                _, predicted = torch.max(outputs['region_logits'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        log.info(f"Student Validation Accuracy: {(100 * correct / total):.2f}%")

    checkpoint_dir = Path(config.system.checkpoints_path)
    torch.save(student_model.state_dict(), checkpoint_dir / "distilled_student_model.pth")
    log.info("Distillation complete. Student model saved.")


if __name__ == '__main__':
    # Run the main training/evaluation pipeline
    main()

# Gradio demo function
def launch_demo():
    """Launch the Gradio demo interface."""
    config = load_config()
    demo.launch(
        share=config.webapp.share,
        server_name=config.webapp.host,
        server_port=config.webapp.port
    )
```python
# model.py
# Contains the unified FiberOpticNet model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        attn = self.channel_gate(x)
        x = x * attn
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.spatial_gate(torch.cat([avg_pool, max_pool], dim=1))
        x = x * attn
        return x

class FiberOpticNet(nn.Module):
    """
    The unified, minimal, and powerful model for fiber optic analysis.
    This single module handles:
    1. Efficient feature extraction using a ResNet backbone.
    2. Hierarchical analysis using a CBAM attention mechanism.
    3. Region classification (core, cladding, ferrule).
    4. Anomaly detection.
    5. Similarity comparison via learned embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Feature Extractor (Backbone)
        resnet = models.get_model(config.backbone, weights='DEFAULT')
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, config.image_size, config.image_size)
            self.backbone_out_channels = self.encoder(dummy_input).shape[1]
            
        # 2. Attention Mechanism
        self.attention = CBAM(self.backbone_out_channels)

        # 3. Heads for different tasks
        self.region_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone_out_channels, config.num_classes)
        )
        self.anomaly_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone_out_channels, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )

    def forward(self, image, ref_image=None, equation_coeffs=None):
        """A single forward pass computes all necessary outputs."""
        features = self.encoder(image)
        attended_features = self.attention(features)
        
        region_logits = self.region_classifier(attended_features)
        anomaly_map = self.anomaly_head(attended_features)
        embedding = self.embedding_head(attended_features)
        
        ref_embedding = None
        if ref_image is not None:
            with torch.no_grad():
                ref_features = self.encoder(ref_image)
                ref_attended = self.attention(ref_features)
                ref_embedding = self.embedding_head(ref_attended)
        
        region_probs = F.softmax(region_logits, dim=1)
        classification_confidence, _ = torch.max(region_probs, dim=1)
        anomaly_score = torch.sigmoid(anomaly_map).mean(dim=[1, 2, 3])
        
        embedding_similarity = torch.zeros_like(classification_confidence)
        if ref_embedding is not None:
            embedding_similarity = F.cosine_similarity(embedding, ref_embedding)

        coeffs = equation_coeffs or {'A': 0.5, 'B': 0.3, 'C': 0.2}
        final_score = (coeffs['A'] * classification_confidence +
                       coeffs['B'] * (1 - anomaly_score) +
                       coeffs['C'] * embedding_similarity)
        
        return {
            "region_logits": region_logits,
            "anomaly_map": anomaly_map,
            "embedding": embedding,
            "ref_embedding": ref_embedding,
            "final_similarity_score": final_score
        }
```python
# dataset.py
# A simplified and efficient dataset and augmentation module.

import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class FiberOpticsDataset(Dataset):
    """
    Handles loading all fiber optic data, including raw images and .pt files.
    Uses the high-performance `albumentations` library for augmentations.
    """
    def __init__(self, config, mode='train'):
        self.config = config.data
        self.mode = mode
        
        self.image_paths = sorted(list(Path(self.config.path).glob("**/*.png")) +
                                  list(Path(self.config.path).glob("**/*.jpg")))
        self.tensor_paths = sorted(list(Path(self.config.path).glob("**/*.pt")))
        
        self.all_files = self.image_paths + self.tensor_paths
        if not self.all_files:
            raise FileNotFoundError(f"No data found in path: {self.config.path}")
            
        self.class_to_idx = {name: i for i, name in enumerate(self.config.class_names)}
        self.transform = self.get_transforms()

    def get_transforms(self):
        """Returns the augmentation pipeline for the current mode."""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.7),
                A.OneOf([A.ElasticTransform(p=0.7), A.GridDistortion(p=0.5)], p=0.8),
                A.RandomBrightnessContrast(p=0.8),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        
        if file_path.suffix == '.pt':
            image_tensor = torch.load(file_path)
            if isinstance(image_tensor, dict):
                image_tensor = image_tensor.get('tensor', torch.zeros(3, self.config.image_size, self.config.image_size))
            image = image_tensor.permute(1, 2, 0).numpy()
        else:
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        class_name = file_path.parent.name
        label = self.class_to_idx.get(class_name, 0)

        augmented = self.transform(image=image)
        image = augmented['image']
        
        ref_idx = random.randint(0, len(self) - 1)
        ref_path = self.all_files[ref_idx]
        if ref_path.suffix == '.pt':
            ref_tensor = torch.load(ref_path)
            if isinstance(ref_tensor, dict):
                ref_tensor = ref_tensor.get('tensor', torch.zeros(3, self.config.image_size, self.config.image_size))
            ref_image = ref_tensor.permute(1, 2, 0).numpy()
        else:
            ref_image = cv2.imread(str(ref_path))
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        
        ref_transform = self.get_transforms()
        ref_image = ref_transform(image=ref_image)['image']
        
        return {"image": image, "label": torch.tensor(label, dtype=torch.long), "reference": ref_image}
```python
# app.py
# A modern, interactive web UI for real-time analysis and visualization.
# To run: `pip install gradio` then `python app.py`

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from box import Box
import yaml
import cv2

from model import FiberOpticNet

# --- Load Model and Config ---
def load_model_and_config():
    """Loads the trained model and configuration for the app."""
    with open("config.yaml", 'r') as f:
        config_data = yaml.safe_load(f)
    config = Box(config_data)

    model = FiberOpticNet(config.model)
    # Load your best checkpoint here
    checkpoint_path = f"{config.system.checkpoints_path}/epoch_{config.training.num_epochs}.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using an untrained model.")
    
    model.eval()
    return model, config

model, config = load_model_and_config()

# --- Pre- and Post-processing Functions ---
def preprocess_image(image_pil):
    """Converts a PIL image to a tensor suitable for the model."""
    image_np = np.array(image_pil)
    image_resized = cv2.resize(image_np, (config.data.image_size, config.data.image_size))
    tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)

def create_segmentation_overlay(image_pil, logits):
    """Creates a colored segmentation mask overlay."""
    image_np = np.array(image_pil.resize((config.data.image_size, config.data.image_size)))
    
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
    
    colors = np.array(config.visualization.segmentation_colors)
    seg_map = np.zeros_like(image_np)
    for i, color in enumerate(colors):
        seg_map[preds == i] = color
        
    overlay = cv2.addWeighted(image_np, 0.6, seg_map, 0.4, 0)
    return Image.fromarray(overlay)

def create_anomaly_heatmap(anomaly_map):
    """Creates a visual heatmap from the anomaly map."""
    heatmap = torch.sigmoid(anomaly_map).squeeze().cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))

# --- Gradio Inference Function ---
def analyze_fiber(image, coeff_A, coeff_B, coeff_C):
    """The main function that runs inference and returns results for the UI."""
    if image is None:
        return None, None, "Please upload an image.", {}
        
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        coeffs = {'A': coeff_A, 'B': coeff_B, 'C': coeff_C}
        output = model(input_tensor, equation_coeffs=coeffs)
        
    seg_overlay = create_segmentation_overlay(image, output['region_logits'])
    anomaly_heatmap = create_anomaly_heatmap(output['anomaly_map'])
    
    score = output['final_similarity_score'].item()
    meets_threshold = score >= config.similarity.threshold
    pred_class_idx = torch.argmax(output['region_logits']).item()
    pred_class_name = config.data.class_names[pred_class_idx]
    status = "PASS" if meets_threshold else "FAIL"
    
    results_text = (
        f"**Analysis Complete**\n"
        f"Final Similarity Score: **{score:.4f}**\n"
        f"Status: **<span style='color:{'green' if meets_threshold else 'red'};'>{status}</span>**\n"
        f"Predicted Primary Region: **{pred_class_name.capitalize()}**"
    )
    
    labels = {name: prob.item() for name, prob in zip(config.data.class_names, F.softmax(output['region_logits'], dim=1).squeeze())}
    
    return seg_overlay, anomaly_heatmap, results_text, labels

# --- Build the Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Fiber Optic Analysis") as demo:
    gr.Markdown("# 🔬 Real-time Fiber Optic Analysis UI")
    gr.Markdown("Upload a fiber optic image to see the model's analysis. Adjust the equation coefficients to see how they affect the final similarity score.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            gr.Markdown("### ⚙️ Adjust Equation Coefficients (I=Ax1+Bx2+Cx3)")
            coeff_A = gr.Slider(minimum=0, maximum=2, value=config.equation.coefficients.A, label="A (Classification Confidence)")
            coeff_B = gr.Slider(minimum=0, maximum=2, value=config.equation.coefficients.B, label="B (1 - Anomaly Score)")
            coeff_C = gr.Slider(minimum=0, maximum=2, value=config.equation.coefficients.C, label="C (Embedding Similarity)")
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")
            with gr.Row():
                output_segmentation = gr.Image(label="Segmentation Overlay")
                output_anomaly = gr.Image(label="Anomaly Heatmap")
            results_markdown = gr.Markdown()
            output_labels = gr.Label(label="Region Probabilities")

    inputs = [input_image, coeff_A, coeff_B, coeff_C]
    outputs = [output_segmentation, output_anomaly, results_markdown, output_labels]
    submit_btn.click(fn=analyze_fiber, inputs=inputs, outputs=outputs)

# Different entry points for different functionalities
if __name__ == "__main__":
    # By default, launch the Gradio demo
    config = load_config()
    demo.launch(
        share=config.webapp.share,
        server_name=config.webapp.host,
        server_port=config.webapp.port
    )
