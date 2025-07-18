#!/usr/bin/env python3
"""
Robust Learning Auto-Clicker with PyTorch Capabilities
Learns from user demonstration, follows click sequence, and uses deep learning for detection
"""

import time
import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from pynput import mouse, keyboard
import tkinter as tk
from threading import Thread
import random
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('learning_autoclicker.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class ClickDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class ClickDetector(nn.Module):
    def __init__(self, num_classes):
        super(ClickDetector, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Unfreeze the last two layers for better adaptation
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class LearningAutoClicker:
    def __init__(self):
        self.crop_size = 224  # For ResNet
        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.click_patches = []  # List of positive patches for each click in sequence
        self.click_positions = []  # Recorded positions
        self.click_times = []  # Timestamps of clicks
        self.delays = []  # Computed delays between clicks
        self.negative_samples = []  # Shared negatives
        self.additional_positives = []  # Additional positives per class, dynamic
        self.additional_negatives = []
        self.recording = False
        self.listener = None
        self.screen_width, self.screen_height = pyautogui.size()
        self.scan_step = 32  # Smaller step for better detection
        self.is_moving = True
        self.uniform_interval = 0
        self.use_recorded_duration = False
        self.running = False
        self.ref_window = None
        self.ref_x = 0
        self.ref_y = 0
        self.ref_size = 50
        self.esc_listener = None
        self.scans = 0
        self.detections = 0
        self.clicks = 0
        self.start_time = time.time()
        self.retrain_interval = 5  # Retrain more frequently
        self.cycle_count = 0

    def create_reference_button(self):
        self.ref_window = tk.Tk()
        self.ref_window.title("Ref")
        self.ref_window.geometry("50x50")
        self.ref_window.attributes('-topmost', True)
        canvas = tk.Canvas(self.ref_window, bg="black", height=50, width=50)
        canvas.pack()
        canvas.create_oval(20, 20, 30, 30, fill="red")
        self.ref_window.bind("<Button-1>", self.toggle_recording)
        self.ref_window.update_idletasks()
        self.ref_x = self.ref_window.winfo_x()
        self.ref_y = self.ref_window.winfo_y()
        logging.info(f"Reference button created at ({self.ref_x}, {self.ref_y})")

    def toggle_recording(self, event=None):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.click_patches = []
        self.click_positions = []
        self.click_times = []
        self.negative_samples = []
        self.additional_positives = []
        logging.info("Started recording user actions.")
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()
        Thread(target=self.capture_background).start()

    def stop_recording(self):
        self.recording = False
        if self.listener:
            self.listener.stop()
        logging.info("Stopped recording. Processing data...")
        self.process_recorded_data()
        self.show_config_ui()

    def is_ref_click(self, x, y):
        return (self.ref_x <= x <= self.ref_x + self.ref_size) and (self.ref_y <= y <= self.ref_y + self.ref_size)

    def on_click(self, x, y, button, pressed):
        if pressed and button == mouse.Button.left and self.recording:
            if self.is_ref_click(x, y):
                logging.info("Reference button clicked during recording - ignoring for data.")
                return True
            current_time = time.time()
            if self.click_times:
                duration = current_time - self.click_times[-1]
                logging.info(f"Recorded click at ({x}, {y}) after {duration:.2f}s")
            else:
                logging.info(f"Recorded first click at ({x}, {y})")
            self.click_times.append(current_time)
            self.click_positions.append((x, y))
            screenshot = ImageGrab.grab()
            crop_left = max(0, x - self.crop_size // 2)
            crop_top = max(0, y - self.crop_size // 2)
            crop_right = min(self.screen_width, x + self.crop_size // 2)
            crop_bottom = min(self.screen_height, y + self.crop_size // 2)
            pos_crop = screenshot.crop((crop_left, crop_top, crop_right, crop_bottom)).resize((self.crop_size, self.crop_size))
            self.click_patches.append(pos_crop)
            self.additional_positives.append([])  # Add empty list for this class
        return True

    def capture_background(self):
        while self.recording:
            screenshot = ImageGrab.grab()
            for _ in range(20):  # More negatives for better training
                rand_x = random.randint(0, self.screen_width - self.crop_size)
                rand_y = random.randint(0, self.screen_height - self.crop_size)
                if not self.is_ref_click(rand_x + self.crop_size // 2, rand_y + self.crop_size // 2):
                    neg_crop = screenshot.crop((rand_x, rand_y, rand_x + self.crop_size, rand_y + self.crop_size)).resize((self.crop_size, self.crop_size))
                    self.negative_samples.append(neg_crop)
                    logging.info(f"Captured negative sample at ({rand_x}, {rand_y})")
            time.sleep(0.3)  # More frequent captures

    def process_recorded_data(self):
        if len(self.click_times) > 1:
            diffs = [self.click_times[i] - self.click_times[i-1] for i in range(1, len(self.click_times))]
            avg_delay = sum(diffs) / len(diffs)
            self.delays = diffs + [avg_delay]  # Append avg for closing the loop
            logging.info(f"Recorded delays: {self.delays}")
        elif len(self.click_times) == 1:
            self.delays = [0.0]
        if self.click_patches:
            self.train_model()

    def get_training_data(self):
        images = []
        labels = []
        num_targets = len(self.click_patches)
        # Original positives
        for i, patch in enumerate(self.click_patches):
            images.append(patch)
            labels.append(i + 1)  # Class 1 to num_targets
        # Additional positives
        for i in range(num_targets):
            for add_patch in self.additional_positives[i]:
                images.append(add_patch)
                labels.append(i + 1)
        # Negatives
        for neg in self.negative_samples + self.additional_negatives:
            images.append(neg)
            labels.append(0)  # Class 0
        return images, labels

    def train_model(self, retrain=False):
        if retrain:
            logging.info("Retraining model with new data...")
        else:
            logging.info("Starting initial model training with PyTorch...")
        num_classes = len(self.click_patches) + 1  # 0 for neg, 1-N for targets
        self.model = ClickDetector(num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='loss', factor=0.1, patience=3)

        images, labels = self.get_training_data()
        dataset = ClickDataset(images, labels, transform=self.aug_transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # Larger batch

        self.model.train()
        epochs = 20 if retrain else 15  # More epochs
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(dataloader)
            scheduler.step(avg_loss)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")
        logging.info("Model training completed.")

    def detect_target(self, target_class):
        self.scans += 1
        screenshot = ImageGrab.grab()
        max_prob = 0
        best_pos = None
        self.model.eval()
        with torch.no_grad():
            for y in range(0, self.screen_height - self.crop_size, self.scan_step):
                for x in range(0, self.screen_width - self.crop_size, self.scan_step):
                    crop = screenshot.crop((x, y, x + self.crop_size, y + self.crop_size))
                    input_tensor = self.test_transform(crop).unsqueeze(0).to(self.device)
                    outputs = self.model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    prob = probs[target_class].item()
                    if prob > max_prob and prob > 0.75:  # Slightly lower threshold for detection
                        max_prob = prob
                        best_pos = (x + self.crop_size // 2, y + self.crop_size // 2)
                        logging.info(f"Detected target class {target_class} at ({best_pos}) with prob {prob:.2f}")
        if best_pos:
            self.detections += 1
            # Add as additional positive
            crop_left = max(0, best_pos[0] - self.crop_size // 2)
            crop_top = max(0, best_pos[1] - self.crop_size // 2)
            crop_right = min(self.screen_width, best_pos[0] + self.crop_size // 2)
            crop_bottom = min(self.screen_height, best_pos[1] + self.crop_size // 2)
            pos_crop = screenshot.crop((crop_left, crop_top, crop_right, crop_bottom)).resize((self.crop_size, self.crop_size))
            self.additional_positives[target_class - 1].append(pos_crop)
            # Add some new negatives
            for _ in range(5):
                rand_x = random.randint(0, self.screen_width - self.crop_size)
                rand_y = random.randint(0, self.screen_height - self.crop_size)
                neg_crop = screenshot.crop((rand_x, rand_y, rand_x + self.crop_size, rand_y + self.crop_size)).resize((self.crop_size, self.crop_size))
                self.additional_negatives.append(neg_crop)
        return best_pos, max_prob

    def click_target(self, pos):
        if pos:
            x, y = pos
            logging.info(f"Clicking at ({x}, {y})")
            pyautogui.click(x, y)
            self.clicks += 1
            return True
        return False

    def show_config_ui(self):
        config_window = tk.Tk()
        config_window.title("Auto-Clicker Settings")
        config_window.resizable(True, True)

        tk.Label(config_window, text="Uniform Click Interval (seconds, 0 for immediate):").pack()
        interval_entry = tk.Entry(config_window)
        interval_entry.pack()
        interval_entry.insert(0, "0")

        use_recorded_var = tk.BooleanVar()
        tk.Checkbutton(config_window, text="Use durations from recording", variable=use_recorded_var).pack()

        moving_var = tk.BooleanVar(value=True)
        tk.Radiobutton(config_window, text="Moving Targets", variable=moving_var, value=True).pack()
        tk.Radiobutton(config_window, text="Fixed Positions", variable=moving_var, value=False).pack()

        def confirm():
            self.uniform_interval = float(interval_entry.get())
            self.use_recorded_duration = use_recorded_var.get()
            self.is_moving = moving_var.get()
            if self.use_recorded_duration:
                logging.info(f"Using recorded delays: {self.delays}")
            else:
                logging.info(f"Using uniform interval: {self.uniform_interval}s")
            logging.info(f"Settings confirmed: Moving={self.is_moving}")
            config_window.destroy()
            self.start_clicking()

        tk.Button(config_window, text="Confirm and Start", command=confirm).pack()
        config_window.mainloop()

    def start_clicking(self):
        self.running = True
        self.ref_window.withdraw()  # Hide reference button
        logging.info("Starting auto-clicking sequence. Press ESC to stop.")
        self.esc_listener = keyboard.Listener(on_press=self.on_key_press)
        self.esc_listener.start()
        num_steps = len(self.click_positions)
        while self.running:
            for step in range(num_steps):
                if not self.running:
                    break
                target_class = step + 1
                if self.is_moving:
                    pos, prob = self.detect_target(target_class)
                else:
                    pos = self.click_positions[step]
                    prob = 1.0
                    logging.info(f"Using fixed position {pos} for step {step}")
                clicked = self.click_target(pos)
                if clicked:
                    if self.use_recorded_duration:
                        sleep_time = self.delays[min(step, len(self.delays)-1)]
                    else:
                        sleep_time = self.uniform_interval if self.uniform_interval > 0 else 0.1
                    logging.info(f"Sleeping for {sleep_time:.2f}s after click")
                    time.sleep(sleep_time)
                else:
                    logging.warning(f"No detection for step {step}, retrying after 0.5s")
                    time.sleep(0.5)
            self.cycle_count += 1
            if self.cycle_count % self.retrain_interval == 0 and (any(len(lst) > 0 for lst in self.additional_positives) or len(self.additional_negatives) > 0):
                self.train_model(retrain=True)
            if math.floor(time.time() - self.start_time) % 30 == 0:
                self.print_stats()

    def on_key_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            logging.info("ESC pressed. Stopping auto-clicker.")
            self.print_stats()
            sys.exit(0)

    def print_stats(self):
        runtime = time.time() - self.start_time
        logging.info(f"\n--- Statistics ---")
        logging.info(f"Runtime: {runtime:.1f}s")
        logging.info(f"Scans: {self.scans}")
        logging.info(f"Detections: {self.detections}")
        logging.info(f"Clicks: {self.clicks}")
        logging.info(f"Success rate: {(self.clicks / max(1, self.detections) * 100):.1f}%")
        logging.info(f"Scan rate: {self.scans / max(1, runtime):.1f}/s")

def main():
    try:
        import torch
        import torchvision
        import cv2
        import numpy
        import pyautogui
        import pynput
        import tkinter
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install torch torchvision opencv-python numpy pillow pyautogui pynput")
        return

    clicker = LearningAutoClicker()
    clicker.create_reference_button()
    clicker.ref_window.mainloop()

if __name__ == "__main__":
    main()