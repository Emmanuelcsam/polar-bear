#!/usr/bin/env python3
"""
Robust Learning Screen Agent with PyTorch Capabilities
Learns from user demonstration, follows event sequence, and uses deep learning for detection
"""

import time
import numpy as np
import cv2
from PIL import ImageGrab, Image
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
import psutil
import pickle
import datetime

if os.name == 'nt':
    import win32gui
    import win32process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('learning_screen_agent.log'),
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
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class LearningScreenAgent:
    def __init__(self):
        self.crop_size = 224
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
        self.click_patches = []
        self.click_positions = []
        self.events = []
        self.diffs = []
        self.negative_samples = []
        self.additional_positives = []
        self.additional_negatives = []
        self.recording = False
        self.mouse_listener = None
        self.keyboard_listener = None
        self.screen_width, self.screen_height = pyautogui.size()
        self.scan_step = 32
        self.is_moving = True
        self.uniform_interval = 0
        self.use_recorded_duration = False
        self.running = False
        self.ui_window = None
        self.record_button = None
        self.selected_run = None
        self.esc_listener = None
        self.scans = 0
        self.detections = 0
        self.clicks = 0
        self.start_time = time.time()
        self.retrain_interval = 5
        self.cycle_count = 0
        self.last_click_pos = None
        self.runs_dir = "runs"
        if not os.path.exists(self.runs_dir):
            os.makedirs(self.runs_dir)
        self.past_runs = sorted([d for d in os.listdir(self.runs_dir) if os.path.isdir(os.path.join(self.runs_dir, d))], reverse=True)
        self.create_ui()

    def create_ui(self):
        self.ui_window = tk.Tk()
        self.ui_window.title("Screen Agent")
        self.ui_window.geometry("200x150")
        self.ui_window.attributes('-topmost', True)
        self.record_button = tk.Button(self.ui_window, text="New Run", command=self.toggle_recording_ui)
        self.record_button.pack()
        if self.past_runs:
            tk.Label(self.ui_window, text="Past Runs:").pack()
            self.selected_run = tk.StringVar(value=self.past_runs[0])
            tk.OptionMenu(self.ui_window, self.selected_run, *self.past_runs).pack()
            tk.Button(self.ui_window, text="Replay Run", command=self.replay_run).pack()
        self.ui_window.mainloop()

    def toggle_recording_ui(self):
        if not self.recording:
            self.reset()
            self.start_recording()
            self.record_button.config(text="Stop Recording")
        else:
            self.stop_recording()
            self.record_button.config(text="New Run")

    def reset(self):
        self.events = []
        self.click_patches = []
        self.click_positions = []
        self.negative_samples = []
        self.additional_positives = []
        self.additional_negatives = []
        self.diffs = []
        self.model = None

    def start_recording(self):
        self.recording = True
        logging.info("Started recording user actions.")
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_keyboard_press, on_release=self.on_keyboard_release)
        self.mouse_listener.start()
        self.keyboard_listener.start()
        Thread(target=self.capture_background).start()

    def stop_recording(self):
        self.recording = False
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        logging.info("Stopped recording. Processing data...")
        self.events.sort(key=lambda e: e['time'])
        self.process_recorded_data()
        self.save_run()
        self.show_config_ui()

    def on_click(self, x, y, button, pressed):
        if self.recording:
            current_time = time.time()
            focused = self.get_focused_info()
            event_type = 'mouse_press' if pressed else 'mouse_release'
            self.events.append({'type': event_type, 'button': str(button), 'pos': (x, y), 'time': current_time, 'focused': focused})
            if pressed and button == mouse.Button.left:
                logging.info(f"Recorded click at ({x}, {y})")
                self.click_positions.append((x, y))
                screenshot = ImageGrab.grab()
                crop_left = max(0, x - self.crop_size // 2)
                crop_top = max(0, y - self.crop_size // 2)
                crop_right = min(self.screen_width, x + self.crop_size // 2)
                crop_bottom = min(self.screen_height, y + self.crop_size // 2)
                pos_crop = screenshot.crop((crop_left, crop_top, crop_right, crop_bottom)).resize((self.crop_size, self.crop_size))
                self.click_patches.append(pos_crop)
                self.additional_positives.append([])
        return True

    def on_keyboard_press(self, key):
        if self.recording:
            current_time = time.time()
            focused = self.get_focused_info()
            key_str = self.key_to_str(key)
            self.events.append({'type': 'key_press', 'key': key_str, 'time': current_time, 'focused': focused})
            logging.info(f"Recorded key press: {key_str}")

    def on_keyboard_release(self, key):
        if self.recording:
            current_time = time.time()
            focused = self.get_focused_info()
            key_str = self.key_to_str(key)
            self.events.append({'type': 'key_release', 'key': key_str, 'time': current_time, 'focused': focused})
            logging.info(f"Recorded key release: {key_str}")

    def key_to_str(self, key):
        if hasattr(key, 'char') and key.char:
            return key.char
        return str(key).replace('Key.', '')

    def capture_background(self):
        while self.recording:
            screenshot = ImageGrab.grab()
            for _ in range(20):
                rand_x = random.randint(0, self.screen_width - self.crop_size)
                rand_y = random.randint(0, self.screen_height - self.crop_size)
                neg_crop = screenshot.crop((rand_x, rand_y, rand_x + self.crop_size, rand_y + self.crop_size)).resize((self.crop_size, self.crop_size))
                self.negative_samples.append(neg_crop)
            time.sleep(0.3)

    def get_focused_info(self):
        if os.name != 'nt':
            return {'title': '', 'exe': ''}
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            p = psutil.Process(pid)
            exe = p.exe()
            return {'title': title, 'exe': exe}
        except Exception as e:
            logging.warning(f"Failed to get focused info: {e}")
            return {'title': '', 'exe': ''}

    def process_recorded_data(self):
        self.process_times()
        if self.click_patches:
            self.train_model()

    def process_times(self):
        if self.events:
            self.diffs = [0.0]
            for i in range(1, len(self.events)):
                self.diffs.append(self.events[i]['time'] - self.events[i-1]['time'])

    def get_training_data(self):
        images = []
        labels = []
        num_targets = len(self.click_patches)
        for i, patch in enumerate(self.click_patches):
            images.append(patch)
            labels.append(i + 1)
        for i in range(num_targets):
            for add_patch in self.additional_positives[i]:
                images.append(add_patch)
                labels.append(i + 1)
        for neg in self.negative_samples + self.additional_negatives:
            images.append(neg)
            labels.append(0)
        return images, labels

    def train_model(self, retrain=False):
        num_classes = len(self.click_patches) + 1
        if retrain:
            logging.info("Retraining model with new data...")
        else:
            self.model = ClickDetector(num_classes).to(self.device)
            logging.info("Starting initial model training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='loss', factor=0.1, patience=3)
        images, labels = self.get_training_data()
        dataset = ClickDataset(images, labels, transform=self.aug_transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        self.model.train()
        epochs = 20 if retrain else 15
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
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
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
                    if prob > max_prob and prob > 0.75:
                        max_prob = prob
                        best_pos = (x + self.crop_size // 2, y + self.crop_size // 2)
        if best_pos:
            self.detections += 1
            crop_left = max(0, best_pos[0] - self.crop_size // 2)
            crop_top = max(0, best_pos[1] - self.crop_size // 2)
            crop_right = min(self.screen_width, best_pos[0] + self.crop_size // 2)
            crop_bottom = min(self.screen_height, best_pos[1] + self.crop_size // 2)
            pos_crop = screenshot.crop((crop_left, crop_top, crop_right, crop_bottom)).resize((self.crop_size, self.crop_size))
            self.additional_positives[target_class - 1].append(pos_crop)
            for _ in range(5):
                rand_x = random.randint(0, self.screen_width - self.crop_size)
                rand_y = random.randint(0, self.screen_height - self.crop_size)
                neg_crop = screenshot.crop((rand_x, rand_y, rand_x + self.crop_size, rand_y + self.crop_size)).resize((self.crop_size, self.crop_size))
                self.additional_negatives.append(neg_crop)
        return best_pos, max_prob

    def show_config_ui(self):
        config_window = tk.Tk()
        config_window.title("Agent Settings")
        tk.Label(config_window, text="Uniform Interval (s, 0 for immediate):").pack()
        interval_entry = tk.Entry(config_window)
        interval_entry.pack()
        interval_entry.insert(0, str(self.uniform_interval))
        use_recorded_var = tk.BooleanVar(value=self.use_recorded_duration)
        tk.Checkbutton(config_window, text="Use recorded durations", variable=use_recorded_var).pack()
        moving_var = tk.BooleanVar(value=self.is_moving)
        tk.Radiobutton(config_window, text="Moving Targets", variable=moving_var, value=True).pack()
        tk.Radiobutton(config_window, text="Fixed Positions", variable=moving_var, value=False).pack()
        def confirm():
            self.uniform_interval = float(interval_entry.get())
            self.use_recorded_duration = use_recorded_var.get()
            self.is_moving = moving_var.get()
            if not self.use_recorded_duration:
                self.diffs = [self.uniform_interval] * len(self.events)
            config_window.destroy()
            self.start_clicking()
        tk.Button(config_window, text="Confirm and Start", command=confirm).pack()
        config_window.mainloop()

    def start_clicking(self):
        self.running = True
        self.ui_window.withdraw()
        logging.info("Starting agent sequence. Press ESC to stop.")
        self.esc_listener = keyboard.Listener(on_press=self.on_key_press)
        self.esc_listener.start()
        while self.running:
            click_count = 0
            self.last_click_pos = None
            for idx, event in enumerate(self.events):
                if not self.running:
                    break
                time.sleep(self.diffs[idx])
                self.handle_focus(event['focused'])
                event_type = event['type']
                if event_type == 'mouse_press':
                    click_count += 1 if event['button'] == 'Button.left' else 0
                    button = event['button'].replace('Button.', '')
                    if self.is_moving and event['button'] == 'Button.left':
                        pos, _ = self.detect_target(click_count)
                        if pos:
                            pyautogui.mouseDown(pos[0], pos[1], button=button)
                            self.last_click_pos = pos
                            self.clicks += 1
                    else:
                        pos = event['pos']
                        pyautogui.mouseDown(pos[0], pos[1], button=button)
                        self.last_click_pos = pos
                        self.clicks += 1
                elif event_type == 'mouse_release':
                    button = event['button'].replace('Button.', '')
                    pos = self.last_click_pos if self.is_moving and event['button'] == 'Button.left' else event['pos']
                    if pos:
                        pyautogui.mouseUp(pos[0], pos[1], button=button)
                elif event_type == 'key_press':
                    pyautogui.keyDown(event['key'])
                elif event_type == 'key_release':
                    pyautogui.keyUp(event['key'])
            self.cycle_count += 1
            if self.cycle_count % self.retrain_interval == 0 and (any(self.additional_positives) or self.additional_negatives):
                self.train_model(retrain=True)
            if math.floor(time.time() - self.start_time) % 30 == 0:
                self.print_stats()
        self.ui_window.deiconify()

    def handle_focus(self, expected_focused):
        if os.name != 'nt' or not expected_focused['exe']:
            return
        running = any(p.info['exe'] == expected_focused['exe'] for p in psutil.process_iter(['exe']))
        if not running:
            logging.info(f"Launching {expected_focused['exe']}")
            try:
                os.startfile(expected_focused['exe'])
                wait_start = time.time()
                found = False
                while time.time() - wait_start < 10:
                    hwnds = []
                    def enum_cb(hwnd, results):
                        if win32gui.GetWindowText(hwnd) == expected_focused['title']:
                            results.append(hwnd)
                    win32gui.EnumWindows(enum_cb, hwnds)
                    if hwnds:
                        found = True
                        break
                    time.sleep(0.5)
                if not found:
                    logging.warning("Window not found after launch")
                    return
            except Exception as e:
                logging.warning(f"Failed to launch {expected_focused['exe']}: {e}")
                return
        # Activate window
        def activate_cb(hwnd, _):
            if win32gui.GetWindowText(hwnd) == expected_focused['title']:
                win32gui.SetForegroundWindow(hwnd)
                return False  # stop enum
            return True
        win32gui.EnumWindows(activate_cb, None)
        time.sleep(0.5)

    def on_key_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            logging.info("ESC pressed. Stopping agent.")
            self.print_stats()
            if self.esc_listener:
                self.esc_listener.stop()

    def print_stats(self):
        runtime = time.time() - self.start_time
        logging.info(f"\n--- Statistics ---")
        logging.info(f"Runtime: {runtime:.1f}s")
        logging.info(f"Scans: {self.scans}")
        logging.info(f"Detections: {self.detections}")
        logging.info(f"Clicks: {self.clicks}")
        logging.info(f"Success rate: {(self.clicks / max(1, self.detections) * 100):.1f}%")

    def save_run(self):
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(self.runs_dir, ts)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "events.pkl"), "wb") as f:
            pickle.dump(self.events, f)
        if self.model:
            torch.save(self.model.state_dict(), os.path.join(run_dir, "model.pth"))
        with open(os.path.join(run_dir, "settings.pkl"), "wb") as f:
            pickle.dump({
                "uniform_interval": self.uniform_interval,
                "use_recorded_duration": self.use_recorded_duration,
                "is_moving": self.is_moving,
            }, f)
        for i, patch in enumerate(self.click_patches):
            patch.save(os.path.join(run_dir, f"patch_{i}.png"))
        add_pos_dir = os.path.join(run_dir, "additional_positives")
        os.makedirs(add_pos_dir)
        for j, lst in enumerate(self.additional_positives):
            for k, p in enumerate(lst):
                p.save(os.path.join(add_pos_dir, f"{j}_{k}.png"))
        add_neg_dir = os.path.join(run_dir, "additional_negatives")
        os.makedirs(add_neg_dir)
        for k, p in enumerate(self.additional_negatives):
            p.save(os.path.join(add_neg_dir, f"{k}.png"))

    def load_run(self, ts):
        self.reset()
        run_dir = os.path.join(self.runs_dir, ts)
        with open(os.path.join(run_dir, "events.pkl"), "rb") as f:
            self.events = pickle.load(f)
        self.click_patches = []
        i = 0
        while os.path.exists(os.path.join(run_dir, f"patch_{i}.png")):
            patch = Image.open(os.path.join(run_dir, f"patch_{i}.png"))
            self.click_patches.append(patch)
            i += 1
        if self.click_patches:
            num_classes = len(self.click_patches) + 1
            self.model = ClickDetector(num_classes).to(self.device)
            self.model.load_state_dict(torch.load(os.path.join(run_dir, "model.pth")))
        with open(os.path.join(run_dir, "settings.pkl"), "rb") as f:
            settings = pickle.load(f)
            self.uniform_interval = settings["uniform_interval"]
            self.use_recorded_duration = settings["use_recorded_duration"]
            self.is_moving = settings["is_moving"]
        self.additional_positives = [[] for _ in self.click_patches]
        add_pos_dir = os.path.join(run_dir, "additional_positives")
        if os.path.exists(add_pos_dir):
            for j in range(len(self.click_patches)):
                k = 0
                while os.path.exists(os.path.join(add_pos_dir, f"{j}_{k}.png")):
                    p = Image.open(os.path.join(add_pos_dir, f"{j}_{k}.png"))
                    self.additional_positives[j].append(p)
                    k += 1
        self.additional_negatives = []
        add_neg_dir = os.path.join(run_dir, "additional_negatives")
        if os.path.exists(add_neg_dir):
            k = 0
            while os.path.exists(os.path.join(add_neg_dir, f"{k}.png")):
                p = Image.open(os.path.join(add_neg_dir, f"{k}.png"))
                self.additional_negatives.append(p)
                k += 1
        self.process_times()
        if not self.use_recorded_duration:
            self.diffs = [self.uniform_interval] * len(self.events)

    def replay_run(self):
        ts = self.selected_run.get()
        self.load_run(ts)
        self.show_config_ui()

def main():
    try:
        import torch
        import torchvision
        import cv2
        import numpy
        import pyautogui
        import pynput
        import tkinter
        import psutil
        if os.name == 'nt':
            import win32gui
            import win32process
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install torch torchvision opencv-python numpy pillow pyautogui pynput psutil pywin32")
        return

    agent = LearningScreenAgent()

if __name__ == "__main__":
    main()
