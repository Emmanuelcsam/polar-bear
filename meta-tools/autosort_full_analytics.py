#!/usr/bin/env python3
"""
Full Analytics Image Sorter - Complete Integration
Combines multi-modal learning with comprehensive analytics tracking
"""

import sys
import os
import shutil
import subprocess
import importlib
import logging
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import re
import json
import pickle
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
import sqlite3
import csv
from pathlib import Path
import hashlib
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import h5py

# Import analytics module
from autosort_analytics import ComprehensiveAnalytics

# Function to install missing packages
def install_if_missing(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure all required packages are installed
required_packages = ['pygame', 'torch', 'torchvision', 'pillow', 'numpy', 
                    'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'scipy', 'h5py']
for pkg in required_packages:
    install_if_missing(pkg)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_analytics_image_sorter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FilenameAnalyzer:
    """Analyzes filenames with analytics tracking"""
    
    def __init__(self, analytics: Optional[ComprehensiveAnalytics] = None):
        self.analytics = analytics
        self.separators = ['-', '_', ' ', '.', '(', ')', '[', ']']
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                          'img', 'image', 'photo', 'pic', 'picture', 'file', 'copy', 'new'}
        self.date_pattern = re.compile(r'\d{4}[-_]?\d{2}[-_]?\d{2}|\d{2}[-_]?\d{2}[-_]?\d{4}')
        self.number_pattern = re.compile(r'\d+')
    
    def extract_keywords(self, filename):
        """Extract meaningful keywords from a filename"""
        name_without_ext = os.path.splitext(filename)[0]
        
        if not name_without_ext or name_without_ext == '':
            return [], []
        
        name_lower = name_without_ext.lower()
        
        for sep in self.separators:
            name_lower = name_lower.replace(sep, ' ')
        
        dates = self.date_pattern.findall(name_without_ext)
        
        for date in dates:
            name_lower = name_lower.replace(date.lower(), '')
        
        words = name_lower.split()
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        keywords = [k for k in keywords if not k.isdigit()]
        
        # Track keyword extraction in analytics
        if self.analytics:
            self.analytics.session_data['feature_statistics']['keywords_extracted'].extend(keywords)
            self.analytics.session_data['feature_statistics']['dates_found'].extend(dates)
        
        return keywords, dates
    
    def get_filename_features(self, filename):
        """Get a feature vector representing the filename"""
        keywords, dates = self.extract_keywords(filename)
        
        features = {
            'keywords': keywords,
            'has_date': len(dates) > 0,
            'date_count': len(dates),
            'keyword_count': len(keywords),
            'filename_length': len(filename),
            'has_numbers': bool(self.number_pattern.search(filename))
        }
        
        return features

class MultiModalImageClassifier(nn.Module):
    """Neural network with analytics tracking"""
    
    def __init__(self, num_classes, analytics: Optional[ComprehensiveAnalytics] = None):
        super(MultiModalImageClassifier, self).__init__()
        self.analytics = analytics
        
        # Image feature extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()
        
        # Text feature processor
        self.text_embedding = nn.Embedding(10000, 64)
        self.text_lstm = nn.LSTM(64, 128, batch_first=True)
        
        # Fusion layers
        self.fusion_layer = nn.Linear(512 + 128, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, image, text_indices, save_features=False, image_path=None):
        # Extract image features
        image_features = self.resnet(image)
        
        # Process text features
        text_embedded = self.text_embedding(text_indices)
        _, (text_features, _) = self.text_lstm(text_embedded)
        text_features = text_features.squeeze(0)
        
        # Combine features
        combined = torch.cat([image_features, text_features], dim=1)
        
        # Save features if requested
        if save_features and self.analytics and image_path:
            self.analytics.save_model_features(
                image_path, image_features, text_features, combined
            )
        
        # Final classification
        fused = self.relu(self.fusion_layer(combined))
        fused = self.dropout(fused)
        output = self.classifier(fused)
        
        return output

class LearningHistory:
    """Maintains learning history with analytics integration"""
    
    def __init__(self, history_file='learning_history.pkl', 
                 analytics: Optional[ComprehensiveAnalytics] = None):
        self.history_file = history_file
        self.analytics = analytics
        self.keyword_to_folder = defaultdict(dict)
        self.folder_keywords = defaultdict(set)
        self.decisions = []
        self.load_history()
    
    def load_history(self):
        """Load existing history from disk"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'rb') as f:
                    data = pickle.load(f)
                    # Convert back to regular dict structure
                    self.keyword_to_folder = defaultdict(dict)
                    for k, v in data['keyword_to_folder'].items():
                        self.keyword_to_folder[k] = v
                    # Ensure folder_keywords uses sets, not lists
                    self.folder_keywords = defaultdict(set)
                    for k, v in data['folder_keywords'].items():
                        self.folder_keywords[k] = set(v) if isinstance(v, list) else v
                    self.decisions = data['decisions']
                logger.info(f"Loaded learning history with {len(self.decisions)} decisions")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
    
    def save_history(self):
        """Save history to disk"""
        try:
            data = {
                'keyword_to_folder': dict(self.keyword_to_folder),
                'folder_keywords': {k: list(v) for k, v in self.folder_keywords.items()},
                'decisions': self.decisions
            }
            with open(self.history_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info("Saved learning history")
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def record_decision(self, filename, keywords, folder_name):
        """Record a sorting decision"""
        decision = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'keywords': keywords,
            'folder': folder_name
        }
        self.decisions.append(decision)
        
        # Update keyword associations
        for keyword in keywords:
            if keyword not in self.keyword_to_folder:
                self.keyword_to_folder[keyword] = defaultdict(int)
            self.keyword_to_folder[keyword][folder_name] += 1
            self.folder_keywords[folder_name].add(keyword)
        
        # Keep only last 10000 decisions
        if len(self.decisions) > 10000:
            self.decisions = self.decisions[-10000:]
        
        # Track in analytics
        if self.analytics:
            self.analytics.session_data['user_behavior']['learning_history_size'].append(
                len(self.decisions))
    
    def get_keyword_predictions(self, keywords):
        """Get folder predictions based on keywords"""
        folder_scores = defaultdict(float)
        
        for keyword in keywords:
            if keyword in self.keyword_to_folder:
                for folder, count in self.keyword_to_folder[keyword].items():
                    folder_scores[folder] += count
        
        return folder_scores

class Vocabulary:
    """Manages vocabulary with analytics"""
    
    def __init__(self, vocab_file='vocabulary.json', 
                 analytics: Optional[ComprehensiveAnalytics] = None):
        self.vocab_file = vocab_file
        self.analytics = analytics
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.load_vocabulary()
    
    def load_vocabulary(self):
        """Load existing vocabulary"""
        if os.path.exists(self.vocab_file):
            try:
                with open(self.vocab_file, 'r') as f:
                    self.word_to_idx = json.load(f)
                    self.idx_to_word = {int(idx): word for word, idx in self.word_to_idx.items()}
                logger.info(f"Loaded vocabulary with {len(self.word_to_idx)} words")
            except Exception as e:
                logger.error(f"Error loading vocabulary: {e}")
    
    def save_vocabulary(self):
        """Save vocabulary to disk"""
        try:
            with open(self.vocab_file, 'w') as f:
                json.dump(self.word_to_idx, f)
            logger.info("Saved vocabulary")
            
            # Track vocabulary growth in analytics
            if self.analytics:
                self.analytics.session_data['feature_statistics']['vocabulary_size'].append(
                    len(self.word_to_idx))
        except Exception as e:
            logger.error(f"Error saving vocabulary: {e}")
    
    def add_word(self, word):
        """Add a word to vocabulary if not present"""
        if word not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def encode_keywords(self, keywords, max_len=20):
        """Convert keywords to indices"""
        indices = []
        for word in keywords[:max_len]:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(self.word_to_idx['<PAD>'])
        
        return indices[:max_len]

def main():
    """Main function with full analytics integration"""
    logger.info("Starting full analytics image sorter application.")
    
    # Initialize analytics
    analytics = ComprehensiveAnalytics()
    
    # Initialize components with analytics
    filename_analyzer = FilenameAnalyzer(analytics=analytics)
    learning_history = LearningHistory(analytics=analytics)
    vocabulary = Vocabulary(analytics=analytics)
    
    # Create hidden tkinter window for dialogs
    root = tk.Tk()
    root.withdraw()
    
    # Select source directory
    logger.info("Prompting user to select source directory.")
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        logger.info("No source directory selected. Exiting.")
        analytics.close()
        return
    
    logger.info(f"Source directory selected: {source_dir}")
    
    # Select target directories
    target_dirs = []
    target_dir_names = []
    
    while True:
        tdir = filedialog.askdirectory(title="Select a Target Directory (Cancel when all added)")
        if not tdir:
            logger.info("Target directory selection completed.")
            break
        
        if tdir not in target_dirs:
            target_dirs.append(tdir)
            target_dir_names.append(os.path.basename(tdir))
            logger.info(f"Added target directory: {tdir}")
    
    if not target_dirs:
        logger.error("No target directories selected.")
        messagebox.showerror("Error", "No target directories selected.")
        analytics.close()
        return
    
    # Track session configuration
    analytics.session_data['configuration'] = {
        'source_directory': source_dir,
        'target_directories': target_dirs,
        'num_targets': len(target_dirs)
    }
    
    # Create directories if needed
    for tdir in target_dirs:
        if not os.path.exists(tdir):
            logger.info(f"Creating target directory: {tdir}")
            os.makedirs(tdir)
    
    # Find all images
    image_paths = []
    for f in os.listdir(source_dir):
        if f.lower().endswith(('.png', '.jpeg', '.jpg', '.gif', '.bmp')):
            full_path = os.path.join(source_dir, f)
            image_paths.append(full_path)
            
            # Extract keywords and add to vocabulary
            keywords, _ = filename_analyzer.extract_keywords(f)
            for keyword in keywords:
                vocabulary.add_word(keyword)
    
    # Save updated vocabulary
    vocabulary.save_vocabulary()
    
    image_paths = sorted(image_paths)
    logger.info(f"Found {len(image_paths)} images in source directory.")
    
    # Track initial statistics
    analytics.session_data['initial_stats'] = {
        'total_images': len(image_paths),
        'unique_keywords': len(vocabulary.word_to_idx),
        'timestamp': datetime.now().isoformat()
    }
    
    if not image_paths:
        logger.info("No images found.")
        messagebox.showinfo("No Images", "No images found in the source directory.")
        analytics.close()
        return
    
    # Initialize Pygame
    logger.info("Initializing Pygame.")
    pygame.init()
    
    # Start with a resizable window
    screen_width = 1200
    screen_height = 600
    min_width = 800  # Minimum window width
    min_height = 500  # Minimum window height
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    pygame.display.set_caption("Full Analytics Image Sorter - Resizable")
    
    # Dynamic font scaling function
    def get_scaled_fonts(width, height):
        base_font_size = max(16, min(36, width // 40))
        small_font_size = max(12, min(24, width // 60))
        return pygame.font.SysFont(None, base_font_size), pygame.font.SysFont(None, small_font_size)
    
    font, small_font = get_scaled_fonts(screen_width, screen_height)
    
    # Dynamic UI layout class
    class UILayout:
        def __init__(self, width, height, target_count):
            self.width = max(min_width, width)
            self.height = max(min_height, height)
            self.target_count = target_count
            self.update_layout()
        
        def update_layout(self):
            # Ensure minimum content fits - scale based on window size
            base_scale = min(self.width / 1200, self.height / 600)  # Reference size
            
            # Header area for stats - responsive but with minimum
            self.header_height = max(60, int(80 * base_scale))
            
            # Button area - scales with window but maintains usability
            self.button_height = max(35, min(100, int(60 * base_scale)))
            
            # Auto mode button - better responsive sizing
            auto_button_width = max(80, min(180, int(self.width * 0.12)))
            auto_button_height = max(25, min(50, int(self.height * 0.06)))
            self.auto_rect = pygame.Rect(self.width - auto_button_width - 10, 5, 
                                       auto_button_width, auto_button_height)
            
            # Column layout for images
            self.col_width = self.width // 3
            
            # Calculate available space more carefully
            reserved_space = self.header_height + self.button_height + 40  # margins
            self.image_area_height = max(200, self.height - reserved_space)
            
            # Target directory buttons - ensure they fit
            self.button_width = max(60, self.width // max(1, self.target_count))
            self.buttons = []
            for idx in range(self.target_count):
                rect = pygame.Rect(idx * self.button_width, 
                                 self.height - self.button_height, 
                                 self.button_width, 
                                 self.button_height)
                self.buttons.append(rect)
            
            # Image slot positioning - more responsive
            self.image_start_y = self.header_height + 10
            
            # Calculate image area more intelligently
            available_width = self.col_width - 20  # margins
            text_space = int(80 * base_scale)  # space for text below image
            available_height = self.image_area_height - text_space
            
            self.max_image_width = max(100, available_width * 0.9)
            self.max_image_height = max(80, available_height * 0.85)
            
            # Text areas within each slot
            self.text_area_height = text_space
    
    # Initialize UI layout
    ui_layout = UILayout(screen_width, screen_height, len(target_dirs))
    
    # Setup initial buttons with proper labels
    buttons = []
    for idx, tdir in enumerate(target_dirs):
        rect = ui_layout.buttons[idx]
        label = os.path.basename(tdir)
        if len(target_dirs) <= 9:
            label += f" ({idx + 1})"
        buttons.append((rect, tdir, label))
    
    dir_keys = [pygame.K_1 + i for i in range(min(9, len(target_dirs)))] if len(target_dirs) <= 9 else []
    
    image_keys = [pygame.K_a, pygame.K_s, pygame.K_d]
    key_labels = ['A', 'S', 'D']
    auto_mode = False
    
    # Set up ML components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Track hardware info
    analytics.session_data['hardware'] = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create multi-modal model with analytics
    num_classes = len(target_dirs)
    model = MultiModalImageClassifier(num_classes, analytics=analytics)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Load existing model if available
    model_path = 'full_analytics_image_sorter_model.pth'
    epoch_start = 0
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if checkpoint.get('num_classes') == num_classes and checkpoint.get('vocab_size') == len(vocabulary.word_to_idx):
            model.load_state_dict(checkpoint['state_dict'])
            epoch_start = checkpoint.get('epoch', 0)
            logger.info(f"Loaded existing model from epoch {epoch_start}")
        else:
            logger.info("Model configuration mismatch. Starting with a new model.")
    
    # Initialize image queue and slots
    image_queue = list(image_paths)
    current_slots = [None, None, None]
    current_keywords = [[], [], []]  # Initialize with empty lists instead of None
    slot_load_times = [0.0, 0.0, 0.0]  # Initialize with float values
    
    # Load initial images
    for slot in range(3):
        if image_queue:
            img_path = image_queue.pop(0)
            current_slots[slot] = img_path
            keywords, _ = filename_analyzer.extract_keywords(os.path.basename(img_path))
            current_keywords[slot] = keywords
            slot_load_times[slot] = time.time()
            logger.info(f"Loaded initial image into slot {slot}: {img_path}")
    
    selected_slot = None
    decision_count = 0
    correct_predictions = 0
    
    # Performance tracking
    model_accuracy_history = []
    decision_times = []
    
    def draw():
        """Enhanced draw function with properly responsive UI"""
        nonlocal screen_width, screen_height, ui_layout, buttons, font, small_font, auto_mode
        
        # Check if window was resized
        current_width, current_height = screen.get_size()
        if current_width != screen_width or current_height != screen_height:
            # Window was resized, update everything
            screen_width = current_width
            screen_height = current_height
            
            # Update UI layout
            ui_layout = UILayout(screen_width, screen_height, len(target_dirs))
            
            # Update fonts
            font, small_font = get_scaled_fonts(screen_width, screen_height)
            
            # Rebuild buttons with new dimensions
            buttons = []
            for idx, tdir in enumerate(target_dirs):
                rect = ui_layout.buttons[idx]
                label = os.path.basename(tdir)
                if len(target_dirs) <= 9:
                    label += f" ({idx + 1})"
                buttons.append((rect, tdir, label))
        
        screen.fill((0, 0, 0))
        
        # Draw analytics stats in header
        stats_text = f"Decisions: {decision_count} | Accuracy: {correct_predictions/max(1,decision_count)*100:.1f}%"
        stats_surface = small_font.render(stats_text, True, (100, 200, 100))
        screen.blit(stats_surface, (10, 10))
        
        # Draw window size info
        size_text = f"Window: {screen_width}x{screen_height}"
        size_surface = small_font.render(size_text, True, (150, 150, 150))
        screen.blit(size_surface, (10, 30))
        
        # Draw queue info
        queue_text = f"Remaining: {len(image_queue)}"
        queue_surface = small_font.render(queue_text, True, (200, 200, 100))
        screen.blit(queue_surface, (10, 50))
        
        # Draw each image slot with proper responsive positioning
        for slot in range(3):
            if current_slots[slot] is not None:  # Check for None explicitly
                try:
                    # Load and display image
                    img_path = current_slots[slot]
                    img = pygame.image.load(img_path)
                    img_rect = img.get_rect()
                    
                    # Calculate responsive scale
                    scale = min(ui_layout.max_image_width / img_rect.w, 
                              ui_layout.max_image_height / img_rect.h, 1)
                    
                    scaled_w = int(img_rect.w * scale)
                    scaled_h = int(img_rect.h * scale)
                    scaled_image = pygame.transform.scale(img, (scaled_w, scaled_h))
                    
                    # Center image in its column
                    x = slot * ui_layout.col_width + (ui_layout.col_width - scaled_w) // 2
                    y = ui_layout.image_start_y
                    
                    screen.blit(scaled_image, (x, y))
                    
                    # Calculate text positions below image with proper spacing
                    text_start_y = y + scaled_h + 5
                    text_line_height = max(12, small_font.get_height())
                    
                    # Draw filename with proper truncation
                    if img_path:  # Check if img_path is not None
                        filename = os.path.basename(img_path)
                        max_filename_chars = max(8, ui_layout.col_width // 12)
                        if len(filename) > max_filename_chars:
                            display_filename = filename[:max_filename_chars-3] + "..."
                        else:
                            display_filename = filename
                        
                        filename_text = small_font.render(display_filename, True, (200, 200, 200))
                        # Center filename text
                        filename_x = x + max(0, (scaled_w - filename_text.get_width()) // 2)
                        # Ensure text doesn't go below button area
                        if text_start_y < ui_layout.height - ui_layout.button_height - 80:
                            screen.blit(filename_text, (filename_x, text_start_y))
                    
                    # Draw keywords
                    current_y = text_start_y + text_line_height
                    if current_keywords[slot] and current_y < ui_layout.height - ui_layout.button_height - 60:
                        kw_text = ", ".join(current_keywords[slot][:2])  # Show fewer keywords
                        if len(current_keywords[slot]) > 2:
                            kw_text += "..."
                        
                        # Truncate keywords if too long
                        max_kw_chars = max(10, ui_layout.col_width // 10)
                        if len(kw_text) > max_kw_chars:
                            kw_text = kw_text[:max_kw_chars-3] + "..."
                        
                        kw_surface = small_font.render(f"Keys: {kw_text}", True, (150, 150, 255))
                        kw_x = x + max(0, (scaled_w - kw_surface.get_width()) // 2)
                        screen.blit(kw_surface, (kw_x, current_y))
                    
                    # Draw key label
                    current_y += text_line_height
                    if current_y < ui_layout.height - ui_layout.button_height - 40:
                        text_color = (255, 0, 0) if selected_slot == slot else (255, 255, 255)
                        key_text = font.render(key_labels[slot], True, text_color)
                        key_x = x + (ui_layout.col_width - key_text.get_width()) // 2
                        screen.blit(key_text, (key_x, current_y))
                    
                    # Show predictions if available
                    current_y += max(20, font.get_height())
                    if (current_keywords[slot] and not auto_mode and 
                        current_y < ui_layout.height - ui_layout.button_height - 20):
                        predictions = learning_history.get_keyword_predictions(current_keywords[slot])
                        if predictions:
                            top_folder = max(predictions.items(), key=lambda x: x[1])
                            folder_name = top_folder[0]
                            
                            # Truncate folder name if needed
                            max_folder_chars = max(8, ui_layout.col_width // 12)
                            if len(folder_name) > max_folder_chars:
                                folder_name = folder_name[:max_folder_chars-3] + "..."
                            
                            pred_text = small_font.render(f"→ {folder_name}", True, (100, 255, 100))
                            pred_x = x + max(0, (scaled_w - pred_text.get_width()) // 2)
                            screen.blit(pred_text, (pred_x, current_y))
                    
                except Exception as e:
                    logger.error(f"Error loading image for display {img_path}: {e}")
                    current_slots[slot] = None
                    current_keywords[slot] = []  # Reset to empty list instead of None
        
        # Draw folder buttons with responsive text
        for idx, (rect, _, label) in enumerate(buttons):
            pygame.draw.rect(screen, (0, 255, 0), rect)
            
            # Scale button text to fit
            button_text = font.render(label, True, (0, 0, 0))
            if button_text.get_width() > rect.width - 10:
                # Text too wide, try smaller font or truncate
                smaller_font = pygame.font.SysFont(None, max(12, font.get_height() - 4))
                button_text = smaller_font.render(label, True, (0, 0, 0))
                if button_text.get_width() > rect.width - 10:
                    # Still too wide, truncate
                    max_chars = max(3, len(label) * (rect.width - 10) // button_text.get_width())
                    truncated = label[:max_chars-3] + "..." if max_chars < len(label) else label
                    button_text = smaller_font.render(truncated, True, (0, 0, 0))
            
            text_rect = button_text.get_rect(center=rect.center)
            screen.blit(button_text, text_rect)
        
        # Draw auto mode button/indicator with responsive sizing
        if not auto_mode:
            pygame.draw.rect(screen, (0, 0, 255), ui_layout.auto_rect)
            auto_text = small_font.render("Auto Mode", True, (255, 255, 255))
            if auto_text.get_width() > ui_layout.auto_rect.width - 10:
                auto_text = small_font.render("Auto", True, (255, 255, 255))
            auto_text_rect = auto_text.get_rect(center=ui_layout.auto_rect.center)
            screen.blit(auto_text, auto_text_rect)
        else:
            auto_text = small_font.render("Auto Active", True, (255, 0, 0))
            screen.blit(auto_text, (screen_width - auto_text.get_width() - 10, 10))
        
        pygame.display.flip()
    
    def move_slot_to_dir(slot_idx, dir_idx, is_manual=True):
        """Enhanced move function with comprehensive analytics"""
        nonlocal decision_count, correct_predictions
        
        path = current_slots[slot_idx]
        keywords = current_keywords[slot_idx]
        
        if path is not None:  # Check for None explicitly
            # Calculate decision time
            decision_time = time.time() - slot_load_times[slot_idx]
            decision_times.append(decision_time)
            
            # Record the decision
            folder_name = target_dir_names[dir_idx]
            learning_history.record_decision(os.path.basename(path), keywords, folder_name)
            
            # Get model prediction for comparison
            confidence = 0.0
            model_output = None
            
            try:
                # Prepare inputs
                img = Image.open(path).convert('RGB')
                input_tensor = preprocess(img).unsqueeze(0).to(device)
                text_indices = vocabulary.encode_keywords(keywords)
                text_tensor = torch.tensor([text_indices]).to(device)
                
                # Get model prediction
                model.eval()
                with torch.no_grad():
                    model_output = model(input_tensor, text_tensor, save_features=True, image_path=path)
                    probs = torch.softmax(model_output, dim=1)
                    predicted_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, dir_idx].item()
                    
                    # Track if prediction was correct
                    if predicted_idx == dir_idx:
                        correct_predictions += 1
                
                # Train if manual
                if is_manual:
                    label = torch.tensor([dir_idx]).to(device)
                    model.train()
                    optimizer.zero_grad()
                    output = model(input_tensor, text_tensor)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    
                    # Track training metrics
                    analytics.record_model_performance(
                        accuracy=correct_predictions / max(1, decision_count),
                        precision={folder_name: confidence},
                        recall={folder_name: confidence},
                        f1_score={folder_name: confidence},
                        loss=loss.item(),
                        learning_rate=optimizer.param_groups[0]['lr'],
                        epoch=epoch_start + decision_count // len(image_paths)
                    )
                    
                    logger.info(f"Trained on {path} (keywords: {keywords}) with label {dir_idx}, "
                              f"loss: {loss.item():.4f}, confidence: {confidence:.4f}")
                    
            except Exception as e:
                logger.error(f"Error in model processing: {e}")
            
            # Record comprehensive decision analytics
            analytics.record_decision(
                filename=os.path.basename(path),
                source_path=path,
                target_folder=folder_name,
                keywords=keywords,
                is_manual=is_manual,
                confidence=confidence,
                decision_time=decision_time,
                model_output=model_output
            )
            
            # Move the file
            tdir = target_dirs[dir_idx]
            filename = os.path.basename(path)
            target_path = os.path.join(tdir, filename)
            
            logger.info(f"Moving {path} to {target_path}")
            Thread(target=shutil.move, args=(path, target_path)).start()
            
            # Update counts
            decision_count += 1
            
            # Clear slot and load next
            current_slots[slot_idx] = None
            current_keywords[slot_idx] = []  # Reset to empty list
            slot_load_times[slot_idx] = 0.0  # Reset to 0.0
            
            if image_queue:
                next_path = image_queue.pop(0)
                current_slots[slot_idx] = next_path
                keywords, _ = filename_analyzer.extract_keywords(os.path.basename(next_path))
                current_keywords[slot_idx] = keywords
                slot_load_times[slot_idx] = time.time()
                logger.info(f"Loaded next image into slot {slot_idx}: {next_path}")
    
    # Main game loop
    running = True
    while running:
        if not any(current_slots) and not image_queue:
            logger.info("All images processed.")
            break
        
        draw()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            
            # Handle window resizing
            if event.type == pygame.VIDEORESIZE:
                screen_width, screen_height = event.w, event.h
                screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                logger.info(f"Window resized to {screen_width}x{screen_height}")
                continue
            
            if event.type == pygame.KEYDOWN:
                if auto_mode:
                    continue
                
                if selected_slot is None:
                    for idx, key in enumerate(image_keys):
                        if event.key == key:
                            if current_slots[idx] is not None:  # Check for None explicitly
                                selected_slot = idx
                                logger.info(f"Selected slot {idx} via keyboard.")
                            break
                else:
                    if event.key in dir_keys:
                        dir_idx = dir_keys.index(event.key)
                        move_slot_to_dir(selected_slot, dir_idx, is_manual=True)
                        selected_slot = None
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                if auto_mode:
                    continue
                
                if selected_slot is None:
                    # Check auto mode button
                    if ui_layout.auto_rect.collidepoint(pos):
                        auto_mode = True
                        logger.info("Enabled auto mode.")
                        analytics.session_data['user_behavior']['auto_mode_activations'].append(
                            time.time())
                        continue
                    
                    # Check image slots
                    for slot in range(3):
                        slot_start_x = slot * ui_layout.col_width
                        slot_end_x = slot_start_x + ui_layout.col_width
                        
                        if (slot_start_x <= pos[0] < slot_end_x and 
                            ui_layout.image_start_y <= pos[1] < ui_layout.height - ui_layout.button_height):
                            if current_slots[slot] is not None:  # Check for None explicitly
                                selected_slot = slot
                                logger.info(f"Selected slot {slot} via mouse.")
                                break
                else:
                    # Check folder buttons
                    for btn_idx, (rect, _, _) in enumerate(buttons):
                        if rect.collidepoint(pos):
                            move_slot_to_dir(selected_slot, btn_idx, is_manual=True)
                            selected_slot = None
                            break
        
        # Auto mode processing
        if auto_mode:
            for slot in range(3):
                if current_slots[slot] is not None:  # Check for None explicitly
                    try:
                        img_path = current_slots[slot]
                        keywords = current_keywords[slot]
                        
                        # Prepare inputs
                        img = Image.open(img_path).convert('RGB')
                        input_tensor = preprocess(img).unsqueeze(0).to(device)
                        text_indices = vocabulary.encode_keywords(keywords)
                        text_tensor = torch.tensor([text_indices]).to(device)
                        
                        # Get prediction
                        model.eval()
                        with torch.no_grad():
                            output = model(input_tensor, text_tensor, save_features=True, image_path=img_path)
                        
                        # Combine with keyword predictions
                        keyword_predictions = learning_history.get_keyword_predictions(keywords)
                        
                        # Convert model output to probabilities
                        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                        
                        # Combine predictions
                        final_scores = probs.copy()
                        for folder_name, score in keyword_predictions.items():
                            if folder_name in target_dir_names:
                                idx = target_dir_names.index(folder_name)
                                final_scores[idx] += score * 0.1
                        
                        pred = np.argmax(final_scores)
                        
                        move_slot_to_dir(slot, pred, is_manual=False)
                        logger.info(f"Auto sorted {img_path} to directory index {pred}")
                        
                    except Exception as e:
                        logger.error(f"Error in auto sorting for slot {slot}: {e}")
                        current_slots[slot] = None
                        current_keywords[slot] = []  # Reset to empty list
    
    # Save everything
    learning_history.save_history()
    vocabulary.save_vocabulary()
    
    # Save model with analytics
    torch.save({
        'num_classes': num_classes,
        'vocab_size': len(vocabulary.word_to_idx),
        'state_dict': model.state_dict(),
        'epoch': epoch_start + decision_count // len(image_paths),
        'accuracy': correct_predictions / max(1, decision_count),
        'total_decisions': decision_count
    }, model_path)
    logger.info("Saved model with analytics.")
    
    # Final analytics summary
    analytics.session_data['final_stats'] = {
        'total_decisions': decision_count,
        'final_accuracy': correct_predictions / max(1, decision_count),
        'average_decision_time': np.mean(decision_times) if decision_times else 0,
        'total_duration': time.time() - time.mktime(
            datetime.fromisoformat(analytics.session_data['start_time']).timetuple())
    }
    
    # Show completion
    screen.fill((0, 0, 0))
    text = font.render("All images sorted!", True, (255, 255, 255))
    text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2 - 30))
    screen.blit(text, text_rect)
    
    # Show final stats
    stats_text = f"Total: {decision_count} | Accuracy: {correct_predictions/max(1,decision_count)*100:.1f}%"
    stats_surface = font.render(stats_text, True, (100, 255, 100))
    stats_rect = stats_surface.get_rect(center=(screen_width // 2, screen_height // 2 + 30))
    screen.blit(stats_surface, stats_rect)
    
    pygame.display.flip()
    logger.info("Displayed completion message.")
    
    # Wait before closing
    waiting = True
    start_time = pygame.time.get_ticks()
    
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
        if pygame.time.get_ticks() - start_time > 5000:
            waiting = False
    
    pygame.quit()
    
    # Close analytics (generates all reports)
    analytics.close()
    
    logger.info(f"Application closed. Analytics saved to {analytics.analytics_dir}")

if __name__ == "__main__":
    main()