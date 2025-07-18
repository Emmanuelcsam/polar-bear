# Enhanced Image Sorter with Multi-Modal Learning
# This program learns from BOTH image content AND filename patterns

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
from datetime import datetime
import pickle
from collections import defaultdict
import numpy as np

# Function to install missing packages
def install_if_missing(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure all required packages are installed
install_if_missing('pygame')
install_if_missing('torch')
install_if_missing('torchvision')
install_if_missing('pillow')
install_if_missing('numpy')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_image_sorter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FilenameAnalyzer:
    """Analyzes filenames to extract meaningful keywords and patterns"""
    
    def __init__(self):
        # Common separators in filenames
        self.separators = ['-', '_', ' ', '.', '(', ')', '[', ']']
        # Common words to ignore
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                          'img', 'image', 'photo', 'pic', 'picture', 'file', 'copy', 'new'}
        # Pattern for detecting dates
        self.date_pattern = re.compile(r'\d{4}[-_]?\d{2}[-_]?\d{2}|\d{2}[-_]?\d{2}[-_]?\d{4}')
        # Pattern for detecting numbers
        self.number_pattern = re.compile(r'\d+')
    
    def extract_keywords(self, filename):
        """Extract meaningful keywords from a filename"""
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Handle empty filename or just extension
        if not name_without_ext or name_without_ext == '':
            return [], []
        
        # Convert to lowercase for analysis
        name_lower = name_without_ext.lower()
        
        # Replace separators with spaces
        for sep in self.separators:
            name_lower = name_lower.replace(sep, ' ')
        
        # Extract dates
        dates = self.date_pattern.findall(name_without_ext)
        
        # Remove dates from the string for keyword extraction
        for date in dates:
            name_lower = name_lower.replace(date.lower(), '')
        
        # Split into words
        words = name_lower.split()
        
        # Filter out stop words and very short words
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Remove pure numbers (but keep alphanumeric)
        keywords = [k for k in keywords if not k.isdigit()]
        
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
    """Neural network that combines image features and filename features"""
    
    def __init__(self, num_classes, image_feature_dim=512, text_feature_dim=128):
        super(MultiModalImageClassifier, self).__init__()
        
        # Image feature extractor (using ResNet18 backbone)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final classification layer
        self.resnet.fc = nn.Identity()
        
        # Text feature processor
        self.text_embedding = nn.Embedding(10000, 64)  # Vocabulary size of 10000
        self.text_lstm = nn.LSTM(64, text_feature_dim, batch_first=True)
        
        # Fusion layers
        self.fusion_layer = nn.Linear(512 + text_feature_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, image, text_indices):
        # Extract image features
        image_features = self.resnet(image)
        
        # Process text features
        text_embedded = self.text_embedding(text_indices)
        _, (text_features, _) = self.text_lstm(text_embedded)
        text_features = text_features.squeeze(0)
        
        # Combine features
        combined = torch.cat([image_features, text_features], dim=1)
        
        # Final classification
        fused = self.relu(self.fusion_layer(combined))
        fused = self.dropout(fused)
        output = self.classifier(fused)
        
        return output

class LearningHistory:
    """Maintains a persistent history of learning decisions"""
    
    def __init__(self, history_file='learning_history.pkl'):
        self.history_file = history_file
        self.keyword_to_folder = defaultdict(lambda: defaultdict(int))
        self.folder_keywords = defaultdict(set)
        self.decisions = []
        self.load_history()
    
    def load_history(self):
        """Load existing history from disk"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'rb') as f:
                    data = pickle.load(f)
                    self.keyword_to_folder = defaultdict(lambda: defaultdict(int), data['keyword_to_folder'])
                    self.folder_keywords = defaultdict(set, data['folder_keywords'])
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
            self.keyword_to_folder[keyword][folder_name] += 1
            self.folder_keywords[folder_name].add(keyword)
        
        # Keep only last 10000 decisions to prevent unbounded growth
        if len(self.decisions) > 10000:
            self.decisions = self.decisions[-10000:]
    
    def get_keyword_predictions(self, keywords):
        """Get folder predictions based on keywords"""
        folder_scores = defaultdict(float)
        
        for keyword in keywords:
            if keyword in self.keyword_to_folder:
                for folder, count in self.keyword_to_folder[keyword].items():
                    folder_scores[folder] += count
        
        return folder_scores

class Vocabulary:
    """Manages vocabulary for text encoding"""
    
    def __init__(self, vocab_file='vocabulary.json'):
        self.vocab_file = vocab_file
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
    """Enhanced main function with multi-modal learning"""
    logger.info("Starting enhanced image sorter application.")
    
    # Initialize components
    filename_analyzer = FilenameAnalyzer()
    learning_history = LearningHistory()
    vocabulary = Vocabulary()
    
    # Create hidden tkinter window for dialogs
    root = tk.Tk()
    root.withdraw()
    
    # Select source directory
    logger.info("Prompting user to select source directory.")
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        logger.info("No source directory selected. Exiting.")
        return
    
    logger.info(f"Source directory selected: {source_dir}")
    
    # Select target directories
    target_dirs = []
    target_dir_names = []  # Store just the folder names
    
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
        return
    
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
    
    if not image_paths:
        logger.info("No images found.")
        messagebox.showinfo("No Images", "No images found in the source directory.")
        return
    
    # Initialize Pygame
    logger.info("Initializing Pygame.")
    pygame.init()
    screen_width = 1200
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Enhanced Image Sorter")
    
    font = pygame.font.SysFont(None, 30)
    small_font = pygame.font.SysFont(None, 20)
    
    # Set up UI elements
    button_height = 50
    button_width = screen_width // max(1, len(target_dirs))
    
    buttons = []
    for idx, tdir in enumerate(target_dirs):
        rect = pygame.Rect(idx * button_width, screen_height - button_height, button_width, button_height)
        label = os.path.basename(tdir)
        if len(target_dirs) <= 9:
            label += f" ({idx + 1})"
        buttons.append((rect, tdir, label))
    
    dir_keys = [pygame.K_1 + i for i in range(min(9, len(target_dirs)))] if len(target_dirs) <= 9 else []
    
    col_width = screen_width // 3
    image_keys = [pygame.K_a, pygame.K_s, pygame.K_d]
    key_labels = ['A', 'S', 'D']
    avail_height = screen_height - button_height - 60  # Extra space for filename display
    
    auto_rect = pygame.Rect(screen_width - 150, 0, 150, 50)
    auto_mode = False
    
    # Set up ML components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create multi-modal model
    num_classes = len(target_dirs)
    model = MultiModalImageClassifier(num_classes)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Load existing model if available
    model_path = 'enhanced_image_sorter_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if checkpoint.get('num_classes') == num_classes and checkpoint.get('vocab_size') == len(vocabulary.word_to_idx):
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded existing enhanced model for continued learning.")
        else:
            logger.info("Model configuration mismatch. Starting with a new model.")
    
    # Initialize image queue and slots
    image_queue = list(image_paths)
    current_slots = [None, None, None]
    current_keywords = [None, None, None]  # Store keywords for each slot
    
    # Load initial images
    for slot in range(3):
        if image_queue:
            img_path = image_queue.pop(0)
            current_slots[slot] = img_path
            keywords, _ = filename_analyzer.extract_keywords(os.path.basename(img_path))
            current_keywords[slot] = keywords
            logger.info(f"Loaded initial image into slot {slot}: {img_path} with keywords {keywords}")
    
    selected_slot = None
    
    def draw():
        """Enhanced draw function that shows filenames and predictions"""
        screen.fill((0, 0, 0))
        
        # Draw each image slot
        for slot in range(3):
            if current_slots[slot]:
                try:
                    # Load and display image
                    img_path = current_slots[slot]
                    img = pygame.image.load(img_path)
                    img_rect = img.get_rect()
                    
                    scale = min(col_width * 0.9 / img_rect.w, (avail_height - 40) / img_rect.h, 1)
                    scaled_w = int(img_rect.w * scale)
                    scaled_h = int(img_rect.h * scale)
                    scaled_image = pygame.transform.scale(img, (scaled_w, scaled_h))
                    
                    x = slot * col_width + (col_width - scaled_w) // 2
                    y = 10
                    
                    screen.blit(scaled_image, (x, y))
                    
                    # Draw filename
                    filename = os.path.basename(img_path)
                    filename_text = small_font.render(filename[:30] + "..." if len(filename) > 30 else filename, 
                                                    True, (200, 200, 200))
                    filename_pos = (x, y + scaled_h + 5)
                    screen.blit(filename_text, filename_pos)
                    
                    # Draw key label
                    text_color = (255, 0, 0) if selected_slot == slot else (255, 255, 255)
                    text = font.render(key_labels[slot], True, text_color)
                    text_pos = (x + (col_width - text.get_width()) // 2, y + scaled_h + 25)
                    screen.blit(text, text_pos)
                    
                    # Show keyword predictions if available
                    if current_keywords[slot] and not auto_mode:
                        predictions = learning_history.get_keyword_predictions(current_keywords[slot])
                        if predictions:
                            # Show top prediction
                            top_folder = max(predictions.items(), key=lambda x: x[1])
                            pred_text = small_font.render(f"Suggested: {top_folder[0]}", True, (100, 255, 100))
                            pred_pos = (x, y + scaled_h + 45)
                            screen.blit(pred_text, pred_pos)
                    
                except Exception as e:
                    logger.error(f"Error loading image for display {img_path}: {e}")
                    current_slots[slot] = None
                    current_keywords[slot] = None
        
        # Draw folder buttons
        for rect, _, label in buttons:
            pygame.draw.rect(screen, (0, 255, 0), rect)
            text = font.render(label, True, (0, 0, 0))
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
        
        # Draw auto mode button/indicator
        if not auto_mode:
            pygame.draw.rect(screen, (0, 0, 255), auto_rect)
            text = font.render("Auto Mode", True, (255, 255, 255))
            text_rect = text.get_rect(center=auto_rect.center)
            screen.blit(text, text_rect)
        else:
            text = font.render("Auto Active", True, (255, 0, 0))
            screen.blit(text, (screen_width - 140, 10))
        
        pygame.display.flip()
    
    def move_slot_to_dir(slot_idx, dir_idx, is_manual=True):
        """Enhanced move function with multi-modal learning"""
        path = current_slots[slot_idx]
        keywords = current_keywords[slot_idx]
        
        if path:
            # Record the decision
            folder_name = target_dir_names[dir_idx]
            learning_history.record_decision(os.path.basename(path), keywords, folder_name)
            
            # Train the model if manual
            if is_manual:
                try:
                    # Prepare image
                    img = Image.open(path).convert('RGB')
                    input_tensor = preprocess(img).unsqueeze(0).to(device)
                    
                    # Prepare text (keywords)
                    text_indices = vocabulary.encode_keywords(keywords)
                    text_tensor = torch.tensor([text_indices]).to(device)
                    
                    # Create label
                    label = torch.tensor([dir_idx]).to(device)
                    
                    # Train
                    model.train()
                    optimizer.zero_grad()
                    output = model(input_tensor, text_tensor)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    
                    logger.info(f"Trained on {path} (keywords: {keywords}) with label {dir_idx}, loss: {loss.item()}")
                except Exception as e:
                    logger.error(f"Error in training: {e}")
            
            # Move the file
            tdir = target_dirs[dir_idx]
            filename = os.path.basename(path)
            target_path = os.path.join(tdir, filename)
            
            logger.info(f"Moving {path} to {target_path}")
            Thread(target=shutil.move, args=(path, target_path)).start()
            
            # Clear slot and load next
            current_slots[slot_idx] = None
            current_keywords[slot_idx] = None
            
            if image_queue:
                next_path = image_queue.pop(0)
                current_slots[slot_idx] = next_path
                keywords, _ = filename_analyzer.extract_keywords(os.path.basename(next_path))
                current_keywords[slot_idx] = keywords
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
            
            if event.type == pygame.KEYDOWN:
                if auto_mode:
                    continue
                
                if selected_slot is None:
                    for idx, key in enumerate(image_keys):
                        if event.key == key:
                            if current_slots[idx]:
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
                    if auto_rect.collidepoint(pos):
                        auto_mode = True
                        logger.info("Enabled auto mode.")
                        continue
                    
                    for slot in range(3):
                        slot_start_x = slot * col_width
                        slot_end_x = slot_start_x + col_width
                        
                        if slot_start_x <= pos[0] < slot_end_x and 0 <= pos[1] < screen_height - button_height:
                            if current_slots[slot]:
                                selected_slot = slot
                                logger.info(f"Selected slot {slot} via mouse.")
                                break
                else:
                    for btn_idx, (rect, _, _) in enumerate(buttons):
                        if rect.collidepoint(pos):
                            move_slot_to_dir(selected_slot, btn_idx, is_manual=True)
                            selected_slot = None
                            break
        
        # Enhanced auto mode with multi-modal predictions
        if auto_mode:
            for slot in range(3):
                if current_slots[slot]:
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
                            output = model(input_tensor, text_tensor)
                        
                        # Combine with keyword predictions
                        keyword_predictions = learning_history.get_keyword_predictions(keywords)
                        
                        # Convert model output to probabilities
                        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                        
                        # Combine predictions (weighted average)
                        final_scores = probs.copy()
                        for folder_name, score in keyword_predictions.items():
                            if folder_name in target_dir_names:
                                idx = target_dir_names.index(folder_name)
                                # Boost score based on keyword history
                                final_scores[idx] += score * 0.1
                        
                        pred = np.argmax(final_scores)
                        
                        move_slot_to_dir(slot, pred, is_manual=False)
                        logger.info(f"Auto sorted {img_path} to directory index {pred}")
                        
                    except Exception as e:
                        logger.error(f"Error in auto sorting for slot {slot}: {e}")
                        current_slots[slot] = None
                        current_keywords[slot] = None
    
    # Save everything
    learning_history.save_history()
    
    torch.save({
        'num_classes': num_classes,
        'vocab_size': len(vocabulary.word_to_idx),
        'state_dict': model.state_dict()
    }, model_path)
    logger.info("Saved enhanced model for future runs.")
    
    # Show completion
    screen.fill((0, 0, 0))
    text = font.render("All images sorted!", True, (255, 255, 255))
    text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(text, text_rect)
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
    logger.info("Application closed.")

if __name__ == "__main__":
    main()