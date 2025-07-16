import sys
import os
import shutil
import subprocess
import importlib
import logging
from threading import Thread
from typing import Optional, List
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim

def install_if_missing(package):
    try:
        importlib.import_module(package)
    except ImportError:
        logging.info(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_if_missing('pygame')
install_if_missing('torch')
install_if_missing('torchvision')
install_if_missing('pillow')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_sorter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting image sorter application.")
    root = tk.Tk()
    root.withdraw()
    logger.info("Prompting user to select source directory.")
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        logger.info("No source directory selected. Exiting.")
        return

    logger.info(f"Source directory selected: {source_dir}")

    target_dirs = []
    while True:
        tdir = filedialog.askdirectory(title="Select a Target Directory (Cancel when all added)")
        if not tdir:
            logger.info("Target directory selection completed.")
            break
        if tdir not in target_dirs:
            target_dirs.append(tdir)
            logger.info(f"Added target directory: {tdir}")

    if not target_dirs:
        logger.error("No target directories selected.")
        messagebox.showerror("Error", "No target directories selected.")
        return

    for tdir in target_dirs:
        if not os.path.exists(tdir):
            logger.info(f"Creating target directory: {tdir}")
            os.makedirs(tdir)

    image_paths = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpeg', '.jpg', '.gif', '.bmp'))])
    logger.info(f"Found {len(image_paths)} images in source directory.")
    if not image_paths:
        logger.info("No images found.")
        messagebox.showinfo("No Images", "No images found in the source directory.")
        return

    logger.info("Initializing Pygame.")
    pygame.init()
    screen_width = 1200
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    pygame.display.set_caption("Image Sorter - Resizable")

    font = pygame.font.SysFont(None, 30)

    # Initialize layout variables
    button_height = 50
    button_width = screen_width // max(1, len(target_dirs))
    buttons = []
    col_width = screen_width // 3
    avail_height = screen_height - button_height - 30
    auto_rect = pygame.Rect(screen_width - 150, 0, 150, 50)

    # Function to recalculate layout based on current window size
    def recalculate_layout():
        nonlocal button_height, button_width, buttons, col_width, avail_height, auto_rect, font
        
        # Adaptive font size based on window size
        font_size = max(20, min(40, screen_width // 40))
        font = pygame.font.SysFont(None, font_size)
        
        button_height = max(40, min(80, screen_height // 10))  # Adaptive button height
        button_width = screen_width // max(1, len(target_dirs))
        buttons = []
        for idx, tdir in enumerate(target_dirs):
            rect = pygame.Rect(idx * button_width, screen_height - button_height, button_width, button_height)
            label = os.path.basename(tdir)
            if len(target_dirs) <= 9:
                label += f" ({idx + 1})"
            buttons.append((rect, tdir, label))

        col_width = screen_width // 3
        avail_height = screen_height - button_height - 30
        
        # Auto button - adaptive size and position
        auto_button_width = min(150, max(100, screen_width // 8))
        auto_button_height = min(50, max(30, screen_height // 12))
        auto_rect = pygame.Rect(screen_width - auto_button_width - 10, 10, auto_button_width, auto_button_height)

    # Initial layout calculation
    recalculate_layout()

    dir_keys = [pygame.K_1 + i for i in range(min(9, len(target_dirs)))] if len(target_dirs) <= 9 else []
    image_keys = [pygame.K_a, pygame.K_s, pygame.K_d]
    key_labels = ['A', 'S', 'D']

    auto_mode = False

    # ML setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    num_classes = len(target_dirs)
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model_path = 'image_sorter_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if checkpoint.get('num_classes') == num_classes:
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded existing model for continued learning.")
        else:
            logger.info("Number of classes mismatch. Starting with a new model.")

    image_queue = list(image_paths)
    current_slots: List[Optional[str]] = [None, None, None]
    for slot in range(3):
        if image_queue:
            current_slots[slot] = image_queue.pop(0)
            logger.info(f"Loaded initial image into slot {slot}: {current_slots[slot]}")

    selected_slot = None

    def draw():
        screen.fill((0, 0, 0))
        for slot in range(3):
            if current_slots[slot]:
                try:
                    img_path = current_slots[slot]
                    img = pygame.image.load(img_path)
                    img_rect = img.get_rect()
                    scale = min(col_width * 0.9 / img_rect.w, avail_height / img_rect.h, 1)
                    scaled_w = int(img_rect.w * scale)
                    scaled_h = int(img_rect.h * scale)
                    scaled_image = pygame.transform.scale(img, (scaled_w, scaled_h))
                    x = slot * col_width + (col_width - scaled_w) // 2
                    y = 10
                    screen.blit(scaled_image, (x, y))

                    # Draw key label
                    text = font.render(key_labels[slot], True, (255, 0, 0) if selected_slot == slot else (255, 255, 255))
                    text_pos = (x + (col_width - text.get_width()) // 2, y + scaled_h + 5)
                    screen.blit(text, text_pos)
                except Exception as e:
                    logger.error(f"Error loading image for display {img_path}: {e}")
                    current_slots[slot] = None

        for rect, _, label in buttons:
            pygame.draw.rect(screen, (0, 255, 0), rect)
            
            # Truncate label if it's too long for the button
            max_label_width = rect.width - 10  # Leave some padding
            text = font.render(label, True, (0, 0, 0))
            if text.get_width() > max_label_width:
                # Truncate the label
                truncated_label = label
                while font.render(truncated_label + "...", True, (0, 0, 0)).get_width() > max_label_width and len(truncated_label) > 3:
                    truncated_label = truncated_label[:-1]
                truncated_label += "..."
                text = font.render(truncated_label, True, (0, 0, 0))
            
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)

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
        path = current_slots[slot_idx]
        if path:
            if is_manual:
                try:
                    img = Image.open(path).convert('RGB')
                    input_tensor = preprocess(img).unsqueeze(0).to(device)
                    label = torch.tensor([dir_idx]).to(device)
                    optimizer.zero_grad()
                    output = model(input_tensor)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    logger.info(f"Trained on {path} with label {dir_idx}, loss: {loss.item()}")
                except Exception as e:
                    logger.error(f"Error in training: {e}")

            tdir = target_dirs[dir_idx]
            filename = os.path.basename(path)
            target_path = os.path.join(tdir, filename)
            logger.info(f"Moving {path} to {target_path}")
            Thread(target=shutil.move, args=(path, target_path)).start()
            current_slots[slot_idx] = None
            if image_queue:
                next_path = image_queue.pop(0)
                current_slots[slot_idx] = next_path
                logger.info(f"Loaded next image into slot {slot_idx}: {next_path}")

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
            
            if event.type == pygame.VIDEORESIZE:
                # Enforce minimum window size to ensure UI remains usable
                min_width = max(600, len(target_dirs) * 100)  # Minimum width based on number of directories
                min_height = 400
                screen_width = max(min_width, event.w)
                screen_height = max(min_height, event.h)
                
                screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
                recalculate_layout()
                pygame.display.set_caption(f"Image Sorter - Resizable ({screen_width}x{screen_height})")
                logger.info(f"Window resized to {screen_width}x{screen_height}")
                continue

            if event.type == pygame.KEYDOWN:
                if auto_mode:
                    continue  # Skip manual inputs in auto
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
                    continue  # Skip manual inputs in auto
                if selected_slot is None:
                    # Check for auto button
                    if auto_rect.collidepoint(pos):
                        auto_mode = True
                        logger.info("Enabled auto mode.")
                        continue
                    # Check image areas
                    for slot in range(3):
                        slot_start_x = slot * col_width
                        slot_end_x = slot_start_x + col_width
                        if slot_start_x <= pos[0] < slot_end_x and 0 <= pos[1] < screen_height - button_height:
                            if current_slots[slot]:
                                selected_slot = slot
                                logger.info(f"Selected slot {slot} via mouse.")
                                break
                else:
                    # Check dir buttons
                    for btn_idx, (rect, _, _) in enumerate(buttons):
                        if rect.collidepoint(pos):
                            move_slot_to_dir(selected_slot, btn_idx, is_manual=True)
                            selected_slot = None
                            break

        if auto_mode:
            for slot in range(3):
                img_path = current_slots[slot]
                if img_path is not None:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        input_tensor = preprocess(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = model(input_tensor)
                        pred = output.argmax(1).item()
                        move_slot_to_dir(slot, pred, is_manual=False)
                        logger.info(f"Auto sorted {img_path} to directory index {pred}")
                    except Exception as e:
                        logger.error(f"Error in auto sorting for slot {slot}: {e}")
                        current_slots[slot] = None

    # Save model at the end
    torch.save({
        'num_classes': num_classes,
        'state_dict': model.state_dict()
    }, model_path)
    logger.info("Saved model for future runs.")

    # End message
    screen.fill((0, 0, 0))
    text = font.render("All images sorted!", True, (255, 255, 255))
    text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    logger.info("Displayed completion message.")

    # Wait 5 seconds or quit
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
