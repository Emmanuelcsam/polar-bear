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

def install_if_missing(package):
    try:
        importlib.import_module(package)
    except ImportError:
        logging.info(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_if_missing('pygame')

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
        sys.exit(0)
    logger.info(f"Source directory selected: {source_dir}")

    target_dirs = []
    logger.info("Prompting user to select target directories.")
    while True:
        tdir = filedialog.askdirectory(title="Select a Target Directory (Cancel when all added)")
        if not tdir:
            logger.info("Target directory selection completed.")
            break
        if tdir not in target_dirs:
            target_dirs.append(tdir)
            logger.info(f"Added target directory: {tdir}")

    if len(target_dirs) == 0:
        logger.error("No target directories selected.")
        messagebox.showerror("Error", "No target directories selected.")
        sys.exit(1)

    for tdir in target_dirs:
        if not os.path.exists(tdir):
            logger.info(f"Creating target directory: {tdir}")
            os.makedirs(tdir)

    image_paths = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpeg', '.jpg', '.gif', '.bmp'))])
    logger.info(f"Found {len(image_paths)} images in source directory.")
    if not image_paths:
        logger.info("No images found. Exiting.")
        messagebox.showinfo("No Images", "No images found in the source directory.")
        sys.exit(0)

    logger.info("Initializing Pygame.")
    pygame.init()
    screen_width = 1200
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image Sorter")

    font = pygame.font.SysFont(None, 30)

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
    avail_height = screen_height - button_height - 30  # space for label

    image_queue = list(image_paths)
    current_slots = [None] * 3
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
                    text_rect = (x + (col_width - text.get_width()) // 2, y + scaled_h + 5)
                    screen.blit(text, text_rect)
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
                    current_slots[slot] = None  # Remove invalid image

        for rect, _, label in buttons:
            pygame.draw.rect(screen, (0, 255, 0), rect)
            text = font.render(label, True, (0, 0, 0))
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)

        pygame.display.flip()

    def move_slot_to_dir(slot_idx, dir_idx):
        path = current_slots[slot_idx]
        if path:
            tdir = target_dirs[dir_idx]
            filename = os.path.basename(path)
            target_path = os.path.join(tdir, filename)
            logger.info(f"Moving {path} to {target_path}")
            Thread(target=shutil.move, args=(path, target_path)).start()
            current_slots[slot_idx] = None
            # Load next into this slot
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
                logger.info("User quit the application.")
                running = False
                break

            if event.type == pygame.KEYDOWN:
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
                        move_slot_to_dir(selected_slot, dir_idx)
                        selected_slot = None

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if selected_slot is None:
                    # Check click on image area
                    for slot in range(3):
                        slot_start_x = slot * col_width
                        slot_end_x = slot_start_x + col_width
                        if slot_start_x <= pos[0] < slot_end_x and 0 <= pos[1] < screen_height - button_height:
                            if current_slots[slot]:
                                selected_slot = slot
                                logger.info(f"Selected slot {slot} via mouse.")
                                break
                else:
                    # Check click on buttons
                    for btn_idx, (rect, _, _) in enumerate(buttons):
                        if rect.collidepoint(pos):
                            move_slot_to_dir(selected_slot, btn_idx)
                            selected_slot = None
                            break

    # End message
    screen.fill((0, 0, 0))
    text = font.render("All images sorted!", True, (255, 255, 255))
    text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    logger.info("Displayed completion message.")

    # Wait for 5 seconds or quit
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
