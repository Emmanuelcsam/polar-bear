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
# If we don't have it, it will install it for us
def install_if_missing(package):
    try:
        # Try to import the package
        importlib.import_module(package)
    except ImportError:
        # If we can't import it, that means it's not installed
        print(f"Installing missing package: {package}")
        # Use pip to install the package
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Make sure we have all the packages we need
install_if_missing('pygame')  # For showing images
install_if_missing('torch')  # For AI
install_if_missing('torchvision')  # For AI with images
install_if_missing('pillow')  # For opening images

# Set up logging so we can see what our program is doing
# This will save messages to a file and also show them on screen
logging.basicConfig(
    level=logging.INFO,  # Show informational messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format for messages
    handlers=[
        logging.FileHandler('image_sorter.log'),  # Save to a file
        logging.StreamHandler()  # Also show on screen
    ]
)
# Create a logger object we can use to write messages
logger = logging.getLogger(__name__)

# This is our main function where everything happens
def main():
    # Tell the user we're starting
    logger.info("Starting image sorter application.")

    # Create a hidden tkinter window (we only want the dialog boxes)
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Ask the user to pick the folder with images to sort
    logger.info("Prompting user to select source directory.")
    source_dir = filedialog.askdirectory(title="Select Source Directory")

    # If they didn't pick a folder, stop the program
    if not source_dir:
        logger.info("No source directory selected. Exiting.")
        return

    # Tell the user which folder they picked
    logger.info(f"Source directory selected: {source_dir}")

    # Now we need to get the folders where images will be sorted into
    target_dirs = []  # This will store all the target folders

    # Keep asking for folders until the user cancels
    while True:
        # Ask for a folder
        tdir = filedialog.askdirectory(title="Select a Target Directory (Cancel when all added)")

        # If they clicked cancel, we're done adding folders
        if not tdir:
            logger.info("Target directory selection completed.")
            break

        # If this folder isn't already in our list, add it
        if tdir not in target_dirs:
            target_dirs.append(tdir)
            logger.info(f"Added target directory: {tdir}")

    # Check if they selected at least one target folder
    if not target_dirs:
        logger.error("No target directories selected.")
        messagebox.showerror("Error", "No target directories selected.")
        return

    # Make sure all the target folders exist (create them if they don't)
    for tdir in target_dirs:
        if not os.path.exists(tdir):
            logger.info(f"Creating target directory: {tdir}")
            os.makedirs(tdir)

    # Find all the image files in the source folder
    # We look for files ending with common image extensions
    image_paths = []
    for f in os.listdir(source_dir):
        # Check if the file ends with an image extension (case doesn't matter)
        if f.lower().endswith(('.png', '.jpeg', '.jpg', '.gif', '.bmp')):
            # Add the full path to our list
            full_path = os.path.join(source_dir, f)
            image_paths.append(full_path)

    # Sort the list alphabetically
    image_paths = sorted(image_paths)

    # Tell the user how many images we found
    logger.info(f"Found {len(image_paths)} images in source directory.")

    # If we didn't find any images, tell the user and stop
    if not image_paths:
        logger.info("No images found.")
        messagebox.showinfo("No Images", "No images found in the source directory.")
        return

    # Now we set up the pygame window to show images
    logger.info("Initializing Pygame.")
    pygame.init()  # Start pygame

    # Set the window size
    screen_width = 1200  # Window will be 1200 pixels wide
    screen_height = 600  # Window will be 600 pixels tall

    # Create the window
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image Sorter")  # Set the window title

    # Create a font for drawing text
    font = pygame.font.SysFont(None, 30)  # Size 30 font

    # Set up buttons for each target directory at the bottom of screen
    button_height = 50  # Buttons will be 50 pixels tall
    # Calculate button width so they fill the screen width
    button_width = screen_width // max(1, len(target_dirs))

    # Create button information for each target directory
    buttons = []
    for idx, tdir in enumerate(target_dirs):
        # Create a rectangle for the button
        rect = pygame.Rect(idx * button_width, screen_height - button_height, button_width, button_height)
        # Get just the folder name (not the full path)
        label = os.path.basename(tdir)
        # If we have 9 or fewer folders, add a number key hint
        if len(target_dirs) <= 9:
            label += f" ({idx + 1})"
        # Save the button info
        buttons.append((rect, tdir, label))

    # Set up keyboard shortcuts for directories (1-9 keys)
    dir_keys = []
    if len(target_dirs) <= 9:
        # Create a list of key codes for numbers 1-9
        for i in range(min(9, len(target_dirs))):
            dir_keys.append(pygame.K_1 + i)

    # Set up the image display area (3 columns)
    col_width = screen_width // 3  # Each column is 1/3 of screen width

    # Set up keyboard shortcuts for selecting images (A, S, D keys)
    image_keys = [pygame.K_a, pygame.K_s, pygame.K_d]
    key_labels = ['A', 'S', 'D']  # Labels to show under images

    # Calculate available height for images (excluding buttons and labels)
    avail_height = screen_height - button_height - 30

    # Create auto mode button in top right corner
    auto_rect = pygame.Rect(screen_width - 150, 0, 150, 50)

    # Variable to track if auto mode is on
    auto_mode = False

    # Set up machine learning components
    # Check if we have a GPU (graphics card) for faster processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set up image preprocessing (preparing images for AI)
    preprocess = transforms.Compose([
        transforms.Resize(256),  # Make image 256 pixels on shortest side
        transforms.CenterCrop(224),  # Crop to 224x224 pixels from center
        transforms.ToTensor(),  # Convert to tensor (AI format)
        # Normalize pixel values (standard for this AI model)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the AI model (ResNet18 - a type of image classifier)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Modify the model to work with our number of folders
    num_ftrs = model.fc.in_features  # Get number of input features
    num_classes = len(target_dirs)  # Number of folders to sort into
    model.fc = nn.Linear(num_ftrs, num_classes)  # Replace last layer

    # Move model to GPU if available
    model = model.to(device)

    # Set up the optimizer (helps the AI learn)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Set up the loss function (tells AI how wrong it was)
    criterion = nn.CrossEntropyLoss()

    # Check if we have a saved model from previous runs
    model_path = 'image_sorter_model.pth'
    if os.path.exists(model_path):
        # Load the saved model
        checkpoint = torch.load(model_path, map_location=device)
        # Check if it has the same number of folders
        if checkpoint.get('num_classes') == num_classes:
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded existing model for continued learning.")
        else:
            logger.info("Number of classes mismatch. Starting with a new model.")

    # Create a queue of images to process
    image_queue = list(image_paths)  # Copy the list

    # Create slots for 3 images currently displayed
    current_slots = [None, None, None]  # Start with empty slots

    # Load the first 3 images into the slots
    for slot in range(3):
        if image_queue:  # If we have images left
            # Take the first image from queue and put in slot
            current_slots[slot] = image_queue.pop(0)
            logger.info(f"Loaded initial image into slot {slot}: {current_slots[slot]}")

    # Variable to track which image slot is selected
    selected_slot = None

    # Function to draw everything on screen
    def draw():
        # Fill screen with black
        screen.fill((0, 0, 0))

        # Draw each image slot
        for slot in range(3):
            if current_slots[slot]:  # If there's an image in this slot
                try:
                    # Load the image
                    img_path = current_slots[slot]
                    img = pygame.image.load(img_path)

                    # Get image size
                    img_rect = img.get_rect()

                    # Calculate scale to fit in column
                    scale = min(col_width * 0.9 / img_rect.w, avail_height / img_rect.h, 1)

                    # Calculate new size
                    scaled_w = int(img_rect.w * scale)
                    scaled_h = int(img_rect.h * scale)

                    # Scale the image
                    scaled_image = pygame.transform.scale(img, (scaled_w, scaled_h))

                    # Calculate position (centered in column)
                    x = slot * col_width + (col_width - scaled_w) // 2
                    y = 10

                    # Draw the image
                    screen.blit(scaled_image, (x, y))

                    # Draw the key label (A, S, or D) below the image
                    # Make it red if this slot is selected
                    if selected_slot == slot:
                        text_color = (255, 0, 0)  # Red
                    else:
                        text_color = (255, 255, 255)  # White

                    text = font.render(key_labels[slot], True, text_color)
                    text_pos = (x + (col_width - text.get_width()) // 2, y + scaled_h + 5)
                    screen.blit(text, text_pos)

                except Exception as e:
                    # If there's an error loading the image, log it
                    logger.error(f"Error loading image for display {img_path}: {e}")
                    current_slots[slot] = None  # Clear the slot

        # Draw the folder buttons at the bottom
        for rect, _, label in buttons:
            # Draw green rectangle
            pygame.draw.rect(screen, (0, 255, 0), rect)
            # Draw text centered in button
            text = font.render(label, True, (0, 0, 0))  # Black text
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)

        # Draw auto mode button or indicator
        if not auto_mode:
            # Draw blue auto mode button
            pygame.draw.rect(screen, (0, 0, 255), auto_rect)
            text = font.render("Auto Mode", True, (255, 255, 255))
            text_rect = text.get_rect(center=auto_rect.center)
            screen.blit(text, text_rect)
        else:
            # Show "Auto Active" in red
            text = font.render("Auto Active", True, (255, 0, 0))
            screen.blit(text, (screen_width - 140, 10))

        # Update the display
        pygame.display.flip()

    # Function to move an image to a folder
    def move_slot_to_dir(slot_idx, dir_idx, is_manual=True):
        # Get the image path from the slot
        path = current_slots[slot_idx]

        if path:  # If there's an image in the slot
            # If this is a manual move (user chose), train the AI
            if is_manual:
                try:
                    # Open and prepare the image for AI
                    img = Image.open(path).convert('RGB')
                    input_tensor = preprocess(img).unsqueeze(0).to(device)

                    # Create label (which folder was chosen)
                    label = torch.tensor([dir_idx]).to(device)

                    # Train the model on this choice
                    optimizer.zero_grad()  # Clear previous gradients
                    output = model(input_tensor)  # Get model prediction
                    loss = criterion(output, label)  # Calculate error
                    loss.backward()  # Calculate gradients
                    optimizer.step()  # Update model weights

                    logger.info(f"Trained on {path} with label {dir_idx}, loss: {loss.item()}")
                except Exception as e:
                    logger.error(f"Error in training: {e}")

            # Move the file to the chosen folder
            tdir = target_dirs[dir_idx]
            filename = os.path.basename(path)  # Get just the filename
            target_path = os.path.join(tdir, filename)  # Full target path

            logger.info(f"Moving {path} to {target_path}")

            # Move file in background thread (so UI doesn't freeze)
            Thread(target=shutil.move, args=(path, target_path)).start()

            # Clear the slot
            current_slots[slot_idx] = None

            # Load next image if available
            if image_queue:
                next_path = image_queue.pop(0)
                current_slots[slot_idx] = next_path
                logger.info(f"Loaded next image into slot {slot_idx}: {next_path}")

    # Main game loop
    running = True
    while running:
        # Check if we're done (no images left)
        if not any(current_slots) and not image_queue:
            logger.info("All images processed.")
            break

        # Draw everything
        draw()

        # Handle events (keyboard and mouse input)
        for event in pygame.event.get():
            # Check if user closed the window
            if event.type == pygame.QUIT:
                running = False
                break

            # Handle keyboard input
            if event.type == pygame.KEYDOWN:
                # Skip manual input if in auto mode
                if auto_mode:
                    continue

                # If no slot is selected yet
                if selected_slot is None:
                    # Check if they pressed A, S, or D
                    for idx, key in enumerate(image_keys):
                        if event.key == key:
                            # Select the slot if it has an image
                            if current_slots[idx]:
                                selected_slot = idx
                                logger.info(f"Selected slot {idx} via keyboard.")
                            break
                else:
                    # A slot is selected, check if they pressed a number key
                    if event.key in dir_keys:
                        # Find which folder they chose
                        dir_idx = dir_keys.index(event.key)
                        # Move the image
                        move_slot_to_dir(selected_slot, dir_idx, is_manual=True)
                        # Clear selection
                        selected_slot = None

            # Handle mouse clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position
                pos = pygame.mouse.get_pos()

                # Skip manual input if in auto mode
                if auto_mode:
                    continue

                # If no slot is selected yet
                if selected_slot is None:
                    # Check if they clicked the auto button
                    if auto_rect.collidepoint(pos):
                        auto_mode = True
                        logger.info("Enabled auto mode.")
                        continue

                    # Check if they clicked on an image
                    for slot in range(3):
                        # Calculate the x-range for this slot
                        slot_start_x = slot * col_width
                        slot_end_x = slot_start_x + col_width

                        # Check if click is in this slot's area
                        if slot_start_x <= pos[0] < slot_end_x and 0 <= pos[1] < screen_height - button_height:
                            # Select the slot if it has an image
                            if current_slots[slot]:
                                selected_slot = slot
                                logger.info(f"Selected slot {slot} via mouse.")
                                break
                else:
                    # A slot is selected, check if they clicked a folder button
                    for btn_idx, (rect, _, _) in enumerate(buttons):
                        if rect.collidepoint(pos):
                            # Move the image to this folder
                            move_slot_to_dir(selected_slot, btn_idx, is_manual=True)
                            # Clear selection
                            selected_slot = None
                            break

        # Handle auto mode
        if auto_mode:
            # Process each slot
            for slot in range(3):
                if current_slots[slot]:  # If there's an image
                    try:
                        # Prepare image for AI
                        img_path = current_slots[slot]
                        img = Image.open(img_path).convert('RGB')
                        input_tensor = preprocess(img).unsqueeze(0).to(device)

                        # Get AI prediction (no training in auto mode)
                        with torch.no_grad():
                            output = model(input_tensor)

                        # Get the folder with highest score
                        pred = output.argmax(1).item()

                        # Move to predicted folder
                        move_slot_to_dir(slot, pred, is_manual=False)
                        logger.info(f"Auto sorted {img_path} to directory index {pred}")

                    except Exception as e:
                        logger.error(f"Error in auto sorting for slot {slot}: {e}")
                        current_slots[slot] = None  # Clear the slot

    # Save the trained model for next time
    torch.save({
        'num_classes': num_classes,  # Save number of folders
        'state_dict': model.state_dict()  # Save model weights
    }, model_path)
    logger.info("Saved model for future runs.")

    # Show completion message
    screen.fill((0, 0, 0))  # Clear screen
    text = font.render("All images sorted!", True, (255, 255, 255))
    text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    logger.info("Displayed completion message.")

    # Wait 5 seconds before closing (or until user closes window)
    waiting = True
    start_time = pygame.time.get_ticks()  # Get current time

    while waiting:
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

        # Check if 5 seconds have passed
        if pygame.time.get_ticks() - start_time > 5000:
            waiting = False

    # Clean up and close
    pygame.quit()
    logger.info("Application closed.")

# This makes sure the main function only runs when we run this file directly
if __name__ == "__main__":
    main()
