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
        importlib.import_module(package)  # Attempt to dynamically import the package to check if it exists
    except ImportError:
        logging.info(f"Installing missing package: {package}")  # Log that package installation is starting
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])  # Execute pip install using current Python interpreter

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
    logger.info("Starting image sorter application.")  # Log application startup for debugging/monitoring
    root = tk.Tk()  # Create main tkinter window for file dialogs
    root.withdraw()  # Hide the main window since we only need dialogs
    logger.info("Prompting user to select source directory.")  # Log user interaction step
    source_dir = filedialog.askdirectory(title="Select Source Directory")  # Open folder selection dialog for images
    if not source_dir:  # Check if user cancelled directory selection
        logger.info("No source directory selected. Exiting.")  # Log early exit condition
        return  # Exit function if no source directory chosen

    logger.info(f"Source directory selected: {source_dir}")  # Log the selected source path

    target_dirs = []  # Initialize empty list to store destination directories
    while True:  # Loop to collect multiple target directories
        tdir = filedialog.askdirectory(title="Select a Target Directory (Cancel when all added)")  # Prompt for each target directory
        if not tdir:  # Check if user cancelled (indicating they're done adding directories)
            logger.info("Target directory selection completed.")  # Log completion of directory selection
            break  # Exit the directory selection loop
        if tdir not in target_dirs:  # Prevent duplicate directories from being added
            target_dirs.append(tdir)  # Add unique directory to the list
            logger.info(f"Added target directory: {tdir}")  # Log each directory addition

    if not target_dirs:  # Validate that at least one target directory was selected
        logger.error("No target directories selected.")  # Log error condition
        messagebox.showerror("Error", "No target directories selected.")  # Show error dialog to user
        return  # Exit function if no target directories

    for tdir in target_dirs:  # Iterate through all selected target directories
        if not os.path.exists(tdir):  # Check if directory doesn't exist yet
            logger.info(f"Creating target directory: {tdir}")  # Log directory creation
            os.makedirs(tdir)  # Create the directory and any necessary parent directories

    image_paths = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpeg', '.jpg', '.gif', '.bmp'))])  # Build sorted list of image file paths by filtering for image extensions
    logger.info(f"Found {len(image_paths)} images in source directory.")  # Log count of images found
    if not image_paths:  # Check if no images were found in source directory
        logger.info("No images found.")  # Log the absence of images
        messagebox.showinfo("No Images", "No images found in the source directory.")  # Inform user no images exist
        return  # Exit function if no images to process

    logger.info("Initializing Pygame.")  # Log pygame setup start
    pygame.init()  # Initialize all pygame modules for graphics and input handling
    screen_width = 1200  # Set initial window width in pixels
    screen_height = 600  # Set initial window height in pixels
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)  # Create resizable game window with specified dimensions
    pygame.display.set_caption("Image Sorter - Resizable")  # Set window title text

    font = pygame.font.SysFont(None, 30)  # Create font object for text rendering with size 30

    # Initialize layout variables
    button_height = 50  # Set height for directory selection buttons
    button_width = screen_width // max(1, len(target_dirs))  # Calculate button width by dividing screen width among target directories
    buttons = []  # Initialize empty list to store button rectangle objects and labels
    col_width = screen_width // 3  # Divide screen into 3 columns for image display
    avail_height = screen_height - button_height - 30  # Calculate available height for images (subtract buttons and padding)
    auto_rect = pygame.Rect(screen_width - 150, 0, 150, 50)  # Create rectangle for auto-mode button in top-right corner

    # Function to recalculate layout based on current window size
    def recalculate_layout():
        nonlocal button_height, button_width, buttons, col_width, avail_height, auto_rect, font  # Access parent function variables for modification
        
        # Adaptive font size based on window size
        font_size = max(20, min(40, screen_width // 40))  # Calculate responsive font size between 20-40 pixels based on window width
        font = pygame.font.SysFont(None, font_size)  # Create new font object with calculated size
        
        button_height = max(40, min(80, screen_height // 10))  # Calculate responsive button height between 40-80 pixels based on window height
        button_width = screen_width // max(1, len(target_dirs))  # Recalculate button width to evenly distribute across screen width
        buttons = []  # Clear existing button list for regeneration
        for idx, tdir in enumerate(target_dirs):  # Iterate through target directories with index
            rect = pygame.Rect(idx * button_width, screen_height - button_height, button_width, button_height)  # Create button rectangle positioned at bottom of screen
            label = os.path.basename(tdir)  # Extract directory name from full path for button label
            if len(target_dirs) <= 9:  # Add keyboard shortcuts only when 9 or fewer directories exist
                label += f" ({idx + 1})"  # Append number key shortcut to label
            buttons.append((rect, tdir, label))  # Store button data as tuple (rectangle, directory path, display label)

        col_width = screen_width // 3  # Recalculate column width for 3-column image layout
        avail_height = screen_height - button_height - 30  # Recalculate available height for images (subtract button area and padding)
        
        # Auto button - adaptive size and position
        auto_button_width = min(150, max(100, screen_width // 8))  # Calculate responsive auto button width between 100-150 pixels
        auto_button_height = min(50, max(30, screen_height // 12))  # Calculate responsive auto button height between 30-50 pixels
        auto_rect = pygame.Rect(screen_width - auto_button_width - 10, 10, auto_button_width, auto_button_height)  # Position auto button in top-right with 10px margin

    # Initial layout calculation
    recalculate_layout()  # Call layout function to set initial UI element positions and sizes

    dir_keys = [pygame.K_1 + i for i in range(min(9, len(target_dirs)))] if len(target_dirs) <= 9 else []  # Create keyboard shortcuts (1-9) for directory selection if 9 or fewer directories
    image_keys = [pygame.K_a, pygame.K_s, pygame.K_d]  # Define keys A, S, D for selecting image slots
    key_labels = ['A', 'S', 'D']  # Text labels corresponding to image selection keys

    auto_mode = False  # Initialize manual mode (auto-sorting disabled)

    # ML setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU if available, otherwise use CPU for neural network
    logger.info(f"Using device: {device}")  # Log which computing device will be used
    preprocess = transforms.Compose([  # Create image preprocessing pipeline for neural network input
        transforms.Resize(256),  # Resize image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop center 224x224 region (standard for ImageNet models)
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics (RGB channels)
    ])
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet-18 model with ImageNet weights
    num_ftrs = model.fc.in_features  # Get number of input features to final classification layer
    num_classes = len(target_dirs)  # Set number of output classes equal to number of target directories
    model.fc = nn.Linear(num_ftrs, num_classes)  # Replace final layer to match number of target directories
    model = model.to(device)  # Move model to selected computing device (GPU/CPU)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Create SGD optimizer with learning rate 0.001 and momentum 0.9
    criterion = nn.CrossEntropyLoss()  # Create cross-entropy loss function for classification training

    model_path = 'image_sorter_model.pth'  # Define filename for saving/loading trained model
    if os.path.exists(model_path):  # Check if previously saved model exists
        checkpoint = torch.load(model_path, map_location=device)  # Load saved model checkpoint to current device
        if checkpoint.get('num_classes') == num_classes:  # Verify saved model has same number of classes as current setup
            model.load_state_dict(checkpoint['state_dict'])  # Load the trained model weights
            logger.info("Loaded existing model for continued learning.")  # Log successful model loading
        else:
            logger.info("Number of classes mismatch. Starting with a new model.")  # Log when saved model incompatible with current directory count

    image_queue = list(image_paths)  # Create working copy of image paths list for processing queue
    current_slots: List[Optional[str]] = [None, None, None]  # Initialize three image display slots as empty
    for slot in range(3):  # Loop through the three image display slots
        if image_queue:  # Check if images remain in queue
            current_slots[slot] = image_queue.pop(0)  # Remove first image from queue and assign to current slot
            logger.info(f"Loaded initial image into slot {slot}: {current_slots[slot]}")  # Log which image was loaded into which slot

    selected_slot = None  # Initialize no slot selected for user interaction

    def draw():
        screen.fill((0, 0, 0))  # Clear screen with black background
        for slot in range(3):  # Iterate through each of the three image display slots
            if current_slots[slot]:  # Check if current slot contains an image path
                try:
                    img_path = current_slots[slot]  # Get image file path from slot
                    img = pygame.image.load(img_path)  # Load image file into pygame surface
                    img_rect = img.get_rect()  # Get rectangle dimensions of loaded image
                    scale = min(col_width * 0.9 / img_rect.w, avail_height / img_rect.h, 1)  # Calculate scaling factor to fit image in column while maintaining aspect ratio
                    scaled_w = int(img_rect.w * scale)  # Calculate scaled width in pixels
                    scaled_h = int(img_rect.h * scale)  # Calculate scaled height in pixels
                    scaled_image = pygame.transform.scale(img, (scaled_w, scaled_h))  # Create scaled version of image
                    x = slot * col_width + (col_width - scaled_w) // 2  # Calculate x position to center image in column
                    y = 10  # Set y position with small top margin
                    screen.blit(scaled_image, (x, y))  # Draw scaled image at calculated position

                    # Draw key label
                    text = font.render(key_labels[slot], True, (255, 0, 0) if selected_slot == slot else (255, 255, 255))  # Create text surface with red color if selected, white otherwise
                    text_pos = (x + (col_width - text.get_width()) // 2, y + scaled_h + 5)  # Calculate position to center text below image
                    screen.blit(text, text_pos)  # Draw key label text at calculated position
                except Exception as e:
                    logger.error(f"Error loading image for display {img_path}: {e}")  # Log any image loading errors
                    current_slots[slot] = None  # Clear slot if image fails to load

        for rect, _, label in buttons:  # Iterate through directory selection buttons
            pygame.draw.rect(screen, (0, 255, 0), rect)  # Draw green rectangle for button background
            
            # Truncate label if it's too long for the button
            max_label_width = rect.width - 10  # Calculate maximum text width leaving 10px padding
            text = font.render(label, True, (0, 0, 0))  # Render button label text in black
            if text.get_width() > max_label_width:  # Check if text exceeds button width
                # Truncate the label
                truncated_label = label  # Start with full label
                while font.render(truncated_label + "...", True, (0, 0, 0)).get_width() > max_label_width and len(truncated_label) > 3:  # Shorten label until it fits with ellipsis
                    truncated_label = truncated_label[:-1]  # Remove last character
                truncated_label += "..."  # Add ellipsis to indicate truncation
                text = font.render(truncated_label, True, (0, 0, 0))  # Re-render with truncated text
            
            text_rect = text.get_rect(center=rect.center)  # Calculate position to center text in button
            screen.blit(text, text_rect)  # Draw button text at centered position

        if not auto_mode:  # Check if auto mode is disabled
            pygame.draw.rect(screen, (0, 0, 255), auto_rect)  # Draw blue rectangle for auto mode button
            text = font.render("Auto Mode", True, (255, 255, 255))  # Create white text for auto mode button
            text_rect = text.get_rect(center=auto_rect.center)  # Center text in auto button
            screen.blit(text, text_rect)  # Draw auto mode button text
        else:
            text = font.render("Auto Active", True, (255, 0, 0))  # Create red text indicating auto mode is active
            screen.blit(text, (screen_width - 140, 10))  # Draw auto active text in top-right corner

        pygame.display.flip()  # Update entire display with all drawn elements

    def move_slot_to_dir(slot_idx, dir_idx, is_manual=True):
        path = current_slots[slot_idx]  # Get image file path from specified slot
        if path:  # Check if slot contains an image
            if is_manual:  # Check if this is a manual user action (for training)
                try:
                    img = Image.open(path).convert('RGB')  # Load image and ensure RGB color format
                    input_tensor = preprocess(img).unsqueeze(0).to(device)  # Apply preprocessing and add batch dimension, move to computing device
                    label = torch.tensor([dir_idx]).to(device)  # Create target label tensor for chosen directory, move to device
                    optimizer.zero_grad()  # Clear previous gradients from optimizer
                    output = model(input_tensor)  # Forward pass: get model predictions for image
                    loss = criterion(output, label)  # Calculate loss between prediction and true label
                    loss.backward()  # Backward pass: compute gradients
                    optimizer.step()  # Update model weights using computed gradients
                    logger.info(f"Trained on {path} with label {dir_idx}, loss: {loss.item()}")  # Log training details
                except Exception as e:
                    logger.error(f"Error in training: {e}")  # Log any training errors

            tdir = target_dirs[dir_idx]  # Get target directory path from index
            filename = os.path.basename(path)  # Extract filename from full image path
            target_path = os.path.join(tdir, filename)  # Construct full destination path
            logger.info(f"Moving {path} to {target_path}")  # Log file movement operation
            Thread(target=shutil.move, args=(path, target_path)).start()  # Move file in separate thread to prevent UI blocking
            current_slots[slot_idx] = None  # Clear the processed slot
            if image_queue:  # Check if more images remain in queue
                next_path = image_queue.pop(0)  # Get next image from queue
                current_slots[slot_idx] = next_path  # Load next image into the cleared slot
                logger.info(f"Loaded next image into slot {slot_idx}: {next_path}")  # Log next image loading

    running = True  # Initialize main loop control flag
    while running:  # Main application loop
        if not any(current_slots) and not image_queue:  # Check if all slots empty and no images remain in queue
            logger.info("All images processed.")  # Log completion of all image processing
            break  # Exit main loop when all images are processed

        draw()  # Render current frame with images and UI elements

        for event in pygame.event.get():  # Process all pending pygame events
            if event.type == pygame.QUIT:  # Check for window close event
                running = False  # Set flag to exit main loop
                break  # Exit event processing loop
            
            if event.type == pygame.VIDEORESIZE:  # Check for window resize event
                # Enforce minimum window size to ensure UI remains usable
                min_width = max(600, len(target_dirs) * 100)  # Calculate minimum width based on directory count
                min_height = 400  # Set absolute minimum height
                screen_width = max(min_width, event.w)  # Ensure width meets minimum requirement
                screen_height = max(min_height, event.h)  # Ensure height meets minimum requirement
                
                screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)  # Create new display surface with constrained dimensions
                recalculate_layout()  # Recalculate UI element positions for new window size
                pygame.display.set_caption(f"Image Sorter - Resizable ({screen_width}x{screen_height})")  # Update window title with current dimensions
                logger.info(f"Window resized to {screen_width}x{screen_height}")  # Log resize operation
                continue  # Skip remaining event processing for this frame

            if event.type == pygame.KEYDOWN:  # Check for keyboard key press events
                if auto_mode:  # Check if auto mode is active
                    continue  # Skip manual inputs in auto mode
                if selected_slot is None:  # Check if no image slot is currently selected
                    for idx, key in enumerate(image_keys):  # Iterate through A, S, D keys
                        if event.key == key:  # Check if pressed key matches current image selection key
                            if current_slots[idx]:  # Verify the corresponding slot contains an image
                                selected_slot = idx  # Set this slot as selected
                                logger.info(f"Selected slot {idx} via keyboard.")  # Log slot selection
                            break  # Exit key checking loop once match found
                else:  # An image slot is already selected
                    if event.key in dir_keys:  # Check if pressed key is a directory selection key (1-9)
                        dir_idx = dir_keys.index(event.key)  # Get directory index from key position
                        move_slot_to_dir(selected_slot, dir_idx, is_manual=True)  # Move selected image to chosen directory with training
                        selected_slot = None  # Clear selection after move operation

            if event.type == pygame.MOUSEBUTTONDOWN:  # Check for mouse button click events
                pos = pygame.mouse.get_pos()  # Get current mouse cursor position as (x, y) coordinates
                if auto_mode:  # Check if auto mode is active
                    continue  # Skip manual inputs in auto mode
                if selected_slot is None:  # Check if no image slot is currently selected
                    # Check for auto button
                    if auto_rect.collidepoint(pos):  # Check if mouse click is within auto mode button area
                        auto_mode = True  # Enable automatic image sorting mode
                        logger.info("Enabled auto mode.")  # Log auto mode activation
                        continue  # Skip remaining mouse handling for this click
                    # Check image areas
                    for slot in range(3):  # Iterate through the three image display slots
                        slot_start_x = slot * col_width  # Calculate left edge of slot area
                        slot_end_x = slot_start_x + col_width  # Calculate right edge of slot area
                        if slot_start_x <= pos[0] < slot_end_x and 0 <= pos[1] < screen_height - button_height:  # Check if click is within slot boundaries (excluding button area)
                            if current_slots[slot]:  # Verify slot contains an image
                                selected_slot = slot  # Set this slot as selected
                                logger.info(f"Selected slot {slot} via mouse.")  # Log slot selection via mouse
                                break  # Exit slot checking loop once selection made
                else:  # An image slot is already selected
                    # Check dir buttons
                    for btn_idx, (rect, _, _) in enumerate(buttons):  # Iterate through directory selection buttons
                        if rect.collidepoint(pos):  # Check if mouse click is within button rectangle
                            move_slot_to_dir(selected_slot, btn_idx, is_manual=True)  # Move selected image to clicked directory with training
                            selected_slot = None  # Clear selection after move operation
                            break  # Exit button checking loop once action taken

        if auto_mode:  # Check if automatic sorting mode is enabled
            for slot in range(3):  # Iterate through all three image display slots
                img_path = current_slots[slot]  # Get image path from current slot
                if img_path is not None:  # Check if slot contains an image
                    try:
                        img = Image.open(img_path).convert('RGB')  # Load image and ensure RGB color format
                        input_tensor = preprocess(img).unsqueeze(0).to(device)  # Apply preprocessing, add batch dimension, move to computing device
                        with torch.no_grad():  # Disable gradient computation for inference (saves memory and computation)
                            output = model(input_tensor)  # Forward pass: get model predictions for image
                        pred = output.argmax(1).item()  # Find class with highest prediction score and convert to Python integer
                        move_slot_to_dir(slot, pred, is_manual=False)  # Move image to predicted directory without training
                        logger.info(f"Auto sorted {img_path} to directory index {pred}")  # Log automatic sorting decision
                    except Exception as e:
                        logger.error(f"Error in auto sorting for slot {slot}: {e}")  # Log any errors during automatic sorting
                        current_slots[slot] = None  # Clear slot if error occurs

    # Save model at the end
    torch.save({  # Save trained model data to file
        'num_classes': num_classes,  # Store number of output classes for compatibility checking
        'state_dict': model.state_dict()  # Store all learned model weights and parameters
    }, model_path)  # Save to predefined model file path
    logger.info("Saved model for future runs.")  # Log successful model saving

    # End message
    screen.fill((0, 0, 0))  # Clear screen with black background
    text = font.render("All images sorted!", True, (255, 255, 255))  # Create white completion message text
    text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))  # Calculate position to center text on screen
    screen.blit(text, text_rect)  # Draw completion message at center of screen
    pygame.display.flip()  # Update display to show completion message
    logger.info("Displayed completion message.")  # Log completion message display

    # Wait 5 seconds or quit
    waiting = True  # Initialize waiting state flag
    start_time = pygame.time.get_ticks()  # Record current time for timeout calculation
    while waiting:  # Loop to wait for either user input or timeout
        for event in pygame.event.get():  # Process any pending events during wait
            if event.type == pygame.QUIT:  # Check for window close event
                waiting = False  # Exit waiting loop if user closes window
        if pygame.time.get_ticks() - start_time > 5000:  # Check if 5 seconds (5000 milliseconds) have elapsed
            waiting = False  # Exit waiting loop after timeout

    pygame.quit()  # Clean up pygame resources and close window
    logger.info("Application closed.")  # Log application shutdown

if __name__ == "__main__":
    main()  # Execute main function only when script is run directly (not imported as module)
