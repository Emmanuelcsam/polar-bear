import subprocess
import os
import sys
import numpy as np

log_file_path = 'background_remover.log'
log_file = open(log_file_path, 'w')

def log_message(message):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

log_message("Script started. Checking for required libraries...")

try:
    import rembg
    log_message("rembg is already installed.")
except ImportError:
    log_message("rembg is not installed. Installing the latest version now...")
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'rembg'])
    import rembg
    log_message("rembg installed successfully.")

try:
    import cv2
    log_message("OpenCV is already installed.")
except ImportError:
    log_message("OpenCV is not installed. Installing the latest version now...")
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'opencv-python'])
    import cv2
    log_message("OpenCV installed successfully.")

try:
    import torch
    log_message("PyTorch is already installed.")
except ImportError:
    log_message("PyTorch is not installed. Installing the latest version now...")
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'])
    import torch
    log_message("PyTorch installed successfully.")

log_message("All required libraries are installed and ready.")

log_message("Please enter the full path to the image file you want to remove the background from.")
input_image_path = input()
log_message(f"User provided input image path: {input_image_path}")

if not os.path.exists(input_image_path):
    log_message("Error: The input image file does not exist. Please check the path and run the script again.")
    log_file.close()
    sys.exit(1)

log_message("Please enter the full path where you want to save the output image (e.g., output.png).")
output_image_path = input()
log_message(f"User provided output image path: {output_image_path}")

output_dir = os.path.dirname(output_image_path)
if output_dir and not os.path.exists(output_dir):
    log_message(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir)

try:
    log_message("Loading the input image with OpenCV...")
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if input_image is None:
        raise ValueError("Failed to load image.")
    original_shape = input_image.shape
    log_message("Input image loaded successfully.")

    log_message("Converting image for rembg processing...")
    from PIL import Image
    input_pil = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

    log_message("Removing the background using rembg (powered by PyTorch U2-Net model)...")
    output_pil = rembg.remove(input_pil)

    log_message("Converting back to OpenCV format...")
    output_image = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGRA)

    if output_image.shape != original_shape:
        log_message("Resizing output to match original dimensions...")
        output_image = cv2.resize(output_image, (original_shape[1], original_shape[0]))

    log_message(f"Saving the output image to {output_image_path}...")
    cv2.imwrite(output_image_path, output_image)
    log_message("Output image saved successfully.")

except Exception as e:
    log_message(f"Error during processing: {str(e)}")

log_message("Script completed.")
log_file.close()
