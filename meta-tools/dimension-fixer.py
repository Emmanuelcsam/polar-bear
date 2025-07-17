import logging
import os
import sys
import subprocess
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models.detection
from PIL import Image
import cv2

# Set up logging to terminal and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_processor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting the image processing script.")

# Function to check and install packages
def check_and_install(package_name, import_name=None, extra_flags=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        logger.info(f"{import_name} is already installed.")
    except ImportError:
        install_cmd = [sys.executable, "-m", "pip", "install", package_name]
        if extra_flags:
            install_cmd.extend(extra_flags)
        logger.info(f"Installing {package_name}...")
        subprocess.check_call(install_cmd)
        logger.info(f"{package_name} installed successfully.")
        __import__(import_name)

# Check and install required libraries
check_and_install("pillow", "PIL")
check_and_install("opencv-contrib-python", "cv2")
check_and_install("torch")
check_and_install("torchvision")
check_and_install("numpy")

# Now import after installations
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models.detection
import numpy as np

# Ask user for configurations
logger.info("Asking for user inputs.")

# Get and validate directory path
while True:
    dir_path = input("Enter the directory path containing images: ").strip()
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        break
    else:
        logger.error(f"Directory '{dir_path}' does not exist or is not a directory. Please try again.")

# Get and validate dimensions
while True:
    try:
        target_width = int(input("Enter the target width: ").strip())
        if target_width > 0:
            break
        else:
            logger.error("Width must be positive. Please try again.")
    except ValueError:
        logger.error("Invalid width. Please enter a number.")

while True:
    try:
        target_height = int(input("Enter the target height: ").strip())
        if target_height > 0:
            break
        else:
            logger.error("Height must be positive. Please try again.")
    except ValueError:
        logger.error("Invalid height. Please enter a number.")

# Get output directory
output_dir = input("Enter the output directory (will create if not exists): ").strip()

try:
    os.makedirs(output_dir, exist_ok=True)
    # Test write permissions
    test_file = os.path.join(output_dir, '.test_write')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    logger.info(f"Output directory set to: {output_dir}")
except Exception as e:
    logger.error(f"Cannot create or write to output directory: {e}")
    sys.exit(1)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == 'cpu':
    logger.warning("Using CPU, processing may be slower for large directories.")

# Find all image files
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
try:
    files_in_dir = os.listdir(dir_path)
except Exception as e:
    logger.error(f"Cannot read directory {dir_path}: {e}")
    sys.exit(1)

image_files = [os.path.join(dir_path, f) for f in files_in_dir if os.path.splitext(f)[1].lower() in image_extensions]
logger.info(f"Found {len(image_files)} images in {dir_path}.")

if len(image_files) == 0:
    logger.info("No images found. Exiting.")
    sys.exit(0)

# Load images
images = []
valid_image_files = []
for filename in image_files:
    img = cv2.imread(filename)
    if img is not None:
        images.append(img)
        valid_image_files.append(filename)
        logger.info(f"Loaded image: {filename}")
    else:
        logger.warning(f"Failed to load image: {filename}")

image_files = valid_image_files

# Prepare tensors for batch processing
to_tensor = transforms.ToTensor()
tensors = []
for idx, img in enumerate(images):
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = to_tensor(pil_img)
        tensors.append(tensor)
        logger.info(f"Converted image to tensor: shape {tensor.shape}")
    except Exception as e:
        logger.error(f"Failed to convert image {image_files[idx]} to tensor: {e}")
        # Remove failed image from processing
        images.pop(idx)
        image_files.pop(idx)

# Load detection model
logger.info("Loading pre-trained Faster R-CNN model.")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)
model.eval()
logger.info("Model loaded and set to eval mode.")

# Batch process detections
batch_size = 4 if device.type == 'cuda' else 1
all_preds = []
with torch.no_grad():
    for i in range(0, len(tensors), batch_size):
        try:
            batch = tensors[i:i+batch_size]
            batch_on_device = [t.to(device) for t in batch]
            preds = model(batch_on_device)
            all_preds.extend(preds)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(tensors)+batch_size-1)//batch_size} for object detection.")
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("GPU out of memory, falling back to CPU for this batch")
                torch.cuda.empty_cache()
                batch_on_cpu = [t.to('cpu') for t in batch]
                model_cpu = model.to('cpu')
                preds = model_cpu(batch_on_cpu)
                all_preds.extend(preds)
                model = model_cpu.to(device)
            else:
                raise e

# Function to get center from predictions or saliency
def get_crop_center(pred, image):
    scores = pred['scores'].cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    high_scores_idx = scores > 0.5
    if np.any(high_scores_idx):
        boxes = boxes[high_scores_idx]
        min_x, min_y, max_x, max_y = np.min(boxes[:,0]), np.min(boxes[:,1]), np.max(boxes[:,2]), np.max(boxes[:,3])
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        logger.info("Used object detection for crop center.")
    else:
        # Use saliency
        try:
            # Check if saliency module is available
            if hasattr(cv2, 'saliency'):
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliencyMap) = saliency.computeSaliency(image)
                if success:
                    moments = cv2.moments(saliencyMap)
                    if moments['m00'] != 0:
                        center_x = moments['m10'] / moments['m00']
                        center_y = moments['m01'] / moments['m00']
                    else:
                        center_x = image.shape[1] / 2
                        center_y = image.shape[0] / 2
                    logger.info("Used saliency map for crop center.")
                else:
                    center_x = image.shape[1] / 2
                    center_y = image.shape[0] / 2
                    logger.info("Fallback to center crop (saliency computation failed).")
            else:
                logger.warning("Saliency module not available in OpenCV installation.")
                center_x = image.shape[1] / 2
                center_y = image.shape[0] / 2
                logger.info("Fallback to center crop (saliency module not available).")
        except Exception as e:
            logger.warning(f"Error computing saliency: {e}")
            center_x = image.shape[1] / 2
            center_y = image.shape[0] / 2
            logger.info("Fallback to center crop (saliency error).")
    return center_x, center_y

# Function to smart crop and resize
def smart_crop_and_resize(image, pred, target_width, target_height):
    h, w = image.shape[:2]
    target_ar = target_width / target_height
    orig_ar = w / h
    center_x, center_y = get_crop_center(pred, image)
    
    if orig_ar > target_ar:
        # Crop width
        crop_h = h
        crop_w = int(crop_h * target_ar)
        left = int(center_x - crop_w / 2)
        left = max(0, min(left, w - crop_w))
        cropped = image[:, left:left + crop_w]
    else:
        # Crop height
        crop_w = w
        crop_h = int(crop_w / target_ar)
        top = int(center_y - crop_h / 2)
        top = max(0, min(top, h - crop_h))
        cropped = image[top:top + crop_h, :]
    
    # Resize
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
    logger.info(f"Resized image to {target_width}x{target_height}.")
    return resized

# Process each image: analyze, resize, save
stats_file = "image_stats.json"
stats = {}
if os.path.exists(stats_file):
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    logger.info("Loaded previous image stats.")

for idx, (filename, image, pred) in enumerate(zip(image_files, images, all_preds)):
    base_name = os.path.basename(filename)
    logger.info(f"Analyzing image {idx+1}/{len(images)}: {base_name}")
    
    # Compute statistics
    means, stds = cv2.meanStdDev(image)
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
    # Convert to grayscale for moments calculation
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray_image)
    hu_moments = cv2.HuMoments(moments).flatten().tolist()
    
    image_stats = {
        'means': means.flatten().tolist(),
        'stds': stds.flatten().tolist(),
        'hu_moments': hu_moments,
        # Histograms are large, store summary like mean hist or skip detailed
        'hist_means': [float(np.mean(hist_b)), float(np.mean(hist_g)), float(np.mean(hist_r))]
    }
    stats[base_name] = image_stats
    logger.info(f"Computed stats for {base_name}: {image_stats}")
    
    # For mathematical relations, e.g., correlation between channels
    ch1 = image[:,:,0].flatten()
    ch2 = image[:,:,1].flatten()
    ch3 = image[:,:,2].flatten()
    
    # Add error handling for correlation calculations
    try:
        corr_12 = np.corrcoef(ch1, ch2)[0,1]
        corr_13 = np.corrcoef(ch1, ch3)[0,1]
        corr_23 = np.corrcoef(ch2, ch3)[0,1]
        # Handle NaN values that might occur with constant channels
        if np.isnan(corr_12):
            corr_12 = 0.0
        if np.isnan(corr_13):
            corr_13 = 0.0
        if np.isnan(corr_23):
            corr_23 = 0.0
    except Exception as e:
        logger.warning(f"Error calculating correlations for {base_name}: {e}")
        corr_12 = corr_13 = corr_23 = 0.0
    
    logger.info(f"Channel correlations for {base_name}: 1-2={corr_12:.4f}, 1-3={corr_13:.4f}, 2-3={corr_23:.4f}")
    
    # Resize
    resized = smart_crop_and_resize(image, pred, target_width, target_height)
    
    # Save resized
    output_path = os.path.join(output_dir, base_name)
    try:
        success = cv2.imwrite(output_path, resized)
        if success:
            logger.info(f"Saved resized image to {output_path}")
        else:
            logger.error(f"Failed to save image to {output_path}")
    except Exception as e:
        logger.error(f"Error saving image {base_name}: {e}")

# Save updated stats
try:
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved image stats to image_stats.json")
except Exception as e:
    logger.error(f"Failed to save stats file: {e}")

# Deep learning: Train autoencoder to learn representations
class ImageDataset(Dataset):
    def __init__(self, images):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create dataset and loader
if len(images) > 0:
    dataset = ImageDataset(images)
    # Adjust batch size based on number of images
    batch_size = min(32, len(images))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info("Prepared dataset for autoencoder training.")
else:
    logger.warning("No valid images to train autoencoder")
    dataloader = None

# Load or create autoencoder
autoencoder = ConvAutoencoder().to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

model_path = "autoencoder.pth"
if os.path.exists(model_path):
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))
    logger.info("Loaded existing autoencoder model for continued learning.")
else:
    logger.info("Created new autoencoder model.")

# Train
if dataloader is not None:
    num_epochs = 10  # Adjustable, for learning
    autoencoder.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        for data in dataloader:
            try:
                data = data.to(device)
                output = autoencoder(data)
                loss = criterion(output, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("GPU out of memory during training, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            logger.info(f"Autoencoder training epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")
        else:
            logger.warning(f"No batches processed in epoch {epoch+1}")

    # Save model
    try:
        torch.save(autoencoder.state_dict(), model_path)
        logger.info("Saved updated autoencoder model after learning from current images.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
else:
    logger.info("Skipped autoencoder training due to no valid images")

logger.info("Script completed successfully.")
