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
    level=logging.INFO,  # Set minimum log level to capture info, warning, and error messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define log message format with timestamp and level
    handlers=[
        logging.FileHandler("image_processor.log"),  # Write logs to a persistent file for debugging
        logging.StreamHandler(sys.stdout)  # Also display logs in terminal for real-time monitoring
    ]
)
logger = logging.getLogger(__name__)  # Create logger instance for this module

logger.info("Starting the image processing script.")  # Log script initialization

# Function to check and install packages
def check_and_install(package_name, import_name=None, extra_flags=None):
    if import_name is None:  # Use package name as import name if not specified
        import_name = package_name
    try:
        __import__(import_name)  # Attempt to import the package to check if it exists
        logger.info(f"{import_name} is already installed.")  # Log successful import
    except ImportError:  # Package not found, need to install it
        install_cmd = [sys.executable, "-m", "pip", "install", package_name]  # Build pip install command
        if extra_flags:  # Add any additional installation flags if provided
            install_cmd.extend(extra_flags)
        logger.info(f"Installing {package_name}...")  # Log installation start
        subprocess.check_call(install_cmd)  # Execute pip install command and wait for completion
        logger.info(f"{package_name} installed successfully.")  # Log successful installation
        __import__(import_name)  # Import the newly installed package to verify

# Check and install required libraries
check_and_install("pillow", "PIL")  # Install PIL/Pillow for image processing capabilities
check_and_install("opencv-contrib-python", "cv2")  # Install OpenCV with contrib modules for computer vision
check_and_install("torch")  # Install PyTorch for deep learning operations
check_and_install("torchvision")  # Install torchvision for pre-trained models and transforms
check_and_install("numpy")  # Install NumPy for numerical computations and array operations

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
logger.info("Asking for user inputs.")  # Log start of user input collection phase

# Get and validate directory path
while True:  # Loop until valid directory path is provided
    dir_path = input("Enter the directory path containing images: ").strip()  # Request input directory path
    if os.path.exists(dir_path) and os.path.isdir(dir_path):  # Check if path exists and is a directory
        break  # Exit loop when valid directory is found
    else:
        logger.error(f"Directory '{dir_path}' does not exist or is not a directory. Please try again.")  # Log error for invalid path

# Get and validate dimensions
while True:  # Loop until valid width is provided
    try:
        target_width = int(input("Enter the target width: ").strip())  # Convert user input to integer
        if target_width > 0:  # Ensure width is positive
            break  # Exit loop when valid width is provided
        else:
            logger.error("Width must be positive. Please try again.")  # Log error for non-positive width
    except ValueError:  # Handle non-numeric input
        logger.error("Invalid width. Please enter a number.")  # Log error for invalid numeric format

while True:  # Loop until valid height is provided
    try:
        target_height = int(input("Enter the target height: ").strip())  # Convert user input to integer
        if target_height > 0:  # Ensure height is positive
            break  # Exit loop when valid height is provided
        else:
            logger.error("Height must be positive. Please try again.")  # Log error for non-positive height
    except ValueError:  # Handle non-numeric input
        logger.error("Invalid height. Please enter a number.")  # Log error for invalid numeric format

# Get output directory
output_dir = input("Enter the output directory (will create if not exists): ").strip()  # Request output directory path

try:
    os.makedirs(output_dir, exist_ok=True)  # Create output directory and all parent directories if they don't exist
    # Test write permissions
    test_file = os.path.join(output_dir, '.test_write')  # Create path for temporary test file
    with open(test_file, 'w') as f:  # Open test file for writing
        f.write('test')  # Write test content to verify write permissions
    os.remove(test_file)  # Delete test file after successful write
    logger.info(f"Output directory set to: {output_dir}")  # Log successful directory setup
except Exception as e:  # Handle any errors in directory creation or permission testing
    logger.error(f"Cannot create or write to output directory: {e}")  # Log error details
    sys.exit(1)  # Exit script with error code

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU if available, otherwise use CPU
logger.info(f"Using device: {device}")  # Log which compute device is being used
if device.type == 'cpu':  # Check if CPU is being used
    logger.warning("Using CPU, processing may be slower for large directories.")  # Warn about potential slower performance

# Find all image files
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}  # Define supported image file extensions
try:
    files_in_dir = os.listdir(dir_path)  # Get list of all files in the input directory
except Exception as e:  # Handle directory reading errors
    logger.error(f"Cannot read directory {dir_path}: {e}")  # Log error details
    sys.exit(1)  # Exit script with error code

image_files = [os.path.join(dir_path, f) for f in files_in_dir if os.path.splitext(f)[1].lower() in image_extensions]  # Filter files by extension and create full paths
logger.info(f"Found {len(image_files)} images in {dir_path}.")  # Log number of images found

if len(image_files) == 0:  # Check if any images were found
    logger.info("No images found. Exiting.")  # Log no images found
    sys.exit(0)  # Exit script normally

# Load images
images = []  # Initialize list to store successfully loaded image arrays
valid_image_files = []  # Initialize list to store paths of successfully loaded images
for filename in image_files:  # Iterate through all found image files
    img = cv2.imread(filename)  # Load image using OpenCV (returns BGR format)
    if img is not None:  # Check if image was loaded successfully
        images.append(img)  # Add image array to list
        valid_image_files.append(filename)  # Add filename to valid files list
        logger.info(f"Loaded image: {filename}")  # Log successful image loading
    else:
        logger.warning(f"Failed to load image: {filename}")  # Log failed image loading

image_files = valid_image_files  # Update image_files to only contain successfully loaded images

# Prepare tensors for batch processing
to_tensor = transforms.ToTensor()  # Create transform to convert PIL images to PyTorch tensors
tensors = []  # Initialize list to store image tensors
for idx, img in enumerate(images):  # Iterate through loaded images with index
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB and create PIL Image
        tensor = to_tensor(pil_img)  # Convert PIL image to normalized tensor (0-1 range)
        tensors.append(tensor)  # Add tensor to list
        logger.info(f"Converted image to tensor: shape {tensor.shape}")  # Log tensor conversion with shape
    except Exception as e:  # Handle conversion errors
        logger.error(f"Failed to convert image {image_files[idx]} to tensor: {e}")  # Log error details
        # Remove failed image from processing
        images.pop(idx)  # Remove image from images list
        image_files.pop(idx)  # Remove filename from files list

# Load detection model
logger.info("Loading pre-trained Faster R-CNN model.")  # Log model loading start
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")  # Load pre-trained Faster R-CNN with ResNet-50 backbone
model.to(device)  # Move model to selected compute device (GPU or CPU)
model.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)
logger.info("Model loaded and set to eval mode.")  # Log successful model setup

# Batch process detections
batch_size = 4 if device.type == 'cuda' else 1  # Use larger batch size for GPU, smaller for CPU
all_preds = []  # Initialize list to store all prediction results
with torch.no_grad():  # Disable gradient computation for inference (saves memory and computation)
    for i in range(0, len(tensors), batch_size):  # Process tensors in batches
        try:
            batch = tensors[i:i+batch_size]  # Extract current batch of tensors
            batch_on_device = [t.to(device) for t in batch]  # Move batch tensors to compute device
            preds = model(batch_on_device)  # Run object detection inference on batch
            all_preds.extend(preds)  # Add predictions to results list
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(tensors)+batch_size-1)//batch_size} for object detection.")  # Log batch processing progress
        except RuntimeError as e:  # Handle runtime errors (typically GPU memory issues)
            if "out of memory" in str(e):  # Check if error is GPU out of memory
                logger.warning("GPU out of memory, falling back to CPU for this batch")  # Log memory issue
                torch.cuda.empty_cache()  # Clear GPU memory cache
                batch_on_cpu = [t.to('cpu') for t in batch]  # Move batch to CPU
                model_cpu = model.to('cpu')  # Move model to CPU
                preds = model_cpu(batch_on_cpu)  # Run inference on CPU
                all_preds.extend(preds)  # Add CPU predictions to results
                model = model_cpu.to(device)  # Move model back to original device
            else:
                raise e  # Re-raise other runtime errors

# Function to get center from predictions or saliency
def get_crop_center(pred, image):
    scores = pred['scores'].cpu().numpy()  # Extract detection confidence scores and convert to numpy
    boxes = pred['boxes'].cpu().numpy()  # Extract bounding box coordinates and convert to numpy
    high_scores_idx = scores > 0.5  # Create boolean mask for high-confidence detections
    if np.any(high_scores_idx):  # Check if any high-confidence detections exist
        boxes = boxes[high_scores_idx]  # Filter boxes to only high-confidence detections
        min_x, min_y, max_x, max_y = np.min(boxes[:,0]), np.min(boxes[:,1]), np.max(boxes[:,2]), np.max(boxes[:,3])  # Find bounding box of all detections
        center_x = (min_x + max_x) / 2  # Calculate horizontal center of detection region
        center_y = (min_y + max_y) / 2  # Calculate vertical center of detection region
        logger.info("Used object detection for crop center.")  # Log that object detection was used
    else:
        # Use saliency
        try:
            # Check if saliency module is available
            if hasattr(cv2, 'saliency'):  # Check if OpenCV saliency module is available
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()  # Create fine-grained saliency detector
                (success, saliencyMap) = saliency.computeSaliency(image)  # Compute saliency map for image
                if success:  # Check if saliency computation was successful
                    moments = cv2.moments(saliencyMap)  # Calculate image moments from saliency map
                    if moments['m00'] != 0:  # Check if total intensity (area) is non-zero
                        center_x = moments['m10'] / moments['m00']  # Calculate x-coordinate of centroid
                        center_y = moments['m01'] / moments['m00']  # Calculate y-coordinate of centroid
                    else:
                        center_x = image.shape[1] / 2  # Use image center x if no salient features
                        center_y = image.shape[0] / 2  # Use image center y if no salient features
                    logger.info("Used saliency map for crop center.")  # Log saliency usage
                else:
                    center_x = image.shape[1] / 2  # Fallback to geometric center x
                    center_y = image.shape[0] / 2  # Fallback to geometric center y
                    logger.info("Fallback to center crop (saliency computation failed).")  # Log fallback reason
            else:
                logger.warning("Saliency module not available in OpenCV installation.")  # Log missing saliency module
                center_x = image.shape[1] / 2  # Use geometric center x as fallback
                center_y = image.shape[0] / 2  # Use geometric center y as fallback
                logger.info("Fallback to center crop (saliency module not available).")  # Log fallback reason
        except Exception as e:  # Handle any saliency computation errors
            logger.warning(f"Error computing saliency: {e}")  # Log error details
            center_x = image.shape[1] / 2  # Use geometric center x as final fallback
            center_y = image.shape[0] / 2  # Use geometric center y as final fallback
            logger.info("Fallback to center crop (saliency error).")  # Log fallback reason
    return center_x, center_y  # Return calculated or fallback center coordinates

# Function to smart crop and resize
def smart_crop_and_resize(image, pred, target_width, target_height):
    h, w = image.shape[:2]  # Extract image height and width from shape
    target_ar = target_width / target_height  # Calculate target aspect ratio
    orig_ar = w / h  # Calculate original image aspect ratio
    center_x, center_y = get_crop_center(pred, image)  # Get optimal crop center using object detection or saliency
    
    if orig_ar > target_ar:  # Original image is wider than target aspect ratio
        # Crop width
        crop_h = h  # Keep full height
        crop_w = int(crop_h * target_ar)  # Calculate crop width based on target aspect ratio
        left = int(center_x - crop_w / 2)  # Calculate left edge of crop, centered on optimal point
        left = max(0, min(left, w - crop_w))  # Clamp left edge to valid range [0, w-crop_w]
        cropped = image[:, left:left + crop_w]  # Crop image horizontally, keeping full height
    else:
        # Crop height
        crop_w = w  # Keep full width
        crop_h = int(crop_w / target_ar)  # Calculate crop height based on target aspect ratio
        top = int(center_y - crop_h / 2)  # Calculate top edge of crop, centered on optimal point
        top = max(0, min(top, h - crop_h))  # Clamp top edge to valid range [0, h-crop_h]
        cropped = image[top:top + crop_h, :]  # Crop image vertically, keeping full width
    
    # Resize
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)  # Resize cropped image to exact target dimensions using area interpolation
    logger.info(f"Resized image to {target_width}x{target_height}.")  # Log resize operation
    return resized  # Return processed image

# Process each image: analyze, resize, save
stats_file = "image_stats.json"  # Define filename for storing image statistics
stats = {}  # Initialize dictionary to store statistics
if os.path.exists(stats_file):  # Check if previous statistics file exists
    with open(stats_file, 'r') as f:  # Open existing statistics file for reading
        stats = json.load(f)  # Load previous statistics from JSON file
    logger.info("Loaded previous image stats.")  # Log successful loading of previous stats

for idx, (filename, image, pred) in enumerate(zip(image_files, images, all_preds)):  # Iterate through images with their filenames and predictions
    base_name = os.path.basename(filename)  # Extract filename without path for logging
    logger.info(f"Analyzing image {idx+1}/{len(images)}: {base_name}")  # Log current image processing progress
    
    # Compute statistics
    means, stds = cv2.meanStdDev(image)  # Calculate mean and standard deviation for each color channel
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])  # Calculate histogram for blue channel
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])  # Calculate histogram for green channel
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])  # Calculate histogram for red channel
    # Convert to grayscale for moments calculation
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert BGR image to grayscale for moment analysis
    moments = cv2.moments(gray_image)  # Calculate spatial moments of grayscale image
    hu_moments = cv2.HuMoments(moments).flatten().tolist()  # Calculate Hu invariant moments and convert to list
    
    image_stats = {  # Create dictionary to store computed image statistics
        'means': means.flatten().tolist(),  # Convert mean values to flat list for JSON serialization
        'stds': stds.flatten().tolist(),  # Convert standard deviation values to flat list
        'hu_moments': hu_moments,  # Include Hu invariant moments for shape analysis
        # Histograms are large, store summary like mean hist or skip detailed
        'hist_means': [float(np.mean(hist_b)), float(np.mean(hist_g)), float(np.mean(hist_r))]  # Store mean histogram values for each color channel
    }
    stats[base_name] = image_stats  # Store statistics for this image in main stats dictionary
    logger.info(f"Computed stats for {base_name}: {image_stats}")  # Log computed statistics
    
    # For mathematical relations, e.g., correlation between channels
    ch1 = image[:,:,0].flatten()  # Flatten blue channel to 1D array
    ch2 = image[:,:,1].flatten()  # Flatten green channel to 1D array
    ch3 = image[:,:,2].flatten()  # Flatten red channel to 1D array
    
    # Add error handling for correlation calculations
    try:
        corr_12 = np.corrcoef(ch1, ch2)[0,1]  # Calculate correlation coefficient between blue and green channels
        corr_13 = np.corrcoef(ch1, ch3)[0,1]  # Calculate correlation coefficient between blue and red channels
        corr_23 = np.corrcoef(ch2, ch3)[0,1]  # Calculate correlation coefficient between green and red channels
        # Handle NaN values that might occur with constant channels
        if np.isnan(corr_12):  # Check if blue-green correlation is NaN (happens with constant channels)
            corr_12 = 0.0  # Set to zero if NaN
        if np.isnan(corr_13):  # Check if blue-red correlation is NaN
            corr_13 = 0.0  # Set to zero if NaN
        if np.isnan(corr_23):  # Check if green-red correlation is NaN
            corr_23 = 0.0  # Set to zero if NaN
    except Exception as e:  # Handle any errors in correlation calculation
        logger.warning(f"Error calculating correlations for {base_name}: {e}")  # Log error details
        corr_12 = corr_13 = corr_23 = 0.0  # Set all correlations to zero on error
    
    logger.info(f"Channel correlations for {base_name}: 1-2={corr_12:.4f}, 1-3={corr_13:.4f}, 2-3={corr_23:.4f}")  # Log correlation values
    
    # Resize
    resized = smart_crop_and_resize(image, pred, target_width, target_height)  # Apply intelligent cropping and resizing
    
    # Save resized
    output_path = os.path.join(output_dir, base_name)  # Create full output path for processed image
    try:
        success = cv2.imwrite(output_path, resized)  # Save processed image to output directory
        if success:  # Check if save operation was successful
            logger.info(f"Saved resized image to {output_path}")  # Log successful save
        else:
            logger.error(f"Failed to save image to {output_path}")  # Log save failure
    except Exception as e:  # Handle any errors during image saving
        logger.error(f"Error saving image {base_name}: {e}")  # Log error details

# Save updated stats
try:
    with open(stats_file, 'w') as f:  # Open stats file for writing
        json.dump(stats, f, indent=2)  # Write statistics dictionary to JSON file with formatting
    logger.info("Saved image stats to image_stats.json")  # Log successful stats save
except Exception as e:  # Handle any errors during stats file saving
    logger.error(f"Failed to save stats file: {e}")  # Log error details

# Deep learning: Train autoencoder to learn representations
class ImageDataset(Dataset):  # Define custom PyTorch dataset class for image data
    def __init__(self, images):  # Constructor takes list of image arrays
        self.transform = transforms.Compose([  # Define image preprocessing pipeline
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((128, 128)),  # Resize all images to consistent 128x128 size
            transforms.ToTensor()  # Convert PIL Image to normalized tensor (0-1 range)
        ])
        self.images = images  # Store reference to image list
    
    def __len__(self):  # Required method: return size of dataset
        return len(self.images)  # Return number of images in dataset
    
    def __getitem__(self, idx):  # Required method: return item at given index
        img = self.images[idx]  # Get image array at specified index
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB color space
        return self.transform(img)  # Apply transformations and return tensor

class ConvAutoencoder(nn.Module):  # Define convolutional autoencoder neural network class
    def __init__(self):  # Constructor for initializing network architecture
        super(ConvAutoencoder, self).__init__()  # Initialize parent Module class
        self.encoder = nn.Sequential(  # Define encoder part that compresses input
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # First conv layer: 3→16 channels, halves spatial dimensions
            nn.ReLU(),  # Apply ReLU activation for non-linearity
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Second conv layer: 16→32 channels, halves dimensions again
            nn.ReLU(),  # Apply ReLU activation
            nn.Conv2d(32, 64, 7)  # Final encoder layer: 32→64 channels with 7x7 kernel
        )
        self.decoder = nn.Sequential(  # Define decoder part that reconstructs from compressed representation
            nn.ConvTranspose2d(64, 32, 7),  # First deconv layer: 64→32 channels, expands spatial dimensions
            nn.ReLU(),  # Apply ReLU activation
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Second deconv: 32→16 channels, doubles dimensions
            nn.ReLU(),  # Apply ReLU activation
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # Final deconv: 16→3 channels, doubles dimensions
            nn.Sigmoid()  # Apply sigmoid to constrain output to [0,1] range
        )
    
    def forward(self, x):  # Define forward pass through the network
        x = self.encoder(x)  # Pass input through encoder to get compressed representation
        x = self.decoder(x)  # Pass compressed representation through decoder to reconstruct
        return x  # Return reconstructed image

# Create dataset and loader
if len(images) > 0:  # Check if any images were successfully loaded
    dataset = ImageDataset(images)  # Create dataset instance with loaded images
    # Adjust batch size based on number of images
    batch_size = min(32, len(images))  # Use smaller batch size if fewer images available
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create data loader with shuffling for training
    logger.info("Prepared dataset for autoencoder training.")  # Log successful dataset preparation
else:
    logger.warning("No valid images to train autoencoder")  # Log warning if no images available
    dataloader = None  # Set dataloader to None to skip training

# Load or create autoencoder
autoencoder = ConvAutoencoder().to(device)  # Create autoencoder instance and move to compute device
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)  # Create Adam optimizer with learning rate 0.001
criterion = nn.MSELoss()  # Define Mean Squared Error loss function for reconstruction

model_path = "autoencoder.pth"  # Define path for saving/loading model weights
if os.path.exists(model_path):  # Check if pre-trained model exists
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))  # Load existing model weights
    logger.info("Loaded existing autoencoder model for continued learning.")  # Log model loading
else:
    logger.info("Created new autoencoder model.")  # Log creation of new model

# Train
if dataloader is not None:  # Only train if we have valid data
    num_epochs = 10  # Adjustable, for learning  # Set number of training epochs
    autoencoder.train()  # Set model to training mode (enables dropout, batch norm updates)
    for epoch in range(num_epochs):  # Iterate through specified number of epochs
        epoch_loss = 0.0  # Initialize loss accumulator for this epoch
        batch_count = 0  # Initialize batch counter for averaging loss
        for data in dataloader:  # Iterate through batches in the dataloader
            try:
                data = data.to(device)  # Move batch data to compute device
                output = autoencoder(data)  # Forward pass: generate reconstruction
                loss = criterion(output, data)  # Calculate reconstruction loss between output and input
                optimizer.zero_grad()  # Clear gradients from previous iteration
                loss.backward()  # Backpropagate gradients through network
                optimizer.step()  # Update model parameters using computed gradients
                epoch_loss += loss.item()  # Accumulate loss for epoch averaging
                batch_count += 1  # Increment batch counter
            except RuntimeError as e:  # Handle runtime errors during training
                if "out of memory" in str(e):  # Check for GPU memory issues
                    logger.warning("GPU out of memory during training, skipping batch")  # Log memory warning
                    torch.cuda.empty_cache()  # Clear GPU memory cache
                    continue  # Skip this batch and continue with next
                else:
                    raise e  # Re-raise other runtime errors
        if batch_count > 0:  # Check if any batches were processed successfully
            avg_loss = epoch_loss / batch_count  # Calculate average loss for this epoch
            logger.info(f"Autoencoder training epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")  # Log training progress
        else:
            logger.warning(f"No batches processed in epoch {epoch+1}")  # Log warning if no batches processed

    # Save model
    try:
        torch.save(autoencoder.state_dict(), model_path)  # Save trained model weights to disk
        logger.info("Saved updated autoencoder model after learning from current images.")  # Log successful save
    except Exception as e:  # Handle any errors during model saving
        logger.error(f"Failed to save model: {e}")  # Log error details
else:
    logger.info("Skipped autoencoder training due to no valid images")  # Log skip reason

logger.info("Script completed successfully.")  # Log successful completion of entire script
