import subprocess
import os
import sys
import numpy as np
import time
import logging
from typing import List, Optional
import urllib.request
import torch.nn.functional as F
import cv2
from PIL import Image
import pygame
import torch
import torchvision
import concurrent.futures
import functools
import gc
import platform
import psutil

log_file_path = 'feature_crop_learner.log'
log_file = open(log_file_path, 'w')

def log_message(message):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

log_message("Script started. Checking for required libraries...")

# Check if running on Windows
is_windows = platform.system() == 'Windows'
log_message(f"Running on {platform.system()} system")

# Install psutil for memory monitoring
try:
    import psutil
    log_message("psutil is already installed.")
except ImportError:
    log_message("psutil is not installed. Installing now...")
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'psutil'])
    import psutil
    log_message("psutil installed successfully.")

# Install onnxruntime first as it's required by rembg
try:
    import onnxruntime
    log_message("onnxruntime is already installed.")
except ImportError:
    log_message("onnxruntime is not installed. Installing now...")
    # Force CPU version on Windows with Intel
    pkg = 'onnxruntime'
    subprocess.call([sys.executable, '-m', 'pip', 'install', pkg])
    import onnxruntime
    log_message(f"{pkg} installed successfully.")

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
    import torchvision
    log_message("PyTorch and torchvision are already installed.")
except ImportError:
    log_message("PyTorch is not installed. Installing the latest version now...")
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'])
    import torch
    import torchvision
    log_message("PyTorch installed successfully.")

try:
    from PIL import Image
    log_message("Pillow is already installed.")
except ImportError:
    log_message("Pillow is not installed. Installing the latest version now...")
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'pillow'])
    from PIL import Image
    log_message("Pillow installed successfully.")

try:
    import pygame
    log_message("Pygame is already installed.")
except ImportError:
    log_message("Pygame is not installed. Installing the latest version now...")
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'pygame'])
    import pygame
    log_message("Pygame installed successfully.")

try:
    from scipy import stats
    log_message("scipy is already installed.")
except ImportError:
    log_message("scipy is not installed. Installing now...")
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'scipy'])
    from scipy import stats
    log_message("scipy installed successfully.")

# tkinter is usually included with Python, but check anyway
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    log_message("tkinter is available.")
except ImportError:
    log_message("tkinter is not available. File dialog functionality may be limited.")

log_message("All required libraries are installed and ready.")

# Updated to use rembg sessions instead of raw models
# Use fewer models for better performance
models_list = ['u2net', 'u2netp', 'u2net_human_seg']
num_methods = len(models_list)

# Force CPU usage on Windows for stability
if is_windows:
    device = torch.device('cpu')
    torch.set_num_threads(2)  # Limit CPU threads for better stability
    log_message("Using CPU device (optimized for Windows)")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")

# Lazy load models instead of loading all at startup
models = {}
current_loaded_model = None

def load_model(model_name):
    global current_loaded_model
    if model_name not in models:
        try:
            # Unload previous model to save memory
            if current_loaded_model and current_loaded_model != model_name:
                if current_loaded_model in models:
                    del models[current_loaded_model]
                    gc.collect()
                    log_message(f"Unloaded model {current_loaded_model}")
            
            session = rembg.new_session(model_name)
            models[model_name] = session
            current_loaded_model = model_name
            log_message(f"Loaded rembg session for {model_name}")
        except Exception as e:
            log_message(f"Error loading rembg session for {model_name}: {e}")
            return None
    return models.get(model_name)

# Smaller image size for faster processing
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(128),
    torchvision.transforms.CenterCrop(112),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Use lighter model for feature extraction
feature_extractor = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor.fc = torch.nn.Identity()
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

input_dim = 512 + 512 + 17  # resnet18(512) + hist512 + aspect,entropy,edge(3) + mean*3,std*3(6) + skew_rgb*3,kurt_rgb*3,skew_gray,kurt_gray(8)=17
classifier = torch.nn.Sequential(
    torch.nn.Linear(input_dim, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, num_methods)
).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

model_path = 'crop_method_classifier.pth'
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        classifier.load_state_dict(checkpoint['state_dict'])
        log_message("Loaded existing classifier model.")
    except Exception as e:
        log_message(f"Error loading saved model: {e}")
        log_message("Starting with fresh classifier model.")
        # Remove the incompatible saved model
        os.remove(model_path)

def extract_comprehensive_features(img_path):
    """Extract comprehensive features including differential profiles"""
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_cv = cv2.imread(img_path)
        height, width, _ = img_cv.shape
        
        # Basic ResNet features
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            resnet_features = feature_extractor(input_tensor).squeeze()
        
        # Convert to grayscale for intensity analysis
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Intensity profiles
        horizontal_profile = np.mean(gray, axis=0)
        vertical_profile = np.mean(gray, axis=1)
        
        # First differential (edge strength)
        h_diff1 = np.diff(horizontal_profile)
        v_diff1 = np.diff(vertical_profile)
        
        # Second differential (curvature)
        h_diff2 = np.diff(h_diff1)
        v_diff2 = np.diff(v_diff1)
        
        # Third differential (rate of curvature change)
        h_diff3 = np.diff(h_diff2)
        v_diff3 = np.diff(v_diff2)
        
        # Compute statistics from differential profiles
        diff_stats = np.array([
            np.mean(np.abs(h_diff1)), np.std(h_diff1),
            np.mean(np.abs(v_diff1)), np.std(v_diff1),
            np.mean(np.abs(h_diff2)), np.std(h_diff2),
            np.mean(np.abs(v_diff2)), np.std(v_diff2),
            np.mean(np.abs(h_diff3)), np.std(h_diff3),
            np.mean(np.abs(v_diff3)), np.std(v_diff3)
        ])
        
        # Geometry features
        aspect_ratio = width / float(height)
        
        # Contour analysis
        edges = cv2.Canny(img_cv, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Fit ellipse if possible
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
                eccentricity = np.sqrt(1 - (min(ellipse[1]) / max(ellipse[1]))**2)
            else:
                ellipse_area = area
                eccentricity = 0
                
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Moments
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                # Normalized centroid position
                cx_norm = cx / width
                cy_norm = cy / height
            else:
                cx_norm = cy_norm = 0.5
        else:
            area = perimeter = ellipse_area = eccentricity = solidity = 0
            cx_norm = cy_norm = 0.5
        
        geometry_features = np.array([
            aspect_ratio, area / (width * height), perimeter / (2 * (width + height)),
            ellipse_area / (width * height), eccentricity, solidity,
            cx_norm, cy_norm
        ])
        
        # Color statistics
        img_flat = img_cv.reshape(-1, 3).astype(np.float64)
        mean = np.mean(img_flat, axis=0) / 255.0
        std = np.std(img_flat, axis=0) / 255.0
        skew_rgb = stats.skew(img_flat, axis=0)
        kurt_rgb = stats.kurtosis(img_flat, axis=0)
        
        # Grayscale statistics
        gray_flat = gray.flatten()
        skew_gray = stats.skew(gray_flat)
        kurt_gray = stats.kurtosis(gray_flat)
        
        # Histogram features
        hist = cv2.calcHist([img_cv], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten()
        hist = hist / (height * width)
        
        # Entropy
        hist_gray, _ = np.histogram(gray, bins=256, range=(0,255))
        hist_gray = hist_gray / (height * width)
        entropy = -np.sum(hist_gray * np.log2(hist_gray + 1e-7))
        
        # Edge density
        edge_density = np.sum(edges) / (height * width * 255.0)
        
        # Texture features (using Sobel gradients)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_energy = np.mean(gradient_magnitude)
        texture_entropy = -np.sum(gradient_magnitude * np.log2(gradient_magnitude + 1e-7)) / gradient_magnitude.size
        
        # Combine all features
        additional_features = np.concatenate([
            [entropy, edge_density, texture_energy, texture_entropy],
            mean, std, skew_rgb, kurt_rgb, [skew_gray, kurt_gray],
            diff_stats, geometry_features, hist
        ])
        
        additional_t = torch.from_numpy(additional_features).float().to(device)
        full_features = torch.cat((resnet_features, additional_t))
        
        return full_features
        
    except Exception as e:
        log_message(f"Error extracting comprehensive features from {img_path}: {e}")
        return None

def get_features(img_path):
    """Original feature extraction for compatibility"""
    try:
        img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            resnet_features = feature_extractor(input_tensor).squeeze()

        # Additional statistics
        img_cv = cv2.imread(img_path)
        height, width, _ = img_cv.shape
        aspect_ratio = width / float(height)
        img_flat = img_cv.reshape(-1, 3).astype(np.float64)
        mean = np.mean(img_flat, axis=0) / 255.0
        std = np.std(img_flat, axis=0) / 255.0
        skew_rgb = stats.skew(img_flat, axis=0)
        kurt_rgb = stats.kurtosis(img_flat, axis=0)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray_flat = gray.flatten().astype(np.float64)
        skew_gray = stats.skew(gray_flat)
        kurt_gray = stats.kurtosis(gray_flat)
        hist = cv2.calcHist([img_cv], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten() / (height * width)
        hist_gray, _ = np.histogram(gray, bins=256, range=(0,255))
        hist_gray = hist_gray / (height * width)
        entropy = -np.sum(hist_gray * np.log2(hist_gray + 1e-7))
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (height * width * 255.0)
        additional = np.concatenate(([aspect_ratio, entropy, edge_density], mean, std, skew_rgb, kurt_rgb, [skew_gray, kurt_gray], hist))
        additional_t = torch.from_numpy(additional).float().to(device)
        full_features = torch.cat((resnet_features, additional_t))
        return full_features
    except Exception as e:
        log_message(f"Error extracting features from {img_path}: {e}")
        return None

def analyze_reference_directory(directory_path):
    """Analyze all images in reference directory and train classifier"""
    log_message(f"Analyzing reference directory: {directory_path}")
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    ref_images = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                  if f.lower().endswith(image_extensions)]
    
    if not ref_images:
        log_message("No images found in reference directory")
        return False
    
    log_message(f"Found {len(ref_images)} reference images")
    
    # Clear existing knowledge (reset classifier)
    global classifier, optimizer
    
    # Reinitialize with expanded feature dimensions for comprehensive features
    comprehensive_input_dim = input_dim + 12 + 8 + 4  # Original + diff_stats(12) + geometry(8) + texture(4)
    
    # Create new classifier with comprehensive feature support
    classifier = torch.nn.Sequential(
        torch.nn.Linear(comprehensive_input_dim, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, num_methods)
    ).to(device)
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    
    # Extract features from all reference images
    reference_features = []
    for img_path in ref_images:
        features = extract_comprehensive_features(img_path)
        if features is not None:
            reference_features.append(features)
    
    if not reference_features:
        log_message("Failed to extract features from reference images")
        return False
    
    # Stack features
    reference_features = torch.stack(reference_features)
    
    # Compute mean and std of reference features
    ref_mean = torch.mean(reference_features, dim=0)
    ref_std = torch.std(reference_features, dim=0) + 1e-6
    
    # Train classifier to prefer method that produces similar statistics
    # This is done by creating synthetic training data
    log_message("Training classifier on reference statistics...")
    
    # Create synthetic targets - we'll test each method and see which produces
    # results most similar to our reference statistics
    best_method_scores = torch.zeros(num_methods)
    
    # Test a subset of reference images with each method
    test_images = ref_images[:min(10, len(ref_images))]  # Test up to 10 images
    
    for method_idx in range(num_methods):
        method_score = 0
        tested = 0
        
        for img_path in test_images:
            try:
                # Try this method on the image
                result = remove_bg(img_path, method_idx)
                if result[0] is not None:
                    # Extract features from result
                    # Save temporary result to extract features
                    temp_path = "temp_result.png"
                    result[0].save(temp_path, 'PNG')
                    result_features = extract_comprehensive_features(temp_path)
                    os.remove(temp_path)
                    
                    if result_features is not None:
                        # Compare to reference statistics
                        normalized_diff = (result_features - ref_mean) / ref_std
                        similarity = torch.exp(-torch.mean(normalized_diff ** 2))
                        method_score += similarity.item()
                        tested += 1
            except Exception as e:
                log_message(f"Error testing method {method_idx}: {e}")
        
        if tested > 0:
            best_method_scores[method_idx] = method_score / tested
    
    # Normalize scores to create probability distribution
    best_method_scores = F.softmax(best_method_scores, dim=0)
    
    log_message(f"Method preference scores: {best_method_scores}")
    
    # Train classifier with synthetic data based on reference statistics
    num_synthetic_samples = 100
    for epoch in range(10):
        total_loss = 0
        
        for _ in range(num_synthetic_samples):
            # Generate synthetic feature near reference distribution
            noise = torch.randn_like(ref_mean) * ref_std * 0.3
            synthetic_feature = ref_mean + noise
            
            # Train to predict best method distribution
            optimizer.zero_grad()
            output = classifier(synthetic_feature.unsqueeze(0))
            loss = soft_cross_entropy(output, best_method_scores.unsqueeze(0))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_synthetic_samples
        log_message(f"Reference training epoch {epoch+1}/10, loss: {avg_loss:.4f}")
    
    # Save the trained model
    save_model()
    log_message("Reference directory analysis complete")
    return True

def show_reference_popup():
    """Show popup asking about loading reference directory"""
    try:
        # Initialize Tk
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        # Ask user
        result = messagebox.askyesno(
            "Load Reference Directory",
            "Would you like to load a past directory as reference?\n\n"
            "This will analyze all images in the directory and train the system "
            "to produce similar crops. The existing knowledge base will be cleared.",
            icon='question'
        )
        
        if result:
            # Ask for directory
            directory = filedialog.askdirectory(
                title="Select Reference Directory with Cropped Images"
            )
            
            if directory:
                # Show progress
                progress_window = tk.Toplevel()
                progress_window.title("Analyzing Reference Directory")
                progress_window.geometry("400x100")
                
                label = tk.Label(progress_window, text="Analyzing reference images...", pady=20)
                label.pack()
                
                progress_window.update()
                
                # Analyze directory
                success = analyze_reference_directory(directory)
                
                progress_window.destroy()
                
                if success:
                    messagebox.showinfo("Success", "Reference directory analyzed successfully!")
                else:
                    messagebox.showerror("Error", "Failed to analyze reference directory")
                
                root.destroy()
                return True
        
        root.destroy()
        return False
        
    except Exception as e:
        log_message(f"Error showing reference popup: {e}")
        # Fallback to console
        try:
            response = input("Would you like to load a past directory as reference? (yes/no): ").strip().lower()
            if response == 'yes':
                directory = input("Enter the full path to the reference directory: ").strip()
                if directory and os.path.exists(directory):
                    return analyze_reference_directory(directory)
        except:
            pass
        return False

def soft_cross_entropy(inputs, targets):
    return -(targets * F.log_softmax(inputs, dim=-1)).sum(dim=-1).mean()

def remove_bg(img_path, method_idx):
    try:
        model_name = models_list[method_idx]
        session = load_model(model_name)
        if session is None:
            return None, None, None
        
        # Read original image
        img = Image.open(img_path)
        original_size = img.size
        
        # Convert to bytes (preserving original size)
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        input_data = buffer.getvalue()
        
        # Use rembg to remove background
        output_data = rembg.remove(input_data, session=session)
        
        # Convert to PIL Image
        output_image = Image.open(BytesIO(output_data))
        
        # Convert to RGBA if not already
        if output_image.mode != 'RGBA':
            output_image = output_image.convert('RGBA')
        
        # Extract RGB and alpha channels
        rgb_array = np.array(output_image)[:, :, :3]
        alpha_array = np.array(output_image)[:, :, 3]
        
        # Check if the alpha channel is not all zeros
        if np.all(alpha_array == 0):
            return None, None, None
            
        return output_image, rgb_array, alpha_array
        
    except Exception as e:
        log_message(f"Error processing image with method {models_list[method_idx]}: {e}")
        return None, None, None

def save_model():
    torch.save({'state_dict': classifier.state_dict()}, model_path)
    log_message("Saved classifier model.")

# Show reference popup before main program
log_message("Showing reference directory popup...")
show_reference_popup()

log_message("Please enter the full path to the directory containing images to process.")
input_dir = input().strip()
log_message(f"User provided input directory: {input_dir}")

if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
    log_message("Error: The input directory does not exist or is not a directory. Please check the path and run the script again.")
    log_file.close()
    sys.exit(1)

log_message("Please enter the full path to the output directory for cropped images.")
output_dir = input().strip()
log_message(f"User provided output directory: {output_dir}")

if not os.path.exists(output_dir):
    log_message(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir)

image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
image_paths.sort()
log_message(f"Found {len(image_paths)} images in the input directory.")

if not image_paths:
    log_message("No images found. Exiting.")
    log_file.close()
    sys.exit(0)

pygame.init()
screen_width, screen_height = 1200, 600
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Feature Crop Learner")
font = pygame.font.SysFont(None, 30)

auto_mode = False
refresh_lock = False
paint_mode = False
mask_modified = False
brush_radius = 20
image_queue = list(image_paths)
current_image_path = None
current_features = None
current_method_idx = 0
current_preview = None
current_rgb = None
current_alpha = None
current_removals = None
last_paint_pos = None
last_refresh_time = 0
last_render_time = time.time()
selected = False

# Reference image variables
reference_mode = False
reference_image_path = None
reference_mask = None
reference_features = None

preloaded = {}
# Reduce workers on Windows
max_workers = 1 if is_windows else 2
preload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

def preload_image(path):
    # Check memory usage
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        log_message(f"High memory usage ({memory_percent}%), skipping preload")
        return None
        
    features = get_features(path)
    if features is None:
        return None
    removals = [None] * num_methods
    # Process sequentially on Windows for stability
    if is_windows:
        for i in range(num_methods):
            try:
                removals[i] = remove_bg(path, i)
            except Exception as e:
                log_message(f"Error in preload removal for method {i}: {e}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(remove_bg, path, i): i for i in range(num_methods)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    removals[i] = future.result()
                except Exception as e:
                    log_message(f"Error in preload removal for method {i}: {e}")
    return {'features': features, 'removals': removals}

def set_preloaded(path, future):
    try:
        data = future.result()
        if data:
            preloaded[path] = data
    except Exception as e:
        log_message(f"Error setting preloaded for {path}: {e}")

def preload_next(n):
    # Check memory before preloading
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 70:
        log_message(f"Memory usage at {memory_percent}%, reducing preload")
        n = min(1, n)
    
    for i in range(min(n, len(image_queue))):
        path = image_queue[i]
        if path not in preloaded:
            future = preload_executor.submit(preload_image, path)
            future.add_done_callback(functools.partial(set_preloaded, path))

def load_next_image():
    global current_image_path, current_features, current_method_idx, current_preview, current_rgb, current_alpha, mask_modified, current_removals
    mask_modified = False
    
    # Clean up memory before loading next image
    gc.collect()
    
    while image_queue:
        current_image_path = image_queue.pop(0)
        if current_image_path in preloaded:
            data = preloaded.pop(current_image_path)
            current_features = data['features']
            current_removals = data['removals']
        else:
            current_features = get_features(current_image_path)
            if current_features is None:
                log_message(f"Skipping image due to feature extraction failure: {current_image_path}")
                continue
            current_removals = [None] * num_methods
            # Process sequentially on Windows
            if is_windows:
                for i in range(num_methods):
                    try:
                        current_removals[i] = remove_bg(current_image_path, i)
                    except Exception as e:
                        log_message(f"Error in removal for method {i}: {e}")
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {executor.submit(remove_bg, current_image_path, i): i for i in range(num_methods)}
                    for future in concurrent.futures.as_completed(futures):
                        i = futures[future]
                        try:
                            current_removals[i] = future.result()
                        except Exception as e:
                            log_message(f"Error in removal for method {i}: {e}")
        current_method_idx = 0
        update_preview()
        log_message(f"Loaded image: {current_image_path}")
        # Reduce preload on Windows
        preload_count = 2 if is_windows else 5
        preload_next(preload_count)
        return
    current_image_path = None
    current_features = None
    current_preview = None
    current_rgb = None
    current_alpha = None
    current_removals = None

def update_preview():
    global current_method_idx, current_preview, current_rgb, current_alpha
    if current_features is None:
        log_message("Error: No features available for current image")
        return
    
    # Handle different feature dimensions
    try:
        with torch.no_grad():
            logits = classifier(current_features.unsqueeze(0))[0]
            sorted_indices = torch.argsort(logits, descending=True)
    except RuntimeError as e:
        # If dimensions don't match, it might be using old features with new classifier
        # Try to pad features to match expected dimensions
        log_message(f"Feature dimension mismatch, attempting to adjust: {e}")
        expected_dim = classifier[0].in_features
        current_dim = current_features.shape[0]
        
        if current_dim < expected_dim:
            # Pad with zeros
            padding = torch.zeros(expected_dim - current_dim, device=device)
            adjusted_features = torch.cat([current_features, padding])
            
            with torch.no_grad():
                logits = classifier(adjusted_features.unsqueeze(0))[0]
                sorted_indices = torch.argsort(logits, descending=True)
        else:
            # Can't handle this case, use default ordering
            sorted_indices = torch.arange(num_methods)
    
    found = False
    for idx in sorted_indices:
        idx = idx.item()
        res = current_removals[idx]
        if res is not None and res[0] is not None:
            current_preview, current_rgb, current_alpha = res
            current_method_idx = idx
            found = True
            break
    if not found:
        log_message(f"Warning: No background removal method worked for {current_image_path}")

def refresh_preview():
    global current_method_idx, current_preview, current_rgb, current_alpha, mask_modified
    original_method = current_method_idx
    
    # Only apply negative training if we have features and we're not in reference mode
    if current_features is not None and not reference_mode:
        # Create target that reduces probability of current method
        target = torch.zeros(1, num_methods, device=device)
        # Set all other methods to have higher probability
        for i in range(num_methods):
            if i != current_method_idx:
                target[0, i] = 1.0 / (num_methods - 1)
        # Current method gets zero probability (negative feedback)
        target[0, current_method_idx] = 0.0
        
        # Handle dimension mismatch
        try:
            optimizer.zero_grad()
            output = classifier(current_features.unsqueeze(0))
            loss = soft_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            log_message(f"NEGATIVE FEEDBACK: Rejected method {models_list[current_method_idx]}, loss: {loss.item():.4f}")
        except RuntimeError as e:
            log_message(f"Dimension mismatch during negative feedback: {e}")
        
        # Save the updated model after negative feedback
        save_model()
    
    # Find the next best method that actually works
    if current_features is not None:
        try:
            with torch.no_grad():
                logits = classifier(current_features.unsqueeze(0))[0]
                sorted_indices = torch.argsort(logits, descending=True)
        except:
            # Fallback to sequential order
            sorted_indices = list(range(num_methods))
    else:
        # Fallback: try methods in order
        sorted_indices = list(range(num_methods))
    
    found = False
    for idx in sorted_indices:
        temp_idx = idx.item() if hasattr(idx, 'item') else idx
        # Skip the current method that we just rejected
        if temp_idx == original_method:
            continue
        res = current_removals[temp_idx]
        if res is not None and res[0] is not None:
            current_preview, current_rgb, current_alpha = res
            current_method_idx = temp_idx
            found = True
            log_message(f"Switched to method {models_list[current_method_idx]}")
            break
    
    if not found:
        log_message(f"Warning: No alternative background removal method worked for {current_image_path}")
        # If no alternative worked, try the original method again
        if original_method < len(current_removals) and current_removals[original_method] is not None:
            res = current_removals[original_method]
            if res[0] is not None:
                current_preview, current_rgb, current_alpha = res
                current_method_idx = original_method
    
    mask_modified = False

def invert_mask():
    global current_alpha, current_preview, mask_modified
    if current_alpha is not None and current_rgb is not None:
        current_alpha = 255 - current_alpha
        current_preview = Image.fromarray(np.dstack((current_rgb, current_alpha)), 'RGBA')
        mask_modified = True
        log_message("Inverted mask")

def accept_and_train():
    global mask_modified
    if current_features is None:
        log_message("Error: No features available for training")
        return
    
    # Positive feedback: train the classifier to prefer this method for this type of image
    if not reference_mode:
        label = torch.tensor([current_method_idx]).to(device)
        
        try:
            optimizer.zero_grad()
            output = classifier(current_features.unsqueeze(0))
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            # Log with method name for clarity
            method_name = models_list[current_method_idx] if current_method_idx < len(models_list) else f"method_{current_method_idx}"
            log_message(f"POSITIVE FEEDBACK: Accepted method {method_name} for image {os.path.basename(current_image_path)}, loss: {loss.item():.4f}")
            save_model()
        except RuntimeError as e:
            log_message(f"Dimension mismatch during positive feedback: {e}")
    else:
        log_message(f"Accepted result from reference-based cropping for image {os.path.basename(current_image_path)}")
    
    # Note: Fine-tuning of rembg models is not supported in this version
    # The original fine-tuning code has been removed as it's not compatible with rembg sessions
    if mask_modified:
        log_message("Mask was modified but fine-tuning is not supported with rembg sessions")
    
    mask_modified = False

def save_cropped():
    if current_preview and current_image_path:
        output_path = os.path.join(output_dir, os.path.basename(current_image_path))
        current_preview.save(output_path, 'PNG')
        log_message(f"Saved image to {output_path}")

def select_reference_image():
    """Open file dialog to select a reference image"""
    global reference_image_path, reference_mask, reference_features
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        if file_path:
            reference_image_path = file_path
            reference_features = extract_reference_features(file_path)
            reference_mask = create_reference_mask(file_path)
            log_message(f"Selected reference image: {file_path}")
            return True
        return False
    except Exception as e:
        log_message(f"Error selecting reference image: {e}")
        # Fallback: ask for manual input
        try:
            log_message("Please enter the full path to the reference image:")
            file_path = input().strip()
            if file_path and os.path.exists(file_path):
                reference_image_path = file_path
                reference_features = extract_reference_features(file_path)
                reference_mask = create_reference_mask(file_path)
                log_message(f"Selected reference image: {file_path}")
                return True
        except Exception as e2:
            log_message(f"Error with manual input: {e2}")
        return False

def extract_reference_features(img_path):
    """Extract features from reference image for similarity matching"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract various features
        features = {}
        
        # Color histograms
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
        features['color_hist'] = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features['contour_area'] = cv2.contourArea(largest_contour)
            features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
        else:
            features['contour_area'] = 0
            features['contour_perimeter'] = 0
            
        return features
    except Exception as e:
        log_message(f"Error extracting reference features: {e}")
        return None

def create_reference_mask(img_path):
    """Create a mask from reference image (assuming it's already cropped/processed)"""
    try:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            # Use alpha channel as mask
            alpha = np.array(img)[:, :, 3]
            return alpha > 0
        else:
            # For RGB images, create mask based on background detection
            img_array = np.array(img)
            # Simple background detection - assume corners are background
            corner_samples = [
                img_array[0, 0],
                img_array[0, -1], 
                img_array[-1, 0],
                img_array[-1, -1]
            ]
            bg_color = np.mean(corner_samples, axis=0)
            
            # Create mask where pixels are significantly different from background
            diff = np.linalg.norm(img_array - bg_color, axis=2)
            threshold = np.std(diff) * 2
            return diff > threshold
    except Exception as e:
        log_message(f"Error creating reference mask: {e}")
        return None

def apply_reference_based_crop(img_path):
    """Apply cropping based on reference image similarity"""
    global current_preview, current_rgb, current_alpha
    
    if reference_features is None:
        log_message("No reference image selected")
        return None, None, None
        
    try:
        # Extract features from current image
        current_features_ref = extract_reference_features(img_path)
        if current_features_ref is None:
            return None, None, None
            
        # Load current image
        img = Image.open(img_path)
        img_array = np.array(img.convert('RGB'))
        
        # Simple similarity-based segmentation
        # This is a basic implementation - can be enhanced with more sophisticated methods
        
        # Compare color histograms
        ref_hist = reference_features['color_hist']
        curr_hist = current_features_ref['color_hist']
        hist_similarity = cv2.compareHist(ref_hist.astype(np.float32), curr_hist.astype(np.float32), cv2.HISTCMP_CORREL)
        
        # Create mask based on color similarity and reference mask pattern
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Use adaptive thresholding and morphological operations
        if reference_mask is not None:
            # Analyze reference mask properties
            ref_is_bright = np.mean(reference_mask) > 0.5
            
            if ref_is_bright:
                # Looking for bright objects
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # Looking for dark objects  
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Default: use edge-based segmentation
            edges = cv2.Canny(gray, 50, 150)
            mask = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10,10), np.uint8))
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Convert mask to alpha channel
        alpha_array = mask.astype(np.uint8)
        
        # Create RGBA image
        rgba_array = np.dstack((img_array, alpha_array))
        output_image = Image.fromarray(rgba_array, 'RGBA')
        
        # Check if mask is valid
        if np.all(alpha_array == 0):
            return None, None, None
            
        return output_image, img_array, alpha_array
        
    except Exception as e:
        log_message(f"Error applying reference-based crop: {e}")
        return None, None, None

def recalculate_layout():
    global button_height, button_width, accept_rect, refresh_rect, lock_rect, auto_rect, paint_rect, invert_rect, reference_rect
    button_height = 50
    button_width = screen.get_width() // 7  # Changed from 6 to 7 to accommodate new button
    accept_rect = pygame.Rect(0, screen.get_height() - button_height, button_width, button_height)
    refresh_rect = pygame.Rect(button_width, screen.get_height() - button_height, button_width, button_height)
    lock_rect = pygame.Rect(2 * button_width, screen.get_height() - button_height, button_width, button_height)
    auto_rect = pygame.Rect(3 * button_width, screen.get_height() - button_height, button_width, button_height)
    paint_rect = pygame.Rect(4 * button_width, screen.get_height() - button_height, button_width, button_height)
    invert_rect = pygame.Rect(5 * button_width, screen.get_height() - button_height, button_width, button_height)
    reference_rect = pygame.Rect(6 * button_width, screen.get_height() - button_height, button_width, button_height)

load_next_image()
recalculate_layout()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            recalculate_layout()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                if current_image_path and current_preview:
                    accept_and_train()
                    save_cropped()
                    load_next_image()
            elif event.key == pygame.K_r:
                if current_image_path:
                    if reference_mode and reference_image_path:
                        # Use reference-based cropping
                        result = apply_reference_based_crop(current_image_path)
                        if result[0] is not None:
                            current_preview, current_rgb, current_alpha = result
                            mask_modified = True
                            log_message("Applied reference-based cropping")
                        else:
                            log_message("Reference-based cropping failed, trying normal refresh")
                            refresh_preview()
                    else:
                        refresh_preview()
            elif event.key == pygame.K_l:
                refresh_lock = not refresh_lock
                log_message(f"Refresh lock toggled to {refresh_lock}")
            elif event.key == pygame.K_t:
                auto_mode = not auto_mode
                log_message(f"Auto mode toggled to {auto_mode}")
            elif event.key == pygame.K_p:
                paint_mode = not paint_mode
                log_message(f"Paint mode toggled to {paint_mode}")
            elif event.key == pygame.K_i:
                invert_mask()
            elif event.key == pygame.K_s:
                if reference_mode:
                    # Exit reference mode
                    reference_mode = False
                    log_message("Exited reference mode")
                else:
                    # Enter reference mode by selecting a reference image
                    if select_reference_image():
                        reference_mode = True
                        log_message("Entered reference mode")
                    else:
                        log_message("Reference mode not activated - no image selected")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if accept_rect.collidepoint(pos):
                if current_image_path and current_preview:
                    accept_and_train()
                    save_cropped()
                    load_next_image()
            elif refresh_rect.collidepoint(pos):
                if current_image_path:
                    if reference_mode and reference_image_path:
                        # Use reference-based cropping
                        result = apply_reference_based_crop(current_image_path)
                        if result[0] is not None:
                            current_preview, current_rgb, current_alpha = result
                            mask_modified = True
                            log_message("Applied reference-based cropping")
                        else:
                            refresh_preview()
                    else:
                        refresh_preview()
            elif lock_rect.collidepoint(pos):
                refresh_lock = not refresh_lock
                log_message(f"Refresh lock toggled to {refresh_lock}")
            elif auto_rect.collidepoint(pos):
                auto_mode = not auto_mode
                log_message(f"Auto mode toggled to {auto_mode}")
            elif paint_rect.collidepoint(pos):
                paint_mode = not paint_mode
                log_message(f"Paint mode toggled to {paint_mode}")
            elif invert_rect.collidepoint(pos):
                invert_mask()
            elif reference_rect.collidepoint(pos):
                if reference_mode:
                    # Exit reference mode
                    reference_mode = False
                    log_message("Exited reference mode")
                else:
                    # Enter reference mode by selecting a reference image
                    if select_reference_image():
                        reference_mode = True
                        log_message("Entered reference mode")
                    else:
                        log_message("Reference mode not activated - no image selected")

    if auto_mode and current_image_path:
        if current_preview:
            save_cropped()
        load_next_image()

    current_time = pygame.time.get_ticks()
    if refresh_lock and current_image_path and current_time - last_refresh_time > 500:
        refresh_preview()
        last_refresh_time = current_time

    pressed = pygame.mouse.get_pressed()
    pos = pygame.mouse.get_pos()
    if paint_mode and current_alpha is not None and current_rgb is not None and (pressed[0] or pressed[2]):
        prev_start_x = screen.get_width() // 2 + 10
        prev_start_y = 10
        prev_rect = pygame.Rect(prev_start_x, prev_start_y, screen.get_width() // 2 - 20, screen.get_height() - button_height - 20)
        if prev_rect.collidepoint(pos):
            rel_x = int((pos[0] - prev_start_x) / scale_prev)
            rel_y = int((pos[1] - prev_start_y) / scale_prev)
            if 0 <= rel_x < current_alpha.shape[1] and 0 <= rel_y < current_alpha.shape[0]:
                value = 255 if pressed[0] else 0
                
                # Ensure current_alpha is in the correct format for OpenCV
                if not current_alpha.flags.c_contiguous:
                    current_alpha = np.ascontiguousarray(current_alpha)
                
                if last_paint_pos is not None:
                    cv2.line(current_alpha, last_paint_pos, (rel_x, rel_y), (value,), thickness=brush_radius * 2)
                else:
                    cv2.circle(current_alpha, (rel_x, rel_y), brush_radius, (value,), -1)
                last_paint_pos = (rel_x, rel_y)
                current_preview = Image.fromarray(np.dstack((current_rgb, current_alpha)), 'RGBA')
                mask_modified = True
    else:
        last_paint_pos = None

    # Add small delay to reduce CPU usage
    if is_windows:
        pygame.time.wait(10)
    
    current_time = time.time()
    if not auto_mode or (current_time - last_render_time > 1):
        screen.fill((0, 0, 0))

        if current_image_path:
            try:
                orig_surf = pygame.image.load(current_image_path)
                orig_rect = orig_surf.get_rect()
                scale = min((screen.get_width() // 2 - 20) / orig_rect.w, (screen.get_height() - button_height - 20) / orig_rect.h)
                scaled_orig = pygame.transform.scale(orig_surf, (int(orig_rect.w * scale), int(orig_rect.h * scale)))
                screen.blit(scaled_orig, (10, 10))

                if current_preview:
                    prev_surf = pygame.image.fromstring(current_preview.tobytes(), current_preview.size, current_preview.mode)
                    prev_rect = prev_surf.get_rect()
                    scale_prev = min((screen.get_width() // 2 - 20) / prev_rect.w, (screen.get_height() - button_height - 20) / prev_rect.h)
                    scaled_prev = pygame.transform.scale(prev_surf, (int(prev_rect.w * scale_prev), int(prev_rect.h * scale_prev)))
                    screen.blit(scaled_prev, (screen.get_width() // 2 + 10, 10))

                    if paint_mode:
                        mouse_pos = pygame.mouse.get_pos()
                        if prev_rect.move(screen.get_width() // 2 + 10, 10).collidepoint(mouse_pos):
                            pygame.draw.circle(screen, (0, 255, 255), mouse_pos, brush_radius * scale_prev, 2)
            except Exception as e:
                log_message(f"Error displaying image: {e}")

        pygame.draw.rect(screen, (0, 255, 0), accept_rect)
        accept_text = font.render("Accept (A)", True, (0, 0, 0))
        screen.blit(accept_text, (accept_rect.centerx - accept_text.get_width() // 2, accept_rect.centery - accept_text.get_height() // 2))

        pygame.draw.rect(screen, (255, 165, 0), refresh_rect)
        refresh_text = font.render("Refresh (R)", True, (0, 0, 0))
        screen.blit(refresh_text, (refresh_rect.centerx - refresh_text.get_width() // 2, refresh_rect.centery - refresh_text.get_height() // 2))

        lock_color = (255, 0, 0) if refresh_lock else (0, 0, 255)
        pygame.draw.rect(screen, lock_color, lock_rect)
        lock_text = font.render("Unlock Refresh (L)" if refresh_lock else "Lock Refresh (L)", True, (255, 255, 255))
        screen.blit(lock_text, (lock_rect.centerx - lock_text.get_width() // 2, lock_rect.centery - lock_text.get_height() // 2))

        auto_color = (0, 0, 255) if not auto_mode else (255, 0, 0)
        pygame.draw.rect(screen, auto_color, auto_rect)
        auto_text = font.render("Auto (T)" if not auto_mode else "Manual (T)", True, (255, 255, 255))
        screen.blit(auto_text, (auto_rect.centerx - auto_text.get_width() // 2, auto_rect.centery - auto_text.get_height() // 2))

        paint_color = (255, 255, 0) if paint_mode else (0, 0, 255)
        pygame.draw.rect(screen, paint_color, paint_rect)
        paint_text = font.render("Exit Paint (P)" if paint_mode else "Paint (P)", True, (0, 255, 255))
        screen.blit(paint_text, (paint_rect.centerx - paint_text.get_width() // 2, paint_rect.centery - paint_text.get_height() // 2))

        pygame.draw.rect(screen, (128, 0, 128), invert_rect)
        invert_text = font.render("Invert (I)", True, (255, 255,255))
        screen.blit(invert_text, (invert_rect.centerx - invert_text.get_width() // 2, invert_rect.centery - invert_text.get_height() // 2))

        reference_color = (255, 0, 255) if reference_mode else (128, 128, 128)
        pygame.draw.rect(screen, reference_color, reference_rect)
        reference_text = font.render("Ref Mode (S)" if not reference_mode else "Exit Ref (S)", True, (255, 255, 255))
        screen.blit(reference_text, (reference_rect.centerx - reference_text.get_width() // 2, reference_rect.centery - reference_text.get_height() // 2))

        pygame.display.flip()
        last_render_time = current_time

    if not current_image_path and not image_queue:
        running = False

preload_executor.shutdown(wait=True)
pygame.quit()
save_model()
log_message("Script completed.")
log_file.close()