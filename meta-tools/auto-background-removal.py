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

def soft_cross_entropy(inputs, targets):
    return -(targets * F.log_softmax(inputs, dim=-1)).sum(dim=-1).mean()

def get_features(img_path):
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

def remove_bg(img_path, method_idx):
    try:
        model_name = models_list[method_idx]
        session = load_model(model_name)
        if session is None:
            return None, None, None
        
        # Read and resize image for faster processing
        img = Image.open(img_path)
        
        # Resize if image is too large
        max_size = 1024
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            log_message(f"Resized image from {img.size} for processing")
        
        # Convert to bytes
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
    with torch.no_grad():
        logits = classifier(current_features.unsqueeze(0))[0]
        sorted_indices = torch.argsort(logits, descending=True)
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
    if current_features is not None:
        output = classifier(current_features.unsqueeze(0))
        target = torch.full((1, num_methods), 1.0 / (num_methods - 1), device=device)
        target[0, current_method_idx] = 0
        optimizer.zero_grad()
        loss = soft_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        log_message(f"Rejected method {current_method_idx}, trained with loss {loss.item()}")
    with torch.no_grad():
        logits = classifier(current_features.unsqueeze(0))[0]
        sorted_indices = torch.argsort(logits, descending=True)
    found = False
    for idx in sorted_indices:
        temp_idx = idx.item()
        if temp_idx == current_method_idx:
            continue
        res = current_removals[temp_idx]
        if res is not None and res[0] is not None:
            current_preview, current_rgb, current_alpha = res
            current_method_idx = temp_idx
            found = True
            break
    if not found:
        log_message(f"Warning: No alternative background removal method worked for {current_image_path}")
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
    label = torch.tensor([current_method_idx]).to(device)
    optimizer.zero_grad()
    output = classifier(current_features.unsqueeze(0))
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    log_message(f"Trained on method {current_method_idx} for image {current_image_path}, loss: {loss.item()}")
    save_model()
    
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

def recalculate_layout():
    global button_height, button_width, accept_rect, refresh_rect, lock_rect, auto_rect, paint_rect, invert_rect
    button_height = 50
    button_width = screen.get_width() // 6
    accept_rect = pygame.Rect(0, screen.get_height() - button_height, button_width, button_height)
    refresh_rect = pygame.Rect(button_width, screen.get_height() - button_height, button_width, button_height)
    lock_rect = pygame.Rect(2 * button_width, screen.get_height() - button_height, button_width, button_height)
    auto_rect = pygame.Rect(3 * button_width, screen.get_height() - button_height, button_width, button_height)
    paint_rect = pygame.Rect(4 * button_width, screen.get_height() - button_height, button_width, button_height)
    invert_rect = pygame.Rect(5 * button_width, screen.get_height() - button_height, button_width, button_height)

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
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if accept_rect.collidepoint(pos):
                if current_image_path and current_preview:
                    accept_and_train()
                    save_cropped()
                    load_next_image()
            elif refresh_rect.collidepoint(pos):
                if current_image_path:
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

        pygame.display.flip()
        last_render_time = current_time

    if not current_image_path and not image_queue:
        running = False

preload_executor.shutdown(wait=True)
pygame.quit()
save_model()
log_message("Script completed.")
log_file.close()