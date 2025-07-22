#!/usr/bin/env python3

import sys
import os
import subprocess
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Global variables for dependencies
REQUIRED_LIBRARIES = {
    'numpy': 'numpy',
    'opencv-python': 'cv2',
    'matplotlib': 'matplotlib',
    'torch': 'torch',
    'torchvision': 'torchvision.models',  # For deep learning models
    'lpips': 'lpips'  # Learned Perceptual Image Patch Similarity for deep metric
}

# Function to check and install dependencies
def check_and_install_dependencies():
    log_action("Starting dependency check and installation process.")
    for pip_name, import_name in REQUIRED_LIBRARIES.items():
        try:
            __import__(import_name)
            log_action(f"Dependency '{pip_name}' is already installed.")
        except ImportError:
            log_action(f"Dependency '{pip_name}' not found. Attempting to install latest version.")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', pip_name])
                log_action(f"Successfully installed '{pip_name}'.")
            except subprocess.CalledProcessError as e:
                log_action(f"Failed to install '{pip_name}'. Error: {str(e)}")
                sys.exit(1)
    # Re-import after potential installations
    global np, cv2, plt, torch, models, lpips
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import torch
    from torchvision import models
    import lpips  # Deep perceptual metric

# Function to setup logging to terminal and file
def setup_logging():
    log_dir = Path("ssim_logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ssim_extreme_{timestamp}.log"
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

# Wrapper to log every action
def log_action(message):
    logging.info(message)

# Function to ask user questions interactively
def ask_user(question, default=None, is_path=False, is_dir=False, is_bool=False):
    log_action(f"Asking user: {question}")
    if default:
        prompt = f"{question} (default: {default}): "
    else:
        prompt = f"{question}: "
    
    while True:
        response = input(prompt).strip()
        if not response and default:
            log_action(f"User accepted default: {default}")
            return default
        if response:
            if is_path or is_dir:
                path = Path(response)
                if is_dir and path.is_dir():
                    log_action(f"Valid directory provided: {path}")
                    return str(path)
                elif is_path and path.is_file():
                    log_action(f"Valid file provided: {path}")
                    return str(path)
                else:
                    print("Invalid path. Please try again.")
                    continue
            elif is_bool:
                if response.lower() in ['y', 'yes', 'true']:
                    log_action("User responded yes.")
                    return True
                elif response.lower() in ['n', 'no', 'false']:
                    log_action("User responded no.")
                    return False
                else:
                    print("Please respond with y/n or yes/no.")
                    continue
            log_action(f"User provided: {response}")
            return response

# Enhanced SSIM computation with multi-scale, deep integration, and extreme functionality
def compute_extreme_ssim(test_img_path, ref_img_path, output_dir):
    log_action(f"Starting extreme SSIM computation between {test_img_path} and {ref_img_path}.")
    
    # Load images
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    if test_img is None or ref_img is None:
        log_action("Failed to load one or both images.")
        sys.exit(1)
    
    # Ensure same size (resize ref to match test if needed)
    if test_img.shape != ref_img.shape:
        log_action("Resizing reference image to match test image dimensions.")
        ref_img = cv2.resize(ref_img, (test_img.shape[1], test_img.shape[0]))
    
    log_action("Images loaded and prepared.")
    
    # Base SSIM parameters (configurable via questions if needed, but hardcoded for intensity)
    gaussian_sigma = float(ask_user("Enter Gaussian sigma for local statistics (default: 1.5)", default="1.5"))
    window_size = int(ask_user("Enter Gaussian window size (odd number, default: 11)", default="11"))
    C1_scale = float(ask_user("Enter C1 scale factor (default: 0.01)", default="0.01"))
    C2_scale = float(ask_user("Enter C2 scale factor (default: 0.03)", default="0.03"))
    
    # Constants
    C1 = (C1_scale * 255) ** 2
    C2 = (C2_scale * 255) ** 2
    
    # Gaussian window
    kernel = cv2.getGaussianKernel(window_size, gaussian_sigma)
    window = kernel * kernel.T
    
    # Local statistics (intense: use replicate border for edge handling)
    log_action("Computing local means.")
    mu1 = cv2.filter2D(test_img.astype(float), -1, window, borderType=cv2.BORDER_REPLICATE)
    mu2 = cv2.filter2D(ref_img.astype(float), -1, window, borderType=cv2.BORDER_REPLICATE)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    log_action("Computing local variances and covariance.")
    sigma1_sq = cv2.filter2D(test_img.astype(float)**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(ref_img.astype(float)**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(test_img.astype(float)*ref_img.astype(float), -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2
    
    # Enhanced SSIM components (extreme: add safeguards for negative variances)
    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)
    
    log_action("Computing luminance, contrast, and structure maps.")
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * np.sqrt(sigma1_sq * sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C2/2) / (np.sqrt(sigma1_sq * sigma2_sq) + C2/2)
    
    ssim_map = luminance * contrast * structure
    ssim_index = np.mean(ssim_map)
    
    # Multi-scale SSIM (intense: more scales, better downsampling)
    log_action("Computing multi-scale SSIM.")
    ms_ssim_values = [ssim_index]
    scales = [2, 4, 8, 16]  # Extreme: more scales for robustness
    for scale in scales:
        img1_scaled = cv2.resize(test_img, (test_img.shape[1]//scale, test_img.shape[0]//scale), interpolation=cv2.INTER_AREA)
        img2_scaled = cv2.resize(ref_img, (ref_img.shape[1]//scale, ref_img.shape[0]//scale), interpolation=cv2.INTER_AREA)
        
        # Full SSIM at each scale (incredible: not simplified, compute full for accuracy)
        mu1_s = cv2.filter2D(img1_scaled.astype(float), -1, window, borderType=cv2.BORDER_REPLICATE)
        mu2_s = cv2.filter2D(img2_scaled.astype(float), -1, window, borderType=cv2.BORDER_REPLICATE)
        mu1_sq_s = mu1_s ** 2
        mu2_sq_s = mu2_s ** 2
        mu1_mu2_s = mu1_s * mu2_s
        sigma1_sq_s = cv2.filter2D(img1_scaled.astype(float)**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq_s
        sigma2_sq_s = cv2.filter2D(img2_scaled.astype(float)**2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq_s
        sigma12_s = cv2.filter2D(img1_scaled.astype(float)*img2_scaled.astype(float), -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2_s
        
        sigma1_sq_s = np.maximum(sigma1_sq_s, 0)
        sigma2_sq_s = np.maximum(sigma2_sq_s, 0)
        
        luminance_s = (2 * mu1_mu2_s + C1) / (mu1_sq_s + mu2_sq_s + C1)
        contrast_s = (2 * np.sqrt(sigma1_sq_s * sigma2_sq_s) + C2) / (sigma1_sq_s + sigma2_sq_s + C2)
        structure_s = (sigma12_s + C2/2) / (np.sqrt(sigma1_sq_s * sigma2_sq_s) + C2/2)
        
        ssim_map_s = luminance_s * contrast_s * structure_s
        ms_ssim_values.append(np.mean(ssim_map_s))
    
    # Deep perceptual integration (PyTorch: use LPIPS)
    log_action("Integrating deep learning perceptual metric (LPIPS).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net='vgg').to(device)  # Incredible: use VGG for better perceptual
    
    # Convert to RGB and tensor (extreme: handle grayscale to RGB conversion)
    test_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
    test_tensor = torch.from_numpy(test_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(device)
    ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(device)
    
    lpips_score = lpips_model(test_tensor, ref_tensor).item()
    log_action(f"Deep LPIPS score computed: {lpips_score:.4f}")
    
    # Fuse SSIM with deep metric (robust: weighted average for ultimate effectiveness)
    fused_score = (ssim_index * 0.6 + (1 - lpips_score) * 0.4)  # Normalize LPIPS to similarity
    
    # Additional extreme features: anomaly highlighting, thresholded maps
    log_action("Generating enhanced maps and visualizations.")
    anomaly_map = 1 - ssim_map  # Inverted for anomalies
    threshold = float(ask_user("Enter anomaly highlight threshold (0-1, default: 0.3)", default="0.3"))
    anomaly_binary = (anomaly_map > threshold).astype(np.uint8)
    
    # Save maps as images
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path / "ssim_map.jpg"), (ssim_map * 255).astype(np.uint8))
    cv2.imwrite(str(output_path / "luminance_map.jpg"), (luminance * 255).astype(np.uint8))
    cv2.imwrite(str(output_path / "contrast_map.jpg"), (contrast * 255).astype(np.uint8))
    cv2.imwrite(str(output_path / "structure_map.jpg"), (structure * 255).astype(np.uint8))
    cv2.imwrite(str(output_path / "anomaly_map.jpg"), (anomaly_map * 255).astype(np.uint8))
    # Visualization (incredible: comprehensive plot)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs[0, 0].imshow(test_img, cmap='gray")
    axs[0, 0].set_title("Test Image")
    axs[0, 1].imshow(ref_img, cmap='gray')
    axs[0, 1].set_title("Reference Image")
    axs[0, 2].imshow(ssim_map, cmap='viridis', vmin=0, vmax=1)
    axs[0, 2].set_title(f"SSIM Map (Index={ssim_index:.4f})")
    axs[1, 0].imshow(anomaly_map, cmap='hot', vmin=0, vmax=1)
    axs[1, 0].set_title("Anomaly Map")
    axs[1, 1].imshow(luminance, cmap='plasma', vmin=0, vmax=1)
    axs[1, 1].set_title("Luminance Map")
    axs[1, 2].imshow(contrast, cmap='cividis', vmin=0, vmax=1)
    axs[1, 2].set_title("Contrast Map")
    plt.savefig(output_path / "visualization.png")
    plt.close()
    
    # Save report
    report = {
        'ssim_index': ssim_index,
        'ms_ssim': ms_ssim_values,
        'lpips_score': lpips_score,
        'fused_score': fused_score,
        'mean_luminance': np.mean(luminance),
        'mean_contrast': np.mean(contrast),
        'mean_structure': np.mean(structure),
    }
    with open(output_path / "report.json", 'w') as f:
        json.dump(report, f, indent=4)
    log_action(f"Results saved to {output_dir}.")
    
    return report

# Main execution
if __name__ == "__main__":
    setup_logging()
    check_and_install_dependencies()
    
    log_action("Welcome to the Extreme Robust SSIM System.")
    
    test_img_path = ask_user("Enter path to test image", is_path=True)
    ref_img_path = ask_user("Enter path to reference image", is_path=True)
    output_dir = ask_user("Enter output directory (will be created if not exists)", is_dir=False, default="ssim_output")
    
    compute_extreme_ssim(test_img_path, ref_img_path, output_dir)
    
    while ask_user("Run another analysis? (y/n)", is_bool=True, default="n"):
        test_img_path = ask_user("Enter path to test image", is_path=True)
        ref_img_path = ask_user("Enter path to reference image", is_path=True)
        output_dir = ask_user("Enter output directory", default=output_dir)
        compute_extreme_ssim(test_img_path, ref_img_path, output_dir)
    
    log_action("System shutdown.")
