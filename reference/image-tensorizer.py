import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import numpy as np
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageTensorizer:
    def __init__(self, 
                 resize=None, 
                 normalize=True, 
                 save_format='pt',
                 preserve_structure=True):
        """
        Initialize the image tensorizer
        
        Args:
            resize: Tuple (height, width) to resize images, None to keep original size
            normalize: Whether to normalize images to [0, 1] range
            save_format: Format to save tensors ('pt' for PyTorch, 'npy' for NumPy)
            preserve_structure: Whether to preserve directory structure in output
        """
        self.resize = resize
        self.normalize = normalize
        self.save_format = save_format
        self.preserve_structure = preserve_structure
        
        # Define transforms
        transform_list = []
        if resize:
            transform_list.append(transforms.Resize(resize))
        transform_list.append(transforms.ToTensor())
        if not normalize:
            # ToTensor already normalizes to [0, 1], so we scale back to [0, 255]
            transform_list.append(transforms.Lambda(lambda x: x * 255))
        
        self.transform = transforms.Compose(transform_list)
        
        # Supported image extensions
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', 
                               '.tiff', '.tif', '.ico', '.jfif')
        
        logger.info(f"Initialized tensorizer - Resize: {resize}, Normalize: {normalize}, Format: {save_format}")
    
    def find_all_images(self, root_dir):
        """Recursively find all image files in directory and subdirectories"""
        root_path = Path(root_dir)
        image_files = []
        
        logger.info(f"Scanning for images in {root_dir} and all subdirectories...")
        
        # Walk through all directories and subdirectories
        for path in root_path.rglob('*'):
            if path.is_file() and path.suffix.lower() in self.image_extensions:
                image_files.append(path)
        
        logger.info(f"Found {len(image_files)} images across all subdirectories")
        return image_files
    
    def tensorize_image(self, img_path):
        """Convert a single image to tensor"""
        try:
            # Load image
            img = Image.open(img_path)
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    # Create white background for RGBA images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Apply transforms
            tensor = self.transform(img)
            
            # Add metadata
            metadata = {
                'original_size': img.size,
                'original_mode': img.mode,
                'tensor_shape': tensor.shape,
                'source_path': str(img_path)
            }
            
            return tensor, metadata
            
        except Exception as e:
            logger.error(f"Error tensorizing {img_path}: {e}")
            return None, None
    
    def get_output_path(self, input_path, input_root, output_root):
        """Generate output path for tensor file"""
        input_path = Path(input_path)
        input_root = Path(input_root)
        output_root = Path(output_root)
        
        if self.preserve_structure:
            # Preserve directory structure
            relative_path = input_path.relative_to(input_root)
            output_path = output_root / relative_path
        else:
            # Flatten structure - use hash to avoid name conflicts
            file_hash = hashlib.md5(str(input_path).encode()).hexdigest()[:8]
            filename = f"{input_path.stem}_{file_hash}"
            output_path = output_root / filename
        
        # Change extension based on format
        if self.save_format == 'pt':
            output_path = output_path.with_suffix('.pt')
        else:
            output_path = output_path.with_suffix('.npy')
            
        return output_path
    
    def process_single_image(self, img_path, input_root, output_root):
        """Process a single image and save the tensor"""
        try:
            # Tensorize image
            tensor, metadata = self.tensorize_image(img_path)
            
            if tensor is None:
                return False
            
            # Get output path
            output_path = self.get_output_path(img_path, input_root, output_root)
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save tensor
            if self.save_format == 'pt':
                # Save as PyTorch tensor with metadata
                torch.save({
                    'tensor': tensor,
                    'metadata': metadata
                }, output_path)
            else:
                # Save as NumPy array
                np_array = tensor.numpy()
                np.save(output_path, np_array)
                
                # Save metadata separately
                metadata_path = output_path.with_suffix('.json')
                import json
                with open(metadata_path, 'w') as f:
                    # Convert metadata to JSON-serializable format
                    json_metadata = {
                        'original_size': list(metadata['original_size']),
                        'original_mode': metadata['original_mode'],
                        'tensor_shape': list(metadata['tensor_shape']),
                        'source_path': metadata['source_path']
                    }
                    json.dump(json_metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return False
    
    def process_directory(self, input_dir, output_dir, num_workers=4):
        """Process all images in directory tree and save as tensors"""
        # Find all images
        image_files = self.find_all_images(input_dir)
        
        if not image_files:
            logger.error("No images found in directory tree")
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process images in parallel
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self.process_single_image, 
                    img_path,
                    input_dir,
                    output_dir
                ): img_path 
                for img_path in image_files
            }
            
            # Process results with progress bar
            with tqdm(total=len(image_files), desc="Tensorizing images") as pbar:
                for future in as_completed(future_to_file):
                    img_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            successful += 1
                        else:
                            failed += 1
                            logger.warning(f"Failed to process {img_path}")
                    except Exception as e:
                        failed += 1
                        logger.error(f"Error processing {img_path}: {e}")
                    pbar.update(1)
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        # Print summary statistics
        self.print_summary(output_dir)
    
    def print_summary(self, output_dir):
        """Print summary of tensorized dataset"""
        output_path = Path(output_dir)
        tensor_files = list(output_path.rglob('*.pt')) + list(output_path.rglob('*.npy'))
        
        if not tensor_files:
            return
        
        total_size = sum(f.stat().st_size for f in tensor_files)
        
        print("\n=== Dataset Summary ===")
        print(f"Total tensor files: {len(tensor_files)}")
        print(f"Total size: {total_size / (1024**3):.2f} GB")
        print(f"Average file size: {total_size / len(tensor_files) / (1024**2):.2f} MB")
        
        # Sample a tensor to show info
        if self.save_format == 'pt':
            sample = torch.load(tensor_files[0])
            print(f"Sample tensor shape: {sample['tensor'].shape}")
            print(f"Sample tensor dtype: {sample['tensor'].dtype}")

def main():
    print("=== Image to Tensor Dataset Converter ===\n")
    
    # Get input directory
    print("Enter the path to the root directory containing images:")
    input_dir = input().strip()
    
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print("Error: Invalid input directory")
        sys.exit(1)
    
    # Get output directory
    print("\nEnter the path to the output directory for tensors:")
    output_dir = input().strip()
    
    # Configuration options
    print("\nConfiguration Options:")
    
    # Resize option
    print("\nResize images? Enter dimensions as 'height,width' (e.g., '224,224') or press Enter to keep original:")
    resize_input = input().strip()
    resize = None
    if resize_input:
        try:
            h, w = map(int, resize_input.split(','))
            resize = (h, w)
        except:
            print("Invalid resize format, keeping original size")
    
    # Normalization option
    print("\nNormalize to [0,1] range? (y/n, default: y):")
    normalize_input = input().strip().lower()
    normalize = normalize_input != 'n'
    
    # Save format
    print("\nSave format - 'pt' for PyTorch or 'npy' for NumPy (default: pt):")
    format_input = input().strip().lower()
    save_format = format_input if format_input in ['pt', 'npy'] else 'pt'
    
    # Preserve structure
    print("\nPreserve directory structure? (y/n, default: y):")
    preserve_input = input().strip().lower()
    preserve_structure = preserve_input != 'n'
    
    # Number of workers
    print("\nEnter number of parallel workers (default: 4):")
    workers_input = input().strip()
    num_workers = int(workers_input) if workers_input.isdigit() else 4
    
    # Create tensorizer
    tensorizer = ImageTensorizer(
        resize=resize,
        normalize=normalize,
        save_format=save_format,
        preserve_structure=preserve_structure
    )
    
    # Process the directory
    print(f"\nProcessing images with {num_workers} workers...")
    print("This will recursively process ALL subdirectories...\n")
    
    tensorizer.process_directory(input_dir, output_dir, num_workers)
    
    print("\nDone! Tensors saved to output directory.")
    
    # Offer to create a data loader example
    print("\nWould you like to generate example code for loading these tensors? (y/n):")
    if input().strip().lower() == 'y':
        generate_loader_example(output_dir, save_format)

def generate_loader_example(output_dir, save_format):
    """Generate example code for loading the tensorized dataset"""
    example_code = f"""
# Example code for loading your tensorized dataset:

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class TensorDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = Path(tensor_dir)
        self.tensor_files = list(self.tensor_dir.rglob('*.{"pt" if save_format == "pt" else "npy"}'))
    
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx):
        tensor_path = self.tensor_files[idx]
        {"        data = torch.load(tensor_path)" if save_format == "pt" else "        tensor = torch.from_numpy(np.load(tensor_path))"}
        {"        return data['tensor'], data['metadata']" if save_format == "pt" else "        return tensor"}

# Usage:
dataset = TensorDataset('{output_dir}')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch in dataloader:
    {"    tensors, metadata = batch" if save_format == "pt" else "    tensors = batch"}
    # Your training/processing code here
    pass
"""
    
    print(example_code)
    
    # Save to file
    example_path = Path(output_dir) / "dataloader_example.py"
    with open(example_path, 'w') as f:
        f.write(example_code)
    print(f"\nExample code saved to: {example_path}")

if __name__ == "__main__":
    main()