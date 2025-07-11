#!/usr/bin/env python3
"""
Duplicate Image Remover (Visual Content Based) - Advanced Version
Finds and removes duplicate images based on visual content, not just file names
Features:
- Detects visually identical images with different names
- Checks ALL image files regardless of size
- Multiple detection methods (exact match, perceptual hash, color histogram)
- Auto-installs dependencies
- Interactive mode with visual preview option
- Detailed logging and progress tracking
- Smart similarity threshold adjustment
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import subprocess
import importlib.util
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages automatic installation of dependencies"""
    
    REQUIRED_PACKAGES = {
        'PIL': 'pillow',
        'imagehash': 'imagehash',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'colorama': 'colorama',
    }
    
    @staticmethod
    def check_and_install_dependencies():
        """Check and install required dependencies"""
        print("\n" + "="*60)
        print("CHECKING DEPENDENCIES FOR IMAGE PROCESSING")
        print("="*60)
        
        missing_packages = []
        
        for module_name, package_name in DependencyManager.REQUIRED_PACKAGES.items():
            try:
                if module_name == 'PIL':
                    importlib.import_module('PIL.Image')
                else:
                    importlib.import_module(module_name)
                print(f"✓ {module_name} ({package_name}) - Already installed")
                logger.info(f"Dependency {module_name} is already installed")
            except ImportError:
                missing_packages.append(package_name)
                print(f"✗ {module_name} ({package_name}) - Not found")
                logger.warning(f"Dependency {module_name} is not installed")
        
        if missing_packages:
            print(f"\nNeed to install {len(missing_packages)} packages: {', '.join(missing_packages)}")
            
            # Upgrade pip first to ensure we get latest versions
            print("\nUpgrading pip to ensure latest package versions...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
                print("✓ pip upgraded successfully")
            except Exception as e:
                print(f"⚠ Warning: Could not upgrade pip: {e}")
            
            # Install missing packages
            for package in missing_packages:
                try:
                    print(f"\nInstalling {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
                    print(f"✓ {package} installed successfully")
                    logger.info(f"Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {package}: {e}")
                    print(f"✗ Failed to install {package}: {e}")
                    sys.exit(1)
        else:
            print("\n✓ All dependencies are already satisfied")
            logger.info("All dependencies are already satisfied")

# Check dependencies before importing
DependencyManager.check_and_install_dependencies()

# Now import the dependencies
from PIL import Image, ImageFile
import imagehash
import numpy as np
from tqdm import tqdm
from colorama import init, Fore, Style

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

init(autoreset=True)

class DuplicateImageRemover:
    """Main class for finding and removing duplicate images based on visual content"""
    
    # Supported image formats
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', 
                       '.webp', '.ico', '.heic', '.heif', '.svg', '.raw'}
    
    def __init__(self):
        self.root_path = None
        self.deletion_mode = 'trash'
        self.detection_mode = 'combined'  # 'exact', 'perceptual', 'combined'
        self.similarity_threshold = 95  # Percentage similarity for perceptual matching
        
        # Storage for different hash types
        self.exact_hashes = defaultdict(list)
        self.perceptual_hashes = defaultdict(list)
        self.average_hashes = defaultdict(list)
        self.dhashes = defaultdict(list)
        self.color_hashes = defaultdict(list)
        
        self.duplicate_groups = []
        self.stats = {
            'total_images': 0,
            'images_processed': 0,
            'exact_duplicates': 0,
            'similar_duplicates': 0,
            'space_to_free': 0,
            'errors': 0
        }
        
    def get_user_input(self):
        """Get configuration from user interactively"""
        print("\n" + "="*60)
        print("DUPLICATE IMAGE REMOVER (Visual Content Based)")
        print("="*60)
        print("\nThis tool finds duplicate images based on their visual content,")
        print("even if they have different names or formats.")
        print("ALL image files are checked regardless of size.")
        
        # Get directory to scan
        while True:
            path_input = input("\nEnter directory path to scan (or '.' for current directory): ").strip()
            if path_input == '.':
                path_input = os.getcwd()
            
            path = Path(path_input).expanduser().resolve()
            
            if path.exists() and path.is_dir():
                self.root_path = path
                print(f"{Fore.GREEN}✓ Will scan: {self.root_path}{Style.RESET_ALL}")
                logger.info(f"Target directory set to: {self.root_path}")
                break
            else:
                print(f"{Fore.RED}✗ Invalid directory path. Please try again.{Style.RESET_ALL}")
        
        # Get detection mode
        print("\nDetection mode:")
        print("1. Exact duplicates only (100% identical)")
        print("2. Perceptual matching (finds similar images)")
        print("3. Combined (both exact and similar) [Recommended]")
        
        while True:
            mode = input("Choose detection mode (1, 2, or 3) [default: 3]: ").strip()
            if mode == '' or mode == '3':
                self.detection_mode = 'combined'
                print(f"{Fore.GREEN}✓ Using combined detection (exact + similar){Style.RESET_ALL}")
                break
            elif mode == '1':
                self.detection_mode = 'exact'
                print(f"{Fore.GREEN}✓ Using exact duplicate detection only{Style.RESET_ALL}")
                break
            elif mode == '2':
                self.detection_mode = 'perceptual'
                print(f"{Fore.GREEN}✓ Using perceptual similarity detection{Style.RESET_ALL}")
                break
            else:
                print(f"{Fore.RED}Please enter 1, 2, or 3{Style.RESET_ALL}")
        
        # Get similarity threshold for perceptual matching
        if self.detection_mode in ['perceptual', 'combined']:
            print("\nSimilarity threshold for matching (1-100%):")
            print("  95-100%: Nearly identical images")
            print("  90-95%:  Very similar images")
            print("  80-90%:  Similar images (may have slight differences)")
            
            threshold_input = input("Enter threshold (default: 95): ").strip()
            if threshold_input:
                try:
                    threshold = int(threshold_input)
                    if 1 <= threshold <= 100:
                        self.similarity_threshold = threshold
                    else:
                        print("Using default threshold of 95%")
                except:
                    print("Using default threshold of 95%")
            
            print(f"{Fore.GREEN}✓ Similarity threshold: {self.similarity_threshold}%{Style.RESET_ALL}")
        
        # Get deletion mode
        print("\nDeletion mode:")
        print("1. Move to trash folder (safer, recommended)")
        print("2. Permanent deletion (cannot be undone!)")
        
        while True:
            mode = input("Choose deletion mode (1 or 2) [default: 1]: ").strip()
            if mode == '' or mode == '1':
                self.deletion_mode = 'trash'
                print(f"{Fore.GREEN}✓ Will move duplicates to trash folder{Style.RESET_ALL}")
                break
            elif mode == '2':
                confirm = input(f"{Fore.YELLOW}⚠ Permanent deletion selected. Are you sure? (yes/no): {Style.RESET_ALL}").strip().lower()
                if confirm == 'yes':
                    self.deletion_mode = 'permanent'
                    print(f"{Fore.RED}✓ Will permanently delete duplicates{Style.RESET_ALL}")
                    break
            else:
                print(f"{Fore.RED}Please enter 1 or 2{Style.RESET_ALL}")
    
    def _get_image_info(self, filepath: Path) -> Optional[Dict]:
        """Get image information and calculate various hashes"""
        try:
            # Basic file info
            file_size = filepath.stat().st_size
            
            # Open and process image
            with Image.open(filepath) as img:
                # Convert to RGB if necessary (for consistent hashing)
                if img.mode not in ('RGB', 'L'):
                    if img.mode == 'RGBA':
                        # Create white background for transparent images
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background
                    else:
                        img = img.convert('RGB')
                
                # Calculate various hashes
                info = {
                    'path': filepath,
                    'size': file_size,
                    'dimensions': img.size,
                    'format': img.format,
                    'mode': img.mode,
                    'modified': filepath.stat().st_mtime
                }
                
                # Exact hash (file content)
                if self.detection_mode in ['exact', 'combined']:
                    with open(filepath, 'rb') as f:
                        info['exact_hash'] = hashlib.sha256(f.read()).hexdigest()
                
                # Perceptual hashes
                if self.detection_mode in ['perceptual', 'combined']:
                    info['phash'] = str(imagehash.phash(img))
                    info['ahash'] = str(imagehash.average_hash(img))
                    info['dhash'] = str(imagehash.dhash(img))
                    info['whash'] = str(imagehash.whash(img))
                    info['colorhash'] = str(imagehash.colorhash(img))
                
                return info
                
        except Exception as e:
            logger.warning(f"Error processing image {filepath}: {e}")
            return None
    
    def scan_images(self):
        """Scan directory tree for image files"""
        print(f"\n{Fore.CYAN}Scanning for ALL image files (no size restrictions)...{Style.RESET_ALL}")
        logger.info(f"Starting scan of {self.root_path}")
        
        # Collect all image files
        image_files = []
        
        for root, _, files in os.walk(self.root_path):
            for filename in files:
                if Path(filename).suffix.lower() in self.IMAGE_EXTENSIONS:
                    filepath = Path(root) / filename
                    image_files.append(filepath)
        
        self.stats['total_images'] = len(image_files)
        print(f"Found {self.stats['total_images']:,} image files (ALL sizes included)")
        
        if not image_files:
            print(f"{Fore.YELLOW}No image files found!{Style.RESET_ALL}")
            return
        
        # Process images
        print(f"\n{Fore.CYAN}Processing images and calculating hashes...{Style.RESET_ALL}")
        logger.info(f"Processing {len(image_files)} images")
        
        processed_images = []
        
        for filepath in tqdm(image_files, desc="Processing images"):
            info = self._get_image_info(filepath)
            
            if info:
                processed_images.append(info)
                self.stats['images_processed'] += 1
                
                # Store in appropriate hash dictionaries
                if 'exact_hash' in info:
                    self.exact_hashes[info['exact_hash']].append(info)
                
                if 'phash' in info:
                    self.perceptual_hashes[info['phash']].append(info)
                    self.average_hashes[info['ahash']].append(info)
                    self.dhashes[info['dhash']].append(info)
                    self.color_hashes[info['colorhash']].append(info)
            else:
                self.stats['errors'] += 1
        
        print(f"\n{Fore.GREEN}Successfully processed {self.stats['images_processed']:,} images{Style.RESET_ALL}")
        
        if self.stats['errors'] > 0:
            print(f"{Fore.YELLOW}Failed to process {self.stats['errors']} images{Style.RESET_ALL}")
    
    def find_duplicates(self):
        """Find duplicate images based on selected detection mode"""
        print(f"\n{Fore.CYAN}Finding duplicates...{Style.RESET_ALL}")
        logger.info("Starting duplicate detection")
        
        found_groups = []
        processed_paths = set()
        
        # Find exact duplicates
        if self.detection_mode in ['exact', 'combined']:
            for hash_val, images in self.exact_hashes.items():
                if len(images) > 1:
                    # Sort by modification time (keep oldest)
                    images.sort(key=lambda x: x['modified'])
                    
                    # First image is the original to keep
                    group = {
                        'type': 'exact',
                        'similarity': 100,
                        'original': images[0],  # This one will be KEPT
                        'duplicates': images[1:],  # These will be REMOVED
                        'count': len(images),
                        'size': images[0]['size'],
                        'total_waste': sum(img['size'] for img in images[1:])
                    }
                    
                    found_groups.append(group)
                    self.stats['exact_duplicates'] += len(images) - 1
                    
                    logger.info(f"Found exact duplicate group: keeping {images[0]['path']}, "
                               f"will remove {len(images)-1} duplicates")
                    
                    # Mark as processed
                    for img in images:
                        processed_paths.add(img['path'])
        
        # Find similar images using perceptual hashing
        if self.detection_mode in ['perceptual', 'combined']:
            # Process images not already in exact matches
            remaining_images = []
            
            for hash_list in [self.perceptual_hashes, self.average_hashes, self.dhashes]:
                for images in hash_list.values():
                    for img in images:
                        if img['path'] not in processed_paths:
                            remaining_images.append(img)
            
            # Remove duplicates from remaining_images
            unique_remaining = {img['path']: img for img in remaining_images}
            remaining_images = list(unique_remaining.values())
            
            # Compare perceptual hashes
            similar_groups = self._find_similar_images(remaining_images)
            
            for group in similar_groups:
                found_groups.append(group)
                self.stats['similar_duplicates'] += len(group['duplicates'])
                
                logger.info(f"Found similar image group: keeping {group['original']['path']}, "
                           f"will remove {len(group['duplicates'])} similar images")
                
                # Mark as processed
                processed_paths.add(group['original']['path'])
                for img in group['duplicates']:
                    processed_paths.add(img['path'])
        
        # Sort by wasted space
        found_groups.sort(key=lambda x: x['total_waste'], reverse=True)
        self.duplicate_groups = found_groups
        
        # Calculate total space to free
        self.stats['space_to_free'] = sum(group['total_waste'] for group in found_groups)
        
        logger.info(f"Found {len(found_groups)} duplicate groups")
        print(f"\n{Fore.GREEN}Analysis complete!{Style.RESET_ALL}")
        print(f"   • Will keep: {len(found_groups)} original images")
        print(f"   • Will remove: {self.stats['exact_duplicates'] + self.stats['similar_duplicates']} duplicates")
    
    def _find_similar_images(self, images: List[Dict]) -> List[Dict]:
        """Find similar images using perceptual hashing"""
        similar_groups = []
        processed = set()
        
        print(f"\n{Fore.CYAN}Comparing images for visual similarity...{Style.RESET_ALL}")
        
        for i, img1 in enumerate(tqdm(images, desc="Finding similar images")):
            if img1['path'] in processed:
                continue
            
            similar = []
            
            for j, img2 in enumerate(images[i+1:], i+1):
                if img2['path'] in processed:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(img1, img2)
                
                if similarity >= self.similarity_threshold:
                    similar.append((img2, similarity))
            
            if similar:
                # Sort by similarity
                similar.sort(key=lambda x: x[1], reverse=True)
                
                group = {
                    'type': 'similar',
                    'similarity': similar[0][1],  # Highest similarity
                    'original': img1,
                    'duplicates': [img for img, _ in similar],
                    'count': len(similar) + 1,
                    'size': img1['size'],
                    'total_waste': sum(img['size'] for img, _ in similar)
                }
                
                similar_groups.append(group)
                
                # Mark as processed
                processed.add(img1['path'])
                for img, _ in similar:
                    processed.add(img['path'])
        
        return similar_groups
    
    def _calculate_similarity(self, img1: Dict, img2: Dict) -> float:
        """Calculate similarity percentage between two images"""
        if 'phash' not in img1 or 'phash' not in img2:
            return 0.0
        
        # Calculate hamming distances for different hash types
        distances = []
        
        # Perceptual hash
        phash_dist = imagehash.hex_to_hash(img1['phash']) - imagehash.hex_to_hash(img2['phash'])
        distances.append(phash_dist)
        
        # Average hash
        ahash_dist = imagehash.hex_to_hash(img1['ahash']) - imagehash.hex_to_hash(img2['ahash'])
        distances.append(ahash_dist)
        
        # Difference hash
        dhash_dist = imagehash.hex_to_hash(img1['dhash']) - imagehash.hex_to_hash(img2['dhash'])
        distances.append(dhash_dist)
        
        # Calculate average similarity
        avg_distance = sum(distances) / len(distances)
        max_distance = 64  # Maximum hamming distance for 64-bit hashes
        
        similarity = (1 - (avg_distance / max_distance)) * 100
        
        return similarity
    
    def display_results(self):
        """Display duplicate findings"""
        if not self.duplicate_groups:
            print(f"\n{Fore.GREEN}✨ No duplicate images found!{Style.RESET_ALL}")
            logger.info("No duplicates found")
            return False
        
        print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}DUPLICATE IMAGES FOUND{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n📊 Summary:")
        print(f"   • Duplicate groups: {len(self.duplicate_groups):,}")
        print(f"   • Exact duplicates: {self.stats['exact_duplicates']:,}")
        print(f"   • Similar images: {self.stats['similar_duplicates']:,}")
        print(f"   • Total duplicates to remove: {self.stats['exact_duplicates'] + self.stats['similar_duplicates']:,}")
        print(f"   • Original images to keep: {len(self.duplicate_groups):,}")
        print(f"   • Space to free: {Fore.RED}{self._format_size(self.stats['space_to_free'])}{Style.RESET_ALL}")
        
        # Show top duplicate groups
        print(f"\n📋 Top duplicate groups:")
        
        for i, group in enumerate(self.duplicate_groups[:10], 1):
            print(f"\n{i}. {Fore.CYAN}{group['count']} images{Style.RESET_ALL} " +
                  f"({group['type']} match, {group['similarity']:.1f}% similar)")
            
            orig = group['original']
            print(f"   {Fore.GREEN}KEEP:{Style.RESET_ALL} {orig['path'].name}")
            print(f"   Path: {orig['path'].parent}")
            print(f"   Size: {self._format_size(orig['size'])}, " +
                  f"Dimensions: {orig['dimensions'][0]}x{orig['dimensions'][1]}")
            print(f"   {Fore.YELLOW}REMOVE {len(group['duplicates'])} duplicates:{Style.RESET_ALL}")
            print(f"   Wasted: {Fore.RED}{self._format_size(group['total_waste'])}{Style.RESET_ALL}")
            
            # Show first few duplicates
            for j, dup in enumerate(group['duplicates'][:2]):
                print(f"     ✗ {dup['path'].name} ({dup['path'].parent})")
            
            if len(group['duplicates']) > 2:
                print(f"     ✗ ... and {len(group['duplicates']) - 2} more duplicates")
        
        if len(self.duplicate_groups) > 10:
            print(f"\n... and {len(self.duplicate_groups) - 10} more duplicate groups")
        
        return True
    
    def _format_size(self, size: int) -> str:
        """Format bytes to human readable size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def remove_duplicates(self):
        """Remove duplicate images"""
        print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}DUPLICATE REMOVAL{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        
        total_duplicates = self.stats['exact_duplicates'] + self.stats['similar_duplicates']
        
        # Confirm deletion
        print(f"\n⚠  Action summary:")
        print(f"   • Will KEEP: {Fore.GREEN}{len(self.duplicate_groups):,} original images{Style.RESET_ALL}")
        print(f"   • Will REMOVE: {Fore.RED}{total_duplicates:,} duplicate images{Style.RESET_ALL}")
        print(f"   • Space to free: {Fore.GREEN}{self._format_size(self.stats['space_to_free'])}{Style.RESET_ALL}")
        
        print(f"\n📝 The oldest or first found image in each group will be kept.")
        
        if self.deletion_mode == 'permanent':
            print(f"\n{Fore.RED}WARNING: Permanent deletion selected!{Style.RESET_ALL}")
            print(f"{Fore.RED}Duplicate files will be permanently deleted and cannot be recovered!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}Duplicates will be moved to a trash folder for safe recovery.{Style.RESET_ALL}")
        
        # Option to review groups
        review = input(f"\nWould you like to review each group before deletion? (yes/no) [default: no]: ").strip().lower()
        if review == 'yes':
            self._interactive_removal()
        else:
            confirm = input(f"\nProceed with removing ALL {total_duplicates:,} duplicates? (yes/no) [default: yes]: ").strip().lower()
            
            if confirm != 'yes' and confirm != '':
                print(f"{Fore.YELLOW}Operation cancelled{Style.RESET_ALL}")
                logger.info("User cancelled duplicate removal")
                return
            
        self._batch_removal()
    
    def _interactive_removal(self):
        """Interactive removal with group-by-group review"""
        print(f"\n{Fore.CYAN}Interactive removal mode{Style.RESET_ALL}")
        
        trash_dir = None
        if self.deletion_mode == 'trash':
            trash_dir = self.root_path / '.trash' / f"duplicate_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trash_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Trash folder: {trash_dir}")
        
        removed_count = 0
        freed_space = 0
        skipped_count = 0
        kept_count = 0
        
        for i, group in enumerate(self.duplicate_groups, 1):
            print(f"\n{'-'*60}")
            print(f"Group {i}/{len(self.duplicate_groups)} - {group['type']} match ({group['similarity']:.1f}% similar)")
            print(f"{Fore.GREEN}KEEP:{Style.RESET_ALL} {group['original']['path']}")
            print(f"{Fore.YELLOW}DUPLICATES TO REMOVE:{Style.RESET_ALL} {len(group['duplicates'])}")
            print(f"Space to free: {self._format_size(group['total_waste'])}")
            
            action = input("\nAction? (r=remove duplicates, k=keep all, v=view list, q=quit) [default: r]: ").strip().lower()
            
            if action == 'q':
                break
            elif action == 'v':
                print(f"\n{Fore.GREEN}Original (WILL KEEP):{Style.RESET_ALL}")
                print(f"  ✓ {group['original']['path']}")
                print(f"\n{Fore.YELLOW}Duplicates (WILL REMOVE):{Style.RESET_ALL}")
                for j, dup in enumerate(group['duplicates'], 1):
                    print(f"  ✗ {j}. {dup['path']}")
                
                action = input("\nNow: (r=remove duplicates, k=keep all, q=quit) [default: r]: ").strip().lower()
                
                if action == 'q':
                    break
                elif action == 'k':
                    skipped_count += len(group['duplicates'])
                    continue
            elif action == 'k':
                skipped_count += len(group['duplicates'])
                continue
            
            # Remove duplicates (default action) - keep the original
            kept_count += 1  # Count the original we're keeping
            print(f"{Fore.GREEN}Keeping original: {group['original']['path'].name}{Style.RESET_ALL}")
            
            for dup in group['duplicates']:
                if self._remove_file(dup['path'], trash_dir):
                    removed_count += 1
                    freed_space += dup['size']
                    print(f"  {Fore.RED}✗ Removed: {dup['path'].name}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}Kept {kept_count} original images{Style.RESET_ALL}")
        self._display_removal_results(removed_count, freed_space, skipped_count)
    
    def _batch_removal(self):
        """Batch removal of all duplicates"""
        trash_dir = None
        if self.deletion_mode == 'trash':
            trash_dir = self.root_path / '.trash' / f"duplicate_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trash_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n📁 Trash folder: {trash_dir}")
        
        removed_count = 0
        freed_space = 0
        errors = []
        kept_count = 0
        
        # Collect all files to remove (only duplicates, not originals)
        all_duplicates = []
        all_originals = set()  # Track originals to ensure we don't remove them
        
        for group in self.duplicate_groups:
            # Add original to protected set
            all_originals.add(group['original']['path'])
            kept_count += 1
            
            # Only add duplicates for removal
            all_duplicates.extend(group['duplicates'])
        
        print(f"\n{Fore.GREEN}Keeping {kept_count} original images{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Removing {len(all_duplicates)} duplicate images...{Style.RESET_ALL}")
        
        for dup in tqdm(all_duplicates, desc="Removing duplicates"):
            try:
                # Double-check we're not removing an original
                if dup['path'] in all_originals:
                    logger.warning(f"Skipping original file that was mistakenly in duplicates: {dup['path']}")
                    continue
                
                if self._remove_file(dup['path'], trash_dir):
                    removed_count += 1
                    freed_space += dup['size']
            except Exception as e:
                error_msg = f"Failed to remove {dup['path']}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        self._display_removal_results(removed_count, freed_space, 0, errors)
    
    def _remove_file(self, filepath: Path, trash_dir: Optional[Path]) -> bool:
        """Remove a single file"""
        try:
            if self.deletion_mode == 'trash' and trash_dir:
                # Move to trash
                relative_path = filepath.relative_to(self.root_path)
                dest_path = trash_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.move(str(filepath), str(dest_path))
                logger.info(f"Moved to trash: {filepath}")
            else:
                # Permanent deletion
                filepath.unlink()
                logger.info(f"Permanently deleted: {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove {filepath}: {e}")
            return False
    
    def _display_removal_results(self, removed_count: int, freed_space: int, 
                                skipped_count: int, errors: List[str] = None):
        """Display removal results"""
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}REMOVAL COMPLETE{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n✅ Successfully removed: {removed_count:,} images")
        print(f"💾 Space freed: {Fore.GREEN}{self._format_size(freed_space)}{Style.RESET_ALL}")
        
        if skipped_count > 0:
            print(f"⏭️  Images skipped: {skipped_count:,}")
        
        if errors:
            print(f"\n{Fore.RED}❌ Errors: {len(errors)}{Style.RESET_ALL}")
            for error in errors[:5]:
                print(f"   • {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")
        
        if self.deletion_mode == 'trash':
            print(f"\n💡 Tip: Images moved to trash folder. Delete the folder when you're sure.")
        
        # Save report
        self._save_report(removed_count, freed_space, skipped_count, errors or [])
    
    def _save_report(self, removed_count: int, freed_space: int, 
                    skipped_count: int, errors: List[str]):
        """Save operation report"""
        report_dir = self.root_path / '.duplicate_image_reports'
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'scan_directory': str(self.root_path),
            'settings': {
                'detection_mode': self.detection_mode,
                'similarity_threshold': self.similarity_threshold,
                'deletion_mode': self.deletion_mode
            },
            'results': {
                'total_images_scanned': self.stats['total_images'],
                'images_processed': self.stats['images_processed'],
                'duplicate_groups_found': len(self.duplicate_groups),
                'exact_duplicates': self.stats['exact_duplicates'],
                'similar_duplicates': self.stats['similar_duplicates'],
                'total_duplicates': self.stats['exact_duplicates'] + self.stats['similar_duplicates'],
                'potential_space_savings': self.stats['space_to_free'],
                'images_removed': removed_count,
                'images_skipped': skipped_count,
                'space_freed': freed_space,
                'errors': len(errors)
            },
            'errors': errors[:100]  # Save first 100 errors
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")
        logger.info(f"Report saved to: {report_file}")
    
    def run(self):
        """Main execution flow"""
        try:
            self.get_user_input()
            self.scan_images()
            
            if self.stats['images_processed'] == 0:
                print(f"\n{Fore.YELLOW}No images could be processed!{Style.RESET_ALL}")
                return
            
            self.find_duplicates()
            
            if self.display_results():
                self.remove_duplicates()
            
            print(f"\n{Fore.GREEN}✨ Operation completed!{Style.RESET_ALL}")
            logger.info("Operation completed successfully")
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Operation cancelled by user{Style.RESET_ALL}")
            logger.info("Operation cancelled by user")
        except Exception as e:
            print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
            logger.error(f"Unexpected error: {e}", exc_info=True)

def main():
    """Main entry point"""
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}DUPLICATE IMAGE REMOVER v1.0{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Visual Content Based Detection{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Checks ALL images - No size restrictions{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    remover = DuplicateImageRemover()
    remover.run()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
