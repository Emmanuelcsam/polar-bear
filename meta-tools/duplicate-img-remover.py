#!/usr/bin/env python3
"""
Duplicate Image Remover (Exact Duplicates Only)
Finds and removes duplicate images based on exact file content matching
Features:
- Detects 100% identical images (exact byte-for-byte matches)
- Checks ALL image files regardless of size
- Auto-installs dependencies
- Interactive mode with visual preview option
- Detailed logging and progress tracking
- Simple, fast, and reliable exact duplicate detection
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
from tqdm import tqdm
from colorama import init, Fore, Style

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

init(autoreset=True)

class DuplicateImageRemover:
    """Main class for finding and removing duplicate images based on exact file content"""
    
    # Supported image formats
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', 
                       '.webp', '.ico', '.heic', '.heif', '.svg', '.raw'}
    
    def __init__(self):
        self.root_path = None
        self.deletion_mode = 'trash'
        
        # Storage for exact file hashes only
        self.exact_hashes = defaultdict(list)
        
        self.duplicate_groups = []
        self.stats = {
            'total_images': 0,
            'images_processed': 0,
            'exact_duplicates': 0,
            'space_to_free': 0,
            'errors': 0
        }
        
    def get_user_input(self):
        """Get configuration from user interactively"""
        print("\n" + "="*60)
        print("DUPLICATE IMAGE REMOVER (Exact Duplicates Only)")
        print("="*60)
        print("\nThis tool finds duplicate images based on exact file content.")
        print("Only 100% identical images (byte-for-byte) are detected.")
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
        """Get image information and calculate exact file hash"""
        try:
            # Basic file info
            file_size = filepath.stat().st_size
            
            # Calculate exact hash (file content)
            with open(filepath, 'rb') as f:
                exact_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Get basic image info
            with Image.open(filepath) as img:
                info = {
                    'path': filepath,
                    'size': file_size,
                    'dimensions': img.size,
                    'format': img.format,
                    'mode': img.mode,
                    'modified': filepath.stat().st_mtime,
                    'exact_hash': exact_hash
                }
                
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
                
                # Store exact hash only
                self.exact_hashes[info['exact_hash']].append(info)
            else:
                self.stats['errors'] += 1
        
        print(f"\n{Fore.GREEN}Successfully processed {self.stats['images_processed']:,} images{Style.RESET_ALL}")
        
        if self.stats['errors'] > 0:
            print(f"{Fore.YELLOW}Failed to process {self.stats['errors']} images{Style.RESET_ALL}")
    
    def find_duplicates(self):
        """Find duplicate images based on exact file content matching"""
        print(f"\n{Fore.CYAN}Finding exact duplicates...{Style.RESET_ALL}")
        logger.info("Starting exact duplicate detection")
        
        found_groups = []
        
        # Find exact duplicates only
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
        
        # Sort by wasted space
        found_groups.sort(key=lambda x: x['total_waste'], reverse=True)
        self.duplicate_groups = found_groups
        
        # Calculate total space to free
        self.stats['space_to_free'] = sum(group['total_waste'] for group in found_groups)
        
        logger.info(f"Found {len(found_groups)} duplicate groups")
        print(f"\n{Fore.GREEN}Analysis complete!{Style.RESET_ALL}")
        print(f"   • Will keep: {len(found_groups)} original images")
        print(f"   • Will remove: {self.stats['exact_duplicates']} exact duplicates")
    
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
        print(f"   • Total duplicates to remove: {self.stats['exact_duplicates']:,}")
        print(f"   • Original images to keep: {len(self.duplicate_groups):,}")
        print(f"   • Space to free: {Fore.RED}{self._format_size(self.stats['space_to_free'])}{Style.RESET_ALL}")
        
        # Show top duplicate groups
        print(f"\n📋 Top duplicate groups:")
        
        for i, group in enumerate(self.duplicate_groups[:10], 1):
            print(f"\n{i}. {Fore.CYAN}{group['count']} images{Style.RESET_ALL} " +
                  f"({group['type']} match, {group['similarity']:.1f}% identical)")
            
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
        size_float = float(size)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_float < 1024:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024
        return f"{size_float:.1f} PB"
    
    def remove_duplicates(self):
        """Remove duplicate images"""
        print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}DUPLICATE REMOVAL{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        
        total_duplicates = self.stats['exact_duplicates']
        
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
                                skipped_count: int, errors: Optional[List[str]] = None):
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
                'detection_mode': 'exact',
                'deletion_mode': self.deletion_mode
            },
            'results': {
                'total_images_scanned': self.stats['total_images'],
                'images_processed': self.stats['images_processed'],
                'duplicate_groups_found': len(self.duplicate_groups),
                'exact_duplicates': self.stats['exact_duplicates'],
                'total_duplicates': self.stats['exact_duplicates'],
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
    print(f"{Fore.CYAN}Exact Duplicates Only - Fast & Reliable{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Checks ALL images - No size restrictions{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    remover = DuplicateImageRemover()
    remover.run()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()