#!/usr/bin/env python3
"""
Duplicate File Finder and Remover
Finds and removes duplicate files based on content hash
Features:
- Auto-installs dependencies
- Interactive mode (no command line args)
- Detailed logging to terminal
- Safe deletion with trash folder option
- Progress tracking with tqdm
"""

import os
import sys
import hashlib
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import subprocess
import importlib.util

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
        'tqdm': 'tqdm',
        'colorama': 'colorama',
    }
    
    @staticmethod
    def check_and_install_dependencies():
        """Check and install required dependencies"""
        print("\n" + "="*60)
        print("CHECKING DEPENDENCIES")
        print("="*60)
        
        missing_packages = []
        
        for module_name, package_name in DependencyManager.REQUIRED_PACKAGES.items():
            try:
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
from tqdm import tqdm
from colorama import init, Fore, Style
init(autoreset=True)

class DuplicateRemover:
    """Main class for finding and removing duplicate files"""
    
    def __init__(self):
        self.root_path = None
        self.file_hashes = defaultdict(list)
        self.duplicates = []
        self.total_files = 0
        self.total_duplicates = 0
        self.space_to_free = 0
        self.min_file_size = 1  # Minimum 1 byte by default
        self.deletion_mode = 'trash'  # 'trash' or 'permanent'
        
    def get_user_input(self):
        """Get configuration from user interactively"""
        print("\n" + "="*60)
        print("DUPLICATE FILE FINDER AND REMOVER")
        print("="*60)
        
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
        
        # Get minimum file size
        print("\nMinimum file size to check (skip small files for performance)")
        print("Examples: 1024 (bytes), 1KB, 10MB, 1GB")
        size_input = input("Enter minimum size (default: 1 byte): ").strip()
        
        if size_input:
            self.min_file_size = self._parse_size(size_input)
        
        print(f"{Fore.GREEN}✓ Will check files >= {self._format_size(self.min_file_size)}{Style.RESET_ALL}")
        logger.info(f"Minimum file size set to: {self.min_file_size} bytes")
        
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
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes"""
        size_str = size_str.strip().upper()
        
        if size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        else:
            return int(size_str.replace('B', ''))
    
    def _format_size(self, size: int) -> str:
        """Format bytes to human readable size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def _calculate_hash(self, filepath: Path, chunk_size: int = 8192) -> Optional[str]:
        """Calculate SHA256 hash of file"""
        try:
            logger.debug(f"Calculating hash for: {filepath}")
            sha256_hash = hashlib.sha256()
            
            with open(filepath, "rb") as f:
                while chunk := f.read(chunk_size):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {filepath}: {e}")
            return None
    
    def find_duplicates(self):
        """Scan directory tree and find duplicate files"""
        print(f"\n{Fore.CYAN}Scanning for files...{Style.RESET_ALL}")
        logger.info(f"Starting scan of {self.root_path}")
        
        # First, collect all files
        files_to_check = []
        
        for root, _, files in os.walk(self.root_path):
            for filename in files:
                filepath = Path(root) / filename
                
                try:
                    if filepath.stat().st_size >= self.min_file_size:
                        files_to_check.append(filepath)
                except Exception as e:
                    logger.warning(f"Cannot access {filepath}: {e}")
        
        self.total_files = len(files_to_check)
        print(f"Found {self.total_files:,} files to check")
        logger.info(f"Found {self.total_files} files >= {self.min_file_size} bytes")
        
        if self.total_files == 0:
            print(f"{Fore.YELLOW}No files found matching criteria{Style.RESET_ALL}")
            return
        
        # Group files by size first (optimization)
        print(f"\n{Fore.CYAN}Grouping files by size...{Style.RESET_ALL}")
        size_groups = defaultdict(list)
        
        for filepath in tqdm(files_to_check, desc="Analyzing sizes"):
            try:
                size = filepath.stat().st_size
                size_groups[size].append(filepath)
            except Exception as e:
                logger.warning(f"Cannot get size of {filepath}: {e}")
        
        # Only hash files that have same size (potential duplicates)
        files_to_hash = []
        for size, paths in size_groups.items():
            if len(paths) > 1:
                files_to_hash.extend(paths)
        
        print(f"\n{Fore.CYAN}Hashing {len(files_to_hash):,} potential duplicates...{Style.RESET_ALL}")
        logger.info(f"Hashing {len(files_to_hash)} files with duplicate sizes")
        
        # Calculate hashes
        for filepath in tqdm(files_to_hash, desc="Calculating hashes"):
            file_hash = self._calculate_hash(filepath)
            
            if file_hash:
                self.file_hashes[file_hash].append({
                    'path': filepath,
                    'size': filepath.stat().st_size,
                    'modified': filepath.stat().st_mtime
                })
        
        # Identify duplicate groups
        for hash_value, files in self.file_hashes.items():
            if len(files) > 1:
                # Sort by modification time (keep oldest)
                files.sort(key=lambda x: x['modified'])
                
                duplicate_group = {
                    'hash': hash_value,
                    'original': files[0],
                    'duplicates': files[1:],
                    'count': len(files),
                    'size': files[0]['size'],
                    'total_waste': files[0]['size'] * (len(files) - 1)
                }
                
                self.duplicates.append(duplicate_group)
                self.total_duplicates += len(files) - 1
                self.space_to_free += duplicate_group['total_waste']
        
        # Sort by wasted space
        self.duplicates.sort(key=lambda x: x['total_waste'], reverse=True)
        
        print(f"\n{Fore.GREEN}Scan complete!{Style.RESET_ALL}")
        logger.info(f"Found {len(self.duplicates)} duplicate groups, {self.total_duplicates} duplicate files")
    
    def display_results(self):
        """Display duplicate findings"""
        if not self.duplicates:
            print(f"\n{Fore.GREEN}✨ No duplicate files found!{Style.RESET_ALL}")
            logger.info("No duplicates found")
            return False
        
        print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}DUPLICATE FILES FOUND{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n📊 Summary:")
        print(f"   • Duplicate groups: {len(self.duplicates):,}")
        print(f"   • Duplicate files: {self.total_duplicates:,}")
        print(f"   • Space to free: {Fore.RED}{self._format_size(self.space_to_free)}{Style.RESET_ALL}")
        
        # Show top 10 duplicate groups
        print(f"\n📋 Top duplicate groups by wasted space:")
        for i, dup in enumerate(self.duplicates[:10], 1):
            print(f"\n{i}. {Fore.CYAN}{dup['count']} copies{Style.RESET_ALL} of {self._format_size(dup['size'])} file")
            print(f"   Original: {dup['original']['path']}")
            print(f"   Wasted: {Fore.RED}{self._format_size(dup['total_waste'])}{Style.RESET_ALL}")
            
            # Show first 3 duplicates
            for j, d in enumerate(dup['duplicates'][:3]):
                print(f"   Duplicate {j+1}: {d['path']}")
            
            if len(dup['duplicates']) > 3:
                print(f"   ... and {len(dup['duplicates']) - 3} more duplicates")
        
        if len(self.duplicates) > 10:
            print(f"\n... and {len(self.duplicates) - 10} more duplicate groups")
        
        return True
    
    def remove_duplicates(self):
        """Remove duplicate files"""
        print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}DUPLICATE REMOVAL{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        
        # Confirm deletion
        print(f"\n⚠  You are about to remove {self.total_duplicates:,} duplicate files")
        print(f"   This will free up {Fore.GREEN}{self._format_size(self.space_to_free)}{Style.RESET_ALL}")
        
        if self.deletion_mode == 'permanent':
            print(f"\n{Fore.RED}WARNING: Permanent deletion selected!{Style.RESET_ALL}")
            print(f"{Fore.RED}Files will be permanently deleted and cannot be recovered!{Style.RESET_ALL}")
        
        confirm = input(f"\nProceed with removal? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print(f"{Fore.YELLOW}Operation cancelled{Style.RESET_ALL}")
            logger.info("User cancelled duplicate removal")
            return
        
        # Create trash directory if using trash mode
        trash_dir = None
        if self.deletion_mode == 'trash':
            trash_dir = self.root_path / '.trash' / f"duplicates_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trash_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n📁 Trash folder: {trash_dir}")
            logger.info(f"Created trash directory: {trash_dir}")
        
        # Remove duplicates
        removed_count = 0
        freed_space = 0
        errors = []
        
        all_duplicates = []
        for group in self.duplicates:
            all_duplicates.extend(group['duplicates'])
        
        print(f"\n{Fore.CYAN}Removing duplicates...{Style.RESET_ALL}")
        
        for dup in tqdm(all_duplicates, desc="Removing files"):
            filepath = dup['path']
            
            try:
                if self.deletion_mode == 'trash':
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
                
                removed_count += 1
                freed_space += dup['size']
                
            except Exception as e:
                error_msg = f"Failed to remove {filepath}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Report results
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}REMOVAL COMPLETE{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n✅ Successfully removed: {removed_count:,} files")
        print(f"💾 Space freed: {Fore.GREEN}{self._format_size(freed_space)}{Style.RESET_ALL}")
        
        if errors:
            print(f"\n{Fore.RED}❌ Errors: {len(errors)}{Style.RESET_ALL}")
            for error in errors[:5]:
                print(f"   • {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")
        
        if self.deletion_mode == 'trash':
            print(f"\n💡 Tip: Files moved to trash folder. Delete the folder when you're sure:")
            print(f"   {trash_dir}")
        
        # Save report
        self._save_report(removed_count, freed_space, errors)
    
    def _save_report(self, removed_count: int, freed_space: int, errors: List[str]):
        """Save operation report"""
        report_dir = self.root_path / '.duplicate_reports'
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'scan_directory': str(self.root_path),
            'settings': {
                'min_file_size': self.min_file_size,
                'deletion_mode': self.deletion_mode
            },
            'results': {
                'total_files_scanned': self.total_files,
                'duplicate_groups_found': len(self.duplicates),
                'duplicate_files_found': self.total_duplicates,
                'potential_space_savings': self.space_to_free,
                'files_removed': removed_count,
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
    print(f"{Fore.CYAN}DUPLICATE FILE FINDER AND REMOVER v1.0{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    remover = DuplicateRemover()
    remover.run()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
