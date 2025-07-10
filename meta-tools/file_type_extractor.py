#!/usr/bin/env python3
"""
File Type Extractor
Extracts all files of specified types from a directory tree
Features:
- Auto-installs dependencies
- Interactive mode (no command line args)
- Detailed logging to terminal
- Multiple file type support
- Duplicate handling with renaming
- Progress tracking
"""

import os
import sys
import shutil
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Set, Dict, Optional
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

class FileTypeExtractor:
    """Main class for extracting files by type"""
    
    # Common file type groups
    FILE_TYPE_GROUPS = {
        'images': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico', '.tiff'},
        'videos': {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'},
        'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus'},
        'documents': {'.pdf', '.doc', '.docx', '.txt', '.odt', '.rtf', '.tex'},
        'spreadsheets': {'.xls', '.xlsx', '.csv', '.ods'},
        'archives': {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'},
        'code': {'.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs'},
    }
    
    def __init__(self):
        self.source_path = None
        self.dest_path = None
        self.file_types = set()
        self.copy_mode = 'copy'  # 'copy' or 'move'
        self.handle_duplicates = 'rename'  # 'rename', 'skip', or 'overwrite'
        self.found_files = []
        self.stats = {
            'total_scanned': 0,
            'files_found': 0,
            'files_processed': 0,
            'duplicates_handled': 0,
            'errors': 0,
            'total_size': 0
        }
        
    def get_user_input(self):
        """Get configuration from user interactively"""
        print("\n" + "="*60)
        print("FILE TYPE EXTRACTOR")
        print("="*60)
        
        # Get source directory
        while True:
            path_input = input("\nEnter source directory path (or '.' for current directory): ").strip()
            if path_input == '.':
                path_input = os.getcwd()
            
            path = Path(path_input).expanduser().resolve()
            
            if path.exists() and path.is_dir():
                self.source_path = path
                print(f"{Fore.GREEN}✓ Source directory: {self.source_path}{Style.RESET_ALL}")
                logger.info(f"Source directory set to: {self.source_path}")
                break
            else:
                print(f"{Fore.RED}✗ Invalid directory path. Please try again.{Style.RESET_ALL}")
        
        # Get destination directory
        while True:
            dest_input = input("\nEnter destination directory path: ").strip()
            dest_path = Path(dest_input).expanduser().resolve()
            
            # Create destination if it doesn't exist
            if not dest_path.exists():
                create = input(f"\nDirectory doesn't exist. Create it? (yes/no): ").strip().lower()
                if create == 'yes':
                    try:
                        dest_path.mkdir(parents=True, exist_ok=True)
                        print(f"{Fore.GREEN}✓ Created directory: {dest_path}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}✗ Failed to create directory: {e}{Style.RESET_ALL}")
                        continue
                else:
                    continue
            
            if dest_path.exists() and dest_path.is_dir():
                self.dest_path = dest_path
                print(f"{Fore.GREEN}✓ Destination directory: {self.dest_path}{Style.RESET_ALL}")
                logger.info(f"Destination directory set to: {self.dest_path}")
                break
            else:
                print(f"{Fore.RED}✗ Invalid directory path. Please try again.{Style.RESET_ALL}")
        
        # Get file types
        self._get_file_types()
        
        # Get copy/move mode
        print("\nOperation mode:")
        print("1. Copy files (keep originals)")
        print("2. Move files (remove originals)")
        
        while True:
            mode = input("Choose mode (1 or 2) [default: 1]: ").strip()
            if mode == '' or mode == '1':
                self.copy_mode = 'copy'
                print(f"{Fore.GREEN}✓ Will copy files{Style.RESET_ALL}")
                break
            elif mode == '2':
                self.copy_mode = 'move'
                print(f"{Fore.YELLOW}✓ Will move files{Style.RESET_ALL}")
                break
            else:
                print(f"{Fore.RED}Please enter 1 or 2{Style.RESET_ALL}")
        
        # Get duplicate handling
        print("\nHow to handle duplicate filenames:")
        print("1. Rename with hash suffix (recommended)")
        print("2. Skip duplicates")
        print("3. Overwrite existing files")
        
        while True:
            dup_mode = input("Choose mode (1, 2, or 3) [default: 1]: ").strip()
            if dup_mode == '' or dup_mode == '1':
                self.handle_duplicates = 'rename'
                print(f"{Fore.GREEN}✓ Will rename duplicates{Style.RESET_ALL}")
                break
            elif dup_mode == '2':
                self.handle_duplicates = 'skip'
                print(f"{Fore.YELLOW}✓ Will skip duplicates{Style.RESET_ALL}")
                break
            elif dup_mode == '3':
                confirm = input(f"{Fore.YELLOW}⚠ This will overwrite existing files. Are you sure? (yes/no): {Style.RESET_ALL}").strip().lower()
                if confirm == 'yes':
                    self.handle_duplicates = 'overwrite'
                    print(f"{Fore.RED}✓ Will overwrite duplicates{Style.RESET_ALL}")
                    break
            else:
                print(f"{Fore.RED}Please enter 1, 2, or 3{Style.RESET_ALL}")
    
    def _get_file_types(self):
        """Get file types to extract from user"""
        print("\n" + "-"*60)
        print("FILE TYPE SELECTION")
        print("-"*60)
        
        print("\nCommon file type groups:")
        for i, (group_name, extensions) in enumerate(self.FILE_TYPE_GROUPS.items(), 1):
            print(f"{i}. {group_name.capitalize()}: {', '.join(sorted(extensions)[:5])}...")
        
        print(f"\n0. Custom file types")
        print("You can select multiple groups (e.g., '1,3,5') or choose custom")
        
        selection = input("\nSelect file type groups: ").strip()
        
        if selection == '0':
            # Custom file types
            print("\nEnter file extensions separated by spaces (e.g., .txt .log .dat)")
            custom_types = input("Extensions: ").strip().split()
            
            for ext in custom_types:
                if not ext.startswith('.'):
                    ext = '.' + ext
                self.file_types.add(ext.lower())
        else:
            # Parse group selections
            try:
                selected_groups = [int(x.strip()) for x in selection.split(',')]
                group_list = list(self.FILE_TYPE_GROUPS.items())
                
                for group_idx in selected_groups:
                    if 1 <= group_idx <= len(group_list):
                        group_name, extensions = group_list[group_idx - 1]
                        self.file_types.update(extensions)
                        print(f"{Fore.GREEN}✓ Added {group_name} file types{Style.RESET_ALL}")
            except:
                print(f"{Fore.RED}Invalid selection. Using custom input mode.{Style.RESET_ALL}")
                print("\nEnter file extensions separated by spaces (e.g., .txt .log .dat)")
                custom_types = input("Extensions: ").strip().split()
                
                for ext in custom_types:
                    if not ext.startswith('.'):
                        ext = '.' + ext
                    self.file_types.add(ext.lower())
        
        if not self.file_types:
            print(f"{Fore.RED}No file types selected!{Style.RESET_ALL}")
            sys.exit(1)
        
        print(f"\n{Fore.GREEN}Selected file types: {', '.join(sorted(self.file_types))}{Style.RESET_ALL}")
        logger.info(f"File types to extract: {self.file_types}")
    
    def scan_files(self):
        """Scan source directory for matching files"""
        print(f"\n{Fore.CYAN}Scanning for files...{Style.RESET_ALL}")
        logger.info(f"Starting scan of {self.source_path}")
        
        for root, _, files in os.walk(self.source_path):
            self.stats['total_scanned'] += len(files)
            
            for filename in files:
                filepath = Path(root) / filename
                
                # Check if file extension matches
                if filepath.suffix.lower() in self.file_types:
                    try:
                        file_info = {
                            'path': filepath,
                            'size': filepath.stat().st_size,
                            'modified': filepath.stat().st_mtime
                        }
                        self.found_files.append(file_info)
                        self.stats['files_found'] += 1
                        self.stats['total_size'] += file_info['size']
                        
                    except Exception as e:
                        logger.warning(f"Cannot access {filepath}: {e}")
        
        print(f"\n{Fore.GREEN}Scan complete!{Style.RESET_ALL}")
        print(f"   • Total files scanned: {self.stats['total_scanned']:,}")
        print(f"   • Matching files found: {self.stats['files_found']:,}")
        print(f"   • Total size: {self._format_size(self.stats['total_size'])}")
        
        logger.info(f"Found {self.stats['files_found']} matching files")
    
    def _format_size(self, size: int) -> str:
        """Format bytes to human readable size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    
    def _get_file_hash(self, filepath: Path, sample_size: int = 65536) -> str:
        """Get partial hash of file for unique naming"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read(sample_size)).hexdigest()[:8]
        except:
            return hashlib.md5(str(filepath).encode()).hexdigest()[:8]
    
    def _generate_unique_filename(self, original_path: Path, dest_dir: Path) -> Path:
        """Generate unique filename to avoid overwrites"""
        dest_path = dest_dir / original_path.name
        
        if not dest_path.exists() or self.handle_duplicates == 'overwrite':
            return dest_path
        
        if self.handle_duplicates == 'skip':
            return None
        
        # Generate unique name with hash
        file_hash = self._get_file_hash(original_path)
        stem = original_path.stem
        suffix = original_path.suffix
        
        new_name = f"{stem}_{file_hash}{suffix}"
        dest_path = dest_dir / new_name
        
        # If still exists, add counter
        counter = 1
        while dest_path.exists():
            new_name = f"{stem}_{file_hash}_{counter}{suffix}"
            dest_path = dest_dir / new_name
            counter += 1
        
        return dest_path
    
    def extract_files(self):
        """Extract (copy/move) files to destination"""
        if not self.found_files:
            print(f"\n{Fore.YELLOW}No files to extract!{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}FILE EXTRACTION{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n📋 Summary:")
        print(f"   • Files to {self.copy_mode}: {self.stats['files_found']:,}")
        print(f"   • Total size: {self._format_size(self.stats['total_size'])}")
        print(f"   • Source: {self.source_path}")
        print(f"   • Destination: {self.dest_path}")
        
        confirm = input(f"\nProceed with extraction? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print(f"{Fore.YELLOW}Operation cancelled{Style.RESET_ALL}")
            logger.info("User cancelled extraction")
            return
        
        # Process files
        print(f"\n{Fore.CYAN}Extracting files...{Style.RESET_ALL}")
        
        errors = []
        skipped = []
        
        for file_info in tqdm(self.found_files, desc=f"{self.copy_mode.capitalize()}ing files"):
            source_path = file_info['path']
            
            try:
                # Generate destination path
                dest_path = self._generate_unique_filename(source_path, self.dest_path)
                
                if dest_path is None:
                    skipped.append(str(source_path))
                    logger.info(f"Skipped duplicate: {source_path}")
                    continue
                
                # Create subdirectories if preserving structure
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy or move file
                if self.copy_mode == 'copy':
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"Copied: {source_path} -> {dest_path}")
                else:
                    shutil.move(str(source_path), str(dest_path))
                    logger.info(f"Moved: {source_path} -> {dest_path}")
                
                self.stats['files_processed'] += 1
                
                if dest_path.name != source_path.name:
                    self.stats['duplicates_handled'] += 1
                
            except Exception as e:
                error_msg = f"Failed to {self.copy_mode} {source_path}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                self.stats['errors'] += 1
        
        # Report results
        self._display_results(errors, skipped)
        self._save_report(errors, skipped)
    
    def _display_results(self, errors: List[str], skipped: List[str]):
        """Display extraction results"""
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}EXTRACTION COMPLETE{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n✅ Successfully {self.copy_mode}d: {self.stats['files_processed']:,} files")
        
        if self.stats['duplicates_handled'] > 0:
            print(f"🔄 Duplicates renamed: {self.stats['duplicates_handled']:,}")
        
        if skipped:
            print(f"⏭️  Files skipped: {len(skipped):,}")
        
        if errors:
            print(f"\n{Fore.RED}❌ Errors: {self.stats['errors']}{Style.RESET_ALL}")
            for error in errors[:5]:
                print(f"   • {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")
        
        print(f"\n📁 Files extracted to: {self.dest_path}")
    
    def _save_report(self, errors: List[str], skipped: List[str]):
        """Save extraction report"""
        report_dir = self.dest_path / '.extraction_reports'
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'source_directory': str(self.source_path),
            'destination_directory': str(self.dest_path),
            'settings': {
                'file_types': sorted(list(self.file_types)),
                'operation_mode': self.copy_mode,
                'duplicate_handling': self.handle_duplicates
            },
            'statistics': self.stats,
            'errors': errors[:100],  # Save first 100 errors
            'skipped_files': skipped[:100]  # Save first 100 skipped
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")
        logger.info(f"Report saved to: {report_file}")
    
    def run(self):
        """Main execution flow"""
        try:
            self.get_user_input()
            self.scan_files()
            self.extract_files()
            
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
    print(f"{Fore.CYAN}FILE TYPE EXTRACTOR v1.0{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    extractor = FileTypeExtractor()
    extractor.run()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
