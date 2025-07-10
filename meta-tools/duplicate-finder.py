#!/usr/bin/env python3
"""
Duplicate File Finder - Find and manage duplicate files
Identifies duplicates by content hash, suggests cleanup options
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
from utils.config_loader import ConfigLoader
from utils.interactive_config import get_interactive_project_config

class DuplicateFinder:
    def __init__(self, path=".", config=None, min_size=1024):
        self.root = Path(path).resolve()
        self.config = config or ConfigLoader()
        self.min_size = min_size  # Minimum file size to check
        self.file_hashes = defaultdict(list)
        self.duplicates = []
        self.total_waste = 0
        
    def find_duplicates(self):
        """Scan directory for duplicate files"""
        print(f"🔍 Scanning for duplicates in: {self.root}")
        print(f"   (Ignoring files smaller than {self._format_size(self.min_size)})")
        print("-" * 50)
        
        files_checked = 0
        
        # First pass: collect files by size
        size_map = defaultdict(list)
        
        for root, dirs, files in os.walk(self.root):
            # Skip hidden and ignored directories
            ignore_patterns = self.config.get_ignore_patterns()
            dirs[:] = [d for d in dirs if not d.startswith('.') and not any(d == pattern.replace('*', '') for pattern in ignore_patterns)]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                path = Path(root) / file
                try:
                    stat = path.stat()
                    if stat.st_size >= self.min_size:
                        size_map[stat.st_size].append(path)
                except:
                    pass
        
        # Second pass: hash files with same size
        for size, paths in size_map.items():
            if len(paths) > 1:  # Only check if multiple files have same size
                for path in paths:
                    files_checked += 1
                    print(f"\rChecking file {files_checked}...", end='')
                    
                    file_hash = self._calculate_hash(path)
                    if file_hash:
                        self.file_hashes[file_hash].append({
                            'path': path,
                            'size': size,
                            'modified': path.stat().st_mtime
                        })
        
        print(f"\r✓ Checked {files_checked} files")
        
        # Identify duplicates
        for hash_val, files in self.file_hashes.items():
            if len(files) > 1:
                # Sort by modification time (keep oldest)
                files.sort(key=lambda x: x['modified'])
                
                duplicate_group = {
                    'hash': hash_val,
                    'original': files[0],
                    'duplicates': files[1:],
                    'count': len(files),
                    'size': files[0]['size'],
                    'waste': files[0]['size'] * (len(files) - 1)
                }
                
                self.duplicates.append(duplicate_group)
                self.total_waste += duplicate_group['waste']
        
        # Sort by wasted space
        self.duplicates.sort(key=lambda x: x['waste'], reverse=True)
        
    def display_results(self):
        """Display duplicate findings"""
        if not self.duplicates:
            print("\n✨ No duplicate files found!")
            return
        
        print(f"\n🗂️  Found {len(self.duplicates)} sets of duplicates")
        print(f"💾 Total wasted space: {self._format_size(self.total_waste)}")
        print("\n" + "=" * 80)
        
        # Show top duplicates
        for i, dup in enumerate(self.duplicates[:10], 1):
            print(f"\n{i}. {dup['count']} copies of {self._format_size(dup['size'])} file:")
            print(f"   Original: {self._relative_path(dup['original']['path'])}")
            print(f"   Modified: {self._format_date(dup['original']['modified'])}")
            print(f"   Duplicates:")
            
            for d in dup['duplicates'][:3]:
                print(f"     - {self._relative_path(d['path'])}")
            
            if len(dup['duplicates']) > 3:
                print(f"     ... and {len(dup['duplicates']) - 3} more")
            
            print(f"   💾 Space wasted: {self._format_size(dup['waste'])}")
        
        if len(self.duplicates) > 10:
            print(f"\n... and {len(self.duplicates) - 10} more duplicate groups")
        
        self._save_report()
        self._interactive_cleanup()
        
    def _interactive_cleanup(self):
        """Offer interactive cleanup options"""
        print("\n" + "=" * 80)
        print("🧹 CLEANUP OPTIONS")
        print("=" * 80)
        
        if input("\nWould you like to review duplicates for deletion? (y/n): ").lower() != 'y':
            return
        
        total_cleaned = 0
        
        for i, dup in enumerate(self.duplicates[:10], 1):
            print(f"\n--- Duplicate Set {i}/{min(10, len(self.duplicates))} ---")
            print(f"Original: {self._relative_path(dup['original']['path'])}")
            print("Duplicates:")
            
            for j, d in enumerate(dup['duplicates'], 1):
                print(f"  {j}. {self._relative_path(d['path'])}")
            
            action = input("\nAction? (k=keep all, d=delete duplicates, s=skip, q=quit): ").lower()
            
            if action == 'q':
                break
            elif action == 'd':
                print("🗑️  Moving duplicates to .trash folder...")
                trash_dir = self.root / '.trash' / datetime.now().strftime('%Y%m%d_%H%M%S')
                trash_dir.mkdir(parents=True, exist_ok=True)
                
                for d in dup['duplicates']:
                    try:
                        dest = trash_dir / d['path'].name
                        d['path'].rename(dest)
                        total_cleaned += d['size']
                        print(f"   ✓ Moved: {d['path'].name}")
                    except Exception as e:
                        print(f"   ✗ Error moving {d['path']}: {e}")
        
        if total_cleaned > 0:
            print(f"\n✨ Cleaned up {self._format_size(total_cleaned)} of disk space!")
            print("   (Files moved to .trash folder - delete permanently when ready)")
    
    def _save_report(self):
        """Save detailed duplicate report"""
        report_dir = self.config.get_stats_directory(self.root)
        report_dir.mkdir(exist_ok=True, parents=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'path': str(self.root),
            'total_duplicates': len(self.duplicates),
            'total_waste': self.total_waste,
            'duplicates': []
        }
        
        for dup in self.duplicates[:50]:  # Save top 50
            report['duplicates'].append({
                'size': dup['size'],
                'count': dup['count'],
                'waste': dup['waste'],
                'original': str(self._relative_path(dup['original']['path'])),
                'duplicates': [str(self._relative_path(d['path'])) for d in dup['duplicates']]
            })
        
        report_file = report_dir / f"duplicates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
    
    def _calculate_hash(self, filepath):
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                # Read in chunks for large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def _format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _relative_path(self, path):
        try:
            return path.relative_to(self.root)
        except:
            return path
    
    def _format_date(self, timestamp):
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')

def main():
    # Use interactive configuration
    project_path, config = get_interactive_project_config("Duplicate File Finder")
    
    if project_path is None:
        return
    
    # Ask for minimum file size
    print("\n⚙️  Additional Options")
    print("-"*30)
    size_input = input("Minimum file size to check (default: 1KB, press Enter for default): ").strip()
    
    min_size = 1024  # Default 1KB
    if size_input:
        if size_input.endswith('KB'):
            min_size = int(size_input[:-2]) * 1024
        elif size_input.endswith('MB'):
            min_size = int(size_input[:-2]) * 1024 * 1024
        elif size_input.isdigit():
            min_size = int(size_input)
    
    print(f"\n🔍 Starting duplicate search...")
    print("="*60)
    
    finder = DuplicateFinder(project_path, config, min_size)
    finder.find_duplicates()
    finder.display_results()
    
    # Ask if user wants to analyze another project
    print("\n" + "-"*50)
    another = input("Find duplicates in another project? (y/n): ").strip().lower()
    if another == 'y':
        main()

if __name__ == "__main__":
    main()
