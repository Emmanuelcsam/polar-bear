#!/usr/bin/env python3
"""
Quick Project Stats - Get instant overview of your project
Provides fast statistics without deep analysis
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
from utils.config_loader import ConfigLoader
from utils.interactive_config import get_interactive_project_config

class QuickStats:
    def __init__(self, path=".", config=None):
        self.root = Path(path).resolve()
        self.config = config or ConfigLoader()
        self.stats = {
            'files': 0,
            'dirs': 0,
            'total_size': 0,
            'languages': Counter(),
            'recent_files': [],
            'large_files': []
        }
        
    def scan(self):
        """Quick scan of directory"""
        print(f"📊 Quick Stats for: {self.root}")
        print("-" * 50)
        
        all_files = []
        
        for root, dirs, files in os.walk(self.root):
            # Skip hidden and ignored directories
            ignore_patterns = self.config.get_ignore_patterns()
            dirs[:] = [d for d in dirs if not d.startswith('.') and not any(d == pattern.replace('*', '') for pattern in ignore_patterns)]
            
            self.stats['dirs'] += len(dirs)
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                path = Path(root) / file
                try:
                    stat = path.stat()
                    self.stats['files'] += 1
                    self.stats['total_size'] += stat.st_size
                    
                    # Track language
                    ext = path.suffix.lower()
                    if ext:
                        self.stats['languages'][ext] += 1
                    
                    # Collect file info
                    all_files.append({
                        'path': str(path.relative_to(self.root)),
                        'size': stat.st_size,
                        'modified': stat.st_mtime
                    })
                except:
                    pass
        
        # Get recent and large files
        all_files.sort(key=lambda x: x['modified'], reverse=True)
        self.stats['recent_files'] = all_files[:5]
        
        all_files.sort(key=lambda x: x['size'], reverse=True)
        self.stats['large_files'] = all_files[:5]
        
    def display(self):
        """Display quick stats"""
        print(f"📁 Directories: {self.stats['dirs']:,}")
        print(f"📄 Files: {self.stats['files']:,}")
        print(f"💾 Total Size: {self._format_size(self.stats['total_size'])}")
        
        print("\n🔤 Top Languages:")
        for ext, count in self.stats['languages'].most_common(5):
            print(f"  {ext}: {count} files")
        
        print("\n🕐 Recently Modified:")
        for f in self.stats['recent_files']:
            mod_date = datetime.fromtimestamp(f['modified']).strftime('%Y-%m-%d %H:%M')
            print(f"  {f['path'][:50]:<50} {mod_date}")
        
        print("\n📏 Largest Files:")
        for f in self.stats['large_files']:
            print(f"  {f['path'][:50]:<50} {self._format_size(f['size'])}")
        
        # Save snapshot
        self._save_snapshot()
        
    def _format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _save_snapshot(self):
        """Save stats snapshot for tracking"""
        snapshot_dir = self.config.get_stats_directory(self.root)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'path': str(self.root),
            'stats': {
                'files': self.stats['files'],
                'dirs': self.stats['dirs'],
                'size': self.stats['total_size'],
                'languages': dict(self.stats['languages'].most_common())
            }
        }
        
        snapshot_file = snapshot_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print(f"\n💾 Snapshot saved to: {snapshot_file}")

def main():
    # Use interactive configuration
    project_path, config = get_interactive_project_config("Quick Project Statistics")
    
    if project_path is None:
        return
    
    print(f"\n📊 Starting analysis...")
    print("="*60)
    
    stats = QuickStats(project_path, config)
    stats.scan()
    stats.display()
    
    # Ask if user wants to analyze another project
    print("\n" + "-"*50)
    another = input("Analyze another project? (y/n): ").strip().lower()
    if another == 'y':
        main()

if __name__ == "__main__":
    main()
