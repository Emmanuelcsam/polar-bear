#!/usr/bin/env python3
"""
Merge existing scripts with generated OpenCV scripts
"""

import os
import shutil
from pathlib import Path

def merge_scripts():
    """Copy existing scripts to opencv_scripts directory structure"""
    source_dir = Path("scripts")
    target_dir = Path("opencv_scripts")
    
    if not source_dir.exists():
        print("❌ Source scripts directory not found")
        return
    
    if not target_dir.exists():
        print("❌ Target opencv_scripts directory not found")
        print("Run generate_opencv_scripts.py first")
        return
    
    # Create 'existing' subdirectory
    existing_dir = target_dir / "existing"
    existing_dir.mkdir(exist_ok=True)
    
    copied = 0
    skipped = 0
    
    # Copy all .py files from scripts
    for script_path in source_dir.rglob("*.py"):
        if script_path.name.startswith("_"):
            continue
        
        # Determine target path
        rel_path = script_path.relative_to(source_dir)
        target_path = existing_dir / rel_path
        
        # Create parent directories
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy if doesn't exist
        if not target_path.exists():
            shutil.copy2(script_path, target_path)
            copied += 1
        else:
            skipped += 1
    
    print(f"✅ Merged scripts:")
    print(f"   - Copied: {copied}")
    print(f"   - Skipped: {skipped}")
    print(f"   - Total in opencv_scripts: {len(list(target_dir.rglob('*.py')))}")

if __name__ == "__main__":
    merge_scripts()