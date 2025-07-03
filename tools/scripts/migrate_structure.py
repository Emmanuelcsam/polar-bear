#!/usr/bin/env python3
"""
Migration script to reorganize Martin project structure
This script helps move remaining files to their new locations
"""

import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectMigrator:
    def __init__(self, project_root):
        self.root = Path(project_root)
        self.migrations = []
        self.define_migrations()
    
    def define_migrations(self):
        """Define all file migrations"""
        
        # Current process utility scripts
        self.add_migration(
            "current-process/utility_scripts/*.py",
            "src/core/utils/scripts/"
        )
        
        # Legacy backups
        self.add_migration(
            "current-process/legacy_backup-*",
            "legacy/backups/"
        )
        
        # GUI from old processes
        self.add_migration(
            "old-processes/studio/image_processor_gui.py",
            "src/gui/studio/"
        )
        self.add_migration(
            "old-processes/studio/app.py",
            "src/gui/studio/"
        )
        
        # ML components
        self.add_migration(
            "old-processes/C++ Method/ml_classifier.py",
            "src/ml/models/"
        )
        self.add_migration(
            "old-processes/C++ Method/train_advanced_models.py",
            "src/ml/training/"
        )
        self.add_migration(
            "old-processes/C++ Method/unet_segmentation.py",
            "src/ml/models/"
        )
        
        # Research experiments
        self.add_migration(
            "old-processes/research_focused_method/*",
            "research/experiments/research_focused/"
        )
        self.add_migration(
            "old-processes/test*.py",
            "research/experiments/iterations/"
        )
        
        # Useful scripts
        self.add_migration(
            "old-processes/useful scripts/*",
            "tools/scripts/analysis/"
        )
        
        # Config files
        self.add_migration(
            "**/config.json",
            "config/default/",
            preserve_path=True
        )
        self.add_migration(
            "**/calibration.json",
            "config/default/"
        )
        
        # Documentation
        self.add_migration(
            "old-processes/studio/guides/*",
            "docs/user_guide/studio/"
        )
        
        # Test data
        self.add_migration(
            "current-process/create_test_images.py",
            "tests/fixtures/"
        )
        
        # Potential upgrades
        self.add_migration(
            "potential_upgrades/*.txt",
            "research/experiments/proposals/"
        )
        self.add_migration(
            "potential_upgrades/fiber_defect_inspection/*",
            "research/prototypes/fiber_defect_inspection/"
        )
        
    def add_migration(self, source_pattern, destination, preserve_path=False):
        """Add a migration rule"""
        self.migrations.append({
            'source': source_pattern,
            'destination': destination,
            'preserve_path': preserve_path
        })
    
    def find_files(self, pattern):
        """Find files matching pattern"""
        if '**' in pattern:
            return list(self.root.rglob(pattern.replace('**/', '')))
        else:
            return list(self.root.glob(pattern))
    
    def migrate_file(self, source, destination, preserve_path=False):
        """Migrate a single file"""
        src_path = Path(source)
        
        if preserve_path:
            # Preserve relative path structure
            rel_path = src_path.relative_to(self.root)
            dest_path = self.root / destination / rel_path
        else:
            dest_path = self.root / destination / src_path.name
        
        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        try:
            if src_path.is_file():
                shutil.copy2(src_path, dest_path)
                logger.info(f"Copied: {src_path} -> {dest_path}")
            elif src_path.is_dir():
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                logger.info(f"Copied directory: {src_path} -> {dest_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy {src_path}: {e}")
            return False
    
    def run_migrations(self, dry_run=True):
        """Run all migrations"""
        logger.info(f"{'DRY RUN' if dry_run else 'RUNNING'} migrations...")
        
        total_files = 0
        migrated_files = 0
        
        for migration in self.migrations:
            files = self.find_files(migration['source'])
            
            for file_path in files:
                total_files += 1
                
                if dry_run:
                    logger.info(f"Would migrate: {file_path}")
                else:
                    if self.migrate_file(
                        file_path, 
                        migration['destination'],
                        migration['preserve_path']
                    ):
                        migrated_files += 1
        
        logger.info(f"\nMigration {'preview' if dry_run else 'complete'}:")
        logger.info(f"Total files: {total_files}")
        if not dry_run:
            logger.info(f"Migrated: {migrated_files}")
            logger.info(f"Failed: {total_files - migrated_files}")
    
    def create_init_files(self):
        """Create __init__.py files for all Python packages"""
        python_dirs = [
            "src/core/detection",
            "src/core/separation",
            "src/core/processing", 
            "src/core/utils",
            "src/ml",
            "src/ml/models",
            "src/ml/training",
            "src/ml/inference",
            "src/api",
            "src/api/rest",
            "src/api/realtime",
            "src/gui",
            "src/gui/studio"
        ]
        
        for dir_path in python_dirs:
            init_path = self.root / dir_path / "__init__.py"
            if not init_path.exists():
                init_path.parent.mkdir(parents=True, exist_ok=True)
                init_path.write_text('"""Package initialization"""')
                logger.info(f"Created: {init_path}")
    
    def update_imports(self):
        """Update import statements in Python files"""
        logger.info("Updating import statements...")
        
        # Define import replacements
        replacements = [
            ("from enhanced_logging import", "from core.utils.logging import"),
            ("from config_manager import", "from core.utils.config import"),
            ("from enhanced_process import", "from core.processing.process import"),
            ("from enhanced_separation import", "from core.separation.separation import"),
            ("from enhanced_detection import", "from core.detection.detection import"),
            ("from realtime_processor import", "from api.realtime.processor import"),
            ("import enhanced_logging", "import core.utils.logging"),
            ("import config_manager", "import core.utils.config"),
        ]
        
        # Find all Python files in src
        python_files = list((self.root / "src").rglob("*.py"))
        
        for py_file in python_files:
            try:
                content = py_file.read_text()
                original_content = content
                
                for old_import, new_import in replacements:
                    content = content.replace(old_import, new_import)
                
                if content != original_content:
                    py_file.write_text(content)
                    logger.info(f"Updated imports in: {py_file}")
                    
            except Exception as e:
                logger.error(f"Failed to update {py_file}: {e}")
    
    def cleanup_empty_dirs(self):
        """Remove empty directories"""
        logger.info("Cleaning up empty directories...")
        
        for dirpath, dirnames, filenames in os.walk(self.root, topdown=False):
            if not dirnames and not filenames:
                try:
                    Path(dirpath).rmdir()
                    logger.info(f"Removed empty directory: {dirpath}")
                except:
                    pass


def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Martin project structure")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview migrations without executing")
    parser.add_argument("--update-imports", action="store_true", help="Update import statements")
    parser.add_argument("--create-inits", action="store_true", help="Create __init__.py files")
    parser.add_argument("--cleanup", action="store_true", help="Remove empty directories")
    
    args = parser.parse_args()
    
    migrator = ProjectMigrator(args.project_root)
    
    if args.create_inits:
        migrator.create_init_files()
    
    if args.update_imports:
        migrator.update_imports()
    
    if not args.create_inits and not args.update_imports and not args.cleanup:
        migrator.run_migrations(dry_run=args.dry_run)
    
    if args.cleanup:
        migrator.cleanup_empty_dirs()


if __name__ == "__main__":
    main()