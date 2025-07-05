#!/usr/bin/env python3
"""
Script to update all remaining Python scripts to use the config system
"""

import os
from pathlib import Path

# Scripts that still need updating
scripts_to_update = [
    'code-analyzer.py',
    'duplicate-finder.py',
    'growth-monitor.py',
    'timeline-tracker.py',
    'project-tracker.py',
    'stats-viewer.py',
    'setup-tracker.py',
    'habitica-integration.py',
    'productivity-tracker.py'
]

# Common imports to add
config_import = "from config_loader import get_project_config, ConfigLoader"

# Update patterns
update_patterns = [
    # Update class __init__
    {
        'find': r'def __init__\(self, path=["\']\.["\']?\):',
        'replace': 'def __init__(self, path=".", config=None):'
    },
    {
        'find': r'self\.root = Path\(path\)\.resolve\(\)',
        'replace': '''self.root = Path(path).resolve()
        self.config = config or ConfigLoader()'''
    },
    # Update stats directory references
    {
        'find': r"Path\(['\"]\.project-stats['\"]\)",
        'replace': "self.config.get_stats_directory(self.root)"
    },
    {
        'find': r"self\.stats_dir = Path\(['\"]\.project-stats['\"]\)",
        'replace': "self.stats_dir = self.config.get_stats_directory(self.root)"
    },
    # Update directory skipping
    {
        'find': r"dirs\[:] = \[d for d in dirs if not d\.startswith\('\.'\) and d not in \['node_modules', '__pycache__'.*?\]\]",
        'replace': '''ignore_patterns = self.config.get_ignore_patterns()
            dirs[:] = [d for d in dirs if not d.startswith('.') and not any(d == pattern.replace('*', '') for pattern in ignore_patterns)]'''
    }
]

def update_script(script_path):
    """Update a single script to use config system"""
    print(f"\nUpdating {script_path.name}...")
    
    try:
        content = script_path.read_text()
        original_content = content
        
        # Add config import if not present
        if 'from config_loader import' not in content:
            # Find where to insert the import
            import_lines = []
            lines = content.split('\n')
            insert_pos = 0
            
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_pos = i + 1
                elif line.strip() and not line.startswith('#') and not line.startswith('"""'):
                    break
            
            lines.insert(insert_pos, config_import)
            content = '\n'.join(lines)
        
        # Update main function
        if 'def main():' in content:
            # Find and update main function
            main_start = content.find('def main():')
            if main_start != -1:
                # Look for the argparse section
                parser_section = content[main_start:main_start+2000]
                
                if 'argparse.ArgumentParser' in parser_section and 'get_project_config' not in parser_section:
                    # Replace the standard argparse with config system
                    new_main = '''def main():
    # Use configuration system
    config, project_path, args = get_project_config("Project analysis tool")
    
    if project_path is None:
        # Config command was handled (like --list-projects)
        return
    
    print(f"🎯 Analyzing project: {project_path}")
    print("="*60)'''
                    
                    # Find the class instantiation pattern
                    class_names = ['Analyzer', 'Finder', 'Monitor', 'Tracker', 'Dashboard', 'Viewer', 'Checker']
                    for class_name in class_names:
                        for variant in [class_name, f'{class_name}()', f'Code{class_name}', f'Duplicate{class_name}', 
                                      f'Growth{class_name}', f'Timeline{class_name}', f'Project{class_name}',
                                      f'Stats{class_name}', f'Setup{class_name}']:
                            pattern = f'{variant}(args.path)'
                            if pattern in content:
                                content = content.replace(pattern, f'{variant}(project_path, config)')
                                break
        
        # Apply general patterns
        for pattern in update_patterns:
            if pattern['find'] in content:
                content = content.replace(pattern['find'], pattern['replace'])
        
        # Save if changed
        if content != original_content:
            script_path.write_text(content)
            print(f"✓ Updated {script_path.name}")
            return True
        else:
            print(f"⚠️  No changes needed for {script_path.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {script_path.name}: {e}")
        return False

def main():
    """Update all scripts"""
    print("🔧 Updating all scripts to use configuration system")
    print("="*50)
    
    updated = 0
    failed = 0
    
    for script_name in scripts_to_update:
        script_path = Path(script_name)
        if script_path.exists():
            if update_script(script_path):
                updated += 1
            else:
                failed += 1
        else:
            print(f"⚠️  Script not found: {script_name}")
            failed += 1
    
    print(f"\n✅ Updated {updated} scripts")
    if failed > 0:
        print(f"❌ Failed to update {failed} scripts")
    
    print("\n📝 Note: Some scripts may need manual review and adjustment")
    print("   Run each script with --help to see new config options")

if __name__ == "__main__":
    main()