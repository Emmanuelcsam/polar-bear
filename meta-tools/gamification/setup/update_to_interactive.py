#!/usr/bin/env python3
"""
Update remaining scripts to use interactive configuration
"""

import os
from pathlib import Path

# Template for updated main function
MAIN_TEMPLATE = '''def main():
    # Use interactive configuration
    project_path, config = get_interactive_project_config("{script_title}")
    
    if project_path is None:
        return
    
    print(f"\\n{emoji} Starting analysis...")
    print("="*60)
    
    {class_init}
    {run_method}
    
    # Ask if user wants to analyze another project
    print("\\n" + "-"*50)
    another = input("{continue_prompt} (y/n): ").strip().lower()
    if another == 'y':
        main()

if __name__ == "__main__":
    main()'''

# Scripts to update with their configurations
scripts_config = {
    'duplicate-finder.py': {
        'title': 'Duplicate File Finder',
        'emoji': '🔍',
        'class_init': 'finder = DuplicateFinder(project_path, config)',
        'run_method': 'finder.find_duplicates()\n    finder.display_results()',
        'continue_prompt': 'Find duplicates in another project?'
    },
    'growth-monitor.py': {
        'title': 'Project Growth Monitor',
        'emoji': '📈',
        'class_init': 'monitor = GrowthMonitor(project_path, config)',
        'run_method': 'monitor.analyze_growth()\n    monitor.display_analysis()',
        'continue_prompt': 'Monitor growth of another project?'
    },
    'timeline-tracker.py': {
        'title': 'Project Timeline Tracker',
        'emoji': '📅',
        'class_init': 'tracker = TimelineTracker(project_path, config)',
        'run_method': 'tracker.track_timeline()\n    tracker.display_timeline()',
        'continue_prompt': 'Track timeline of another project?'
    },
    'stats-viewer.py': {
        'title': 'Statistics Viewer',
        'emoji': '📊',
        'class_init': 'viewer = StatsViewer(project_path, config)',
        'run_method': 'viewer.run()',
        'continue_prompt': 'View stats for another project?'
    },
    'productivity-tracker.py': {
        'title': 'Productivity Tracker',
        'emoji': '⚡',
        'class_init': 'tracker = ProductivityTracker(project_path, config)',
        'run_method': 'tracker.run_interactive()',
        'continue_prompt': 'Track productivity for another project?'
    },
    'habitica-integration.py': {
        'title': 'Habitica Integration',
        'emoji': '🎮',
        'class_init': 'integration = HabiticaIntegration(project_path, config)',
        'run_method': 'integration.run()',
        'continue_prompt': 'Update Habitica for another project?'
    }
}

def update_script(script_path, config):
    """Update a script to use interactive configuration"""
    print(f"\nUpdating {script_path}...")
    
    try:
        content = script_path.read_text()
        
        # Add interactive import if not present
        if 'from interactive_config import' not in content:
            # Find config_loader import
            if 'from config_loader import' in content:
                content = content.replace(
                    'from config_loader import get_project_config, ConfigLoader',
                    'from config_loader import ConfigLoader\nfrom interactive_config import get_interactive_project_config'
                )
                content = content.replace(
                    'from config_loader import ConfigLoader',
                    'from config_loader import ConfigLoader\nfrom interactive_config import get_interactive_project_config'
                )
            else:
                # Add after other imports
                lines = content.split('\n')
                import_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        import_idx = i + 1
                    elif line.strip() and not line.startswith('#'):
                        break
                lines.insert(import_idx, 'from config_loader import ConfigLoader')
                lines.insert(import_idx + 1, 'from interactive_config import get_interactive_project_config')
                content = '\n'.join(lines)
        
        # Replace main function
        main_start = content.find('def main():')
        if main_start != -1:
            # Find the end of main function
            main_end = content.find('\n\nif __name__', main_start)
            if main_end == -1:
                main_end = len(content)
            
            # Generate new main function
            new_main = MAIN_TEMPLATE.format(
                script_title=config['title'],
                emoji=config['emoji'],
                class_init=config['class_init'],
                run_method=config['run_method'],
                continue_prompt=config['continue_prompt']
            )
            
            # Replace old main with new main
            content = content[:main_start] + new_main + content[main_end:]
        
        # Save updated content
        script_path.write_text(content)
        print(f"✓ Updated {script_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating {script_path.name}: {e}")
        return False

def main():
    print("🔧 Updating scripts to use interactive configuration")
    print("="*50)
    
    updated = 0
    failed = 0
    
    for script_name, config in scripts_config.items():
        script_path = Path(script_name)
        if script_path.exists():
            if update_script(script_path, config):
                updated += 1
            else:
                failed += 1
        else:
            print(f"⚠️  Script not found: {script_name}")
            failed += 1
    
    print(f"\n✅ Updated {updated} scripts")
    if failed > 0:
        print(f"❌ Failed to update {failed} scripts")
    
    # Also update project-tracker.py specially since it's a menu system
    print("\n📝 Note: project-tracker.py may need manual adjustment as it's a menu system")
    print("     setup-tracker.py may also need manual review")

if __name__ == "__main__":
    main()