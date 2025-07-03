#!/usr/bin/env python3
"""
Project Tracker Setup Script
Helps set up the Project Tracker Suite
"""

import os
import sys
from pathlib import Path

def setup_tracker():
    """Setup the Project Tracker Suite"""
    print("="*60)
    print("🚀 PROJECT TRACKER SUITE SETUP")
    print("="*60)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"\nSetting up in: {current_dir}")
    
    # Create tools directory
    tools_dir = current_dir / "project-tracker-tools"
    if not tools_dir.exists():
        tools_dir.mkdir()
        print(f"✓ Created directory: {tools_dir}")
    else:
        print(f"📁 Using existing directory: {tools_dir}")
    
    # List of required scripts
    scripts = {
        'quick-stats.py': 'Quick project overview and snapshots',
        'duplicate-finder.py': 'Find and remove duplicate files',
        'timeline-tracker.py': 'Track project evolution over time',
        'code-analyzer.py': 'Analyze code structure and dependencies',
        'health-checker.py': 'Check project health and best practices',
        'growth-monitor.py': 'Monitor and predict project growth',
        'project-dashboard.py': 'Comprehensive project overview',
        'project-tracker.py': 'Master control script'
    }
    
    print("\n📋 REQUIRED SCRIPTS:")
    print("-" * 40)
    
    missing_scripts = []
    for script, description in scripts.items():
        script_path = tools_dir / script
        if script_path.exists():
            print(f"✓ {script:<25} - {description}")
        else:
            print(f"✗ {script:<25} - {description}")
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"\n⚠️  Missing {len(missing_scripts)} scripts!")
        print("\nTo complete setup:")
        print("1. Copy all 8 Python scripts to:", tools_dir)
        print("2. Run this setup script again")
        print("\nScripts needed:")
        for script in missing_scripts:
            print(f"   - {script}")
    else:
        print("\n✅ All scripts found!")
        create_shortcuts(tools_dir)
        show_quick_start(tools_dir)
    
    # Create example .gitignore
    create_gitignore_example(current_dir)
    
    # Test imports
    test_dependencies()

def create_shortcuts(tools_dir):
    """Create convenient shortcuts"""
    print("\n🔗 CREATING SHORTCUTS...")
    print("-" * 40)
    
    # Create a simple runner script in current directory
    runner_content = f'''#!/usr/bin/env python3
"""Quick runner for Project Tracker"""
import subprocess
import sys
import os

tools_dir = r"{tools_dir}"
tracker_script = os.path.join(tools_dir, "project-tracker.py")

if os.path.exists(tracker_script):
    subprocess.run([sys.executable, tracker_script] + sys.argv[1:])
else:
    print("Error: Project Tracker not found at:", tracker_script)
'''
    
    runner_path = Path.cwd() / "track"
    with open(runner_path, 'w') as f:
        f.write(runner_content)
    
    if os.name != 'nt':  # Unix-like systems
        os.chmod(runner_path, 0o755)
    
    print(f"✓ Created shortcut: {runner_path}")
    print("  You can now run: ./track or python track")

def create_gitignore_example(base_dir):
    """Create example .gitignore entries"""
    gitignore_example = base_dir / ".gitignore.example"
    
    content = """# Project Tracker Suite
.project-stats/
project-tracker-tools/

# Or if you want to track some data:
# .project-stats/duplicates_*.json
# .project-stats/health_report_*.json
# But keep snapshots:
# !.project-stats/snapshot_*.json
"""
    
    with open(gitignore_example, 'w') as f:
        f.write(content)
    
    print(f"\n📄 Created: {gitignore_example}")
    print("   (Add these entries to your .gitignore if needed)")

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n🔍 CHECKING DEPENDENCIES...")
    print("-" * 40)
    
    dependencies = {
        'json': 'Built-in JSON support',
        'pathlib': 'Path handling',
        'collections': 'Data structures',
        'datetime': 'Date/time handling',
        'hashlib': 'File hashing',
        'ast': 'Python code analysis',
        're': 'Regular expressions'
    }
    
    optional = {
        'matplotlib': 'Advanced visualizations',
        'networkx': 'Network graphs',
        'plotly': 'Interactive charts'
    }
    
    # Check required modules
    all_good = True
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {module:<15} - {description}")
        except ImportError:
            print(f"✗ {module:<15} - {description}")
            all_good = False
    
    print("\nOptional modules (for enhanced features):")
    for module, description in optional.items():
        try:
            __import__(module)
            print(f"✓ {module:<15} - {description}")
        except ImportError:
            print(f"ℹ {module:<15} - {description} (not installed)")
    
    if all_good:
        print("\n✅ All required dependencies are available!")
    else:
        print("\n❌ Some required dependencies are missing!")
        print("   This shouldn't happen with standard Python.")

def show_quick_start(tools_dir):
    """Show quick start guide"""
    print("\n" + "="*60)
    print("🎯 QUICK START GUIDE")
    print("="*60)
    
    print("\n1. FIRST RUN - Create initial snapshot:")
    print(f"   cd {tools_dir}")
    print("   python quick-stats.py /path/to/your/project")
    
    print("\n2. GET FULL OVERVIEW:")
    print("   python project-dashboard.py /path/to/your/project")
    
    print("\n3. USE MASTER CONTROL:")
    print("   python project-tracker.py /path/to/your/project")
    print("   (Or use ./track from current directory)")
    
    print("\n4. COMMON WORKFLOWS:")
    print("   • Daily: Run quick-stats.py")
    print("   • Weekly: Run project-dashboard.py")
    print("   • Monthly: Run health-checker.py")
    print("   • As needed: Run duplicate-finder.py")
    
    print("\n5. VIEW REPORTS:")
    print("   Reports are saved in: <project>/.project-stats/")
    print("   Use project-tracker.py option 9 to browse")
    
    print("\n💡 TIP: Start with project-tracker.py for menu-driven interface!")
    
    # Create example automation script
    create_automation_example(tools_dir)

def create_automation_example(tools_dir):
    """Create example automation script"""
    auto_script = tools_dir / "automate-example.sh"
    
    content = f'''#!/bin/bash
# Example automation script for Project Tracker Suite

TOOLS_DIR="{tools_dir}"
PROJECT_DIR="$1"

if [ -z "$PROJECT_DIR" ]; then
    echo "Usage: $0 <project-directory>"
    exit 1
fi

echo "Running automated analysis for: $PROJECT_DIR"
echo "========================================"

# Run all analyses
cd "$TOOLS_DIR"

echo "1. Quick Stats..."
python quick-stats.py "$PROJECT_DIR"

echo -e "\\n2. Health Check..."
python health-checker.py "$PROJECT_DIR"

echo -e "\\n3. Growth Monitor..."
python growth-monitor.py "$PROJECT_DIR"

echo -e "\\n4. Dashboard..."
python project-dashboard.py "$PROJECT_DIR"

echo -e "\\n========================================"
echo "Analysis complete! Check $PROJECT_DIR/.project-stats/ for reports."
'''
    
    with open(auto_script, 'w') as f:
        f.write(content)
    
    if os.name != 'nt':
        os.chmod(auto_script, 0o755)
    
    print(f"\n📜 Created automation example: {auto_script}")

def main():
    """Main setup function"""
    print("\n🚀 Welcome to Project Tracker Suite Setup!\n")
    
    try:
        setup_tracker()
        
        print("\n" + "="*60)
        print("✨ SETUP COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. If scripts are missing, copy them to project-tracker-tools/")
        print("2. Run: python track (or ./track on Unix)")
        print("3. Start tracking your projects!")
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
