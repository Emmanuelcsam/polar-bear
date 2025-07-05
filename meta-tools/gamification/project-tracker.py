#!/usr/bin/env python3
"""
Project Tracker Master Script - Central hub for all project analysis tools
Provides menu interface and can run all analysis scripts
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
from utils.config_loader import ConfigLoader
from utils.interactive_config import get_interactive_project_config

class ProjectTracker:
    def __init__(self, project_path=None, config=None):
        self.config = config or ConfigLoader()
        self.project_path = project_path or "."
        self.stats_dir = self.config.get_stats_directory(Path(self.project_path))
        self.scripts = {
            '1': {
                'name': 'Quick Stats',
                'file': 'quick-stats.py',
                'desc': 'Get instant overview of your project'
            },
            '2': {
                'name': 'Duplicate Finder',
                'file': 'duplicate-finder.py',
                'desc': 'Find and remove duplicate files'
            },
            '3': {
                'name': 'Timeline Tracker',
                'file': 'timeline-tracker.py',
                'desc': 'Visualize project growth over time'
            },
            '4': {
                'name': 'Code Analyzer',
                'file': 'code-analyzer.py',
                'desc': 'Analyze code structure and dependencies'
            },
            '5': {
                'name': 'Health Checker',
                'file': 'health-checker.py',
                'desc': 'Check for issues and best practices'
            },
            '6': {
                'name': 'Growth Monitor',
                'file': 'growth-monitor.py',
                'desc': 'Track and predict project growth'
            },
            '7': {
                'name': 'Project Dashboard',
                'file': 'project-dashboard.py',
                'desc': 'Comprehensive overview of all metrics'
            }
        }
        
    def display_menu(self):
        """Display main menu"""
        while True:
            self._clear_screen()
            print("="*80)
            print("🚀 PROJECT TRACKER - Master Analysis Suite")
            print("="*80)
            print(f"\nAnalyzing: {Path(self.project_path).resolve()}")
            
            # Show quick stats if available
            self._show_quick_summary()
            
            print("\n📊 AVAILABLE TOOLS:")
            print("-" * 40)
            
            for key, tool in self.scripts.items():
                print(f"  {key}. {tool['name']:<20} - {tool['desc']}")
            
            print("\n🔧 SPECIAL COMMANDS:")
            print("-" * 40)
            print("  8. Run All Analyses")
            print("  9. View Reports")
            print("  0. Setup Scripts")
            print("  C. Change Directory")
            print("  H. Help")
            print("  Q. Quit")
            
            choice = input("\nSelect option: ").strip().lower()
            
            if choice == 'q':
                print("\n👋 Thanks for using Project Tracker!")
                break
            elif choice in self.scripts:
                self._run_script(self.scripts[choice])
            elif choice == '8':
                self._run_all_analyses()
            elif choice == '9':
                self._view_reports()
            elif choice == '0':
                self._setup_scripts()
            elif choice == 'c':
                self._change_directory()
            elif choice == 'h':
                self._show_help()
            else:
                print("\n❌ Invalid option. Press Enter to continue...")
                input()
    
    def _clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _show_quick_summary(self):
        """Show quick summary from latest dashboard"""
        latest_dashboard = self.stats_dir / 'latest_dashboard.json'
        
        if latest_dashboard.exists():
            try:
                with open(latest_dashboard, 'r') as f:
                    data = json.load(f)
                
                overview = data.get('overview', {})
                health = data.get('health_score', 0)
                
                print("\n📈 Quick Summary (from last scan):")
                print(f"  Files: {overview.get('total_files', 0):,}")
                print(f"  Size: {self._format_size(overview.get('total_size', 0))}")
                print(f"  Health: {health}/100")
                
                # Show age of data
                if 'timestamp' in data:
                    age = datetime.now() - datetime.fromisoformat(data['timestamp'])
                    print(f"  Last scan: {self._format_time_ago(age.total_seconds())} ago")
            except:
                pass
    
    def _run_script(self, script_info):
        """Run a specific analysis script"""
        script_path = Path(script_info['file'])
        
        if not script_path.exists():
            print(f"\n❌ Script not found: {script_info['file']}")
            print("   Run 'Setup Scripts' (option 0) first!")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n🚀 Running {script_info['name']}...")
        print("-" * 80)
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path), self.project_path],
                capture_output=False
            )
            
            if result.returncode != 0:
                print(f"\n⚠️  Script exited with error code: {result.returncode}")
        except Exception as e:
            print(f"\n❌ Error running script: {e}")
        
        input("\nPress Enter to continue...")
    
    def _run_all_analyses(self):
        """Run all analysis scripts in sequence"""
        print("\n🔄 Running all analyses...")
        print("="*80)
        
        # Order matters - quick stats first, dashboard last
        run_order = ['1', '6', '3', '5', '4', '2', '7']
        
        for i, key in enumerate(run_order, 1):
            if key in self.scripts:
                script = self.scripts[key]
                print(f"\n[{i}/{len(run_order)}] Running {script['name']}...")
                print("-" * 40)
                
                script_path = Path(script['file'])
                if script_path.exists():
                    try:
                        subprocess.run(
                            [sys.executable, str(script_path), self.project_path],
                            capture_output=False
                        )
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    print("\n" + "-"*40)
                else:
                    print(f"Script not found: {script['file']}")
        
        print("\n✅ All analyses complete!")
        input("\nPress Enter to continue...")
    
    def _view_reports(self):
        """View generated reports"""
        if not self.stats_dir.exists():
            print("\n📁 No reports found. Run some analyses first!")
            input("\nPress Enter to continue...")
            return
        
        while True:
            self._clear_screen()
            print("="*80)
            print("📄 VIEW REPORTS")
            print("="*80)
            
            # List available reports
            reports = {
                '1': ('Latest Dashboard', 'latest_dashboard.json'),
                '2': ('Health History', 'health_history.json'),
                '3': ('All Snapshots', 'snapshot_*.json'),
                '4': ('Duplicate Reports', 'duplicates_*.json'),
                '5': ('Timeline Reports', 'timeline_*.json'),
                '6': ('Code Analysis', 'code_analysis_*.json'),
                '7': ('Health Reports', 'health_report_*.json')
            }
            
            print("\nAvailable reports:")
            for key, (name, pattern) in reports.items():
                files = list(self.stats_dir.glob(pattern))
                print(f"  {key}. {name:<20} ({len(files)} files)")
            
            print("\n  B. Back to main menu")
            
            choice = input("\nSelect report type: ").strip().lower()
            
            if choice == 'b':
                break
            elif choice in reports:
                self._display_report_files(reports[choice][1])
    
    def _display_report_files(self, pattern):
        """Display specific report files"""
        files = sorted(self.stats_dir.glob(pattern), reverse=True)
        
        if not files:
            print("\nNo files found.")
            input("Press Enter to continue...")
            return
        
        print(f"\nFound {len(files)} files:")
        for i, file in enumerate(files[:10], 1):
            size = file.stat().st_size
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"  {i}. {file.name} ({self._format_size(size)}, {mtime.strftime('%Y-%m-%d %H:%M')})")
        
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        
        choice = input("\nEnter number to view (or Enter to go back): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= min(10, len(files)):
            file_path = files[int(choice) - 1]
            self._view_json_file(file_path)
    
    def _view_json_file(self, file_path):
        """View a JSON report file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"\n📄 {file_path.name}")
            print("="*80)
            
            # Pretty print based on file type
            if 'health_score' in data:
                self._print_health_report(data)
            elif 'overview' in data:
                self._print_dashboard(data)
            elif 'timeline' in data:
                self._print_timeline(data)
            else:
                # Generic pretty print
                import pprint
                pprint.pprint(data, width=80, compact=True)
            
        except Exception as e:
            print(f"\n❌ Error reading file: {e}")
        
        input("\nPress Enter to continue...")
    
    def _print_health_report(self, data):
        """Pretty print health report"""
        print(f"Health Score: {data.get('health_score', 0)}/100")
        print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
        
        if 'issues' in data:
            print("\nIssues:")
            for level, items in data['issues'].items():
                if items:
                    print(f"\n{level.upper()}:")
                    for item in items[:5]:
                        print(f"  - {item}")
    
    def _print_dashboard(self, data):
        """Pretty print dashboard"""
        overview = data.get('overview', {})
        print(f"Files: {overview.get('total_files', 0):,}")
        print(f"Size: {self._format_size(overview.get('total_size', 0))}")
        print(f"Health: {data.get('health_score', 0)}/100")
        
        if 'languages' in data and data['languages'].get('counts'):
            print("\nTop Languages:")
            for lang, count in list(data['languages']['counts'].items())[:5]:
                print(f"  {lang}: {count} files")
    
    def _print_timeline(self, data):
        """Pretty print timeline"""
        print(f"Timeline entries: {len(data.get('timeline', {}))}")
        if 'date_range' in data:
            print(f"Date range: {data['date_range'].get('oldest', 'Unknown')} to {data['date_range'].get('newest', 'Unknown')}")
    
    def _setup_scripts(self):
        """Create all analysis scripts"""
        print("\n🔧 Setting up analysis scripts...")
        print("-" * 40)
        
        # Script contents would be embedded here in a real implementation
        print("\n⚠️  Script setup requires manual creation.")
        print("   Please create the following Python scripts:")
        
        for script in self.scripts.values():
            print(f"   - {script['file']}")
        
        print("\n   Copy the script contents from the individual artifacts.")
        
        input("\nPress Enter to continue...")
    
    def _change_directory(self):
        """Change the project directory to analyze"""
        print("\n📁 Change Project Directory")
        print("-" * 40)
        print(f"Current: {Path(self.project_path).resolve()}")
        
        # Use interactive config to select new project
        project_path, _ = get_interactive_project_config("Select New Project")
        
        if project_path:
            self.project_path = str(project_path)
            self.stats_dir = self.config.get_stats_directory(project_path)
            print(f"\n✅ Changed to: {project_path}")
            
        input("\nPress Enter to continue...")
    
    def _show_help(self):
        """Show help information"""
        self._clear_screen()
        print("="*80)
        print("📚 PROJECT TRACKER HELP")
        print("="*80)
        
        print("\n🎯 PURPOSE:")
        print("  Track and analyze your project development over time.")
        print("  Get insights into code quality, growth patterns, and potential issues.")
        
        print("\n📊 TOOLS OVERVIEW:")
        print("  • Quick Stats      - Fast overview, creates snapshots")
        print("  • Duplicate Finder - Find duplicate files, save disk space")
        print("  • Timeline Tracker - See how your project evolved")
        print("  • Code Analyzer    - Understand code structure and dependencies")
        print("  • Health Checker   - Find issues and improve project quality")
        print("  • Growth Monitor   - Track progress and predict future growth")
        print("  • Dashboard        - All metrics in one comprehensive view")
        
        print("\n💡 TIPS:")
        print("  1. Run Quick Stats regularly to create snapshots")
        print("  2. Use Dashboard for a complete overview")
        print("  3. Reports are saved in .project-stats/ directory")
        print("  4. Run 'All Analyses' for a complete checkup")
        print("  5. Check Health regularly to maintain code quality")
        
        print("\n📁 DATA STORAGE:")
        print("  All data is stored locally in .project-stats/")
        print("  Add this to .gitignore if you don't want to track it")
        
        input("\nPress Enter to continue...")
    
    def _format_size(self, size):
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _format_time_ago(self, seconds):
        """Format seconds to human readable time ago"""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds/60)} minutes"
        elif seconds < 86400:
            return f"{int(seconds/3600)} hours"
        else:
            return f"{int(seconds/86400)} days"

def main():
    """Main entry point"""
    print("\n🚀 Welcome to Project Tracker!")
    print("   Your comprehensive project analysis suite\n")
    
    # Use interactive configuration
    project_path, config = get_interactive_project_config("Project Tracker Master")
    
    if project_path is None:
        return
    
    tracker = ProjectTracker(str(project_path), config)
    
    try:
        tracker.display_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
