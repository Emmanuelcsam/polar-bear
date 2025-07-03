#!/usr/bin/env python3
"""
Project Dashboard - Comprehensive overview of all project statistics
Combines all metrics into a single, easy-to-read dashboard
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import hashlib

class ProjectDashboard:
    def __init__(self, path="."):
        self.root = Path(path).resolve()
        self.stats_dir = Path('.project-stats')
        self.dashboard_data = {
            'overview': {},
            'languages': {},
            'recent_activity': [],
            'health_score': 0,
            'issues': [],
            'growth': {},
            'duplicates': [],
            'dependencies': []
        }
        
    def generate_dashboard(self):
        """Generate complete project dashboard"""
        print("🎯 GENERATING PROJECT DASHBOARD")
        print("="*80)
        print(f"Project: {self.root.name}")
        print(f"Path: {self.root}")
        print("="*80)
        
        # Run all analyses
        self._analyze_overview()
        self._analyze_languages()
        self._analyze_recent_activity()
        self._analyze_health()
        self._analyze_growth()
        self._find_quick_issues()
        self._check_dependencies()
        
        # Display dashboard
        self._display_dashboard()
        
        # Save dashboard
        self._save_dashboard()
    
    def _analyze_overview(self):
        """Get basic project overview"""
        print("\n📊 Analyzing project overview...", end='')
        
        total_files = 0
        total_dirs = 0
        total_size = 0
        file_types = Counter()
        
        for root, dirs, files in os.walk(self.root):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
            
            total_dirs += len(dirs)
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                path = Path(root) / file
                try:
                    stat = path.stat()
                    total_files += 1
                    total_size += stat.st_size
                    
                    # Categorize file
                    ext = path.suffix.lower()
                    if ext in ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs']:
                        file_types['code'] += 1
                    elif ext in ['.md', '.txt', '.rst', '.doc', '.pdf']:
                        file_types['docs'] += 1
                    elif ext in ['.jpg', '.png', '.gif', '.svg']:
                        file_types['images'] += 1
                    elif ext in ['.json', '.yml', '.yaml', '.xml', '.conf']:
                        file_types['config'] += 1
                    else:
                        file_types['other'] += 1
                except:
                    pass
        
        self.dashboard_data['overview'] = {
            'total_files': total_files,
            'total_dirs': total_dirs,
            'total_size': total_size,
            'file_types': dict(file_types),
            'avg_file_size': total_size // total_files if total_files > 0 else 0
        }
        
        # Check for git
        if (self.root / '.git').exists():
            try:
                result = subprocess.run(
                    ['git', 'rev-list', '--count', 'HEAD'],
                    cwd=self.root,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    self.dashboard_data['overview']['commits'] = int(result.stdout.strip())
            except:
                pass
        
        print(" ✓")
    
    def _analyze_languages(self):
        """Analyze programming languages used"""
        print("🔤 Analyzing languages...", end='')
        
        languages = Counter()
        language_lines = defaultdict(int)
        
        # Language mappings
        ext_to_lang = {
            '.py': 'Python', '.js': 'JavaScript', '.java': 'Java',
            '.cpp': 'C++', '.c': 'C', '.cs': 'C#', '.go': 'Go',
            '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby',
            '.swift': 'Swift', '.kt': 'Kotlin', '.ts': 'TypeScript',
            '.html': 'HTML', '.css': 'CSS', '.sql': 'SQL',
            '.sh': 'Shell', '.r': 'R', '.m': 'MATLAB'
        }
        
        for ext, lang in ext_to_lang.items():
            files = list(self.root.rglob(f'*{ext}'))
            if files:
                languages[lang] = len(files)
                
                # Count lines for top languages
                if len(files) < 100:  # Don't count lines for too many files
                    for file in files:
                        try:
                            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = len(f.readlines())
                                language_lines[lang] += lines
                        except:
                            pass
        
        self.dashboard_data['languages'] = {
            'counts': dict(languages.most_common()),
            'lines': dict(language_lines),
            'primary': languages.most_common(1)[0][0] if languages else 'Unknown'
        }
        
        print(" ✓")
    
    def _analyze_recent_activity(self):
        """Analyze recent file activity"""
        print("🕐 Analyzing recent activity...", end='')
        
        recent_files = []
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                path = Path(root) / file
                try:
                    stat = path.stat()
                    mod_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    if mod_time > cutoff_date:
                        recent_files.append({
                            'path': str(path.relative_to(self.root)),
                            'modified': mod_time,
                            'size': stat.st_size
                        })
                except:
                    pass
        
        # Sort by modification time
        recent_files.sort(key=lambda x: x['modified'], reverse=True)
        self.dashboard_data['recent_activity'] = recent_files[:20]
        
        print(" ✓")
    
    def _analyze_health(self):
        """Quick health check"""
        print("🏥 Checking project health...", end='')
        
        health_score = 100
        issues = []
        
        # Check for essential files
        if not any((self.root / f).exists() for f in ['README.md', 'README.txt', 'README']):
            health_score -= 10
            issues.append("Missing README file")
        
        if (self.root / '.git').exists() and not (self.root / '.gitignore').exists():
            health_score -= 5
            issues.append("Missing .gitignore file")
        
        # Check for large files
        large_files = []
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                path = Path(root) / file
                try:
                    size = path.stat().st_size
                    if size > 50 * 1024 * 1024:  # 50MB
                        large_files.append(path)
                except:
                    pass
        
        if large_files:
            health_score -= 5
            issues.append(f"{len(large_files)} large files (>50MB)")
        
        # Check for potential security issues
        env_file = self.root / '.env'
        if env_file.exists() and (self.root / '.git').exists():
            if (self.root / '.gitignore').exists():
                with open(self.root / '.gitignore', 'r') as f:
                    if '.env' not in f.read():
                        health_score -= 15
                        issues.append("⚠️  .env file not in .gitignore!")
        
        self.dashboard_data['health_score'] = health_score
        self.dashboard_data['issues'] = issues
        
        print(" ✓")
    
    def _analyze_growth(self):
        """Analyze project growth"""
        print("📈 Analyzing growth...", end='')
        
        growth_data = {
            'snapshots': 0,
            'first_snapshot': None,
            'growth_rate': 0,
            'trend': 'unknown'
        }
        
        if self.stats_dir.exists():
            snapshots = list(self.stats_dir.glob('snapshot_*.json'))
            growth_data['snapshots'] = len(snapshots)
            
            if len(snapshots) >= 2:
                # Get first and last snapshot
                snapshots.sort()
                
                try:
                    with open(snapshots[0], 'r') as f:
                        first = json.load(f)
                    with open(snapshots[-1], 'r') as f:
                        last = json.load(f)
                    
                    growth_data['first_snapshot'] = first['timestamp']
                    
                    # Calculate growth
                    file_growth = last['stats']['files'] - first['stats']['files']
                    days = (datetime.fromisoformat(last['timestamp']) - 
                           datetime.fromisoformat(first['timestamp'])).days
                    
                    if days > 0:
                        growth_data['growth_rate'] = file_growth / days
                        
                        if growth_data['growth_rate'] > 1:
                            growth_data['trend'] = 'rapid growth'
                        elif growth_data['growth_rate'] > 0:
                            growth_data['trend'] = 'steady growth'
                        elif growth_data['growth_rate'] < -0.5:
                            growth_data['trend'] = 'declining'
                        else:
                            growth_data['trend'] = 'stable'
                except:
                    pass
        
        self.dashboard_data['growth'] = growth_data
        
        print(" ✓")
    
    def _find_quick_issues(self):
        """Find quick issues like duplicates"""
        print("🔍 Checking for issues...", end='')
        
        # Quick duplicate check (only for small files)
        file_hashes = defaultdict(list)
        duplicate_count = 0
        duplicate_size = 0
        
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files[:100]:  # Check first 100 files only for speed
                path = Path(root) / file
                try:
                    size = path.stat().st_size
                    if size < 1024 * 1024:  # Only files < 1MB
                        with open(path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        file_hashes[file_hash].append((path, size))
                except:
                    pass
        
        for hash_val, files in file_hashes.items():
            if len(files) > 1:
                duplicate_count += len(files) - 1
                duplicate_size += files[0][1] * (len(files) - 1)
        
        if duplicate_count > 0:
            self.dashboard_data['duplicates'] = {
                'count': duplicate_count,
                'wasted_space': duplicate_size
            }
        
        print(" ✓")
    
    def _check_dependencies(self):
        """Check for dependency files"""
        print("📦 Checking dependencies...", end='')
        
        dep_files = {
            'requirements.txt': 'Python',
            'package.json': 'Node.js',
            'Gemfile': 'Ruby',
            'pom.xml': 'Java/Maven',
            'Cargo.toml': 'Rust',
            'go.mod': 'Go'
        }
        
        found = []
        for dep_file, tech in dep_files.items():
            if (self.root / dep_file).exists():
                found.append(tech)
        
        self.dashboard_data['dependencies'] = found
        
        print(" ✓")
    
    def _display_dashboard(self):
        """Display the complete dashboard"""
        print("\n" + "="*80)
        print("📊 PROJECT DASHBOARD")
        print("="*80)
        
        # Overview section
        overview = self.dashboard_data['overview']
        print("\n📋 OVERVIEW")
        print("-" * 40)
        print(f"Files: {overview['total_files']:,}")
        print(f"Directories: {overview['total_dirs']:,}")
        print(f"Total Size: {self._format_size(overview['total_size'])}")
        print(f"Average File Size: {self._format_size(overview['avg_file_size'])}")
        if 'commits' in overview:
            print(f"Git Commits: {overview['commits']:,}")
        
        # File type breakdown
        print("\nFile Types:")
        for ftype, count in overview['file_types'].items():
            percent = (count / overview['total_files'] * 100) if overview['total_files'] > 0 else 0
            bar = '█' * int(percent / 2)
            print(f"  {ftype:8}: {bar:<25} {count:5} ({percent:.1f}%)")
        
        # Languages section
        langs = self.dashboard_data['languages']
        if langs['counts']:
            print("\n🔤 LANGUAGES")
            print("-" * 40)
            print(f"Primary Language: {langs['primary']}")
            print("\nLanguage Distribution:")
            
            total_files = sum(langs['counts'].values())
            for lang, count in list(langs['counts'].items())[:5]:
                percent = (count / total_files * 100)
                bar = '█' * int(percent)
                print(f"  {lang:12}: {bar:<20} {count:4} files ({percent:.1f}%)")
            
            if langs['lines']:
                print("\nLines of Code:")
                for lang, lines in sorted(langs['lines'].items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  {lang}: {lines:,} lines")
        
        # Health section
        health_score = self.dashboard_data['health_score']
        print("\n🏥 HEALTH CHECK")
        print("-" * 40)
        
        # Health bar
        health_bar = '█' * (health_score // 5) + '░' * ((100 - health_score) // 5)
        if health_score >= 90:
            status = "Excellent"
            color = '\033[92m'  # Green
        elif health_score >= 70:
            status = "Good"
            color = '\033[93m'  # Yellow
        else:
            status = "Needs Attention"
            color = '\033[91m'  # Red
        
        print(f"Health Score: {color}[{health_bar}] {health_score}/100 - {status}\033[0m")
        
        if self.dashboard_data['issues']:
            print("\nIssues Found:")
            for issue in self.dashboard_data['issues'][:5]:
                print(f"  • {issue}")
        
        # Recent activity
        if self.dashboard_data['recent_activity']:
            print("\n🕐 RECENT ACTIVITY (Last 7 days)")
            print("-" * 40)
            print(f"Files modified: {len(self.dashboard_data['recent_activity'])}")
            print("\nMost recently modified:")
            for item in self.dashboard_data['recent_activity'][:5]:
                mod_date = item['modified'].strftime('%Y-%m-%d %H:%M')
                print(f"  {item['path'][:40]:<40} {mod_date}")
        
        # Growth section
        growth = self.dashboard_data['growth']
        if growth['snapshots'] > 0:
            print("\n📈 GROWTH TRACKING")
            print("-" * 40)
            print(f"Tracking snapshots: {growth['snapshots']}")
            if growth['first_snapshot']:
                first_date = datetime.fromisoformat(growth['first_snapshot'])
                age_days = (datetime.now() - first_date).days
                print(f"Tracking since: {first_date.strftime('%Y-%m-%d')} ({age_days} days)")
                print(f"Growth trend: {growth['trend']}")
                if growth['growth_rate'] != 0:
                    print(f"Average growth: {abs(growth['growth_rate']):.1f} files/day")
        
        # Quick wins section
        print("\n💡 QUICK WINS")
        print("-" * 40)
        
        quick_wins = []
        
        if 'README' not in [issue for issue in self.dashboard_data['issues'] if 'README' in issue]:
            if (self.root / 'README.md').exists():
                size = (self.root / 'README.md').stat().st_size
                if size < 500:
                    quick_wins.append("Expand your README file (currently very small)")
        
        if self.dashboard_data.get('duplicates'):
            dup = self.dashboard_data['duplicates']
            quick_wins.append(f"Remove duplicate files to save {self._format_size(dup['wasted_space'])}")
        
        if not self.dashboard_data['dependencies']:
            quick_wins.append("Add dependency management (requirements.txt, package.json, etc.)")
        
        if health_score < 90:
            quick_wins.append("Address health check issues to improve project quality")
        
        if quick_wins:
            for i, win in enumerate(quick_wins[:3], 1):
                print(f"  {i}. {win}")
        else:
            print("  ✨ Your project is in great shape!")
        
        # Summary
        print("\n" + "="*80)
        print("📝 SUMMARY")
        print("="*80)
        
        # Create a one-line summary
        summary_parts = []
        summary_parts.append(f"{overview['total_files']:,} files")
        summary_parts.append(f"{self._format_size(overview['total_size'])}")
        if langs['counts']:
            summary_parts.append(f"primarily {langs['primary']}")
        summary_parts.append(f"health: {health_score}/100")
        
        print(f"Your project: {', '.join(summary_parts)}")
        
        # Next steps
        print("\n🚀 Next Steps:")
        print("  1. Run individual analyzers for detailed insights:")
        print("     - quick-stats.py     : Fast overview")
        print("     - duplicate-finder.py: Find duplicate files")
        print("     - timeline-tracker.py: See project evolution")
        print("     - code-analyzer.py   : Analyze code structure")
        print("     - health-checker.py  : Detailed health check")
        print("     - growth-monitor.py  : Track growth over time")
        print("\n  2. Run this dashboard regularly to track progress!")
    
    def _save_dashboard(self):
        """Save dashboard data"""
        self.stats_dir.mkdir(exist_ok=True)
        
        dashboard_file = self.stats_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare data for JSON
        save_data = self.dashboard_data.copy()
        
        # Convert datetime objects
        if save_data['recent_activity']:
            for item in save_data['recent_activity']:
                item['modified'] = item['modified'].isoformat()
        
        with open(dashboard_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Dashboard saved to: {dashboard_file}")
        
        # Also save a "latest" version
        latest_file = self.stats_dir / 'latest_dashboard.json'
        with open(latest_file, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def _format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive project dashboard')
    parser.add_argument('path', nargs='?', default='.', help='Directory to analyze')
    
    args = parser.parse_args()
    
    dashboard = ProjectDashboard(args.path)
    dashboard.generate_dashboard()

if __name__ == "__main__":
    main()
