#!/usr/bin/env python3
"""
Project Growth Monitor - Track project evolution over time
Compares snapshots to show growth trends and progress
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import subprocess
from utils.interactive_config import get_interactive_project_config
from utils.config_loader import ConfigLoader


class GrowthMonitor:
    def __init__(self, path=".", config=None):
        self.root = Path(path).resolve()
        self.config = config or ConfigLoader()
        self.stats_dir = self.config.get_stats_directory(self.root)
        self.current_stats = {}
        self.snapshots = []
        self.growth_data = {}
        
    def collect_current_stats(self):
        """Collect current project statistics"""
        print(f"📊 Collecting current stats for: {self.root}")
        print("-" * 50)
        
        self.current_stats = {
            'timestamp': datetime.now().isoformat(),
            'files': 0,
            'dirs': 0,
            'size': 0,
            'languages': defaultdict(int),
            'commits': self._get_git_stats(),
            'recent_activity': []
        }
        
        # Scan directory
        for root, dirs, files in os.walk(self.root):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
            
            self.current_stats['dirs'] += len(dirs)
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                path = Path(root) / file
                try:
                    stat = path.stat()
                    self.current_stats['files'] += 1
                    self.current_stats['size'] += stat.st_size
                    
                    # Track language
                    ext = path.suffix.lower()
                    if ext:
                        self.current_stats['languages'][ext] += 1
                    
                    # Track recent files
                    if datetime.fromtimestamp(stat.st_mtime) > datetime.now() - timedelta(days=7):
                        self.current_stats['recent_activity'].append({
                            'file': str(path.relative_to(self.root)),
                            'modified': stat.st_mtime
                        })
                except:
                    pass
        
        print(f"✓ Current stats collected: {self.current_stats['files']} files, {self._format_size(self.current_stats['size'])}")
        
        # Save current snapshot
        self._save_snapshot()
    
    def load_snapshots(self):
        """Load all previous snapshots"""
        if not self.stats_dir.exists():
            print("No previous snapshots found. This is your first scan!")
            return
        
        snapshot_files = list(self.stats_dir.glob('snapshot_*.json'))
        
        for snapshot_file in sorted(snapshot_files):
            try:
                with open(snapshot_file, 'r') as f:
                    snapshot = json.load(f)
                    self.snapshots.append(snapshot)
            except:
                pass
        
        print(f"📂 Loaded {len(self.snapshots)} previous snapshots")
    
    def analyze_growth(self):
        """Analyze growth patterns"""
        if not self.snapshots:
            print("\n📈 No historical data to compare. Run this tool regularly to track growth!")
            return
        
        print("\n" + "="*80)
        print("📈 GROWTH ANALYSIS")
        print("="*80)
        
        # Get first and last snapshot for overall growth
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        
        # Overall growth
        self._show_overall_growth(first_snapshot, self.current_stats)
        
        # Recent growth (last snapshot to current)
        if len(self.snapshots) > 0:
            self._show_recent_growth(last_snapshot, self.current_stats)
        
        # Growth trend
        self._show_growth_trend()
        
        # Language evolution
        self._show_language_evolution()
        
        # Activity patterns
        self._show_activity_patterns()
        
        # Milestones
        self._check_milestones()
    
    def _show_overall_growth(self, first, current):
        """Show overall project growth"""
        print("\n🌱 OVERALL GROWTH")
        print("-" * 40)
        
        # Time span
        first_date = datetime.fromisoformat(first['timestamp'])
        current_date = datetime.fromisoformat(current['timestamp'])
        days = (current_date - first_date).days
        
        print(f"📅 Project age: {days} days ({days//30} months)")
        
        # File growth
        file_growth = current['files'] - first['stats']['files']
        file_percent = (file_growth / first['stats']['files'] * 100) if first['stats']['files'] > 0 else 0
        
        print(f"📄 Files: {first['stats']['files']} → {current['files']} (+{file_growth}, {file_percent:.1f}% growth)")
        
        # Size growth
        size_growth = current['size'] - first['stats']['size']
        size_percent = (size_growth / first['stats']['size'] * 100) if first['stats']['size'] > 0 else 0
        
        print(f"💾 Size: {self._format_size(first['stats']['size'])} → {self._format_size(current['size'])} "
              f"(+{self._format_size(size_growth)}, {size_percent:.1f}% growth)")
        
        # Average growth rate
        if days > 0:
            avg_files_per_day = file_growth / days
            avg_size_per_day = size_growth / days
            print(f"📊 Average growth: {avg_files_per_day:.1f} files/day, {self._format_size(avg_size_per_day)}/day")
    
    def _show_recent_growth(self, last, current):
        """Show recent growth since last snapshot"""
        print("\n🚀 RECENT CHANGES")
        print("-" * 40)
        
        last_date = datetime.fromisoformat(last['timestamp'])
        current_date = datetime.fromisoformat(current['timestamp'])
        days = (current_date - last_date).days
        
        print(f"Since last snapshot ({days} days ago):")
        
        # Files
        file_change = current['files'] - last['stats']['files']
        if file_change > 0:
            print(f"  ✅ Added {file_change} files")
        elif file_change < 0:
            print(f"  ❌ Removed {abs(file_change)} files")
        else:
            print(f"  ➖ No change in file count")
        
        # Size
        size_change = current['size'] - last['stats']['size']
        if size_change > 0:
            print(f"  ✅ Size increased by {self._format_size(size_change)}")
        elif size_change < 0:
            print(f"  ❌ Size decreased by {self._format_size(abs(size_change))}")
        
        # Recent activity
        if current['recent_activity']:
            print(f"\n📝 Recent activity ({len(current['recent_activity'])} files modified in last 7 days):")
            for item in sorted(current['recent_activity'], key=lambda x: x['modified'], reverse=True)[:5]:
                mod_date = datetime.fromtimestamp(item['modified']).strftime('%Y-%m-%d')
                print(f"  • {item['file']} ({mod_date})")
    
    def _show_growth_trend(self):
        """Show growth trend over time"""
        print("\n📊 GROWTH TREND")
        print("-" * 40)
        
        if len(self.snapshots) < 3:
            print("Need more snapshots to show trend")
            return
        
        # Calculate monthly growth rates
        monthly_growth = []
        
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            
            prev_date = datetime.fromisoformat(prev['timestamp'])
            curr_date = datetime.fromisoformat(curr['timestamp'])
            days = (curr_date - prev_date).days
            
            if days > 0:
                file_growth_rate = (curr['stats']['files'] - prev['stats']['files']) / days * 30
                monthly_growth.append({
                    'date': curr_date.strftime('%Y-%m'),
                    'rate': file_growth_rate,
                    'files': curr['stats']['files']
                })
        
        # Show trend
        print("Monthly file growth rate:")
        for item in monthly_growth[-6:]:  # Last 6 months
            bar = '█' * int(abs(item['rate']) / 2)
            if item['rate'] >= 0:
                print(f"  {item['date']}: +{bar} ({item['rate']:.1f} files/month)")
            else:
                print(f"  {item['date']}: -{bar} ({item['rate']:.1f} files/month)")
        
        # Trend analysis
        if len(monthly_growth) >= 3:
            recent_rates = [item['rate'] for item in monthly_growth[-3:]]
            avg_recent = sum(recent_rates) / len(recent_rates)
            
            if avg_recent > 10:
                print("\n🔥 Project is growing rapidly!")
            elif avg_recent > 0:
                print("\n📈 Project is growing steadily")
            elif avg_recent < -5:
                print("\n📉 Project is shrinking")
            else:
                print("\n➡️  Project size is stable")
    
    def _show_language_evolution(self):
        """Show how language usage has evolved"""
        print("\n🔤 LANGUAGE EVOLUTION")
        print("-" * 40)
        
        if not self.snapshots:
            return
        
        # Get language data from first and current
        first_langs = dict(self.snapshots[0]['stats'].get('languages', {}))
        current_langs = dict(self.current_stats['languages'])
        
        # Find all languages
        all_langs = set(first_langs.keys()) | set(current_langs.keys())
        
        # Show changes
        changes = []
        for lang in all_langs:
            first_count = first_langs.get(lang, 0)
            current_count = current_langs.get(lang, 0)
            change = current_count - first_count
            
            if change != 0:
                changes.append((lang, first_count, current_count, change))
        
        # Sort by absolute change
        changes.sort(key=lambda x: abs(x[3]), reverse=True)
        
        for lang, first, current, change in changes[:10]:
            if change > 0:
                print(f"  {lang}: {first} → {current} (+{change} files)")
            else:
                print(f"  {lang}: {first} → {current} ({change} files)")
        
        # Show top languages
        print("\nCurrent top languages:")
        for lang, count in sorted(current_langs.items(), key=lambda x: x[1], reverse=True)[:5]:
            percent = (count / self.current_stats['files'] * 100) if self.current_stats['files'] > 0 else 0
            print(f"  {lang}: {count} files ({percent:.1f}%)")
    
    def _show_activity_patterns(self):
        """Show activity patterns"""
        print("\n⏱️  ACTIVITY PATTERNS")
        print("-" * 40)
        
        # Analyze commit patterns if git is available
        if self.current_stats.get('commits'):
            commits = self.current_stats['commits']
            print(f"Total commits: {commits.get('total', 0)}")
            
            if commits.get('recent_commits'):
                print("\nRecent commit activity:")
                for date, count in commits['recent_commits'][:7]:
                    bar = '█' * min(20, count)
                    print(f"  {date}: {bar} ({count} commits)")
        
        # Show busiest periods
        if len(self.snapshots) >= 5:
            growth_periods = []
            for i in range(1, len(self.snapshots)):
                prev = self.snapshots[i-1]
                curr = self.snapshots[i]
                growth = curr['stats']['files'] - prev['stats']['files']
                growth_periods.append({
                    'date': datetime.fromisoformat(curr['timestamp']).strftime('%Y-%m-%d'),
                    'growth': growth
                })
            
            growth_periods.sort(key=lambda x: x['growth'], reverse=True)
            
            print("\nMost productive periods:")
            for period in growth_periods[:3]:
                if period['growth'] > 0:
                    print(f"  {period['date']}: +{period['growth']} files")
    
    def _check_milestones(self):
        """Check for project milestones"""
        print("\n🏆 MILESTONES")
        print("-" * 40)
        
        milestones = []
        
        # File count milestones
        file_milestones = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        for milestone in file_milestones:
            if self.current_stats['files'] >= milestone:
                # Check if recently achieved
                if self.snapshots:
                    last_files = self.snapshots[-1]['stats']['files']
                    if last_files < milestone <= self.current_stats['files']:
                        milestones.append(f"🎉 Just reached {milestone} files!")
                    elif self.current_stats['files'] >= milestone:
                        milestones.append(f"✓ {milestone}+ files")
        
        # Size milestones
        size_milestones = [
            (1024*1024, "1 MB"),
            (10*1024*1024, "10 MB"),
            (100*1024*1024, "100 MB"),
            (1024*1024*1024, "1 GB")
        ]
        
        for size, label in size_milestones:
            if self.current_stats['size'] >= size:
                if self.snapshots:
                    last_size = self.snapshots[-1]['stats']['size']
                    if last_size < size <= self.current_stats['size']:
                        milestones.append(f"🎉 Project size exceeded {label}!")
                    elif self.current_stats['size'] >= size:
                        milestones.append(f"✓ {label}+ project size")
        
        # Time milestones
        if self.snapshots:
            first_date = datetime.fromisoformat(self.snapshots[0]['timestamp'])
            age_days = (datetime.now() - first_date).days
            
            time_milestones = [
                (7, "1 week"),
                (30, "1 month"),
                (90, "3 months"),
                (180, "6 months"),
                (365, "1 year")
            ]
            
            for days, label in time_milestones:
                if age_days >= days:
                    milestones.append(f"✓ {label} old")
        
        if milestones:
            for milestone in milestones:
                print(f"  {milestone}")
        else:
            print("  Keep going! Milestones will appear as your project grows.")
        
        # Predictions
        self._make_predictions()
    
    def _make_predictions(self):
        """Make growth predictions"""
        if len(self.snapshots) < 3:
            return
        
        print("\n🔮 PREDICTIONS")
        print("-" * 40)
        
        # Calculate average growth rate
        total_days = 0
        total_file_growth = 0
        
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]
            
            prev_date = datetime.fromisoformat(prev['timestamp'])
            curr_date = datetime.fromisoformat(curr['timestamp'])
            
            days = (curr_date - prev_date).days
            file_growth = curr['stats']['files'] - prev['stats']['files']
            
            total_days += days
            total_file_growth += file_growth
        
        if total_days > 0:
            avg_growth_rate = total_file_growth / total_days
            
            # Predictions
            current_files = self.current_stats['files']
            
            print("At current growth rate:")
            
            predictions = [
                (30, "In 1 month"),
                (90, "In 3 months"),
                (180, "In 6 months"),
                (365, "In 1 year")
            ]
            
            for days, label in predictions:
                predicted_files = int(current_files + (avg_growth_rate * days))
                predicted_size = int(self.current_stats['size'] * (predicted_files / current_files))
                
                print(f"  {label}: ~{predicted_files} files, ~{self._format_size(predicted_size)}")
    
    def _get_git_stats(self):
        """Get git statistics if available"""
        git_dir = self.root / '.git'
        if not git_dir.exists():
            return None
        
        try:
            # Get total commits
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=self.root,
                capture_output=True,
                text=True
            )
            total_commits = int(result.stdout.strip()) if result.returncode == 0 else 0
            
            # Get recent commits by date
            result = subprocess.run(
                ['git', 'log', '--format=%cd', '--date=short'],
                cwd=self.root,
                capture_output=True,
                text=True
            )
            
            recent_commits = []
            if result.returncode == 0:
                dates = result.stdout.strip().split('\n')
                date_counts = defaultdict(int)
                for date in dates[:100]:  # Last 100 commits
                    if date:
                        date_counts[date] += 1
                
                recent_commits = sorted(date_counts.items(), reverse=True)[:30]
            
            return {
                'total': total_commits,
                'recent_commits': recent_commits
            }
        except:
            return None
    
    def _save_snapshot(self):
        """Save current snapshot"""
        self.stats_dir.mkdir(exist_ok=True)
        
        snapshot = {
            'timestamp': self.current_stats['timestamp'],
            'path': str(self.root),
            'stats': {
                'files': self.current_stats['files'],
                'dirs': self.current_stats['dirs'],
                'size': self.current_stats['size'],
                'languages': dict(self.current_stats['languages'])
            }
        }
        
        if self.current_stats.get('commits'):
            snapshot['commits'] = self.current_stats['commits']['total']
        
        snapshot_file = self.stats_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print(f"\n💾 Snapshot saved to: {snapshot_file}")
    
    def _format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

def main():
    # Get project configuration interactively
    project_path, config = get_interactive_project_config("Growth Monitor")
    
    if not project_path:
        return
    
    monitor = GrowthMonitor(project_path, config)
    monitor.load_snapshots()
    monitor.collect_current_stats()
    monitor.analyze_growth()

if __name__ == "__main__":
    main()
