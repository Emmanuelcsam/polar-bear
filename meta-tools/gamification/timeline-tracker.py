#!/usr/bin/env python3
"""
Project Timeline Tracker - Visualize project growth over time
Shows file creation/modification patterns and project evolution
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import calendar

from utils.interactive_config import get_interactive_project_config
from utils.config_loader import ConfigLoader

class TimelineTracker:
    def __init__(self, path=".", config=None):
        self.root = Path(path).resolve()
        self.config = config or ConfigLoader()
        self.timeline = defaultdict(lambda: {'created': 0, 'modified': 0, 'size': 0})
        self.file_events = []
        self.daily_activity = defaultdict(int)
        self.author_stats = defaultdict(lambda: {'commits': 0, 'files': set()})
        
    def analyze_timeline(self):
        """Analyze project timeline"""
        print(f"📅 Analyzing timeline for: {self.root}")
        print("-" * 50)
        
        total_files = 0
        oldest_date = datetime.now()
        newest_date = datetime(1970, 1, 1)
        
        # Scan all files
        for root, dirs, files in os.walk(self.root):
            # Skip version control and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                path = Path(root) / file
                try:
                    stat = path.stat()
                    total_files += 1
                    
                    # Creation time (use mtime if ctime not available)
                    created = datetime.fromtimestamp(stat.st_ctime)
                    modified = datetime.fromtimestamp(stat.st_mtime)
                    
                    # Update oldest/newest
                    oldest_date = min(oldest_date, created)
                    newest_date = max(newest_date, modified)
                    
                    # Monthly timeline
                    created_key = created.strftime('%Y-%m')
                    modified_key = modified.strftime('%Y-%m')
                    
                    self.timeline[created_key]['created'] += 1
                    self.timeline[modified_key]['modified'] += 1
                    self.timeline[modified_key]['size'] += stat.st_size
                    
                    # Daily activity (day of week)
                    self.daily_activity[created.strftime('%A')] += 1
                    
                    # Store file events
                    self.file_events.append({
                        'path': str(path.relative_to(self.root)),
                        'created': created,
                        'modified': modified,
                        'size': stat.st_size,
                        'ext': path.suffix.lower()
                    })
                    
                except Exception as e:
                    pass
        
        print(f"✓ Analyzed {total_files} files")
        print(f"📅 Project span: {oldest_date.strftime('%Y-%m-%d')} to {newest_date.strftime('%Y-%m-%d')}")
        print(f"⏱️  That's {(newest_date - oldest_date).days} days of development!")
        
        # Check for git history
        self._analyze_git_history()
        
    def display_timeline(self):
        """Display timeline visualization"""
        print("\n" + "="*80)
        print("📊 PROJECT TIMELINE")
        print("="*80)
        
        # Monthly activity chart
        print("\n📈 Monthly Activity (Created/Modified):")
        
        # Get sorted months
        months = sorted(self.timeline.keys())
        if months:
            # Show last 12 months
            for month in months[-12:]:
                data = self.timeline[month]
                created_bar = '█' * min(50, data['created'] // 2)
                modified_bar = '░' * min(50, data['modified'] // 2)
                
                print(f"{month}: {created_bar}{modified_bar} ({data['created']}/{data['modified']})")
        
        # Most active periods
        print("\n🔥 Most Active Months:")
        active_months = sorted(self.timeline.items(), 
                             key=lambda x: x[1]['created'] + x[1]['modified'], 
                             reverse=True)[:5]
        
        for month, data in active_months:
            total_activity = data['created'] + data['modified']
            print(f"  {month}: {total_activity} file operations ({self._format_size(data['size'])} modified)")
        
        # Day of week analysis
        print("\n📅 Most Productive Days:")
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days_order:
            if day in self.daily_activity:
                bar = '█' * min(30, self.daily_activity[day] // 5)
                print(f"  {day:9}: {bar} ({self.daily_activity[day]} files)")
        
        # Recent activity
        print("\n🕐 Recent Activity (Last 30 days):")
        recent_date = datetime.now() - timedelta(days=30)
        recent_files = [f for f in self.file_events if f['modified'] > recent_date]
        
        if recent_files:
            recent_by_ext = Counter(f['ext'] for f in recent_files)
            print(f"  Files modified: {len(recent_files)}")
            print("  By type:", ', '.join(f"{ext or 'no-ext'}:{count}" for ext, count in recent_by_ext.most_common(5)))
        else:
            print("  No recent activity")
        
        # Growth visualization
        self._show_growth_chart()
        
        # Save timeline data
        self._save_timeline()
        
    def _show_growth_chart(self):
        """Show project growth over time"""
        print("\n📈 Project Growth:")
        
        # Calculate cumulative growth
        months = sorted(self.timeline.keys())
        if not months:
            return
            
        cumulative_files = 0
        growth_data = []
        
        for month in months:
            cumulative_files += self.timeline[month]['created']
            growth_data.append((month, cumulative_files))
        
        # Show last 12 months
        for month, total in growth_data[-12:]:
            bar_length = min(50, total // 10)
            bar = '█' * bar_length
            print(f"  {month}: {bar} {total} files")
        
        # Growth rate
        if len(growth_data) >= 2:
            recent_growth = growth_data[-1][1] - growth_data[-6][1] if len(growth_data) >= 6 else growth_data[-1][1]
            months_period = min(6, len(growth_data))
            avg_growth = recent_growth / months_period
            print(f"\n📊 Average growth: {avg_growth:.1f} files/month (last {months_period} months)")
    
    def _analyze_git_history(self):
        """Analyze git history if available"""
        git_dir = self.root / '.git'
        if not git_dir.exists():
            return
        
        try:
            # Try to get git log
            import subprocess
            result = subprocess.run(
                ['git', 'log', '--pretty=format:%H|%an|%ae|%at|%s', '--name-only'],
                cwd=self.root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("\n🔍 Found Git history!")
                # Parse git log (basic parsing)
                lines = result.stdout.split('\n')
                commit_count = len([l for l in lines if '|' in l])
                print(f"   Total commits: {commit_count}")
        except:
            pass
    
    def _save_timeline(self):
        """Save timeline data"""
        stats_dir = self.config.get_stats_directory(self.root) if self.config else Path('.project-stats')
        stats_dir.mkdir(exist_ok=True)
        
        timeline_file = stats_dir / f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'path': str(self.root),
            'timeline': dict(self.timeline),
            'daily_activity': dict(self.daily_activity),
            'total_files': len(self.file_events),
            'date_range': {
                'oldest': min(f['created'] for f in self.file_events).isoformat() if self.file_events else None,
                'newest': max(f['modified'] for f in self.file_events).isoformat() if self.file_events else None
            }
        }
        
        with open(timeline_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Timeline data saved to: {timeline_file}")
        
        # Compare with previous snapshots
        self._compare_growth()
    
    def _compare_growth(self):
        """Compare with previous snapshots"""
        stats_dir = self.config.get_stats_directory(self.root) if self.config else Path('.project-stats')
        if not stats_dir.exists():
            return
        
        snapshots = list(stats_dir.glob('snapshot_*.json'))
        if len(snapshots) < 2:
            return
        
        # Get oldest and newest snapshot
        snapshots.sort()
        
        try:
            with open(snapshots[0], 'r') as f:
                old_data = json.load(f)
            
            with open(snapshots[-1], 'r') as f:
                new_data = json.load(f)
            
            old_files = old_data.get('stats', {}).get('files', 0)
            new_files = new_data.get('stats', {}).get('files', 0)
            
            if old_files > 0:
                growth = ((new_files - old_files) / old_files) * 100
                print(f"\n📈 Growth since first snapshot: {growth:.1f}% ({old_files} → {new_files} files)")
        except:
            pass
    
    def _format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

def main():
    # Get project configuration interactively
    project_path, config = get_interactive_project_config("Timeline Tracker")
    
    if not project_path:
        return
    
    tracker = TimelineTracker(project_path, config)
    tracker.analyze_timeline()
    tracker.display_timeline()

if __name__ == "__main__":
    main()
