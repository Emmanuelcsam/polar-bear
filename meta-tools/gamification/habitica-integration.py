#!/usr/bin/env python3
"""
Habitica Project Gamification Integration
Rewards good development practices and punishes bad habits
Integrates with Project Tracker Suite to gamify your productivity
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import time
import sys
from utils.config_loader import ConfigLoader
from utils.interactive_config import get_interactive_project_config

class HabiticaProjectGamification:
    def __init__(self, user_id=None, api_key=None, config=None):
        """Initialize Habitica integration"""
        self.base_url = "https://habitica.com/api/v3"
        self.config = config or ConfigLoader()
        self.stats_dir = None  # Will be set based on project path
        
        # Load credentials
        self.user_id = user_id or os.environ.get('HABITICA_USER_ID')
        self.api_key = api_key or os.environ.get('HABITICA_API_KEY')
        
        if not self.user_id or not self.api_key:
            self._setup_credentials()
        
        self.headers = {
            'x-api-user': self.user_id,
            'x-api-key': self.api_key,
            'x-client': '3a326108-1895-4c23-874e-37668c75f2ad-ProjectTracker',
            'Content-Type': 'application/json'
        }
        
        # Productivity tracking
        self.productivity_score = 0
        self.productivity_multiplier = 1.0
        self.streak_days = 0
        self.achievements = []
        
        # Task IDs for project habits
        self.task_ids = {
            'daily_commit': None,
            'code_quality': None,
            'documentation': None,
            'cleanup': None,
            'productivity': None,
            'bad_habits': None
        }
        
        # Scoring configuration
        self.scoring = {
            'daily_commit': {'up': 10, 'down': -5},
            'new_files': {'per_file': 2, 'max': 20},
            'documentation': {'readme_update': 15, 'new_docs': 10},
            'tests_added': {'per_test': 5, 'max': 25},
            'duplicates_removed': {'per_mb': 10},
            'health_improvement': {'per_point': 2},
            'complexity_reduction': {'per_point': 3},
            'security_fix': {'per_issue': 20},
            'large_file_added': {'per_file': -10},
            'health_decline': {'per_point': -3},
            'no_activity': {'days': -15},
            'poor_naming': {'per_file': -2}
        }
        
    def _setup_credentials(self):
        """Setup Habitica credentials"""
        print("🎮 HABITICA INTEGRATION SETUP")
        print("="*50)
        print("\nTo find your credentials:")
        print("1. Log in to Habitica")
        print("2. Go to Settings > API")
        print("3. Copy your User ID and API Token")
        print("\n" + "-"*50)
        
        self.user_id = input("Enter your Habitica User ID: ").strip()
        self.api_key = input("Enter your Habitica API Token: ").strip()
        
        # Save to environment file
        save = input("\nSave credentials to .env file? (y/n): ").lower()
        if save == 'y':
            env_file = Path('.env')
            with open(env_file, 'a') as f:
                f.write(f"\n# Habitica Integration\n")
                f.write(f"HABITICA_USER_ID={self.user_id}\n")
                f.write(f"HABITICA_API_KEY={self.api_key}\n")
            print("✅ Credentials saved to .env")
    
    def _get_or_create_tag(self, tag_name):
        """Get existing tag ID or create a new tag"""
        try:
            # Get all tags
            tags = self._api_request('GET', '/tags')
            
            # Check if tag already exists
            for tag in tags:
                if tag.get('name') == tag_name:
                    return tag['id']
            
            # Create new tag if it doesn't exist
            print(f"Creating tag: {tag_name}")
            new_tag = self._api_request('POST', '/tags', {'name': tag_name})
            if new_tag:
                return new_tag['id']
            
        except Exception as e:
            print(f"⚠️  Could not create tag: {e}")
            return None
    
    def setup_project_habits(self):
        """Create or find project-related habits in Habitica"""
        print("\n🎯 Setting up project habits...")
        
        # First, check if we need to create a tag for project-tracker
        tag_id = self._get_or_create_tag('project-tracker')
        
        habits = [
            {
                'text': '💻 Project Development',
                'type': 'habit',
                'up': True,
                'down': True,
                'notes': 'Track your daily project development activities',
                'tags': [tag_id] if tag_id else [],
                'attribute': 'int',
                'alias': 'project-development'
            },
            {
                'text': '📝 Code Documentation',
                'type': 'habit',
                'up': True,
                'down': False,
                'notes': 'Write documentation, comments, and READMEs',
                'tags': [tag_id] if tag_id else [],
                'attribute': 'per',
                'alias': 'code-documentation'
            },
            {
                'text': '🧹 Code Cleanup',
                'type': 'habit',
                'up': True,
                'down': False,
                'notes': 'Remove duplicates, refactor, improve code quality',
                'tags': [tag_id] if tag_id else [],
                'attribute': 'con',
                'alias': 'code-cleanup'
            },
            {
                'text': '🚀 Productivity Streak',
                'type': 'habit',
                'up': True,
                'down': True,
                'notes': 'Maintain consistent development productivity',
                'tags': [tag_id] if tag_id else [],
                'attribute': 'str',
                'alias': 'productivity-streak'
            },
            {
                'text': '⚠️ Bad Development Habits',
                'type': 'habit',
                'up': False,
                'down': True,
                'notes': 'Poor naming, no tests, security issues, etc.',
                'tags': [tag_id] if tag_id else [],
                'attribute': 'str',
                'alias': 'bad-dev-habits'
            }
        ]
        
        # Create a daily for consistent development
        daily = {
            'text': '📅 Daily Project Work',
            'type': 'daily',
            'notes': 'Work on your project every day to maintain streak',
            'tags': [tag_id] if tag_id else [],
            'attribute': 'str',
            'alias': 'daily-project-work'
        }
        
        # Check existing tasks
        existing_tasks = self._api_request('GET', '/tasks/user')
        if not existing_tasks:
            existing_tasks = []
        existing_aliases = {task.get('alias', ''): task['id'] for task in existing_tasks if task.get('alias')}
        
        # Create habits if they don't exist
        for habit in habits:
            habit_alias = habit['alias']
            if habit_alias not in existing_aliases:
                print(f"Creating habit: {habit['text']}")
                response = self._api_request('POST', '/tasks/user', habit)
                if response:
                    self.task_ids[habit_alias.replace('-', '_')] = response['id']
            else:
                self.task_ids[habit_alias.replace('-', '_')] = existing_aliases[habit_alias]
                print(f"Found existing habit: {habit['text']}")
        
        # Create daily
        daily_alias = daily['alias']
        if daily_alias not in existing_aliases:
            print(f"Creating daily: {daily['text']}")
            response = self._api_request('POST', '/tasks/user', daily)
            if response:
                self.task_ids['daily_commit'] = response['id']
        else:
            self.task_ids['daily_commit'] = existing_aliases[daily_alias]
            print(f"Found existing daily: {daily['text']}")
        
        print("✅ Project habits setup complete!")
    
    def analyze_project_changes(self, project_path="."):
        """Analyze project changes and calculate rewards/punishments"""
        print("\n🔍 Analyzing project changes...")
        
        project_path = Path(project_path).expanduser().resolve()
        # Set stats directory based on project path
        self.stats_dir = self.config.get_stats_directory(project_path)
        stats_dir = self.stats_dir
        
        if not stats_dir.exists():
            print("❌ No project stats found. Run project tracker tools first!")
            return None
        
        # Load latest data
        latest_dashboard = stats_dir / 'latest_dashboard.json'
        if not latest_dashboard.exists():
            print("❌ No dashboard data found. Run project-dashboard.py first!")
            return None
        
        with open(latest_dashboard, 'r') as f:
            current_data = json.load(f)
        
        # Load previous snapshot for comparison
        snapshots = sorted(stats_dir.glob('snapshot_*.json'))
        if len(snapshots) < 2:
            print("⚠️  Need at least 2 snapshots for comparison")
            return None
        
        with open(snapshots[-2], 'r') as f:
            previous_data = json.load(f)
        
        # Calculate changes
        changes = {
            'files_added': current_data['overview']['total_files'] - previous_data['stats']['files'],
            'size_change': current_data['overview']['total_size'] - previous_data['stats']['size'],
            'health_change': current_data.get('health_score', 0) - self._get_previous_health(stats_dir),
            'recent_activity': len(current_data.get('recent_activity', [])),
            'duplicates': current_data.get('duplicates', {}).get('count', 0),
            'issues': current_data.get('issues', [])
        }
        
        # Check for specific improvements
        changes['documentation_updated'] = self._check_documentation_updates(project_path)
        changes['tests_added'] = self._check_tests_added(project_path, current_data)
        changes['security_issues'] = self._count_security_issues(current_data.get('issues', []))
        
        return changes, current_data
    
    def calculate_rewards_punishments(self, changes, current_data):
        """Calculate rewards and punishments based on changes"""
        rewards = []
        punishments = []
        
        # File additions (reward)
        if changes['files_added'] > 0:
            points = min(changes['files_added'] * self.scoring['new_files']['per_file'], 
                        self.scoring['new_files']['max'])
            rewards.append({
                'reason': f"Added {changes['files_added']} new files",
                'points': points,
                'task': 'project_development'
            })
        
        # Documentation updates (reward)
        if changes['documentation_updated']:
            points = self.scoring['documentation']['readme_update']
            rewards.append({
                'reason': "Updated documentation",
                'points': points,
                'task': 'code_documentation'
            })
        
        # Health improvements (reward)
        if changes['health_change'] > 0:
            points = changes['health_change'] * self.scoring['health_improvement']['per_point']
            rewards.append({
                'reason': f"Health score improved by {changes['health_change']} points",
                'points': points,
                'task': 'code_cleanup'
            })
        
        # Duplicate removal (reward)
        if changes.get('duplicates_removed', 0) > 0:
            mb_saved = changes['duplicates_removed'] / (1024 * 1024)
            points = int(mb_saved * self.scoring['duplicates_removed']['per_mb'])
            rewards.append({
                'reason': f"Removed duplicates, saved {mb_saved:.1f} MB",
                'points': points,
                'task': 'code_cleanup'
            })
        
        # Recent activity (reward)
        if changes['recent_activity'] > 5:
            rewards.append({
                'reason': f"High activity: {changes['recent_activity']} files modified",
                'points': 10,
                'task': 'productivity_streak'
            })
        
        # --- PUNISHMENTS ---
        
        # Health decline (punishment)
        if changes['health_change'] < 0:
            points = abs(changes['health_change']) * self.scoring['health_decline']['per_point']
            punishments.append({
                'reason': f"Health score declined by {abs(changes['health_change'])} points",
                'points': points,
                'task': 'bad_dev_habits'
            })
        
        # Many duplicates (punishment)
        if changes['duplicates'] > 10:
            punishments.append({
                'reason': f"Too many duplicate files ({changes['duplicates']})",
                'points': 10,
                'task': 'bad_dev_habits'
            })
        
        # Security issues (punishment)
        if changes['security_issues'] > 0:
            points = changes['security_issues'] * self.scoring['security_fix']['per_issue']
            punishments.append({
                'reason': f"Security issues found ({changes['security_issues']})",
                'points': points,
                'task': 'bad_dev_habits'
            })
        
        # No recent activity (punishment)
        if changes['recent_activity'] == 0:
            punishments.append({
                'reason': "No recent activity in the last 7 days",
                'points': self.scoring['no_activity']['days'],
                'task': 'productivity_streak'
            })
        
        return rewards, punishments
    
    def apply_habitica_changes(self, rewards, punishments):
        """Apply rewards and punishments to Habitica"""
        print("\n🎮 Applying Habitica changes...")
        
        total_exp = 0
        total_gold = 0
        
        # Apply rewards
        for reward in rewards:
            task_id = self.task_ids.get(reward['task'])
            if task_id:
                print(f"✅ {reward['reason']} (+{reward['points']} points)")
                response = self._api_request('POST', f"/tasks/{task_id}/score/up")
                if response:
                    total_exp += response.get('exp', 0)
                    total_gold += response.get('gp', 0)
                time.sleep(0.5)  # Rate limiting
        
        # Apply punishments
        for punishment in punishments:
            task_id = self.task_ids.get(punishment['task'])
            if task_id:
                print(f"❌ {punishment['reason']} (-{punishment['points']} damage)")
                response = self._api_request('POST', f"/tasks/{task_id}/score/down")
                time.sleep(0.5)  # Rate limiting
        
        # Update daily if worked today
        if rewards:
            daily_id = self.task_ids.get('daily_commit')
            if daily_id:
                print("📅 Marking daily project work as complete")
                self._api_request('POST', f"/tasks/{daily_id}/score/up")
        
        return total_exp, total_gold
    
    def calculate_productivity_score(self, changes, current_data):
        """Calculate productivity score and multiplier"""
        base_score = 0
        
        # Positive factors
        base_score += min(changes['files_added'] * 5, 50)
        base_score += changes['recent_activity'] * 2
        base_score += max(0, changes['health_change'] * 3)
        
        # Negative factors
        base_score -= changes['duplicates']
        base_score -= changes['security_issues'] * 10
        
        # Time-based multiplier
        hour = datetime.now().hour
        if 6 <= hour <= 9:  # Early morning
            time_multiplier = 1.5
        elif 9 <= hour <= 17:  # Work hours
            time_multiplier = 1.2
        elif 22 <= hour or hour <= 5:  # Late night
            time_multiplier = 0.8
        else:
            time_multiplier = 1.0
        
        # Streak multiplier
        streak_multiplier = 1 + (self.streak_days * 0.1)
        
        # Calculate final score
        self.productivity_score = int(base_score * time_multiplier * streak_multiplier)
        self.productivity_multiplier = time_multiplier * streak_multiplier
        
        return self.productivity_score
    
    def display_productivity_dashboard(self, changes, rewards, punishments, exp_gained, gold_gained):
        """Display productivity dashboard"""
        print("\n" + "="*60)
        print("🎮 PRODUCTIVITY GAMING DASHBOARD")
        print("="*60)
        
        # Productivity gauge
        gauge_width = 40
        filled = int((self.productivity_score / 100) * gauge_width)
        gauge = '█' * filled + '░' * (gauge_width - filled)
        
        print(f"\n📊 Productivity Score: [{gauge}] {self.productivity_score}/100")
        print(f"🔥 Multiplier: {self.productivity_multiplier:.1f}x")
        print(f"⚡ Streak: {self.streak_days} days")
        
        # Session summary
        print("\n📈 Session Summary:")
        print(f"  Experience Gained: +{exp_gained:.1f} XP")
        print(f"  Gold Earned: +{gold_gained:.1f} GP")
        print(f"  Rewards Applied: {len(rewards)}")
        print(f"  Punishments: {len(punishments)}")
        
        # Achievements unlocked
        if self.achievements:
            print("\n🏆 Achievements Unlocked:")
            for achievement in self.achievements[-3:]:
                print(f"  ⭐ {achievement}")
        
        # Productivity tips
        print("\n💡 Productivity Tips:")
        tips = self._get_productivity_tips(changes)
        for tip in tips[:3]:
            print(f"  • {tip}")
        
        # Save session data
        self._save_session_data(changes, rewards, punishments, exp_gained, gold_gained)
    
    def _get_productivity_tips(self, changes):
        """Generate productivity tips based on analysis"""
        tips = []
        
        if changes['duplicates'] > 5:
            tips.append("Run duplicate-finder.py to clean up and earn cleanup points!")
        
        if changes['health_change'] < 0:
            tips.append("Run health-checker.py to identify and fix project issues")
        
        if changes['recent_activity'] < 3:
            tips.append("Try to work on your project daily to maintain streak bonus")
        
        if not changes['documentation_updated']:
            tips.append("Update your README for documentation points")
        
        if changes['files_added'] == 0:
            tips.append("Create new features or modules for development points")
        
        tips.append("Work in the morning for 1.5x productivity multiplier!")
        tips.append("Maintain a daily streak for increasing bonuses")
        tips.append("Fix security issues for major point rewards")
        
        return tips
    
    def _save_session_data(self, changes, rewards, punishments, exp_gained, gold_gained):
        """Save gaming session data"""
        session_dir = Path('.project-stats') / 'habitica_sessions'
        session_dir.mkdir(exist_ok=True)
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'productivity_score': self.productivity_score,
            'multiplier': self.productivity_multiplier,
            'streak_days': self.streak_days,
            'changes': changes,
            'rewards': rewards,
            'punishments': punishments,
            'exp_gained': exp_gained,
            'gold_gained': gold_gained,
            'achievements': self.achievements
        }
        
        session_file = session_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def check_achievements(self, changes, current_data):
        """Check for achievement unlocks"""
        new_achievements = []
        
        # File count achievements
        total_files = current_data['overview']['total_files']
        if total_files >= 100 and "Centurion" not in self.achievements:
            new_achievements.append("Centurion - 100 files reached!")
        if total_files >= 500 and "Half-K Hero" not in self.achievements:
            new_achievements.append("Half-K Hero - 500 files reached!")
        
        # Health achievements
        health = current_data.get('health_score', 0)
        if health >= 95 and "Health Master" not in self.achievements:
            new_achievements.append("Health Master - 95+ health score!")
        
        # Streak achievements
        if self.streak_days >= 7 and "Week Warrior" not in self.achievements:
            new_achievements.append("Week Warrior - 7 day streak!")
        if self.streak_days >= 30 and "Monthly Master" not in self.achievements:
            new_achievements.append("Monthly Master - 30 day streak!")
        
        # Productivity achievements
        if self.productivity_score >= 80:
            new_achievements.append("Productivity Pro - 80+ productivity score!")
        
        self.achievements.extend(new_achievements)
        
        # Award bonus points for achievements
        for achievement in new_achievements:
            print(f"🏆 Achievement Unlocked: {achievement}")
            # Bonus reward
            if self.task_ids.get('productivity_streak'):
                self._api_request('POST', f"/tasks/{self.task_ids['productivity_streak']}/score/up")
    
    def _api_request(self, method, endpoint, data=None):
        """Make API request to Habitica"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=self.headers)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data)
            elif method == 'PUT':
                response = requests.put(url, headers=self.headers, json=data)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers)
            
            response.raise_for_status()
            return response.json().get('data', response.json())
        
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return None
    
    def _check_documentation_updates(self, project_path):
        """Check if documentation was updated"""
        readme_files = ['README.md', 'README.txt', 'README.rst', 'README']
        
        for readme in readme_files:
            readme_path = project_path / readme
            if readme_path.exists():
                # Check if modified recently
                mtime = datetime.fromtimestamp(readme_path.stat().st_mtime)
                if datetime.now() - mtime < timedelta(days=1):
                    return True
        
        return False
    
    def _check_tests_added(self, project_path, current_data):
        """Estimate if tests were added"""
        # Simple heuristic: look for test files
        test_patterns = ['test_*.py', '*_test.py', 'test*.js', '*.test.js']
        test_count = 0
        
        for pattern in test_patterns:
            test_count += len(list(project_path.rglob(pattern)))
        
        return test_count
    
    def _count_security_issues(self, issues):
        """Count security-related issues"""
        security_keywords = ['security', 'password', 'secret', 'api', 'key', 'token', 'credential']
        count = 0
        
        for issue in issues:
            if any(keyword in issue.lower() for keyword in security_keywords):
                count += 1
        
        return count
    
    def _get_previous_health(self, stats_dir):
        """Get previous health score"""
        health_reports = sorted(stats_dir.glob('health_report_*.json'))
        
        if len(health_reports) >= 2:
            with open(health_reports[-2], 'r') as f:
                data = json.load(f)
                return data.get('score', 0)
        
        return 0
    
    def run_gamification(self, project_path="."):
        """Main gamification process"""
        print("\n🎮 HABITICA PROJECT GAMIFICATION")
        print("="*60)
        
        # Analyze changes
        result = self.analyze_project_changes(project_path)
        if not result:
            return
        
        changes, current_data = result
        
        # Calculate rewards and punishments
        rewards, punishments = self.calculate_rewards_punishments(changes, current_data)
        
        # Calculate productivity score
        self.calculate_productivity_score(changes, current_data)
        
        # Check achievements
        self.check_achievements(changes, current_data)
        
        # Apply to Habitica
        exp_gained, gold_gained = self.apply_habitica_changes(rewards, punishments)
        
        # Display dashboard
        self.display_productivity_dashboard(changes, rewards, punishments, exp_gained, gold_gained)
        
        print("\n✨ Gamification complete! Keep coding to level up!")


def main():
    """Main entry point"""
    # Use interactive configuration
    project_path, config = get_interactive_project_config("Habitica Project Gamification")
    
    if project_path is None:
        return
    
    # Initialize with config
    gamification = HabiticaProjectGamification(config=config)
    
    # Ask for setup mode
    setup_mode = input("\nSetup Habitica habits? (y/N): ").strip().lower() == 'y'
    
    # Setup habits if requested
    if setup_mode:
        gamification.setup_project_habits()
        print("\n✅ Setup complete! Run again without setup to start gamification.")
        return
    
    # Check if habits are setup
    print("🎮 Connecting to Habitica...")
    user_data = gamification._api_request('GET', '/user')
    if user_data:
        print(f"✅ Connected as: {user_data.get('profile', {}).get('name', 'Unknown')}")
        print(f"   Level: {user_data.get('stats', {}).get('lvl', 0)}")
        print(f"   HP: {user_data.get('stats', {}).get('hp', 0):.0f}/{user_data.get('stats', {}).get('maxHealth', 0)}")
    else:
        print("❌ Failed to connect to Habitica. Check your credentials.")
        return
    
    # Setup habits if first run
    if not any(gamification.task_ids.values()):
        print("\n🔧 First run detected. Setting up habits...")
        gamification.setup_project_habits()
    
    # Run gamification
    gamification.run_gamification(str(project_path))
    
    # Ask if user wants to analyze another project
    print("\n" + "-"*50)
    another = input("Gamify another project? (y/n): ").strip().lower()
    if another == 'y':
        main()


if __name__ == "__main__":
    main()
