# 🎮 Habitica Project Gamification Guide

Transform your coding habits into an RPG adventure! This guide explains how to use the Habitica integration with your Project Tracker Suite.

## 🚀 Overview

The Habitica integration gamifies your development process by:
- **Rewarding** good development practices (commits, documentation, cleanup)
- **Punishing** bad habits (duplicates, security issues, inactivity)
- **Tracking** productivity with scores and multipliers
- **Motivating** through achievements and leaderboards

## 📋 Prerequisites

1. **Habitica Account**: Create one at [habitica.com](https://habitica.com)
2. **API Credentials**: Get from Habitica Settings > API
3. **Project Tracker Suite**: All scripts from the main suite installed
4. **Python Dependencies**: Standard library only (requests for API calls)

## 🔧 Initial Setup

### 1. Install the Scripts

Save these scripts in your project-tracker-tools directory:
```bash
habitica-integration.py     # Main Habitica gamification
productivity-tracker.py     # Advanced productivity tracking
```

### 2. Configure Habitica Credentials

First run will prompt for credentials:
```bash
python habitica-integration.py --setup
```

Or set environment variables:
```bash
export HABITICA_USER_ID="your-user-id"
export HABITICA_API_KEY="your-api-key"
```

Or create a `.env` file:
```env
HABITICA_USER_ID=your-user-id
HABITICA_API_KEY=your-api-key
```

### 3. Create Project Habits

The script automatically creates these habits in Habitica:
- 💻 **Project Development** (INT) - General coding activities
- 📝 **Code Documentation** (PER) - Writing docs and comments
- 🧹 **Code Cleanup** (CON) - Refactoring and optimization
- 🚀 **Productivity Streak** (STR) - Consistent work
- ⚠️ **Bad Development Habits** (STR) - Poor practices to avoid
- 📅 **Daily Project Work** - Daily task for consistency

## 🎯 How It Works

### Scoring System

**Rewards** (Gain XP and Gold):
- New files added: 2 points per file (max 20)
- Documentation updates: 15 points
- Health score improvements: 2 points per point
- Duplicate removal: 10 points per MB saved
- Security fixes: 20 points per issue
- High activity: 10 points for 5+ files modified

**Punishments** (Lose HP):
- Health score decline: -3 damage per point
- Too many duplicates: -10 damage
- Security issues: -20 damage per issue
- No activity for 7 days: -15 damage
- Poor file naming: -2 damage per file

### Productivity Multipliers

Work at optimal times for bonus rewards:
- 🌅 **Early Morning** (6-9 AM): 1.5x multiplier
- ☀️ **Work Hours** (9-5 PM): 1.0x multiplier
- 🌆 **Evening** (5-10 PM): 0.8x multiplier
- 🌙 **Night Owl** (10 PM+): 0.6x multiplier

### Achievements System

Unlock achievements for milestones:
- **Centurion**: Reach 100 files
- **Half-K Hero**: Reach 500 files
- **Health Master**: Achieve 95+ health score
- **Week Warrior**: 7-day development streak
- **Monthly Master**: 30-day streak
- **Productivity Pro**: 80+ productivity score

## 📖 Usage Guide

### Basic Workflow

1. **Run project analysis** first:
   ```bash
   python quick-stats.py
   python health-checker.py
   python project-dashboard.py
   ```

2. **Apply gamification**:
   ```bash
   python habitica-integration.py /path/to/project
   ```

3. **Track productivity sessions**:
   ```bash
   # Start work session
   python productivity-tracker.py --start
   
   # End session (calculates productivity)
   python productivity-tracker.py --end
   ```

### Daily Routine

1. **Morning Setup**:
   ```bash
   # Check daily goals
   python productivity-tracker.py --check-goals
   
   # Start work session
   python productivity-tracker.py --start
   ```

2. **During Development**:
   - Make regular commits
   - Update documentation
   - Run tests
   - Clean up code

3. **End of Day**:
   ```bash
   # End work session
   python productivity-tracker.py --end
   
   # Run gamification
   python habitica-integration.py
   
   # Check leaderboard
   python productivity-tracker.py --leaderboard --username YourName
   ```

### Weekly Review

```bash
# Generate productivity report
python productivity-tracker.py --report --days 7

# Run full analysis suite
python project-tracker.py
# Select option 8 (Run All)

# Apply weekly gamification
python habitica-integration.py
```

## 📊 Productivity Tracking Features

### Work Sessions
- Tracks focus time and break patterns
- Calculates velocity based on changes
- Measures code quality factors
- Applies time-zone multipliers

### Daily Goals
Set and track daily targets:
```bash
python productivity-tracker.py --setup-goals
```

Default goals:
- 3 commits per day
- 4 productive hours
- 5 files modified
- 70 average productivity score

### Productivity Reports
Generates comprehensive reports with:
- Productivity trends over time
- Work pattern heatmaps
- Daily/weekly statistics
- Achievement tracking
- Visual charts and graphs

### Leaderboard System
Compete with yourself or team members:
- Weekly and monthly scores
- Streak tracking
- Rank changes
- Achievement counts

## 🎮 Productivity Dashboard

The gamification dashboard shows:
```
📊 Productivity Score: [████████████████░░░░] 80/100
🔥 Multiplier: 1.5x
⚡ Streak: 7 days

📈 Session Summary:
  Experience Gained: +45.5 XP
  Gold Earned: +23.2 GP
  Rewards Applied: 4
  Punishments: 1

🏆 Achievements Unlocked:
  ⭐ Week Warrior - 7 day streak!
  
💡 Productivity Tips:
  • Run duplicate-finder.py to clean up and earn cleanup points!
  • Work in the morning for 1.5x productivity multiplier!
  • Update your README for documentation points
```

## 🔥 Pro Tips

### Maximize Rewards
1. **Work in the morning** for 1.5x multiplier
2. **Commit frequently** with good messages
3. **Update documentation** regularly
4. **Clean up duplicates** for easy points
5. **Fix security issues** for major rewards

### Maintain Streaks
1. **Set daily reminders** to work on project
2. **Use small tasks** on busy days
3. **Track progress** with daily goals
4. **Review weekly** to stay motivated

### Avoid Punishments
1. **Run health checks** regularly
2. **Don't let duplicates** accumulate
3. **Address security issues** immediately
4. **Work consistently** to avoid inactivity penalties

## 🛠️ Customization

### Modify Scoring Values

Edit the scoring configuration in `habitica-integration.py`:
```python
self.scoring = {
    'daily_commit': {'up': 10, 'down': -5},
    'new_files': {'per_file': 2, 'max': 20},
    # ... modify as needed
}
```

### Add Custom Achievements

Add new achievements in the `check_achievements` method:
```python
if total_files >= 1000:
    new_achievements.append("Thousand File Master - 1000 files!")
```

### Adjust Time Zones

Modify productivity zones for your schedule:
```python
self.productivity_zones = {
    'deep_work': {'start': 6, 'end': 11, 'multiplier': 1.5},
    # ... adjust times and multipliers
}
```

## 📈 Integration with Project Tracker

The Habitica integration uses data from:
- **Quick Stats**: File counts and recent activity
- **Health Checker**: Project health score
- **Duplicate Finder**: Cleanup opportunities
- **Growth Monitor**: Development trends
- **Code Analyzer**: Code quality metrics

Run these tools first for accurate gamification!

## 🐛 Troubleshooting

### Common Issues

1. **API Authentication Failed**
   - Check User ID and API Key
   - Ensure no extra spaces
   - Verify credentials on Habitica website

2. **No Changes Detected**
   - Run project analysis tools first
   - Ensure at least 2 snapshots exist
   - Check `.project-stats/` directory

3. **Tasks Not Created**
   - Check Habitica task limit
   - Verify API permissions
   - Try manual setup with `--setup`

### Debug Mode

Run with verbose output:
```bash
python habitica-integration.py --debug
```

## 🎯 Best Practices

1. **Consistency is Key**
   - Work on project daily
   - Track all sessions
   - Review progress weekly

2. **Quality Over Quantity**
   - Focus on meaningful changes
   - Write good documentation
   - Maintain clean code

3. **Use All Features**
   - Set daily goals
   - Check leaderboard
   - Generate reports
   - Unlock achievements

4. **Stay Motivated**
   - Celebrate achievements
   - Learn from statistics
   - Compete with yourself
   - Share progress

## 📊 Example Workflow

### Monday Morning
```bash
# 1. Check weekend activity
python timeline-tracker.py

# 2. Set weekly goals
python productivity-tracker.py --setup-goals

# 3. Start work session
python productivity-tracker.py --start

# 4. Work on project...

# 5. End session and gamify
python productivity-tracker.py --end
python habitica-integration.py
```

### Friday Review
```bash
# 1. Generate weekly report
python productivity-tracker.py --report --days 7

# 2. Run full analysis
python project-dashboard.py

# 3. Apply gamification
python habitica-integration.py

# 4. Check achievements
python productivity-tracker.py --leaderboard --username YourName
```

## 🚀 Next Steps

1. **Start Today**: Run your first gamification session
2. **Set Goals**: Configure daily productivity targets
3. **Build Habits**: Work consistently for streaks
4. **Track Progress**: Review reports weekly
5. **Level Up**: Improve scores and unlock achievements

Transform your coding routine into an epic quest for productivity! 🎮✨
