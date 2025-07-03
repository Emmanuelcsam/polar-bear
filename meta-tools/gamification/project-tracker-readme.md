# Project Tracker Suite 🚀

A comprehensive collection of Python scripts to track, analyze, and monitor your project's growth and health over time.

## 🎯 Overview

The Project Tracker Suite consists of 8 specialized tools that work together to give you deep insights into your project:

1. **Quick Stats** - Fast overview and snapshot creation
2. **Duplicate Finder** - Find and remove duplicate files
3. **Timeline Tracker** - Visualize project evolution
4. **Code Analyzer** - Analyze code structure and dependencies
5. **Health Checker** - Check for issues and best practices
6. **Growth Monitor** - Track and predict project growth
7. **Project Dashboard** - Comprehensive overview of all metrics
8. **Project Tracker** - Master script to run all tools

## 📋 Features

### Quick Stats (`quick-stats.py`)
- Instant project overview
- File and directory counts
- Language distribution
- Recent file activity
- Creates snapshots for historical tracking

### Duplicate Finder (`duplicate-finder.py`)
- Finds duplicate files by content hash
- Interactive cleanup options
- Space saving recommendations
- Safe file moving to trash folder

### Timeline Tracker (`timeline-tracker.py`)
- Shows project growth over time
- Monthly/daily activity patterns
- Most productive days analysis
- Git history integration

### Code Analyzer (`code-analyzer.py`)
- Language-specific code analysis
- Import/dependency mapping
- Complexity scoring
- Function and class counting
- Circular dependency detection

### Health Checker (`health-checker.py`)
- Project health scoring (0-100)
- Security issue detection
- Best practices checking
- Missing file detection (README, LICENSE, etc.)
- Actionable recommendations

### Growth Monitor (`growth-monitor.py`)
- Historical growth analysis
- Growth rate calculations
- Trend identification
- Future predictions
- Milestone tracking

### Project Dashboard (`project-dashboard.py`)
- All metrics in one place
- Visual health indicators
- Quick wins suggestions
- Summary statistics

### Project Tracker (`project-tracker.py`)
- Menu-driven interface
- Run individual or all tools
- View historical reports
- Change project directories

## 🚀 Quick Start

### 1. Setup

Save all 8 Python scripts in the same directory:
```bash
# Create a directory for the tools
mkdir project-tracker-tools
cd project-tracker-tools

# Save all the scripts here:
# - quick-stats.py
# - duplicate-finder.py
# - timeline-tracker.py
# - code-analyzer.py
# - health-checker.py
# - growth-monitor.py
# - project-dashboard.py
# - project-tracker.py
```

### 2. Basic Usage

#### Run the master script:
```bash
python project-tracker.py [path-to-your-project]
```

#### Or run individual tools:
```bash
# Quick overview
python quick-stats.py /path/to/project

# Find duplicates
python duplicate-finder.py /path/to/project

# Check project health
python health-checker.py /path/to/project
```

### 3. First Time Setup

1. Navigate to your project directory
2. Run Quick Stats to create your first snapshot:
   ```bash
   python quick-stats.py
   ```
3. Run the Dashboard for a complete overview:
   ```bash
   python project-dashboard.py
   ```

## 📊 Data Storage

All analysis data is stored in a `.project-stats/` directory within your project:

```
your-project/
├── .project-stats/
│   ├── snapshot_*.json          # Historical snapshots
│   ├── health_report_*.json     # Health check reports
│   ├── timeline_*.json          # Timeline data
│   ├── code_analysis_*.json     # Code analysis results
│   ├── duplicates_*.json        # Duplicate file reports
│   ├── dashboard_*.json         # Dashboard snapshots
│   └── latest_dashboard.json    # Most recent dashboard
├── src/
├── docs/
└── ...
```

## 💡 Best Practices

### Regular Monitoring
1. Run Quick Stats daily/weekly to build historical data
2. Check Health monthly to maintain code quality
3. Use Growth Monitor to track long-term trends

### Project Maintenance
1. Use Duplicate Finder to clean up regularly
2. Address Health Checker warnings promptly
3. Review Code Analyzer results for refactoring opportunities

### Tracking Progress
1. Create snapshots before major changes
2. Use Timeline Tracker to identify productive periods
3. Monitor growth trends to predict resource needs

## 🛠️ Advanced Usage

### Command Line Arguments

All scripts accept a path argument:
```bash
python [script-name].py [path-to-analyze]
```

If no path is provided, the current directory is analyzed.

### Duplicate Finder Options
```bash
python duplicate-finder.py /path/to/project --min-size 1024
```
- `--min-size`: Minimum file size to check (default: 1KB)

### Automation

Create a cron job or scheduled task:
```bash
# Daily snapshot
0 9 * * * cd /path/to/project && python /path/to/quick-stats.py

# Weekly health check
0 10 * * 1 cd /path/to/project && python /path/to/health-checker.py
```

## 📈 Interpreting Results

### Health Score
- **90-100**: Excellent - Well-maintained project
- **70-89**: Good - Minor issues to address
- **50-69**: Fair - Several improvements needed
- **0-49**: Needs Attention - Significant issues

### Growth Trends
- **Rapid Growth**: >1 file/day average
- **Steady Growth**: 0.1-1 file/day
- **Stable**: <0.1 file/day change
- **Declining**: Negative growth rate

### Common Issues
1. **No README**: Add project documentation
2. **Large Files**: Consider Git LFS or external storage
3. **Many Duplicates**: Run cleanup to save space
4. **Security Issues**: Check for exposed credentials
5. **Poor Structure**: Reorganize into standard directories

## 🔧 Troubleshooting

### Scripts Not Found
Ensure all scripts are in the same directory or in your PATH.

### Permission Errors
Some operations require write permissions:
```bash
chmod +x *.py  # Make scripts executable
```

### Memory Issues
For very large projects, some tools may need adjustment:
- Duplicate Finder: Increase `--min-size` parameter
- Code Analyzer: Runs on samples of files

## 🚀 Tips for Maximum Value

1. **Start Early**: Begin tracking when project is small
2. **Be Consistent**: Regular snapshots provide best insights
3. **Act on Findings**: Use recommendations to improve
4. **Track Milestones**: Celebrate growth achievements
5. **Share Results**: Use dashboards in team meetings

## 📝 Adding to .gitignore

If you don't want to track analysis data:
```gitignore
# Project tracking data
.project-stats/
```

## 🤝 Integration Ideas

1. **CI/CD Pipeline**: Add health checks to build process
2. **Git Hooks**: Run quick stats on each commit
3. **Documentation**: Include dashboard in project docs
4. **Team Dashboards**: Share growth metrics
5. **Alerts**: Set up notifications for health score drops

## 📊 Example Workflow

### Weekly Project Review
1. Run full analysis suite:
   ```bash
   python project-tracker.py
   # Select option 8 (Run All Analyses)
   ```

2. Review dashboard results
3. Address any critical health issues
4. Clean up duplicates if found
5. Document major changes

### Before Major Release
1. Run Health Checker for quality gate
2. Use Code Analyzer to review dependencies
3. Check Timeline for recent activity
4. Generate Dashboard for release notes

## 🎯 Goal Setting

Use the tools to set and track goals:
- "Reduce duplicates by 50%"
- "Achieve 95+ health score"
- "Maintain steady growth of 5 files/week"
- "Keep project size under 100MB"

## 🔍 Understanding Your Project Better

The suite helps answer questions like:
- How fast is my project growing?
- What languages do I use most?
- When am I most productive?
- Are there quality issues to address?
- How has the project evolved?
- What files are wasting space?

---

Start tracking your project today and gain valuable insights into your development patterns! 🚀
