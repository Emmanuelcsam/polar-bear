# Complete Project Tracker & Habitica Integration Setup Command
# Run this entire block in Git Bash

# Set base directory
BASE_DIR="$HOME/project-tracker-suite" && \
echo "🚀 Setting up Project Tracker Suite with Habitica Integration..." && \
echo "📁 Installing to: $BASE_DIR" && \
echo "=============================================" && \
sleep 2 && \
\
# Create directory structure
mkdir -p "$BASE_DIR"/{scripts,config,templates} && \
mkdir -p "$HOME/.project-productivity" && \
cd "$BASE_DIR" && \
\
# Create setup script that will download/create all files
cat > setup-all-scripts.py << 'EOF'
#!/usr/bin/env python3
"""
Complete Setup Script for Project Tracker Suite
This will create all the tracking scripts with their full content
"""

import os
import sys
from pathlib import Path

print("📦 Creating all Project Tracker scripts...")

# Base directory
scripts_dir = Path("scripts")
scripts_dir.mkdir(exist_ok=True)

# Script contents will be created as empty files with instructions
scripts = {
    "quick-stats.py": "Quick project overview and snapshots",
    "duplicate-finder.py": "Find and remove duplicate files",
    "timeline-tracker.py": "Visualize project evolution",
    "code-analyzer.py": "Analyze code structure and dependencies",
    "health-checker.py": "Check project health and best practices",
    "growth-monitor.py": "Track and predict project growth",
    "project-dashboard.py": "Comprehensive project overview",
    "project-tracker.py": "Master control script",
    "stats-viewer.py": "Web-based stats viewer",
    "habitica-integration.py": "Habitica gamification integration",
    "productivity-tracker.py": "Advanced productivity tracking"
}

print("\n📝 Creating script files...")
for script_name, description in scripts.items():
    script_path = scripts_dir / script_name
    with open(script_path, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
{description}
NOTE: This is a placeholder. Copy the full script content from the artifacts.
"""

print("❌ This script needs to be updated with the full code.")
print("   Please copy the content from the Claude artifacts.")
print(f"   Script: {script_name}")
''')
    print(f"✓ Created: {script_name}")

print("\n✅ All script files created in ./scripts/")
print("\n⚠️  IMPORTANT: The scripts are currently placeholders.")
print("   You need to copy the full code from the Claude artifacts into each file.")

# Create requirements file
with open("requirements.txt", "w") as f:
    f.write("""# Core dependencies (most are optional for enhanced features)
requests>=2.31.0          # For Habitica API integration
matplotlib>=3.7.0         # For charts and visualizations (optional)
plotly>=5.14.0           # For interactive charts (optional)
pandas>=2.0.0            # For data analysis (optional)
numpy>=1.24.0            # For numerical operations (optional)
networkx>=3.1            # For network graphs (optional)
pyvis>=0.3.2             # For network visualizations (optional)
jinja2>=3.1.0            # For HTML report generation (optional)

# Note: The core scripts work without these dependencies
# Install only what you need for enhanced features
""")

print("\n📋 Created requirements.txt")

# Create config templates
config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

# .env template
with open(config_dir / ".env.template", "w") as f:
    f.write("""# Habitica Integration Configuration
HABITICA_USER_ID=8a9e85dc-d8f5-4c60-86ef-d70a19bf225e
HABITICA_API_KEY=a4375d21-0a50-4ceb-a412-ebb70e927349

# Optional: Git integration
GIT_AUTHOR_NAME=emmanuelcsam
GIT_AUTHOR_EMAIL=ecsampson03@gmail.com

# Optional: Productivity settings
PRODUCTIVITY_WORK_START=09:00
PRODUCTIVITY_WORK_END=17:00
PRODUCTIVITY_BREAK_DURATION=15
""")

print("✓ Created config/.env.template")

# Create .gitignore
with open(".gitignore", "w") as f:
    f.write("""# Project Tracker Suite
.project-stats/
.project-productivity/
*.pyc
__pycache__/
.env
config/.env

# OS files
.DS_Store
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp

# Logs and temp files
*.log
*.tmp
*.temp

# Keep example files
!.env.template
!.env.example
""")

print("✓ Created .gitignore")

# Create quick start guide
with open("QUICKSTART.md", "w") as f:
    f.write("""# 🚀 Project Tracker Suite - Quick Start Guide

## ✅ Installation Progress

1. ✓ Directory structure created
2. ✓ Script placeholders created
3. ✓ Configuration templates created
4. ⏳ **Next: Copy script contents from Claude artifacts**

## 📋 Setup Steps

### 1. Copy Script Contents

You need to copy the full code for each script from the Claude artifacts:

**Core Scripts:**
- `quick-stats.py` - Fast project overview
- `duplicate-finder.py` - Find duplicate files
- `timeline-tracker.py` - Project timeline visualization
- `code-analyzer.py` - Code structure analysis
- `health-checker.py` - Project health checking
- `growth-monitor.py` - Growth tracking
- `project-dashboard.py` - Comprehensive dashboard
- `project-tracker.py` - Master control menu

**Additional Scripts:**
- `stats-viewer.py` - Web-based viewer
- `habitica-integration.py` - Habitica gamification
- `productivity-tracker.py` - Productivity tracking

### 2. Install Dependencies

Basic operation (no external dependencies needed):
```bash
# The core scripts work with standard Python!
python scripts/quick-stats.py
