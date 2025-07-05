# Project Gamification Suite

A comprehensive set of tools for tracking, analyzing, and gamifying your software development projects.

## 📁 Folder Structure

```
gamification/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.json                  # Project configuration
│
├── 🎮 Main Scripts (Essential Tools)
│   ├── quick-stats.py          # Quick project statistics
│   ├── project-dashboard.py    # Comprehensive project dashboard
│   ├── health-checker.py       # Project health analysis
│   ├── code-analyzer.py        # Code quality analysis
│   ├── duplicate-finder.py     # Find duplicate files
│   ├── timeline-tracker.py     # Track project timeline
│   ├── growth-monitor.py       # Monitor project growth
│   ├── productivity-tracker.py # Track developer productivity
│   ├── stats-viewer.py         # View collected statistics
│   ├── project-tracker.py      # Main tracking tool
│   ├── habitica-integration.py # Gamify with Habitica
│   └── ai-project-advisor.py   # Get AI-powered insights
│
├── 📂 utils/                   # Utility modules
│   ├── config_loader.py        # Configuration management
│   └── interactive_config.py   # Interactive project selection
│
├── 📂 setup/                   # Setup and configuration scripts
│   ├── setup.sh               # Main setup script
│   ├── setup-environment.sh   # Environment setup
│   ├── api-setup.sh          # API configuration
│   ├── ai-advisor-setup.sh   # AI advisor setup
│   ├── setup-tracker.py      # Tracker setup
│   ├── update_all_scripts.py # Update scripts
│   └── update_to_interactive.py # Interactive mode updater
│
├── 📂 tests/                   # Test scripts
│   ├── test_apis.py           # Test API connections
│   └── test_script_functionality.py # Test script functions
│
└── 📂 docs/                    # Documentation
    └── test_report.md         # API test report

```

## 🚀 Quick Start

1. **First Time Setup**
   ```bash
   cd setup
   ./setup.sh
   ```

2. **Configure APIs** (if using Habitica/AI features)
   ```bash
   # Create .env file with your API keys
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Run Analysis**
   ```bash
   # Quick statistics
   python quick-stats.py
   
   # Full dashboard
   python project-dashboard.py
   
   # Health check
   python health-checker.py
   ```

4. **Gamification Features**
   ```bash
   # Habitica integration
   python habitica-integration.py
   
   # AI insights
   python ai-project-advisor.py
   ```

## 📊 Essential Scripts

### Analysis Tools
- **quick-stats.py** - Get quick project statistics
- **project-dashboard.py** - Generate comprehensive dashboard
- **health-checker.py** - Analyze project health
- **code-analyzer.py** - Analyze code quality and complexity
- **duplicate-finder.py** - Find and manage duplicate files

### Tracking Tools
- **timeline-tracker.py** - Track file creation/modification timeline
- **growth-monitor.py** - Monitor project growth over time
- **productivity-tracker.py** - Track developer productivity
- **stats-viewer.py** - View historical statistics
- **project-tracker.py** - Main project tracking interface

### Gamification Tools
- **habitica-integration.py** - Connect to Habitica for gamification
- **ai-project-advisor.py** - Get AI-powered project insights

## 🔧 Configuration

Edit `config.json` to customize:
- Project directories
- Analysis settings
- Ignore patterns
- AI provider preferences

## 📝 Requirements

- Python 3.8+
- requests library (required)
- python-dotenv (optional)
- google-generativeai (optional, for AI features)

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🧪 Testing

Run tests to verify API connections:
```bash
cd tests
python test_apis.py
python test_script_functionality.py
```

## 🤝 API Integrations

### Habitica
- Tracks development activities as habits
- Rewards good practices
- Punishes technical debt

### Google Gemini
- Provides AI-powered project insights
- Suggests improvements
- Analyzes code patterns

## 📈 Data Storage

All analysis data is stored in `.project-stats/` directory within your project folder.