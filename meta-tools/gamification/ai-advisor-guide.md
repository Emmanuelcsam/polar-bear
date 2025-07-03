# 🤖 AI Project Advisor Guide

Get intelligent, specific suggestions for your project using AI analysis of your project tracking data.

## 🌟 Overview

The AI Project Advisor analyzes all your project statistics and provides:
- **Specific, actionable recommendations** based on real data
- **Focused analysis** in 8 different areas
- **Support for multiple AI providers** (OpenAI, Claude, Gemini, Hugging Face)
- **Comprehensive reports** with prioritized suggestions
- **Saved suggestions** for future reference

## 🔧 Setup Guide

### Prerequisites

1. **Run project analysis first**:
   ```bash
   python quick-stats.py
   python health-checker.py
   python code-analyzer.py
   python project-dashboard.py
   ```

2. **Get an API key** from one of these providers:

### Provider Setup

#### OpenAI (GPT-4/GPT-3.5)
1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Go to API Keys section
3. Create new secret key
4. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

#### Anthropic Claude
1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Go to API Keys
3. Create new key
4. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

#### Google Gemini
1. Get API key from [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Set environment variable:
   ```bash
   export GOOGLE_API_KEY="..."
   ```

#### Hugging Face
1. Sign up at [huggingface.co](https://huggingface.co)
2. Go to Settings > Access Tokens
3. Create new token
4. Set environment variable:
   ```bash
   export HUGGINGFACE_API_KEY="hf_..."
   ```

### Installation

```bash
# The script uses standard libraries plus requests
pip install requests
```

## 📖 Usage

### Interactive Mode (Recommended)

```bash
python ai-project-advisor.py
```

This will:
1. Let you choose your AI provider
2. Select a model
3. Choose analysis focus areas
4. Get detailed suggestions

### Quick Analysis

```bash
# Analyze specific area with specific provider
python ai-project-advisor.py --provider openai --api-key YOUR_KEY --focus health --quick

# Use environment variables
export OPENAI_API_KEY="your-key"
python ai-project-advisor.py --provider openai --focus productivity --quick
```

### Command Line Options

```bash
python ai-project-advisor.py [path] [options]

Options:
  path                  Project path to analyze (default: current directory)
  --provider            AI provider: openai, claude, gemini, huggingface
  --api-key            API key for the provider
  --model              Specific model to use
  --focus              Analysis area: health, productivity, architecture, etc.
  --quick              Skip interactive mode
```

## 🎯 Analysis Focus Areas

### 1. **Health** - Code quality improvements
- Identifies critical issues
- Suggests fixes for warnings
- Recommends best practices

### 2. **Productivity** - Workflow optimization
- Analyzes development patterns
- Suggests efficiency improvements
- Recommends tools and processes

### 3. **Architecture** - Structure recommendations
- Reviews code organization
- Suggests refactoring opportunities
- Identifies design improvements

### 4. **Performance** - Optimization opportunities
- Identifies bottlenecks
- Suggests caching strategies
- Recommends algorithm improvements

### 5. **Security** - Vulnerability detection
- Reviews security issues
- Suggests fixes
- Recommends security practices

### 6. **Growth** - Project scaling strategies
- Predicts future needs
- Suggests preparation steps
- Recommends growth paths

### 7. **Team** - Collaboration improvements
- Suggests documentation needs
- Recommends communication tools
- Identifies knowledge gaps

### 8. **Technical Debt** - Debt reduction
- Identifies accumulated debt
- Prioritizes refactoring
- Suggests incremental improvements

## 💡 Example Sessions

### Basic Health Check
```bash
$ python ai-project-advisor.py

🤖 AI PROJECT ADVISOR SETUP
==================================================
Choose your AI provider:
1. OpenAI
2. Anthropic Claude
3. Google Gemini
4. Hugging Face

Select provider (1-4): 1

📊 Collecting project data...
✓ Loaded dashboard data
✓ Loaded health report
✓ Loaded code analysis

🎯 ANALYSIS FOCUS AREAS
==================================================
1. Health - Project health and code quality improvements
2. Productivity - Developer productivity and workflow optimization
...

Select focus area: 1

🤖 Getting AI suggestions from OpenAI...

🤖 AI SUGGESTIONS - HEALTH
==================================================
Based on your project analysis, here are specific recommendations:

1. **Missing Documentation** (High Priority)
   - Issue: README.md is only 87 bytes
   - Action: Create comprehensive documentation including:
     - Project overview and purpose
     - Installation instructions
     - Usage examples
     - API documentation
   - Impact: Improves onboarding, reduces support questions

2. **Security Vulnerabilities** (Critical)
   - Issue: .env file not in .gitignore
   - Action: Add .env to .gitignore immediately
   - Commands: echo ".env" >> .gitignore
   - Impact: Prevents credential exposure

3. **Duplicate Code** (Medium Priority)
   - Issue: 47 duplicate files wasting 2.3 MB
   - Action: Run duplicate-finder.py and consolidate
   - Impact: Cleaner codebase, easier maintenance
...
```

### Comprehensive Analysis
```bash
# Analyze all areas at once
$ python ai-project-advisor.py

Select focus area: 0

🔄 Generating comprehensive analysis...

📊 Analyzing health...
📊 Analyzing productivity...
📊 Analyzing architecture...
📊 Analyzing performance...

[Generates complete report covering all areas]
```

### Quick Command Line
```bash
# Quick productivity check
$ python ai-project-advisor.py --provider gemini --focus productivity --quick

✓ Using Google Gemini with gemini-pro
📁 Analyzing project: /home/user/myproject

📊 Collecting project data...
✓ Loaded dashboard data
✓ Loaded health report

🤖 Getting AI suggestions from Google Gemini...

[Displays productivity-focused suggestions]
```

## 📊 Understanding the Analysis

The AI advisor uses your actual project data:

### Data Sources
- **Dashboard**: Overall metrics, languages, file counts
- **Health Report**: Issues, warnings, recommendations
- **Code Analysis**: Dependencies, complexity, structure
- **Timeline**: Growth patterns, activity history
- **Duplicates**: Redundancy and waste

### AI Analysis Process
1. **Data Collection**: Gathers all available metrics
2. **Context Building**: Creates comprehensive prompt
3. **AI Processing**: Sends to chosen provider
4. **Suggestion Generation**: Receives specific recommendations
5. **Report Formatting**: Presents actionable items

## 💾 Saved Reports

All suggestions are saved to:
```
.project-stats/
└── ai_suggestions/
    ├── suggestions_health_20240115_143022.md
    ├── suggestions_productivity_20240115_144512.md
    └── suggestions_comprehensive_20240115_150234.md
```

View saved suggestions anytime:
```bash
ls .project-stats/ai_suggestions/
cat .project-stats/ai_suggestions/suggestions_health_*.md
```

## 🎯 Best Practices

### 1. **Run Analysis Tools First**
Always run the tracking tools before AI analysis:
```bash
python quick-stats.py
python health-checker.py
python project-dashboard.py
```

### 2. **Start with Health Check**
Begin with health analysis to fix critical issues:
```bash
python ai-project-advisor.py --focus health
```

### 3. **Regular Reviews**
Schedule weekly AI reviews:
```bash
# Weekly comprehensive analysis
python ai-project-advisor.py --focus comprehensive
```

### 4. **Act on Suggestions**
- Prioritize critical issues
- Implement one suggestion at a time
- Re-run analysis to track improvements

### 5. **Compare Providers**
Try different AI providers for varied perspectives:
```bash
# OpenAI perspective
python ai-project-advisor.py --provider openai --focus architecture

# Claude perspective  
python ai-project-advisor.py --provider claude --focus architecture
```

## 🚀 Advanced Usage

### Custom Analysis Areas
Modify the script to add custom focus areas:
```python
self.analysis_areas['testing'] = 'Test coverage and testing strategies'
```

### Batch Analysis
Analyze multiple projects:
```bash
for project in ~/projects/*; do
    echo "Analyzing $project"
    python ai-project-advisor.py "$project" --focus health --quick
done
```

### Integration with CI/CD
Add to your build pipeline:
```yaml
- name: AI Code Review
  run: |
    python quick-stats.py
    python health-checker.py
    python ai-project-advisor.py --provider openai --focus health --quick
```

## 🔍 Troubleshooting

### API Key Issues
```bash
# Test your API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### No Data Found
Ensure you've run analysis tools:
```bash
ls .project-stats/
# Should contain: latest_dashboard.json, health_report_*.json, etc.
```

### Rate Limits
- Add delays between requests
- Use smaller models for testing
- Upgrade API plan if needed

## 💰 Cost Optimization

### Model Selection
- **Development**: Use cheaper models (gpt-3.5-turbo, claude-3-haiku)
- **Production**: Use advanced models (gpt-4, claude-3-opus)
- **Testing**: Use Hugging Face free tier

### Token Usage
- Focus on specific areas vs comprehensive
- Limit data sent in prompts
- Cache results for repeated analysis

## 🎨 Example Suggestions Output

```markdown
# AI Project Suggestions - Health

**Project**: my-awesome-app
**Date**: 2024-01-15 14:30
**AI Provider**: OpenAI
**Model**: gpt-4-turbo-preview

---

## 🔍 Issues Identified

1. **Critical Security Vulnerability**
   - Your .env file containing API keys is not in .gitignore
   - This could expose sensitive credentials if pushed to GitHub

2. **Poor Test Coverage**
   - No test files detected in the project
   - This increases risk of bugs and regressions

3. **Accumulating Technical Debt**
   - 47 duplicate files found (2.3MB wasted)
   - No documentation beyond a minimal README

## 🎯 Prioritized Recommendations

### Priority 1: Security (Immediate Action)
```bash
echo ".env" >> .gitignore
git rm --cached .env
git commit -m "Remove .env from tracking"
```
**Impact**: Prevents credential exposure, protects user data

### Priority 2: Documentation (This Week)
1. Create comprehensive README.md:
   - Project description and purpose
   - Installation instructions
   - Usage examples
   - Contributing guidelines

2. Add inline code comments:
   - Document complex functions
   - Explain business logic
   - Add TODO comments for future work

**Impact**: 50% reduction in onboarding time, easier maintenance

### Priority 3: Testing (Next Sprint)
1. Set up testing framework:
   ```bash
   pip install pytest pytest-cov
   ```

2. Create initial test suite:
   - Unit tests for core functions
   - Integration tests for APIs
   - Aim for 70% coverage initially

**Impact**: 80% reduction in production bugs, confident deployments

## 📈 Expected Outcomes

Implementing these recommendations will:
- Improve health score from 72 to 90+
- Reduce security vulnerabilities to zero
- Decrease onboarding time by 50%
- Increase development velocity by 30%

## ⚠️ Risks and Considerations

- Documentation effort: ~8 hours initial investment
- Testing setup: ~16 hours for comprehensive suite
- Team training: May need testing workshop

Start with security fixes today, then gradually implement other suggestions over the next sprint.
```

## 🎉 Start Getting AI-Powered Insights!

1. Choose your AI provider
2. Run project analysis tools
3. Get specific, actionable suggestions
4. Implement improvements
5. Track your progress

Transform your project with intelligent AI guidance! 🚀
