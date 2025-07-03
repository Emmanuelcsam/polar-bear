# Complete Setup Command - Copy and run this entire block in Git Bash
# This sets up Project Tracker Suite with AI Advisor and all dependencies

# Set variables
INSTALL_DIR="$HOME/project-tracker-suite" && \
SCRIPTS_DIR="$INSTALL_DIR/scripts" && \
CONFIG_DIR="$INSTALL_DIR/config" && \
echo "🚀 COMPLETE PROJECT TRACKER + AI ADVISOR SETUP" && \
echo "=============================================" && \
echo "Installing to: $INSTALL_DIR" && \
sleep 2 && \
\
# Create directory structure
mkdir -p "$SCRIPTS_DIR" "$CONFIG_DIR" "$INSTALL_DIR/examples" && \
cd "$INSTALL_DIR" && \
\
# Install Python dependencies
echo -e "\n📦 Installing Python dependencies..." && \
pip install requests matplotlib plotly pandas numpy networkx pyvis jinja2 --quiet && \
echo "✓ Dependencies installed" && \
\
# Create a setup completion script
cat > setup-complete.py << 'SETUP_EOF'
#!/usr/bin/env python3
import os
import sys

print("""
✅ SETUP COMPLETE!
================

📁 Installation Directory: {}

🎯 NEXT STEPS:
1. Copy all script contents from Claude artifacts into scripts/
2. Configure AI provider (optional):
   ./setup-ai-advisor.sh
3. Start tracking:
   cd scripts
   python quick-stats.py

📚 Quick Reference:
- Project tracking: python project-tracker.py
- AI advisor: python ai-project-advisor.py
- Habitica integration: python habitica-integration.py
- Productivity tracking: python productivity-tracker.py

💡 TIP: Add to PATH for easy access:
   echo 'export PATH="$HOME/project-tracker-suite/scripts:$PATH"' >> ~/.bashrc
   source ~/.bashrc
""".format(os.path.dirname(os.path.abspath(__file__))))

# Create placeholder scripts
scripts = [
    "quick-stats.py", "duplicate-finder.py", "timeline-tracker.py",
    "code-analyzer.py", "health-checker.py", "growth-monitor.py",
    "project-dashboard.py", "project-tracker.py", "stats-viewer.py",
    "habitica-integration.py", "productivity-tracker.py", "ai-project-advisor.py"
]

print("\n📝 Creating script placeholders...")
for script in scripts:
    script_path = os.path.join("scripts", script)
    with open(script_path, 'w') as f:
        f.write(f'#!/usr/bin/env python3\n# {script} - Copy full content from Claude artifacts\nprint("Placeholder - copy full script from Claude")\n')
    os.chmod(script_path, 0o755)
    print(f"   ✓ {script}")

print("\n⚠️  Remember: Scripts are placeholders - copy actual code from Claude artifacts!")
SETUP_EOF

# Create AI advisor setup script
cat > setup-ai-advisor.sh << 'AI_SETUP_EOF'
#!/bin/bash
# AI Advisor Configuration Script

echo "🤖 AI ADVISOR CONFIGURATION"
echo "=========================="
echo ""
echo "Choose AI provider:"
echo "1. OpenAI (GPT-4/GPT-3.5)"
echo "2. Anthropic Claude"
echo "3. Google Gemini"
echo "4. Hugging Face"
echo ""
read -p "Select (1-4): " choice

ENV_FILE="config/.env"
mkdir -p config

case $choice in
    1)
        echo "Get key from: https://platform.openai.com/api-key
