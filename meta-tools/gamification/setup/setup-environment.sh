#!/bin/bash

# Setup script for gamification tools
echo "🚀 Setting up Gamification Project Tools"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✨ Setup complete!"
echo ""
echo "📋 Environment Configuration:"
echo "- Virtual environment: ./venv"
echo "- API keys configured in: .env"
echo "- Dependencies installed from: requirements.txt"
echo ""
echo "🔑 API Key Configuration:"
echo "- Gemini API key is stored in .env file"
echo "- The key is automatically loaded when running scripts"
echo ""
echo "📝 To activate the environment manually:"
echo "   source venv/bin/activate"
echo ""
echo "🚀 To run the AI advisor with Gemini:"
echo "   python ai-project-advisor.py --provider gemini"
echo ""
echo "📊 Available tools:"
echo "   - quick-stats.py: Fast project overview"
echo "   - project-dashboard.py: Comprehensive project analysis"
echo "   - health-checker.py: Check project health"
echo "   - code-analyzer.py: Analyze code structure"
echo "   - duplicate-finder.py: Find duplicate files"
echo "   - timeline-tracker.py: Track project evolution"
echo "   - ai-project-advisor.py: Get AI-powered suggestions"
echo ""