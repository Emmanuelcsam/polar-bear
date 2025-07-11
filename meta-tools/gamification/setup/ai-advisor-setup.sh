#!/bin/bash
# AI Project Advisor - Complete Setup Script
# Sets up API keys, tests connections, and configures the AI advisor

echo "🤖 AI PROJECT ADVISOR SETUP"
echo "==========================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to test API keys
test_openai() {
    echo -n "Testing OpenAI API key... "
    response=$(curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models \
        -H "Authorization: Bearer $1")
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ Valid${NC}"
        return 0
    else
        echo -e "${RED}✗ Invalid${NC}"
        return 1
    fi
}

test_anthropic() {
    echo -n "Testing Anthropic API key... "
    response=$(curl -s -o /dev/null -w "%{http_code}" https://api.anthropic.com/v1/messages \
        -H "x-api-key: $1" \
        -H "anthropic-version: 2023-06-01" \
        -H "content-type: application/json" \
        -d '{"model":"claude-3-haiku-20240307","messages":[{"role":"user","content":"test"}],"max_tokens":1}')
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ Valid${NC}"
        return 0
    else
        echo -e "${RED}✗ Invalid${NC}"
        return 1
    fi
}

test_google() {
    echo -n "Testing Google API key... "
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        "https://generativelanguage.googleapis.com/v1beta/models?key=$1")
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ Valid${NC}"
        return 0
    else
        echo -e "${RED}✗ Invalid${NC}"
        return 1
    fi
}

test_huggingface() {
    echo -n "Testing Hugging Face API key... "
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        https://api-inference.huggingface.co/models/gpt2 \
        -H "Authorization: Bearer $1" \
        -d '{"inputs":"test"}')
    if [ "$response" = "200" ] || [ "$response" = "503" ]; then
        echo -e "${GREEN}✓ Valid${NC}"
        return 0
    else
        echo -e "${RED}✗ Invalid${NC}"
        return 1
    fi
}

# Create directories
mkdir -p ~/project-tracker-suite/{scripts,config}
cd ~/project-tracker-suite

# Check if AI advisor script exists
if [ ! -f "scripts/ai-project-advisor.py" ]; then
    echo -e "${YELLOW}⚠️  AI advisor script not found!${NC}"
    echo "Creating placeholder - you'll need to copy the full script content"
    
    cat > scripts/ai-project-advisor.py << 'EOF'
#!/usr/bin/env python3
"""
AI Project Advisor - Placeholder
Copy the full script content from Claude artifacts
"""
print("This is a placeholder. Copy the full AI advisor script here.")
EOF
fi

# Create or update .env file
ENV_FILE="config/.env"
echo ""
echo "📋 Configuring API Keys"
echo "======================"
echo ""
echo "Choose which AI providers to configure:"
echo "(You can set up multiple providers)"
echo ""

# OpenAI Setup
echo -e "${YELLOW}1. OpenAI (GPT-4, GPT-3.5)${NC}"
read -p "Configure OpenAI? (y/n): " configure_openai
if [ "$configure_openai" = "y" ]; then
    echo "Get your API key from: https://platform.openai.com/api-keys"
    read -p "Enter OpenAI API key: " openai_key
    if test_openai "$openai_key"; then
        echo "OPENAI_API_KEY=$openai_key" >> "$ENV_FILE"
    fi
    echo ""
fi

# Anthropic Setup
echo -e "${YELLOW}2. Anthropic Claude${NC}"
read -p "Configure Claude? (y/n): " configure_claude
if [ "$configure_claude" = "y" ]; then
    echo "Get your API key from: https://console.anthropic.com/settings/keys"
    read -p "Enter Anthropic API key: " anthropic_key
    if test_anthropic "$anthropic_key"; then
        echo "ANTHROPIC_API_KEY=$anthropic_key" >> "$ENV_FILE"
    fi
    echo ""
fi

# Google Setup
echo -e "${YELLOW}3. Google Gemini${NC}"
read -p "Configure Gemini? (y/n): " configure_gemini
if [ "$configure_gemini" = "y" ]; then
    echo "Get your API key from: https://makersuite.google.com/app/apikey"
    read -p "Enter Google API key: " google_key
    if test_google "$google_key"; then
        echo "GOOGLE_API_KEY=$google_key" >> "$ENV_FILE"
    fi
    echo ""
fi

# Hugging Face Setup
echo -e "${YELLOW}4. Hugging Face${NC}"
read -p "Configure Hugging Face? (y/n): " configure_hf
if [ "$configure_hf" = "y" ]; then
    echo "Get your API token from: https://huggingface.co/settings/tokens"
    read -p "Enter Hugging Face API token: " hf_key
    if test_huggingface "$hf_key"; then
        echo "HUGGINGFACE_API_KEY=$hf_key" >> "$ENV_FILE"
    fi
    echo ""
fi

# Create shell configuration
echo ""
echo "📝 Creating configuration files..."

# Create activation script
cat > config/activate.sh << 'EOF'
#!/bin/bash
# Activate AI Project Advisor environment

# Load environment variables
if [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
    echo "✓ Environment variables loaded"
else
    echo "❌ No .env file found in config directory"
fi

# Add scripts to PATH
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
export PATH="$SCRIPT_DIR:$PATH"
echo "✓ Scripts directory added to PATH"

# Aliases
alias ai-health='python $SCRIPT_DIR/ai-project-advisor.py --focus health'
alias ai-prod='python $SCRIPT_DIR/ai-project-advisor.py --focus productivity'
alias ai-security='python $SCRIPT_DIR/ai-project-advisor.py --focus security'
alias ai-all='python $SCRIPT_DIR/ai-project-advisor.py'

echo ""
echo "🤖 AI Project Advisor activated!"
echo ""
echo "Quick commands:"
echo "  ai-health    - Get health recommendations"
echo "  ai-prod      - Get productivity tips"
echo "  ai-security  - Get security analysis"
echo "  ai-all       - Interactive full analysis"
echo ""
EOF

chmod +x config/activate.sh

# Create quick test script
cat > scripts/test-ai-advisor.py << 'EOF'
#!/usr/bin/env python3
"""Test AI Advisor Configuration"""

import os
import sys

print("🔍 Testing AI Advisor Configuration")
print("=" * 40)

# Check for API keys
providers = {
    'OPENAI_API_KEY': 'OpenAI',
    'ANTHROPIC_API_KEY': 'Anthropic Claude',
    'GOOGLE_API_KEY': 'Google Gemini',
    'HUGGINGFACE_API_KEY': 'Hugging Face'
}

configured = []
for env_var, provider in providers.items():
    if os.environ.get(env_var):
        print(f"✓ {provider} configured")
        configured.append(provider)
    else:
        print(f"✗ {provider} not configured")

if not configured:
    print("\n❌ No AI providers configured!")
    print("   Run the setup script to configure API keys")
    sys.exit(1)

print(f"\n✅ {len(configured)} provider(s) ready:")
for provider in configured:
    print(f"   • {provider}")

print("\n📊 Checking for project data...")
if os.path.exists('.project-stats/latest_dashboard.json'):
    print("✓ Project data found")
    print("\n✅ Ready to use AI advisor!")
    print("   Run: python ai-project-advisor.py")
else:
    print("✗ No project data found")
    print("\n⚠️  Run these commands first:")
    print("   python quick-stats.py")
    print("   python health-checker.py")
    print("   python project-dashboard.py")
EOF

chmod +x scripts/test-ai-advisor.py

# Create example usage script
cat > examples/ai-advisor-examples.sh << 'EOF'
#!/bin/bash
# AI Project Advisor - Example Usage

echo "🤖 AI PROJECT ADVISOR EXAMPLES"
echo "=============================="
echo ""

echo "1. Interactive mode (recommended):"
echo "   python ai-project-advisor.py"
echo ""

echo "2. Quick health check:"
echo "   python ai-project-advisor.py --provider openai --focus health --quick"
echo ""

echo "3. Productivity analysis:"
echo "   python ai-project-advisor.py --provider claude --focus productivity --quick"
echo ""

echo "4. Security audit:"
echo "   python ai-project-advisor.py --provider gemini --focus security --quick"
echo ""

echo "5. Comprehensive analysis:"
echo "   python ai-project-advisor.py --provider openai"
echo "   # Then select option 0 for all areas"
echo ""

echo "6. Specific model selection:"
echo "   python ai-project-advisor.py --provider openai --model gpt-4-turbo-preview"
echo ""

echo "7. Batch analysis:"
echo "   for area in health productivity security architecture; do"
echo "     python ai-project-advisor.py --focus \$area --quick"
echo "     sleep 2"
echo "   done"
echo ""
EOF

chmod +x examples/ai-advisor-examples.sh

# Final summary
echo ""
echo "====================================="
echo -e "${GREEN}✅ AI ADVISOR SETUP COMPLETE!${NC}"
echo "====================================="
echo ""

# Count configured providers
configured_count=$(grep -c "API_KEY=" "$ENV_FILE" 2>/dev/null || echo 0)

if [ $configured_count -gt 0 ]; then
    echo -e "${GREEN}✓ $configured_count AI provider(s) configured${NC}"
    echo ""
    echo "🚀 Quick Start:"
    echo "   1. Load environment: source config/activate.sh"
    echo "   2. Test setup: python scripts/test-ai-advisor.py"
    echo "   3. Run advisor: python scripts/ai-project-advisor.py"
    echo ""
    echo "📚 Examples:"
    echo "   ./examples/ai-advisor-examples.sh"
else
    echo -e "${YELLOW}⚠️  No AI providers configured${NC}"
    echo "   Re-run this script to add API keys"
fi

echo ""
echo "📁 Configuration saved to:"
echo "   ~/project-tracker-suite/config/.env"
echo ""
echo "🔐 Security reminder:"
echo "   • Keep your API keys private"
echo "   • Add .env to .gitignore"
echo "   • Use environment variables in production"
echo ""

# Create quick launcher
cat > ~/project-tracker-suite/ai-advisor << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source config/activate.sh
python scripts/ai-project-advisor.py "$@"
EOF
chmod +x ~/project-tracker-suite/ai-advisor

echo "💡 Created launcher: ~/project-tracker-suite/ai-advisor"
echo "   You can now run: ./ai-advisor from that directory"
