#!/bin/bash
# Enhanced Neural Nexus IDE Setup Script v6.0
# Sets up performance and security enhancements

echo "🚀 Setting up Neural Nexus IDE Server v6.0 with Enhanced Features"
echo "=================================================================="

# Check if Python 3.8+ is available
python3 --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install fastapi uvicorn[standard] websockets aiofiles

# Install performance boosters
echo "🚀 Installing performance enhancements..."
pip install orjson uvloop pydantic[performance] loguru

# Install static analysis and security tools
echo "🔒 Installing security and analysis tools..."
pip install semgrep bandit ruff pyright

# Install additional enhancements
echo "✨ Installing additional enhancements..."
pip install slowapi msgspec psutil

# Install optional AI dependencies
echo "🤖 Installing AI dependencies..."
pip install openai requests

# Create requirements file
echo "📄 Creating enhanced requirements file..."
pip freeze > requirements_enhanced_actual.txt

# Set executable permissions
echo "🔧 Setting up launch scripts..."
chmod +x launch_enhanced.py

# Create additional tools directory
echo "📁 Setting up tools directory..."
mkdir -p tools

# Create a pyproject.toml for modern Python project management
echo "📋 Creating pyproject.toml..."
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "neural-nexus-ide"
version = "6.0.0"
description = "AI-Powered Development Environment with Security & Performance"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Neural Nexus Team"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "websockets>=11.0",
    "orjson>=3.9.0",
    "uvloop>=0.17.0",
    "pydantic>=2.0.0",
    "semgrep>=1.45.0",
    "bandit>=1.7.5",
    "ruff>=0.1.0",
    "loguru>=0.7.0",
    "aiofiles>=23.0.0",
    "psutil>=5.9.0",
    "slowapi>=0.1.8",
    "msgspec>=0.18.0",
]

[project.optional-dependencies]
ai = [
    "openai>=1.0.0",
    "requests>=2.31.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]

[project.urls]
Homepage = "https://github.com/neural-nexus/ide-server"
Repository = "https://github.com/neural-nexus/ide-server.git"
Documentation = "https://neural-nexus.readthedocs.io"

[project.scripts]
neural-nexus = "neural_nexus_server:main"

[tool.ruff]
line-length = 88
target-version = "py38"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # Pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
EOF

# Create a simple test script
echo "🧪 Creating test script..."
cat > test_server.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for Neural Nexus IDE Server v6.0
"""
import asyncio
import aiohttp
import json

async def test_server():
    """Test the enhanced server functionality"""
    base_url = "http://localhost:8765"

    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        print("🔍 Testing health endpoint...")
        async with session.get(f"{base_url}/health") as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✅ Health check passed: {data['message']}")
                print(f"📊 Features: {list(data['features'].keys())}")
            else:
                print(f"❌ Health check failed: {resp.status}")

        # Test format endpoint
        print("\n🎨 Testing code formatting...")
        test_code = """import os,sys
def hello(  ):
    print( "Hello World" )
hello()"""

        async with session.post(
            f"{base_url}/api/format",
            json={"content": test_code}
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data['success']:
                    print("✅ Code formatting test passed")
                    print("📝 Original vs Formatted:")
                    print("Original:", repr(test_code))
                    print("Formatted:", repr(data['formatted_code']))
                else:
                    print(f"❌ Formatting failed: {data.get('error')}")
            else:
                print(f"❌ Format endpoint failed: {resp.status}")

        # Test security scan endpoint
        print("\n🔒 Testing security scanning...")
        unsafe_code = """import os
os.system("echo 'test'")
eval("print('unsafe')")"""

        async with session.post(
            f"{base_url}/api/security-scan",
            json={"content": unsafe_code}
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data['success']:
                    print(f"✅ Security scan completed: {data['total_issues']} issues found")
                    if data['total_issues'] > 0:
                        print("🚨 Security issues detected (expected for test):")
                        for issue in data['security_issues'][:3]:  # Show first 3
                            print(f"  - Line {issue.get('line', 0)}: {issue.get('message', 'Unknown')}")
                else:
                    print(f"❌ Security scan failed: {data.get('error')}")
            else:
                print(f"❌ Security scan endpoint failed: {resp.status}")

if __name__ == "__main__":
    try:
        asyncio.run(test_server())
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure the server is running: python launch_enhanced.py")
EOF

echo ""
echo "🎉 Setup complete! Neural Nexus IDE v6.0 is ready with enhanced features."
echo ""
echo "🚀 Quick Start:"
echo "  1. Start the server: python launch_enhanced.py"
echo "  2. Open browser: http://localhost:8765"
echo "  3. Test features: python test_server.py (in another terminal)"
echo ""
echo "✨ New Features Available:"
echo "  • Ultra-fast JSON processing (orjson)"
echo "  • Enhanced event loop (uvloop)"
echo "  • Security scanning (Semgrep + Bandit)"
echo "  • Auto-formatting (Ruff)"
echo "  • Type checking (Pyright)"
echo "  • Rate limiting & security headers"
echo "  • Structured logging (loguru)"
echo "  • Performance monitoring"
echo ""
echo "🔧 Configuration files created:"
echo "  • pyproject.toml (modern Python project config)"
echo "  • requirements_enhanced_actual.txt (actual installed packages)"
echo "  • test_server.py (functionality tests)"
echo ""
