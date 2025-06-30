# Neural Nexus IDE 4.0 🚀

The most advanced AI-powered Python development environment for rapid prototyping, testing, and neural network visualization.

![Neural Nexus IDE](https://img.shields.io/badge/version-4.0.0-brightgreen) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![License](https://img.shields.io/badge/license-MIT-purple)

## 🌟 Features

### Core Capabilities
- **🎯 Instant Script Execution** - Run Python scripts with one click
- **🔧 Auto-Heal Technology** - Automatically fix errors using AI
- **📦 Smart Dependency Management** - Auto-detect and install missing packages
- **🧠 AI-Powered Analysis** - Get code improvements and suggestions
- **🌐 Multi-Script Management** - Handle multiple scripts with visual relationships
- **💻 Cross-Platform Terminal** - Support for Git Bash, WSL, CMD, and more

### Advanced Features
- **🤖 OpenAI GPT-4 Integration** - Automatic error correction and code improvements
- **🚁 GitHub Copilot Support** - Get AI suggestions while coding
- **📊 Neural Network Visualization** - See script relationships as interactive graphs
- **📝 VS Code Integration** - Open scripts directly in VS Code
- **⚡ Real-time Output Streaming** - See results as they happen
- **🎨 Modern Dark UI** - Beautiful, responsive interface

## 🚀 Quick Start

### 1. Download Files
Save these three files in the same directory:
- `neural_nexus_ide.html` - The web interface
- `neural_nexus_server.py` - The backend server
- `setup.py` - The setup script

### 2. Run Setup
```bash
python setup.py
```

The setup script will:
- Check Python version (3.8+ required)
- Install all dependencies
- Create necessary directories
- Start the server
- Open your browser

### 3. Start Using!
- Drag & drop Python files or paste code
- Click ▶️ Run to execute
- Enable Auto-Heal for automatic error fixing
- Configure OpenAI API key for AI features

## 📋 Requirements

### Minimum Requirements
- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Edge, Safari)
- 4GB RAM
- Internet connection (for AI features)

### Recommended
- Python 3.10+
- UV package manager (`pip install uv`)
- Git Bash (Windows) or Terminal (Mac/Linux)
- VS Code
- GitHub CLI (for Copilot)

## 🛠️ Installation Options

### Option 1: Automated Setup (Recommended)
```bash
python setup.py
```

### Option 2: Manual Installation
```bash
# Install core dependencies
pip install fastapi uvicorn websockets aiofiles psutil

# Install optional dependencies
pip install openai matplotlib networkx requests

# Start the server
python neural_nexus_server.py
```

### Option 3: Using UV (Fastest)
```bash
# Install UV first
pip install uv

# Install dependencies with UV
uv pip install fastapi uvicorn websockets aiofiles psutil openai

# Start server
python neural_nexus_server.py
```

## 🎮 Usage Guide

### Running Scripts
1. **New Script**: Click ➕ New Script or press Ctrl+N
2. **Upload Script**: Click 📁 Upload or drag & drop .py files
3. **Run**: Press F5 or click ▶️ Run
4. **Stop**: Press Shift+F5 or click ⏹️ Stop

### Auto-Heal Feature
1. Run a script that has errors
2. Click 🔧 Auto Heal
3. The AI will analyze and fix the errors
4. Script automatically re-runs

### AI Analysis
1. Configure OpenAI API key in Settings
2. Click 🤖 AI Analysis
3. Get suggestions for:
   - Code quality improvements
   - Performance optimizations
   - Security issues
   - Best practices

### Terminal Commands
- Supports multiple terminal types:
  - Git Bash (Windows)
  - WSL (Windows Subsystem for Linux)
  - CMD (Windows Command Prompt)
  - Bash (Mac/Linux)

### Keyboard Shortcuts
- `Ctrl+N` - New script
- `Ctrl+S` - Save scripts
- `Ctrl+Enter` - Run script
- `F5` - Run script
- `Shift+F5` - Stop script
- `Ctrl+O` - Open file

## ⚙️ Configuration

### Settings File (`config.json`)
```json
{
  "port": 8765,
  "host": "127.0.0.1",
  "auto_open_browser": true,
  "theme": "dark",
  "terminal_type": "auto"
}
```

### OpenAI Integration
1. Get API key from https://platform.openai.com
2. Click ⚙️ Settings in the IDE
3. Enter your API key
4. Save settings

### GitHub Copilot
1. Install GitHub CLI: https://cli.github.com
2. Install Copilot extension:
   ```bash
   gh extension install github/gh-copilot
   ```
3. Authenticate: `gh auth login`

## 🔥 Advanced Features

### Multi-Script Networks
- Visualize relationships between scripts
- See import dependencies
- Track execution status
- Identify bottlenecks

### Continuous Integration Mode
Enable Auto-Heal to:
- Automatically fix errors
- Re-run scripts after fixes
- Learn from error patterns
- Improve code quality

### Custom Terminals
Configure different shells:
```python
# In terminal settings
{
  "terminal_type": "gitbash",  # or "wsl", "cmd", "auto"
  "terminal_env": {
    "PYTHONPATH": "/custom/path"
  }
}
```

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check if port is in use
netstat -an | grep 8765

# Use different port
python neural_nexus_server.py --port 8080
```

### WebSocket Connection Failed
- Check firewall settings
- Ensure server is running
- Try different browser

### Dependencies Won't Install
```bash
# Update pip
python -m pip install --upgrade pip

# Use UV instead
pip install uv
uv pip install -r requirements.txt
```

## 📚 API Reference

### WebSocket Commands

#### Run Script
```javascript
{
  "action": "run",
  "scriptId": "script_123",
  "content": "print('Hello')"
}
```

#### Auto-Heal
```javascript
{
  "action": "autoHeal",
  "scriptId": "script_123",
  "content": "broken code",
  "errors": [...],
  "apiKey": "sk-..."
}
```

#### Terminal Command
```javascript
{
  "action": "terminal",
  "command": "pip list",
  "type": "auto"
}
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional AI model support (Claude, Gemini)
- More visualization options
- Plugin system
- Cloud deployment
- Mobile support

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- Built with FastAPI and modern web technologies
- UI inspired by VS Code and modern IDEs
- AI features powered by OpenAI and GitHub Copilot

## 💡 Tips & Tricks

1. **Speed Tip**: Use UV for 10x faster package installs
2. **Organization**: Group related scripts in projects
3. **Debugging**: Use the terminal for interactive debugging
4. **Performance**: Close unused scripts to free memory
5. **Shortcuts**: Learn keyboard shortcuts for efficiency

## 🚀 Future Roadmap

- [ ] Cloud sync and collaboration
- [ ] Jupyter notebook support
- [ ] Docker integration
- [ ] Real-time collaboration
- [ ] Mobile app
- [ ] Plugin marketplace
- [ ] GPU acceleration support
- [ ] Advanced debugging tools

---

**Need help?** Open an issue or contact support
**Love it?** Star the project and share with others!

Happy coding with Neural Nexus IDE! ⚡🧠