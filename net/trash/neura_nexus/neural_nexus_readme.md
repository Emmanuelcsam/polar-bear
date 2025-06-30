# Neural Nexus IDE v3.0

An advanced IDE specifically designed for rapid development and testing of neural network and AI scripts with automatic error correction, visualization, and seamless AI integration.

![Neural Nexus IDE](banner.png)

## 🚀 Features

### Core Features
- **Multi-Script Management**: Handle multiple interconnected scripts as a neural network
- **Real-time Error Detection**: Syntax and dependency checking as you type
- **Automatic Error Correction**: AI-powered auto-healing for both syntax and runtime errors
- **Dependency Visualization**: See how your scripts connect and interact
- **Integrated Terminal**: Execute commands with Git Bash, WSL, or native terminals
- **Smart Dependency Installation**: Automatic detection and installation of missing packages

### AI Integration
- **OpenAI GPT-4 Integration**: Get code suggestions and automatic fixes
- **GitHub Copilot Support**: Direct integration with GitHub Copilot CLI
- **Continuous Auto-Heal**: Automatically fix errors and re-run until success
- **Smart Code Analysis**: AI-powered code quality suggestions

### Advanced Features
- **Neural Network Visualization**: See your scripts as an interactive graph
- **Performance Monitoring**: Track script execution and resource usage
- **VS Code Integration**: Open scripts directly in VS Code
- **Cross-platform Terminal**: Support for Windows Git Bash, WSL, and Linux terminals
- **Package Management**: Built-in support for `uv` and `pip`

## 📋 Requirements

- Python 3.8 or higher
- Windows, macOS, or Linux
- Optional: Git, VS Code, GitHub CLI

## 🛠️ Installation

1. **Download the files**:
   - `neural_nexus_ide.py` - Main application
   - `setup.py` - Setup script

2. **Run the setup**:
   ```bash
   python setup.py
   ```

3. **Launch the IDE**:
   ```bash
   python neural_nexus_ide.py
   ```

## 🎯 Quick Start

### Basic Workflow

1. **Create/Open a Script**:
   - Click ➕ or press `Ctrl+N` for new script
   - Click 📁 or press `Ctrl+O` to open existing script

2. **Write Your Code**:
   - The editor provides syntax highlighting
   - Line numbers help navigate your code
   - Auto-save keeps your work safe

3. **Run Your Script**:
   - Click ▶️ or press `F5` to run
   - Output appears in the Output tab
   - Errors are highlighted and cataloged

4. **Fix Errors Automatically**:
   - Click 🔧 to auto-fix detected errors
   - Click 🔄 to enable auto-heal mode
   - The IDE will continuously fix and retry

### Using AI Features

1. **Setup OpenAI** (optional):
   - Go to Settings tab
   - Enter your OpenAI API key
   - Click Save

2. **Get AI Suggestions**:
   - Click 💡 or press `Ctrl+Shift+I`
   - AI analyzes your code and suggests improvements

3. **Use GitHub Copilot**:
   - Install GitHub CLI: `gh extension install github/gh-copilot`
   - Click 🚁 to get Copilot suggestions

## 🎮 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New script |
| `Ctrl+O` | Open script |
| `Ctrl+S` | Save script |
| `Ctrl+Shift+S` | Save as |
| `F5` | Run script |
| `Shift+F5` | Stop script |
| `Ctrl+I` | Install dependencies |
| `Ctrl+Shift+I` | AI suggestions |
| `Ctrl+Return` | Quick run |

## 🔧 Terminal Commands

The integrated terminal supports various commands:

### Package Management
```bash
# Using uv (if installed)
uv pip install numpy pandas
uv run python script.py

# Using pip
pip install tensorflow
python script.py

# For shell scripts
chmod +x script.sh
./script.sh
```

### Terminal Types
- **Auto**: Automatically selects best terminal
- **Git Bash**: Uses Git Bash on Windows
- **WSL**: Uses Windows Subsystem for Linux
- **CMD**: Uses native command prompt

## 🧠 Auto-Heal Mode

Auto-heal continuously fixes errors until your script runs successfully:

1. Enable with 🔄 button
2. Run your script
3. If errors occur, AI automatically:
   - Analyzes the error
   - Generates a fix
   - Applies the fix
   - Runs again

This continues until the script succeeds or max attempts reached.

## 📊 Script Dependencies

The IDE automatically:
- Detects imports in your code
- Identifies missing packages
- Shows dependency relationships
- Installs missing packages with one click

## 🌐 Network Visualization

See how your scripts connect:
- Each script is a node
- Imports create connections
- Colors indicate status:
  - Gray: Idle
  - Yellow: Running
  - Green: Success
  - Red: Error

## ⚙️ Configuration

Settings are saved in `~/.neural_nexus/config.json`:

```json
{
  "openai_api_key": "your-key-here",
  "auto_save": true,
  "auto_analyze": true,
  "theme": "dark"
}
```

## 🐛 Troubleshooting

### Missing Dependencies
If packages fail to install:
```bash
# Try manual installation
pip install psutil networkx matplotlib openai
```

### GitHub Copilot Not Working
Install GitHub CLI and Copilot:
```bash
# Install GitHub CLI (varies by OS)
# Windows: winget install GitHub.cli
# Mac: brew install gh
# Linux: See GitHub docs

# Install Copilot extension
gh auth login
gh extension install github/gh-copilot
```

### Terminal Issues on Windows
- Ensure Git is installed for Git Bash
- Enable WSL for WSL support
- Use CMD as fallback

## 📝 Example Workflow

1. **Create a Neural Network Script**:
   ```python
   import torch
   import torch.nn as nn
   
   class SimpleNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(784, 128)
           self.fc2 = nn.Linear(128, 10)
   ```

2. **Run and See Error**:
   - Missing `torch` dependency detected
   - Click "Install Dependencies"

3. **Enable Auto-Heal**:
   - Any runtime errors are automatically fixed
   - Script re-runs until successful

4. **Create Connected Scripts**:
   - Import your model in another script
   - See the connection in Network View

## 🎨 Themes

Switch between Dark and Light themes in Settings:
- Dark: Optimized for long coding sessions
- Light: Better for bright environments

## 🔐 Privacy & Security

- API keys are stored locally in your home directory
- Scripts are saved locally unless you specify otherwise
- No data is sent anywhere without your explicit action

## 🤝 Contributing

This is an open-source project. Feel free to:
- Report issues
- Suggest features
- Submit pull requests

## 📜 License

MIT License - Feel free to use and modify as needed.

## 🙏 Acknowledgments

Built with:
- Python tkinter for the UI
- NetworkX for graph visualization
- OpenAI API for AI features
- GitHub Copilot for enhanced suggestions

---

**Happy Coding with Neural Nexus IDE!** 🚀🧠✨
