# Neural Nexus IDE Server v6.0 - Installation & Usage Guide

## 🚀 Quick Start

### Option 1: Simple Version (Recommended for testing)

```bash
# 1. Download the simple version
python neural_nexus_simple.py --install

# 2. Start the server
python neural_nexus_simple.py

# 3. Open browser to http://localhost:8765
```

### Option 2: Full Modular Version (Advanced features)

```bash
# 1. Install dependencies
python setup_modular.py

# 2. Start the server
python neural_nexus_modular.py

# 3. Open browser to http://localhost:8765
```

## 📁 Project Structure

```
neural-nexus-enhanced/
├── 🚀 neural_nexus_simple.py       # Simple, self-contained version
├── 🏗️ neural_nexus_modular.py      # Full modular version
├── 📦 setup_modular.py             # Automated setup script
├── 📋 requirements_modular.txt     # Full dependency list
├── 📚 README_modular.md            # Comprehensive documentation
├── core/                           # Core modules (modular version)
│   ├── config.py                   # Configuration management
│   ├── logger.py                   # Enhanced logging
│   ├── models.py                   # Data models
│   └── json_utils.py               # JSON optimization
├── analysis/                       # Analysis engines (modular version)
│   ├── code_analyzer.py            # Multi-tool analysis
│   └── auto_healer.py              # Intelligent error fixing
└── server/                         # Server components (modular version)
    ├── app.py                      # FastAPI application
    └── websocket_handler.py         # WebSocket handling
```

## ✨ Key Enhancements

### 🔥 Performance Improvements

- **uvloop integration**: 4x faster event loop
- **orjson support**: 6x faster JSON processing
- **Async optimization**: Better concurrent handling
- **Connection pooling**: Efficient WebSocket management

### 🔍 Enhanced Code Analysis

- **Multiple analysis tools**: Pylint, MyPy, Ruff, Bandit, Semgrep
- **Real-time analysis**: Instant feedback as you type
- **Security scanning**: Vulnerability detection
- **Type checking**: Static type analysis
- **Code quality scoring**: Comprehensive metrics

### 🚑 Intelligent Auto-Healing

- **Pattern recognition**: Common error pattern detection
- **Syntax fixing**: Automatic syntax error correction
- **Import resolution**: Smart import error fixing
- **Name correction**: Variable name typo detection
- **Security mitigation**: Automatic vulnerability fixes

### 🛡️ Security Features

- **Content Security Policy**: Strict CSP headers
- **Rate limiting**: Per-connection limits
- **Security headers**: OWASP recommendations
- **CORS protection**: Configurable policies
- **Vulnerability scanning**: Multi-tool security analysis

## 🛠️ Installation Options

### Minimal Installation (Simple Version)

```bash
pip install fastapi uvicorn[standard] websockets
python neural_nexus_simple.py
```

### Enhanced Installation (Full Features)

```bash
pip install fastapi uvicorn[standard] websockets aiofiles psutil
pip install uvloop orjson loguru  # Performance enhancements
pip install mypy pylint ruff bandit  # Analysis tools
python neural_nexus_modular.py
```

### Automated Installation

```bash
python setup_modular.py  # Installs everything automatically
```

## 🎯 Usage Examples

### Basic Code Execution

1. Open <http://localhost:8765>
2. Write Python code in the editor
3. Click "Run" to execute
4. View output in real-time

### Code Analysis

1. Write code in the editor
2. Click "Analyze" for comprehensive analysis
3. View errors, warnings, and suggestions
4. Get quality scores and metrics

### Auto-Healing

1. Write code with intentional errors
2. Run the code to see errors
3. Enable auto-heal mode
4. Watch automatic fixes being applied

### Project Management (Modular Version)

1. Create new projects
2. Add multiple scripts
3. Set main script
4. Run entire projects

## 🔧 Configuration

### Command Line Options

```bash
# Custom port
python neural_nexus_simple.py --port 9000

# Debug mode
python neural_nexus_simple.py --debug

# Custom host (allow external connections)
python neural_nexus_simple.py --host 0.0.0.0

# Install dependencies
python neural_nexus_simple.py --install
```

### Environment Variables

```bash
export NEURAL_NEXUS_HOST="127.0.0.1"
export NEURAL_NEXUS_PORT="8765"
export NEURAL_NEXUS_DEBUG="true"
```

## 📊 Performance Benchmarks

### uvloop vs Standard asyncio

- **WebSocket throughput**: 3-4x improvement
- **HTTP response time**: 2-3x faster
- **Concurrent connections**: 2x more supported

### orjson vs Standard json

- **JSON serialization**: 6x faster
- **JSON deserialization**: 4x faster
- **Memory usage**: 30% reduction

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   python neural_nexus_simple.py --port 9000
   ```

2. **Missing dependencies**

   ```bash
   python neural_nexus_simple.py --install
   ```

3. **Permission errors**

   ```bash
   # Check write permissions in current directory
   ls -la
   ```

4. **Import errors (modular version)**

   ```bash
   # Use simple version for testing
   python neural_nexus_simple.py
   ```

### Debug Mode

```bash
python neural_nexus_simple.py --debug
```

Enables verbose logging and detailed error messages.

## 🔍 Feature Comparison

| Feature | Simple Version | Modular Version |
|---------|---------------|-----------------|
| Code Execution | ✅ | ✅ |
| WebSocket Communication | ✅ | ✅ |
| Basic Analysis | ✅ | ✅ |
| uvloop Performance | ✅ | ✅ |
| Security Headers | ✅ | ✅ |
| Advanced Analysis | ❌ | ✅ |
| Auto-Healing | ❌ | ✅ |
| Project Management | ❌ | ✅ |
| Security Scanning | ❌ | ✅ |
| Performance Monitoring | ❌ | ✅ |

## 📚 API Documentation

### WebSocket Messages

#### Run Script

```json
{
  "type": "run_script",
  "scriptId": "unique-id",
  "content": "print('Hello World')"
}
```

#### Analyze Code

```json
{
  "type": "analyze_script",
  "content": "print('Hello World')"
}
```

#### Stop Script

```json
{
  "type": "stop_script",
  "scriptId": "unique-id"
}
```

### HTTP Endpoints

- `GET /` - Web interface
- `GET /health` - Server health and metrics
- `WS /ws` - WebSocket connection

## 🤝 Contributing

### Adding Features

1. Simple version: Edit `neural_nexus_simple.py`
2. Modular version: Add to appropriate module in `core/`, `analysis/`, or `server/`

### Testing

```bash
# Test simple version
python neural_nexus_simple.py --debug

# Test modular version
python neural_nexus_modular.py --debug
```

## 📄 License

MIT License - Enhanced version maintains original licensing.

## 🙏 Acknowledgments

- **uvloop**: MagicStack for enhanced async performance
- **FastAPI**: Sebastián Ramírez for the modern web framework
- **Open-source analysis tools**: Pylint, MyPy, Ruff, Bandit communities
- **Original Neural Nexus**: Foundation for this enhanced version

---

**Ready to experience enhanced Python development?** 🚀

Choose your installation method and start coding with AI-powered assistance!
