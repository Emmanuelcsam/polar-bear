# VS Code Copilot Auto-Continue Script Setup Guide

## Prerequisites

### 1. Install Python Dependencies
```bash
pip install opencv-python pyautogui pytesseract python-mss pillow numpy
```

### 2. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Platform-Specific Permissions

**macOS:**
- Go to System Preferences → Security & Privacy → Privacy
- Enable "Screen Recording" for Terminal/Python
- Enable "Accessibility" for Terminal/Python

**Linux:**
- Install additional dependencies:
```bash
sudo apt-get install python3-tk python3-dev scrot
```

**Windows:**
- Run as Administrator for first time
- Windows Defender may flag it - add exception

## Installation

1. Save the main script as `vscode_auto_clicker.py`
2. Make it executable:
```bash
chmod +x vscode_auto_clicker.py
```

## Usage

### Basic Usage
Start the auto-clicker:
```bash
python vscode_auto_clicker.py
```

### Test Mode
Test detection without clicking:
```bash
python vscode_auto_clicker.py --test
```

### Save Templates
Save button templates for better detection:
```bash
python vscode_auto_clicker.py --save-template
```

## How It Works

The script uses three detection methods in order:

1. **Template Matching**: Fastest and most accurate if templates exist
2. **Color Detection**: Looks for blue buttons (typical Continue button color)
3. **OCR Detection**: Reads text to find "Continue" buttons

## Safety Features

- **Failsafe**: Move mouse to top-left corner to stop
- **Click Limiting**: Max 30 clicks per minute
- **Position Cooldown**: 2-second cooldown per position
- **Logging**: All actions logged to `~/.vscode-auto-clicker/`

## Configuration

Edit the `Config` class in the script to adjust:
- `SCAN_INTERVAL`: How often to check (default: 0.5 seconds)
- `CONFIDENCE_THRESHOLD`: Detection confidence (default: 0.8)
- `CLICK_COOLDOWN`: Time between clicks (default: 2 seconds)

## Troubleshooting

### Button Not Detected
1. Run test mode to capture screenshots
2. Check logs in `~/.vscode-auto-clicker/`
3. Try different VS Code themes (dark theme works best)
4. Save manual templates using `--save-template`

### OCR Not Working
- Ensure Tesseract is installed and in PATH
- Try adjusting `OCR_CONFIDENCE_THRESHOLD`

### Performance Issues
- Increase `SCAN_INTERVAL` to reduce CPU usage
- Disable `MULTI_MONITOR_SUPPORT` if using single monitor

## Advanced Usage

### Running as Service (Linux)
Create systemd service file `/etc/systemd/system/vscode-auto-clicker.service`:
```ini
[Unit]
Description=VS Code Auto Clicker
After=graphical.target

[Service]
Type=simple
User=YOUR_USERNAME
Environment="DISPLAY=:0"
ExecStart=/usr/bin/python3 /path/to/vscode_auto_clicker.py
Restart=on-failure

[Install]
WantedBy=default.target
```

Enable and start:
```bash
sudo systemctl enable vscode-auto-clicker
sudo systemctl start vscode-auto-clicker
```

### Running on Startup (macOS)
Create `~/Library/LaunchAgents/com.vscode.autoclicker.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.vscode.autoclicker</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/path/to/vscode_auto_clicker.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.vscode.autoclicker.plist
```

## Monitoring

View logs:
```bash
tail -f ~/.vscode-auto-clicker/*.log
```

## Stopping the Script

- **Method 1**: Press Ctrl+C in terminal
- **Method 2**: Move mouse to top-left corner (failsafe)
- **Method 3**: Kill the process

## Important Notes

- The script only clicks "Continue" buttons in VS Code
- It respects click rate limits to avoid issues
- All actions are logged for debugging
- Templates improve accuracy significantly

## Tips for Best Results

1. Use a consistent VS Code theme
2. Keep dialog boxes in similar screen positions
3. Save templates when detection works well
4. Monitor logs for any issues
5. Adjust configuration based on your system performance