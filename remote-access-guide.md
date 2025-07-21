# Windows Remote Access from Linux - Complete Setup Guide

## Overview
This guide covers setting up secure remote access to a Windows computer from Linux across different networks using RDP (Remote Desktop Protocol) as the primary method, with SSH and VNC as alternatives.

## Installation & Setup

### Step 1: Install the Script on Linux

1. Save the script as `winremote.sh`
2. Make it executable:
```bash
chmod +x winremote.sh
```

3. Run it once to create the configuration file:
```bash
./winremote.sh
```

4. Install required dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install freerdp2-x11 openssh-client netcat-openbsd
# Optional: tigervnc-viewer for VNC support
```

**Fedora/RHEL:**
```bash
sudo dnf install freerdp openssh-clients nmap-ncat
# Optional: tigervnc for VNC support
```

**Arch Linux:**
```bash
sudo pacman -S freerdp openssh netcat
# Optional: tigervnc for VNC support
```

### Step 2: Configure Windows for Remote Access

#### Enable Remote Desktop (RDP) - Recommended

1. **Windows 10/11 Pro/Enterprise:**
   - Open Settings → System → Remote Desktop
   - Toggle "Enable Remote Desktop" to ON
   - Click "Confirm" when prompted

2. **Configure User Access:**
   - Click "Select users that can remotely access this PC"
   - Add your user account if not already listed
   - Ensure your Windows account has a strong password

3. **Note Your Connection Details:**
   - PC Name: Settings → System → About → Device name
   - IP Address: Open Command Prompt, type `ipconfig`
   - Look for IPv4 Address under your active network adapter

#### Enable SSH (Optional)

1. **Install OpenSSH Server:**
   - Settings → Apps → Optional features
   - Click "Add a feature"
   - Search for "OpenSSH Server" and install

2. **Start SSH Service:**
   - Open PowerShell as Administrator
   - Run: `Start-Service sshd`
   - Enable auto-start: `Set-Service -Name sshd -StartupType 'Automatic'`

### Step 3: Network Configuration for Cross-Network Access

#### Option A: Port Forwarding (Home/Small Office)

1. **Find Your Router's IP:**
   - Windows: `ipconfig` → Default Gateway
   - Usually 192.168.1.1 or 192.168.0.1

2. **Access Router Admin Panel:**
   - Open web browser, navigate to router IP
   - Login with admin credentials

3. **Configure Port Forwarding:**
   - Find "Port Forwarding" or "Virtual Server" section
   - Add rules:
     ```
     RDP: External Port 3389 → Internal IP:3389
     SSH: External Port 22 → Internal IP:22 (if using SSH)
     ```

4. **Get Your Public IP:**
   - Visit: https://whatismyipaddress.com
   - Or use: `curl ifconfig.me`

#### Option B: VPN (Recommended for Security)

Consider using:
- WireGuard (lightweight, fast)
- OpenVPN (widely supported)
- Tailscale (easy setup, zero-config)
- ZeroTier (peer-to-peer)

#### Option C: Dynamic DNS (For Changing IPs)

If your ISP assigns dynamic IPs:
1. Sign up for a DDNS service (No-IP, DuckDNS, etc.)
2. Configure DDNS on your router
3. Use the DDNS hostname instead of IP

### Step 4: Configure the Script

Edit `~/.winremote_config`:

```bash
# Basic configuration
WINDOWS_HOST="your-public-ip-or-ddns"  # Or local IP if same network
WINDOWS_USER="your-windows-username"
WINDOWS_DOMAIN=""  # Leave empty for local accounts

# Connection settings
USE_RDP="yes"
RDP_PORT="3389"  # Change if using non-standard port
RDP_RESOLUTION="1920x1080"  # Adjust to your preference
RDP_CLIPBOARD="yes"  # Enable clipboard sharing

# Optional methods
USE_SSH="no"  # Set to "yes" if SSH is configured
USE_VNC="no"  # Set to "yes" if VNC is installed
```

## Usage

### Basic Connection
```bash
./winremote.sh
```

### View Setup Guide
```bash
./winremote.sh help
```

## Security Best Practices

### 1. **Use Strong Passwords**
- Minimum 12 characters
- Mix of uppercase, lowercase, numbers, symbols
- Unique for each account

### 2. **Change Default Ports**
For better security, use non-standard ports:
```bash
# In router port forwarding:
External Port 50389 → Internal 3389

# In script config:
RDP_PORT="50389"
```

### 3. **Limit Access by IP**
Configure Windows Firewall to allow connections only from specific IPs

### 4. **Enable Network Level Authentication**
- Windows: System Properties → Remote → Check "Allow connections only from computers running Remote Desktop with Network Level Authentication"

### 5. **Use a VPN Instead of Direct Exposure**
This is the most secure option for cross-network access

## Troubleshooting

### Connection Refused
1. **Check Windows Firewall:**
   - Ensure Remote Desktop is allowed
   - Windows Security → Firewall & network protection → Allow an app

2. **Verify Port Forwarding:**
   - Use online port checker: https://www.yougetsignal.com/tools/open-ports/
   - Test from Linux: `nc -zv your-public-ip 3389`

3. **Check Windows Service:**
   - Services.msc → "Remote Desktop Services" should be Running

### Authentication Failures
1. Ensure using correct username format:
   - Local account: `username`
   - Domain account: `DOMAIN\username`
   - Microsoft account: `email@example.com`

2. Verify password doesn't contain special characters that need escaping

### Poor Performance
1. **Adjust Quality Settings:**
   ```bash
   RDP_COLOR_DEPTH="16"  # Reduce from 32
   RDP_RESOLUTION="1280x720"  # Lower resolution
   ```

2. **Check Network:**
   - Test latency: `ping windows-host`
   - Minimum recommended: 5 Mbps upload on Windows side

### Black Screen After Connection
- Press Ctrl+Alt+Del in the RDP session
- Disable bitmap caching if issue persists

## Advanced Configuration

### Multi-Monitor Support
```bash
# In config file:
RDP_RESOLUTION="/multimon"
# Or specific monitors:
RDP_RESOLUTION="/monitors:0,1"
```

### Share Local Drives
```bash
RDP_DRIVE_REDIRECT="yes"
# Shares your home directory with Windows
```

### Audio Redirection
```bash
RDP_SOUND="local"  # Play on Linux
RDP_SOUND="remote"  # Play on Windows
RDP_SOUND="off"  # Disable audio
```

### Custom Performance Flags
Add to the xfreerdp command in the script:
- `/gfx:avc420` - Better compression for slow connections
- `/network:modem` - Optimize for very slow connections
- `/compression-level:2` - Higher compression

## Common Use Cases

### 1. **Home Office Access**
- Set up DDNS for reliable hostname
- Use non-standard ports
- Enable clipboard and drive sharing

### 2. **Technical Support**
- Use temporary port forwarding
- Consider TeamViewer/AnyDesk as alternatives
- Always verify user identity

### 3. **System Administration**
- Prefer SSH for command-line tasks
- Use RDP for GUI requirements
- Implement jump boxes for better security

### 4. **Development Work**
- Enable drive redirection for file transfer
- Use SSH for Git operations
- Consider VS Code Remote Development

## Alternative Solutions

If this setup is too complex, consider:

1. **Commercial Solutions:**
   - TeamViewer
   - AnyDesk
   - Chrome Remote Desktop
   - LogMeIn

2. **Self-Hosted:**
   - Guacamole (web-based)
   - NoMachine
   - X2Go

3. **Cloud-Based:**
   - Windows Virtual Desktop
   - AWS WorkSpaces
   - Azure Virtual Desktop

## Maintenance

### Regular Tasks
1. Update Windows and Linux packages monthly
2. Review access logs: `~/.winremote.log`
3. Test connection after router/firewall changes
4. Rotate passwords quarterly

### Monitoring
Check logs for suspicious activity:
```bash
# On Windows (PowerShell):
Get-EventLog -LogName 'Security' -InstanceId 4624,4625 -Newest 50

# On Linux:
tail -f ~/.winremote.log
```

## Legal and Compliance Notes

- Ensure you have permission to access the target system
- Follow company policies for remote access
- Be aware of data protection regulations (GDPR, HIPAA, etc.)
- Log access for audit purposes if required

This setup provides a robust, secure method for remote Windows access from Linux across different networks. Always prioritize security and use VPN connections when possible for the best protection.