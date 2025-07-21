#!/bin/bash

# Windows Remote Access Script from Linux
# Supports RDP, SSH, and VNC connections across different networks
# No argparse - uses configuration file or environment variables

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration file path
CONFIG_FILE="$HOME/.winremote_config"

# Function to display banner
show_banner() {
    echo -e "${BLUE}====================================${NC}"
    echo -e "${BLUE}Windows Remote Access Tool${NC}"
    echo -e "${BLUE}====================================${NC}"
}

# Function to create default config
create_default_config() {
    cat > "$CONFIG_FILE" << 'EOF'
# Windows Remote Access Configuration
# Edit these values for your setup

# Target Windows Machine
WINDOWS_HOST=""
WINDOWS_USER=""
WINDOWS_DOMAIN=""  # Optional, leave empty if not using domain

# Connection Methods (set to "yes" to enable)
USE_RDP="yes"
USE_SSH="no"
USE_VNC="no"

# RDP Settings
RDP_PORT="3389"
RDP_RESOLUTION="1920x1080"
RDP_COLOR_DEPTH="32"
RDP_SOUND="off"  # off, local, or remote
RDP_CLIPBOARD="yes"
RDP_DRIVE_REDIRECT="no"  # Share local drives
RDP_PRINTER_REDIRECT="no"

# SSH Settings (requires OpenSSH server on Windows)
SSH_PORT="22"
SSH_KEY_FILE=""  # Path to private key, leave empty for password auth

# VNC Settings
VNC_PORT="5900"
VNC_QUALITY="8"  # 0-9, higher is better quality

# Security Settings
USE_ENCRYPTION="yes"
VERIFY_CERTIFICATE="no"  # Set to yes for production
CONNECTION_TIMEOUT="30"

# Logging
LOG_FILE="$HOME/.winremote.log"
ENABLE_LOGGING="yes"
EOF
    echo -e "${GREEN}Created default configuration at: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}Please edit this file with your Windows machine details${NC}"
}

# Function to load configuration
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
    else
        echo -e "${YELLOW}Configuration file not found. Creating default...${NC}"
        create_default_config
        exit 1
    fi
}

# Function to validate configuration
validate_config() {
    local errors=0
    
    if [[ -z "$WINDOWS_HOST" ]]; then
        echo -e "${RED}Error: WINDOWS_HOST not set in configuration${NC}"
        errors=$((errors + 1))
    fi
    
    if [[ -z "$WINDOWS_USER" ]]; then
        echo -e "${RED}Error: WINDOWS_USER not set in configuration${NC}"
        errors=$((errors + 1))
    fi
    
    if [[ "$errors" -gt 0 ]]; then
        echo -e "${RED}Please edit $CONFIG_FILE and set the required values${NC}"
        exit 1
    fi
}

# Function to log messages
log_message() {
    if [[ "$ENABLE_LOGGING" == "yes" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    fi
}

# Function to check dependencies
check_dependencies() {
    local missing_deps=()
    
    # Check for RDP client
    if [[ "$USE_RDP" == "yes" ]]; then
        if ! command -v xfreerdp &> /dev/null && ! command -v rdesktop &> /dev/null; then
            missing_deps+=("xfreerdp or rdesktop (for RDP)")
        fi
    fi
    
    # Check for SSH client
    if [[ "$USE_SSH" == "yes" ]]; then
        if ! command -v ssh &> /dev/null; then
            missing_deps+=("openssh-client (for SSH)")
        fi
    fi
    
    # Check for VNC client
    if [[ "$USE_VNC" == "yes" ]]; then
        if ! command -v vncviewer &> /dev/null && ! command -v tigervnc &> /dev/null; then
            missing_deps+=("tigervnc or realvnc (for VNC)")
        fi
    fi
    
    # Check for network tools
    if ! command -v nc &> /dev/null && ! command -v nmap &> /dev/null; then
        missing_deps+=("netcat or nmap (for port checking)")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo -e "${RED}Missing dependencies:${NC}"
        for dep in "${missing_deps[@]}"; do
            echo -e "  - $dep"
        done
        echo -e "\n${YELLOW}Install missing dependencies:${NC}"
        echo -e "Ubuntu/Debian: sudo apt-get install freerdp2-x11 openssh-client tigervnc-viewer netcat"
        echo -e "Fedora/RHEL: sudo dnf install freerdp openssh-clients tigervnc"
        echo -e "Arch: sudo pacman -S freerdp openssh tigervnc netcat"
        exit 1
    fi
}

# Function to test port connectivity
test_port() {
    local host=$1
    local port=$2
    local service=$3
    
    echo -e "${BLUE}Testing $service connection to $host:$port...${NC}"
    
    if command -v nc &> /dev/null; then
        if nc -z -w5 "$host" "$port" 2>/dev/null; then
            echo -e "${GREEN}✓ Port $port is open${NC}"
            return 0
        else
            echo -e "${RED}✗ Port $port is closed or unreachable${NC}"
            return 1
        fi
    elif command -v nmap &> /dev/null; then
        if nmap -p "$port" "$host" 2>/dev/null | grep -q "open"; then
            echo -e "${GREEN}✓ Port $port is open${NC}"
            return 0
        else
            echo -e "${RED}✗ Port $port is closed or unreachable${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}Warning: Cannot test port connectivity${NC}"
        return 0
    fi
}

# Function to establish RDP connection
connect_rdp() {
    echo -e "${BLUE}Establishing RDP connection...${NC}"
    log_message "Attempting RDP connection to $WINDOWS_HOST"
    
    # Test port connectivity
    if ! test_port "$WINDOWS_HOST" "$RDP_PORT" "RDP"; then
        echo -e "${RED}RDP port is not accessible. Check firewall and port forwarding.${NC}"
        return 1
    fi
    
    # Build connection string
    local rdp_options=""
    local rdp_command=""
    
    # Prefer xfreerdp (FreeRDP 2.x) as it's more modern
    if command -v xfreerdp &> /dev/null; then
        rdp_command="xfreerdp"
        
        # Basic connection parameters
        rdp_options="/v:$WINDOWS_HOST:$RDP_PORT"
        rdp_options="$rdp_options /u:$WINDOWS_USER"
        
        # Add domain if specified
        if [[ -n "$WINDOWS_DOMAIN" ]]; then
            rdp_options="$rdp_options /d:$WINDOWS_DOMAIN"
        fi
        
        # Resolution
        rdp_options="$rdp_options /size:$RDP_RESOLUTION"
        
        # Color depth
        rdp_options="$rdp_options /bpp:$RDP_COLOR_DEPTH"
        
        # Sound
        case "$RDP_SOUND" in
            "local") rdp_options="$rdp_options /sound:sys:alsa" ;;
            "remote") rdp_options="$rdp_options /sound" ;;
            "off") rdp_options="$rdp_options -sound" ;;
        esac
        
        # Clipboard
        if [[ "$RDP_CLIPBOARD" == "yes" ]]; then
            rdp_options="$rdp_options +clipboard"
        fi
        
        # Drive redirect
        if [[ "$RDP_DRIVE_REDIRECT" == "yes" ]]; then
            rdp_options="$rdp_options /drive:home,$HOME"
        fi
        
        # Security options
        if [[ "$USE_ENCRYPTION" == "yes" ]]; then
            rdp_options="$rdp_options /sec:tls"
        else
            rdp_options="$rdp_options /sec:rdp"
        fi
        
        # Certificate verification
        if [[ "$VERIFY_CERTIFICATE" == "no" ]]; then
            rdp_options="$rdp_options /cert-ignore"
        fi
        
        # Performance flags for better experience over WAN
        rdp_options="$rdp_options /compression /network:auto"
        
        # Execute connection
        echo -e "${GREEN}Connecting via xfreerdp...${NC}"
        echo "Command: $rdp_command $rdp_options"
        $rdp_command $rdp_options
        
    elif command -v rdesktop &> /dev/null; then
        # Fallback to rdesktop (older but still works)
        rdp_command="rdesktop"
        
        rdp_options="-u $WINDOWS_USER"
        
        if [[ -n "$WINDOWS_DOMAIN" ]]; then
            rdp_options="$rdp_options -d $WINDOWS_DOMAIN"
        fi
        
        rdp_options="$rdp_options -g $RDP_RESOLUTION"
        rdp_options="$rdp_options -a $RDP_COLOR_DEPTH"
        
        if [[ "$RDP_CLIPBOARD" == "yes" ]]; then
            rdp_options="$rdp_options -r clipboard:PRIMARYCLIPBOARD"
        fi
        
        if [[ "$RDP_SOUND" == "local" ]]; then
            rdp_options="$rdp_options -r sound:local"
        fi
        
        echo -e "${GREEN}Connecting via rdesktop...${NC}"
        echo "Command: $rdp_command $rdp_options $WINDOWS_HOST:$RDP_PORT"
        $rdp_command $rdp_options "$WINDOWS_HOST:$RDP_PORT"
    else
        echo -e "${RED}No RDP client found${NC}"
        return 1
    fi
    
    log_message "RDP connection terminated"
}

# Function to establish SSH connection
connect_ssh() {
    echo -e "${BLUE}Establishing SSH connection...${NC}"
    log_message "Attempting SSH connection to $WINDOWS_HOST"
    
    # Test port connectivity
    if ! test_port "$WINDOWS_HOST" "$SSH_PORT" "SSH"; then
        echo -e "${RED}SSH port is not accessible. Ensure OpenSSH Server is running on Windows.${NC}"
        return 1
    fi
    
    local ssh_options="-p $SSH_PORT"
    
    # Add timeout
    ssh_options="$ssh_options -o ConnectTimeout=$CONNECTION_TIMEOUT"
    
    # Use key file if specified
    if [[ -n "$SSH_KEY_FILE" ]] && [[ -f "$SSH_KEY_FILE" ]]; then
        ssh_options="$ssh_options -i $SSH_KEY_FILE"
    fi
    
    # Build user@host string
    local ssh_target="$WINDOWS_USER@$WINDOWS_HOST"
    
    echo -e "${GREEN}Connecting via SSH...${NC}"
    echo "Command: ssh $ssh_options $ssh_target"
    ssh $ssh_options "$ssh_target"
    
    log_message "SSH connection terminated"
}

# Function to establish VNC connection
connect_vnc() {
    echo -e "${BLUE}Establishing VNC connection...${NC}"
    log_message "Attempting VNC connection to $WINDOWS_HOST"
    
    # Test port connectivity
    if ! test_port "$WINDOWS_HOST" "$VNC_PORT" "VNC"; then
        echo -e "${RED}VNC port is not accessible. Ensure VNC Server is running on Windows.${NC}"
        return 1
    fi
    
    local vnc_command=""
    local vnc_options=""
    
    if command -v vncviewer &> /dev/null; then
        vnc_command="vncviewer"
        vnc_options="$WINDOWS_HOST:$VNC_PORT"
        
        if [[ -n "$VNC_QUALITY" ]]; then
            vnc_options="-quality $VNC_QUALITY $vnc_options"
        fi
    elif command -v tigervnc &> /dev/null; then
        vnc_command="tigervnc"
        vnc_options="$WINDOWS_HOST:$VNC_PORT"
    else
        echo -e "${RED}No VNC client found${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Connecting via VNC...${NC}"
    echo "Command: $vnc_command $vnc_options"
    $vnc_command $vnc_options
    
    log_message "VNC connection terminated"
}

# Function to display quick setup guide
show_setup_guide() {
    echo -e "\n${BLUE}=== Quick Setup Guide ===${NC}"
    echo -e "\n${YELLOW}On your Windows machine:${NC}"
    echo -e "1. ${GREEN}For RDP:${NC}"
    echo -e "   - Enable Remote Desktop: Settings > System > Remote Desktop"
    echo -e "   - Note your PC name or IP address"
    echo -e "   - Ensure user has remote access permissions"
    echo -e "\n2. ${GREEN}For SSH (optional):${NC}"
    echo -e "   - Install OpenSSH Server: Settings > Apps > Optional Features"
    echo -e "   - Start SSH service: PowerShell (Admin) > Start-Service sshd"
    echo -e "\n3. ${GREEN}For VNC (optional):${NC}"
    echo -e "   - Install VNC Server (TightVNC, RealVNC, etc.)"
    echo -e "   - Configure password and start service"
    
    echo -e "\n${YELLOW}Network Configuration:${NC}"
    echo -e "1. ${GREEN}If on same network:${NC} Use local IP address"
    echo -e "2. ${GREEN}If on different networks:${NC}"
    echo -e "   - Configure port forwarding on router:"
    echo -e "     * RDP: Forward external port to internal_ip:3389"
    echo -e "     * SSH: Forward external port to internal_ip:22"
    echo -e "     * VNC: Forward external port to internal_ip:5900"
    echo -e "   - Use public IP or dynamic DNS hostname"
    echo -e "   - Consider using VPN for better security"
    
    echo -e "\n${YELLOW}Security Recommendations:${NC}"
    echo -e "- Use strong passwords"
    echo -e "- Enable Network Level Authentication for RDP"
    echo -e "- Use non-standard ports for public exposure"
    echo -e "- Consider VPN instead of direct port forwarding"
    echo -e "- Enable Windows Firewall with specific rules"
}

# Main execution
main() {
    show_banner
    
    # Check if asking for help
    if [[ "$1" == "help" ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_setup_guide
        exit 0
    fi
    
    # Load and validate configuration
    load_config
    validate_config
    
    # Check dependencies
    check_dependencies
    
    # Display connection info
    echo -e "\n${BLUE}Target System:${NC} $WINDOWS_HOST"
    echo -e "${BLUE}User:${NC} $WINDOWS_USER"
    if [[ -n "$WINDOWS_DOMAIN" ]]; then
        echo -e "${BLUE}Domain:${NC} $WINDOWS_DOMAIN"
    fi
    
    # Try connection methods in order of preference
    connection_established=false
    
    if [[ "$USE_RDP" == "yes" ]]; then
        if connect_rdp; then
            connection_established=true
        fi
    fi
    
    if [[ "$connection_established" == false ]] && [[ "$USE_SSH" == "yes" ]]; then
        if connect_ssh; then
            connection_established=true
        fi
    fi
    
    if [[ "$connection_established" == false ]] && [[ "$USE_VNC" == "yes" ]]; then
        if connect_vnc; then
            connection_established=true
        fi
    fi
    
    if [[ "$connection_established" == false ]]; then
        echo -e "\n${RED}Failed to establish connection${NC}"
        echo -e "${YELLOW}Troubleshooting tips:${NC}"
        echo -e "1. Verify Windows machine is powered on and connected to network"
        echo -e "2. Check firewall settings on both machines"
        echo -e "3. Verify port forwarding if connecting across networks"
        echo -e "4. Run './$(basename $0) help' for setup guide"
        exit 1
    fi
}

# Run main function
main "$@"