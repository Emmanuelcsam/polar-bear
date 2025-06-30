#!/usr/bin/env python3
"""
Interactive .env Setup Script
Helps create and configure the .env file
"""

import os
import sys
import subprocess
from pathlib import Path


def check_command(command):
    """Check if a command exists"""
    try:
        subprocess.run([command, '--version'], capture_output=True, check=True)
        return True
    except:
        return False


def find_chromedriver():
    """Try to find chromedriver or chromium-chromedriver"""
    possible_paths = [
        '/usr/lib/chromium-browser/chromedriver',
        '/usr/bin/chromedriver',
        '/usr/local/bin/chromedriver',
        '/snap/bin/chromium.chromedriver',
        '/usr/lib/chromium/chromedriver',
        'chromedriver',
        './chromedriver'
    ]
    
    # Try 'which' command
    try:
        result = subprocess.run(['which', 'chromedriver'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
        
    try:
        result = subprocess.run(['which', 'chromium-chromedriver'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    # Check common paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return None


def create_env_file():
    """Interactive .env file creation"""
    print("🔧 ChatGPT Analyzer - Environment Setup")
    print("="*50)
    print("\nThis will help you create a .env configuration file.\n")
    
    # Check if .env already exists
    if os.path.exists('.env'):
        response = input(".env file already exists. Overwrite? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return
            
    # Check if .env.example exists
    if not os.path.exists('.env.example'):
        print("❌ .env.example not found!")
        print("Make sure you're running this from the ChatGPT Analyzer directory.")
        return
        
    config = {}
    
    # Mode selection
    print("\n1️⃣  PRIMARY USE CASE")
    print("Which mode will you use most?")
    print("1. Export mode (analyze downloaded JSON)")
    print("2. Live mode (browse ChatGPT)")
    print("3. Both")
    
    mode = input("\nChoice (1-3, default=1): ").strip() or "1"
    
    # Export mode config
    if mode in ["1", "3"]:
        print("\n📥 EXPORT MODE CONFIGURATION")
        
        default_file = input("Default export file path (default: conversations.json): ").strip()
        if default_file:
            config['EXPORT_FILE'] = default_file
            
        output_dir = input("Output directory (default: chatgpt_analysis): ").strip()
        if output_dir:
            config['OUTPUT_DIR'] = output_dir
            
    # Live mode config
    if mode in ["2", "3"]:
        print("\n🌐 LIVE MODE CONFIGURATION")
        
        email = input("ChatGPT email: ").strip()
        if email:
            config['CHATGPT_EMAIL'] = email
            
        # Password warning
        print("\n⚠️  Password Security:")
        print("It's recommended to set passwords as environment variables instead of in .env")
        print("You can set it temporarily with: export CHATGPT_PASSWORD='your_password'")
        
        save_password = input("\nSave password in .env anyway? (y/N): ").lower()
        if save_password == 'y':
            import getpass
            password = getpass.getpass("ChatGPT password: ")
            if password:
                config['CHATGPT_PASSWORD'] = password
                
        # ChromeDriver detection
        print("\n🔍 Detecting ChromeDriver...")
        chromedriver_path = find_chromedriver()
        
        if chromedriver_path:
            print(f"✅ Found ChromeDriver at: {chromedriver_path}")
            use_this = input("Use this path? (Y/n): ").lower()
            if use_this != 'n':
                config['CHROMEDRIVER_PATH'] = chromedriver_path
        else:
            print("❌ ChromeDriver not found")
            print("\nFor Ubuntu with Chromium:")
            print("  sudo apt-get install chromium-chromedriver")
            print("\nYou can add the path manually to .env later")
            
    # Performance settings
    print("\n⚡ PERFORMANCE SETTINGS")
    advanced = input("Configure advanced settings? (y/N): ").lower()
    
    if advanced == 'y':
        workers = input("Max parallel workers (default: 8): ").strip()
        if workers and workers.isdigit():
            config['MAX_WORKERS'] = workers
            
        cache = input("Enable caching? (Y/n): ").lower()
        if cache == 'n':
            config['USE_CACHE'] = 'false'
            
    # Email notifications
    print("\n📧 EMAIL NOTIFICATIONS")
    setup_email = input("Setup email notifications? (y/N): ").lower()
    
    if setup_email == 'y':
        print("\nSMTP Configuration (Gmail example):")
        config['SMTP_SERVER'] = input("SMTP server (default: smtp.gmail.com): ").strip() or "smtp.gmail.com"
        config['SMTP_PORT'] = input("SMTP port (default: 587): ").strip() or "587"
        config['SMTP_USERNAME'] = input("SMTP username (your email): ").strip()
        
        print("\n⚠️  For Gmail, use an App Password, not your regular password")
        print("Create one at: https://myaccount.google.com/apppasswords")
        
        smtp_password = getpass.getpass("SMTP password (app password): ")
        if smtp_password:
            config['SMTP_PASSWORD'] = smtp_password
            
        config['NOTIFY_EMAIL'] = input("Send notifications to: ").strip()
        
    # Output formats
    print("\n📊 DEFAULT OUTPUT FORMATS")
    print("1. HTML only")
    print("2. HTML + CSV")
    print("3. All formats")
    
    format_choice = input("\nChoice (1-3, default=1): ").strip() or "1"
    
    if format_choice == "2":
        config['DEFAULT_FORMATS'] = "html,csv"
    elif format_choice == "3":
        config['DEFAULT_FORMATS'] = "html,csv,json,markdown,pdf"
        
    # Write .env file
    print("\n📝 Creating .env file...")
    
    with open('.env.example', 'r') as template:
        content = template.read()
        
    # Replace values in template
    for key, value in config.items():
        # Find the line with this key
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith(f'# {key}=') or line.startswith(f'{key}='):
                # Replace with actual value
                lines[i] = f'{key}={value}'
                break
                
        content = '\n'.join(lines)
        
    # Write .env file
    with open('.env', 'w') as f:
        f.write(content)
        
    print("✅ .env file created successfully!")
    
    # Check if python-dotenv is installed
    try:
        import dotenv
    except ImportError:
        print("\n⚠️  python-dotenv not installed")
        print("Install it with: pip install python-dotenv")
        
    # Summary
    print("\n📋 CONFIGURATION SUMMARY")
    print("-"*50)
    for key, value in config.items():
        if 'PASSWORD' in key:
            value = '*' * 8
        print(f"{key}: {value}")
        
    print("\n✅ Setup complete!")
    print("\nYou can now run: python gptscraper.py")
    print("The script will use your .env configuration automatically.")


if __name__ == "__main__":
    try:
        create_env_file()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
