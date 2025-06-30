#!/usr/bin/env python3
"""
Fix OpenAI API version for Neural Nexus IDE
"""

import subprocess
import sys

def main():
    print("🔧 Fixing OpenAI API version for Neural Nexus IDE\n")
    
    # Check current OpenAI version
    try:
        import openai
        version = openai.__version__
        print(f"Current OpenAI version: {version}")
        
        if version.startswith("0."):
            print("✅ You have the old API version")
            response = input("\nUpgrade to new OpenAI API (recommended)? (y/n): ")
            if response.lower() == 'y':
                print("Upgrading OpenAI...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "openai"])
                print("✅ OpenAI upgraded successfully!")
                print("\n⚠️  Make sure to update your Neural Nexus server with the latest code!")
        else:
            print("✅ You already have the new OpenAI API")
            
    except ImportError:
        print("OpenAI not installed")
        response = input("\nInstall OpenAI API? (y/n): ")
        if response.lower() == 'y':
            print("Installing OpenAI...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
            print("✅ OpenAI installed successfully!")
    
    print("\n📝 OpenAI API Key Instructions:")
    print("1. Get your API key from: https://platform.openai.com/api-keys")
    print("2. In Neural Nexus IDE, click Settings (⚙️)")
    print("3. Enter your API key and save")
    print("\n✨ You're all set!")

if __name__ == "__main__":
    main()
