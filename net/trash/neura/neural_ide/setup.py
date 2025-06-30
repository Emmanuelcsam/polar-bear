import subprocess
import sys

def install(package):
    """Installs a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    """
    Main function to install all required and optional dependencies.
    """
    print("Starting setup...")
    
    # List of required packages
    required_packages = [
        "pylint",
        "flake8",
        "mypy",
        "bandit",
        "radon",
        "vulture",
        "openai",
        "networkx",
        "matplotlib",
        "psutil",
        "numpy",
        "pyyaml",
        "websockets",
        "beautifulsoup4",
        "requests"
    ]

    print("Installing required packages...")
    for package in required_packages:
        try:
            print(f"Installing {package}...")
            install(package)
            print(f"Successfully installed {package}")
        except Exception as e:
            print(f"Failed to install {package}. Error: {e}")
            
    print("\nSetup complete. Please ensure you have the GitHub CLI installed and authenticated for Copilot features.")
    print("You can do this by running 'gh auth login' and 'gh extension install github/gh-copilot'.")

if __name__ == "__main__":
    main()
