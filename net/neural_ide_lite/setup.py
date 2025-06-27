import subprocess
import sys

def install(package):
    """
    Installs a given package using pip.
    This function is a simple wrapper around the pip install command.

    Args:
        package (str): The name of the package to install.
    """
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}. Pip returned a non-zero exit code.")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred while installing {package}.")
        print(f"   Error: {e}")

def main():
    """
    Main function to set up the environment for Neural Weaver.
    It installs all required and optional dependencies.
    """
    print("--- Starting Neural Weaver Setup ---")

    # A comprehensive list of packages needed for full functionality.
    # We've added ttkthemes for styling and Pillow for better image support in the UI.
    all_packages = [
        # Core UI and Logic
        "ttkthemes",
        "Pillow",
        "networkx",
        "matplotlib",
        "psutil",
        "numpy",
        "pyyaml",
        # Optional AI & Analysis Features
        "openai",
        "requests",
        "beautifulsoup4",
        "pylint",
    ]

    print("\nThis script will install all necessary packages for Neural Weaver.")
    
    # Iterate through the list and install each package
    for package in all_packages:
        install(package)
    
    print("\n--- Setup Complete ---")
    print("\nFor AI features using GitHub Copilot (if you choose to enable it):")
    print("1. Ensure you have the GitHub CLI installed ('gh').")
    print("2. Authenticate by running: gh auth login")
    print("3. Install the Copilot extension: gh extension install github/gh-copilot")
    print("\nFor AI features using OpenAI:")
    print("1. You will need an API key from https://platform.openai.com/")
    print("2. The application will prompt you to enter this key when needed.")
    
    print("\nYou can now run the application using 'launch_neural_weaver.py'.")

if __name__ == "__main__":
    main()
