import os
from setuptools import find_packages

def generate_setup_py(project_path, project_name):
    """
    Generates a basic setup.py file for a given project.
    """
    setup_content = f"""
from setuptools import setup, find_packages
import os

# Basic information
NAME = "{project_name}"
VERSION = "0.1.0"
DESCRIPTION = "A project dynamically packaged by the Neural Framework."

# Find requirements
requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_requires = []
if os.path.exists(requirements_file):
    with open(requirements_file, 'r') as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
)
"""
    setup_file_path = os.path.join(project_path, "setup.py")
    try:
        if not os.path.exists(setup_file_path):
            with open(setup_file_path, 'w') as f:
                f.write(setup_content)
            return True
        return False # Indicates setup.py already existed
    except Exception as e:
        print(f"Error generating setup.py for {project_name}: {e}")
        return False
