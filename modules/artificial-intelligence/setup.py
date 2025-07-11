
from setuptools import setup, find_packages
import os

# Basic information
NAME = "artificial-intelligence"
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
