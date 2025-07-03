"""
Setup script for Martin Fiber Optic Defect Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="martin-fiber-defect-detection",
    version="2.0.0",
    author="Martin Project Team",
    description="Advanced fiber optic defect detection system with ML integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/martin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "pyyaml>=5.4",
        "psutil>=5.8.0",
        "matplotlib>=3.3.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "ml": ["torch>=1.9.0", "torchvision>=0.10.0", "tensorflow>=2.6.0"],
        "dev": ["pytest>=6.2.0", "pytest-cov>=2.12.0", "black>=21.0", "flake8>=3.9.0"],
        "docs": ["sphinx>=4.0.0", "sphinx-rtd-theme>=0.5.0"],
    },
    entry_points={
        "console_scripts": [
            "martin-detect=core.app:main",
            "martin-realtime=api.realtime.processor:main",
        ],
    },
)