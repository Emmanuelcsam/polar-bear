# Installation Guide for UV Users

Since you're using `uv` (which is great - it's much faster than pip!), here are the specific steps to get the geometry detection system working:

## Quick Installation Steps

### Option 1: Using requirements.txt (Simplest)

1. Save the `requirements.txt` file in your project directory

2. Install all dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

### Option 2: Install packages individually

```bash
# Core packages (required)
uv pip install opencv-python opencv-contrib-python numpy psutil

# Optional packages (recommended)
uv pip install scipy matplotlib pandas pillow
```

### Option 3: Using pyproject.toml

1. Save the `pyproject.toml` file in your project directory

2. Sync dependencies:
   ```bash
   uv sync
   ```

## Troubleshooting

### If you get "No module named pip" error:

UV doesn't include pip by default. You have two options:

1. **Use uv commands directly** (recommended):
   ```bash
   uv pip install package-name
   ```

2. **Install pip in the uv environment**:
   ```bash
   uv pip install pip
   ```

### If OpenCV fails to install:

Try installing with specific versions:
```bash
# For Python 3.13, you might need newer versions
uv pip install opencv-python==4.10.0.84
uv pip install opencv-contrib-python==4.10.0.84
```

### Python 3.13 Compatibility

Since you're using Python 3.13 (very new!), some packages might not have wheels yet. If you encounter issues:

1. Try using Python 3.11 or 3.12:
   ```bash
   uv venv --python 3.11
   uv pip install -r requirements.txt
   ```

2. Or wait for package updates (OpenCV usually takes a few weeks to release wheels for new Python versions)

## Verify Installation

After installation, test everything works:

```bash
# Run the UV-compatible setup script
uv run python uv_setup.py

# Or test directly
uv run python -c "import cv2; print(f'OpenCV {cv2.__version__} installed successfully!')"

# Run the main program
uv run python integrated_geometry_system.py
```

## Complete Installation Command

For a one-liner installation of all packages:

```bash
uv pip install opencv-python opencv-contrib-python numpy psutil scipy matplotlib pandas pillow
```

## Using the System

Once installed, always use `uv run` to execute scripts:

```bash
# Main system
uv run python integrated_geometry_system.py

# Examples
uv run python shape_analysis_dashboard.py

# Calibration tool
uv run python realtime_calibration_tool.py

# Benchmark
uv run python performance_benchmark_tool.py
```

## Notes on UV

- UV is much faster than pip for dependency resolution
- It automatically manages virtual environments
- Use `uv pip list` to see installed packages
- Use `uv pip show package-name` for package details
- UV caches packages, so reinstalls are very fast

## Alternative: Use Standard pip

If you prefer traditional pip:

```bash
# Create standard venv
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install with pip
pip install -r requirements.txt
```
