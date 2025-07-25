# Refactoring Summary: Configuration Centralization

## Overview
All command-line flags, arguments, and commands have been removed from the codebase and replaced with centralized configuration through `config.yaml`.

## Changes Made

### 1. Enhanced config.yaml
Added new configuration sections:
- `system.mode`: Execution mode (train/eval/optimize)
- `system.checkpoint_path`: Path to model checkpoint
- `system.seed`: Random seed for reproducibility
- `system.verbose`: Verbose logging flag
- `webapp.*`: Web application settings (host, port, share, default_checkpoint)

### 2. Updated main.py
**Before:**
- Used argparse to handle --config, --mode, --checkpoint, --output-dir, --seed, --verbose
- Required command-line arguments for operation

**After:**
- Loads config.yaml directly
- All settings controlled through configuration file
- No command-line arguments needed

**Usage:**
```bash
# Before
python main.py --mode train --checkpoint checkpoints/best.pth --verbose

# After  
python main.py
# (Configure system.mode, system.checkpoint_path, system.verbose in config.yaml)
```

### 3. Updated app.py
**Before:**
- Used argparse for --config, --checkpoint, --host, --port, --share
- Required command-line arguments for web interface

**After:**
- Loads all settings from config.yaml
- Uses webapp.* configuration section
- No command-line arguments needed

**Usage:**
```bash
# Before
python app.py --host 0.0.0.0 --port 8080 --share

# After
python app.py
# (Configure webapp.host, webapp.port, webapp.share in config.yaml)
```

### 4. Updated base.py
**Before:**
- Had hardcoded Gradio launch settings

**After:**
- Uses webapp configuration from config.yaml
- Configurable host, port, and sharing options

### 5. Updated run_examples.sh
**Before:**
- Showed command-line argument usage
- Multiple complex command examples

**After:**
- Shows simple python commands
- Explains configuration through config.yaml
- Provides clear instructions for different modes

## Configuration Guide

### Setting Execution Mode
Edit `config.yaml`:
```yaml
system:
  mode: "train"  # Options: train, eval, optimize
  checkpoint_path: "checkpoints/best_model.pth"  # Required for eval/optimize
  verbose: true  # Enable debug logging
  seed: 42  # Random seed
```

### Web Application Settings
Edit `config.yaml`:
```yaml
webapp:
  host: "0.0.0.0"  # Host address
  port: 8080       # Port number
  share: true      # Create public link
  default_checkpoint: "best_model.pth"  # Default model to load
```

## Benefits

1. **Simplified Usage**: No need to remember complex command-line arguments
2. **Centralized Configuration**: All settings in one place
3. **Reproducibility**: Configuration files can be versioned and shared
4. **Easier Automation**: Scripts can simply modify config.yaml
5. **Reduced Errors**: No risk of typos in command-line arguments
6. **Better Documentation**: Configuration options are clearly documented in YAML

## Migration Guide

### For Training
```bash
# Old way
python main.py --mode train --checkpoint checkpoints/epoch_10.pth --verbose

# New way
# 1. Edit config.yaml:
#    system.mode: "train"
#    system.checkpoint_path: "checkpoints/epoch_10.pth"
#    system.verbose: true
# 2. Run:
python main.py
```

### For Evaluation
```bash
# Old way
python main.py --mode eval --checkpoint checkpoints/best_model.pth

# New way
# 1. Edit config.yaml:
#    system.mode: "eval"
#    system.checkpoint_path: "checkpoints/best_model.pth"
# 2. Run:
python main.py
```

### For Web Interface
```bash
# Old way
python app.py --host 0.0.0.0 --port 8080 --share

# New way
# 1. Edit config.yaml:
#    webapp.host: "0.0.0.0"
#    webapp.port: 8080
#    webapp.share: true
# 2. Run:
python app.py
```

## Files Modified
- `config.yaml` - Enhanced with new configuration sections
- `main.py` - Removed argparse, uses config-only approach
- `app.py` - Removed argparse, uses webapp config
- `base.py` - Updated Gradio launch to use config
- `run_examples.sh` - Updated with new usage examples
- `REFACTORING_SUMMARY.md` - This documentation file

All command-line arguments and flags have been successfully moved to the centralized configuration file.
