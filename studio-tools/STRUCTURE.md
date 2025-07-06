# Studio Directory Structure

## Essential Files (Root Directory)
- `automated_processing_studio_v2.py` - Main application
- `image_processor_gui.py` - GUI interface
- `README.md` - Main documentation

## Organized Subdirectories

### 📁 scripts/
Original collection of 400+ image processing scripts (legacy)

### 📁 opencv_scripts/
Well-organized OpenCV scripts by category:
- edge_detection/
- effects/
- features/
- filtering/
- histogram/
- morphology/
- noise/
- thresholding/
- transformations/

### 📁 tests/
- `test_automated_studio.py`
- `test_automated_studio_v2.py`
- `test_env/` - Virtual environment for testing

### 📁 setup/
- `setup_dependencies.py`
- `setup_gui.py`
- `requirements.txt`

### 📁 demos/
- `demo_automated_studio.py`
- `demo_fixed_studio.py`
- `demo_output/` - Example outputs

### 📁 utilities/
- `generate_opencv_scripts.py`
- `merge_scripts.py`
- `enhanced_script_cleaner.py`
- `validate_studio.py`

### 📁 guides/
- Script writing guides
- Quick reference
- Examples and templates

### 📁 deprecated/
- `automated_processing_studio_v2_fixed.py`
- `.studio_cache_v2/`
- `text.txt`

## Hidden Directories
- `.claude/` - Claude-specific files
- `.studio_cache/` - Runtime cache
- `__pycache__/` - Python cache

## Quick Commands

```bash
# Run main application
python automated_processing_studio_v2.py

# Run GUI
python image_processor_gui.py

# Run tests
cd tests && python test_automated_studio_v2.py

# Run demo
cd demos && python demo_fixed_studio.py
```