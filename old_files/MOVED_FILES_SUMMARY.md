# Summary of Moved Files

This directory contains all the original files from the Martin project that have been reorganized into the new structure. Here's what was moved and why:

## 📁 old_files/current-process/

### Original Implementation Files (Replaced by Enhanced Versions)
- `app.py` → Replaced by enhanced version in `src/core/app.py`
- `process.py` → Replaced by enhanced version in `src/core/processing/process.py`
- `separation.py` → Replaced by enhanced version in `src/core/separation/separation.py`
- `detection.py` → Replaced by enhanced version in `src/core/detection/detection.py`
- `data_acquisition.py` → Functionality integrated into enhanced modules

### Enhanced Versions (Duplicates)
- `enhanced_*.py` files → Already copied to appropriate locations in `src/`
- `config_manager.py` → Already in `src/core/utils/config.py`
- `realtime_processor.py` → Already in `src/api/realtime/processor.py`

### Support Files
- `debug_utils.py` → Functionality in `src/core/utils/logging.py`
- `create_test_images.py` → Can be moved to `tests/fixtures/` if needed
- `test_pipeline.py`, `integration_test.py` → Testing files

### Directories
- `zone_methods/` → Already copied to `src/core/separation/methods/`
- `legacy_backup-*/` → Historical backups
- `results/` → Old results directory
- `utility_scripts/` → Various utility scripts

## 📁 old_files/old-processes/

All historical implementations including:
- Multiple versions of detection and separation algorithms
- Studio GUI application
- C++ implementations
- Research-focused methods
- Test iterations (test3.py through test8.12)

## 📁 old_files/research_articles/

62 PDF research papers → Should be copied to `research/papers/` if you want to keep them

## 📁 old_files/dataset/

Original image datasets → Should be copied to `data/datasets/raw/` if you want to use them

## 📁 old_files/potential_upgrades/

Enhancement proposals and documentation → Relevant files can be copied to `research/experiments/proposals/`

## 📁 old_files/hpc_tools/

HPC deployment scripts → Can be copied to `tools/deployment/hpc/` if needed

## 📁 old_files/root/

Files that were in the project root:
- `debug_utils.py` → Functionality replaced by enhanced logging

## Original Test Files

- `test_data_acquisition.py`
- `test_debug_utils.py`
- `test_detection_misc.py`
- `test_detection_utils.py`
- `test_process.py`

These are the original unit tests for the non-enhanced versions.

---

## What You Have Now:

1. **Clean new structure** in the main project directories with enhanced versions
2. **All original files preserved** in `old_files/` for reference
3. **No confusion** between old and new implementations

## Next Steps:

If you want to use any of the old files:

1. **Research papers**: Copy from `old_files/research_articles/` to `research/papers/`
2. **Datasets**: Copy from `old_files/dataset/` to `data/datasets/raw/`
3. **Specific algorithms**: Extract from `old_files/old-processes/` as needed
4. **HPC tools**: Copy from `old_files/hpc_tools/` to `tools/deployment/hpc/`

The enhanced versions in the new structure are improved versions of your original code with:
- Machine learning integration
- Better performance
- No argparse dependency
- Full logging
- Real-time capabilities