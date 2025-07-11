# Martin Project Structure Guide

## Overview

The Martin project has been reorganized into a clean, professional structure that separates concerns and improves maintainability. This document explains the new organization and how to navigate it.

## Directory Structure

```
Martin/
├── src/                           # 🚀 Production Source Code
├── data/                          # 📊 Data and Models
├── research/                      # 🔬 Research and Experiments
├── legacy/                        # 📦 Historical Code Archive
├── tools/                         # 🛠️ Development Tools
├── tests/                         # 🧪 Test Suite
├── docs/                          # 📚 Documentation
├── config/                        # ⚙️ Configuration
└── [root files]                   # 📄 Project Files
```

## Detailed Structure

### 🚀 `src/` - Production Code

The heart of the application, containing all production-ready code.

```
src/
├── core/                          # Core functionality
│   ├── app.py                    # Main application entry
│   ├── detection/                # Defect detection algorithms
│   │   └── detection.py
│   ├── separation/               # Zone separation algorithms  
│   │   ├── separation.py
│   │   └── methods/              # Individual separation methods
│   ├── processing/               # Image processing pipeline
│   │   └── process.py
│   └── utils/                    # Shared utilities
│       ├── config.py            # Configuration management
│       ├── logging.py           # Enhanced logging system
│       └── scripts/             # Utility scripts
│
├── ml/                           # Machine learning components
│   ├── models/                   # Model architectures
│   ├── training/                 # Training scripts
│   └── inference/                # Inference engines
│
├── api/                          # API interfaces
│   ├── rest/                     # REST API (future)
│   └── realtime/                 # Real-time processing
│       └── processor.py
│
└── gui/                          # GUI applications
    └── studio/                   # Image processor studio
```

### 📊 `data/` - Data Management

Centralized location for all data files.

```
data/
├── datasets/                     # Image datasets
│   ├── raw/                     # Original images
│   │   ├── clean/               # Good samples
│   │   ├── dirty/               # Defective samples
│   │   └── full_dataset/        # Complete dataset
│   ├── processed/               # Pre-processed data
│   └── augmented/               # Data augmentation results
│
├── models/                      # Trained ML models
│   ├── pytorch/                 # PyTorch models (.pth)
│   └── tensorflow/              # TensorFlow models (.h5)
│
└── cache/                       # Temporary cache files
    └── results/                 # Processing results cache
```

### 🔬 `research/` - Research & Development

Experimental code and research materials.

```
research/
├── papers/                      # Research papers (62 PDFs)
│   ├── fiber_optic_inspection/
│   ├── machine_learning/
│   └── image_processing/
│
├── notebooks/                   # Jupyter notebooks
│   └── experiments/
│
├── experiments/                 # Experimental code
│   ├── iterations/             # test3.py, test4.py, etc.
│   ├── research_focused/       # Research implementations
│   └── proposals/              # Enhancement proposals
│
└── prototypes/                 # Proof of concepts
    └── fiber_defect_inspection/
```

### 📦 `legacy/` - Historical Archive

Previous implementations preserved for reference.

```
legacy/
├── old_processes/              # All old-processes content
│   ├── processing/            # Best performing old version
│   ├── studio/                # Original GUI implementation
│   ├── C++ Method/            # C++ acceleration attempts
│   └── useful scripts/        # Historical utility scripts
│
├── deprecated/                 # Deprecated but referenced
│   └── correlation_methods/
│
└── migration_notes/           # Migration documentation
    └── v1_to_v2.md
```

### 🛠️ `tools/` - Development Tools

Scripts and tools for development and deployment.

```
tools/
├── scripts/                    # Utility scripts
│   ├── migrate_structure.py   # Project reorganization
│   ├── analysis/              # Analysis scripts
│   └── processing/            # Processing utilities
│
├── deployment/                 # Deployment configurations
│   ├── docker/                # Docker files
│   │   └── Dockerfile
│   └── hpc/                   # HPC deployment
│       └── hpc_quick_deploy.sh
│
└── benchmarks/                # Performance testing
    └── speed_test.py
```

### 🧪 `tests/` - Testing

Comprehensive test coverage.

```
tests/
├── unit/                      # Unit tests
│   ├── test_enhanced_process.py
│   ├── test_enhanced_separation.py
│   └── test_enhanced_detection.py
│
├── integration/               # Integration tests
│   └── test_integration.py
│
├── fixtures/                  # Test data
│   └── create_test_images.py
│
├── performance/               # Performance tests
│   └── test_speed.py
│
└── run_all_tests.py          # Test runner
```

### 📚 `docs/` - Documentation

All project documentation.

```
docs/
├── README.md                  # Documentation index
├── api/                       # API documentation
├── user_guide/               # End-user documentation
│   ├── getting_started.md
│   ├── configuration.md
│   └── studio/               # Studio GUI guide
├── developer/                # Developer documentation
│   ├── architecture.md       # System architecture
│   └── ml_integration.md     # ML integration guide
└── algorithms/               # Algorithm explanations
```

### ⚙️ `config/` - Configuration

Centralized configuration management.

```
config/
├── default/                  # Default configurations
│   ├── config.yaml          # Main config
│   └── calibration.json     # Camera calibration
├── environments/            # Environment-specific
│   ├── development.yaml
│   └── production.yaml
└── examples/               # Example configurations
```

## Key Files

### Root Directory Files

- `README.md` - Project overview and quick start
- `setup.py` - Python package setup
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Modern Python project config
- `pytest.ini` - Test configuration
- `.gitignore` - Git ignore rules
- `LICENSE` - Project license

### Entry Points

- **Main Application**: `src/core/app.py`
- **Real-time Processing**: `src/api/realtime/processor.py`
- **Test Runner**: `tests/run_all_tests.py`
- **Migration Tool**: `tools/scripts/migrate_structure.py`

## Navigation Tips

### For Users

1. **Start Here**: `README.md` for overview
2. **Run Application**: `python src/core/app.py`
3. **Documentation**: Check `docs/user_guide/`
4. **Configuration**: See `config/examples/`

### For Developers

1. **Architecture**: `docs/developer/architecture.md`
2. **Source Code**: Browse `src/core/` for main logic
3. **Add Features**: Extend code in appropriate `src/` subdirectory
4. **Run Tests**: `python tests/run_all_tests.py`

### For Researchers

1. **Papers**: Browse `research/papers/`
2. **Experiments**: Check `research/experiments/`
3. **Old Methods**: Reference `legacy/old_processes/`
4. **Prototypes**: See `research/prototypes/`

## Migration from Old Structure

### Finding Old Files

- `current-process/*` → `src/core/`
- `old-processes/*` → `legacy/old_processes/`
- `research_articles/*` → `research/papers/`
- `dataset/*` → `data/datasets/raw/`
- `hpc_tools/*` → `tools/deployment/hpc/`
- `tests/*` → `tests/unit/` or `tests/integration/`

### Import Changes

Old imports:
```python
from enhanced_logging import get_logger
from enhanced_process import EnhancedProcessor
```

New imports:
```python
from core.utils.logging import get_logger
from core.processing.process import EnhancedProcessor
```

## Benefits of New Structure

1. **Clear Separation**: Production, research, and legacy code are clearly separated
2. **Easy Navigation**: Logical grouping makes finding files intuitive
3. **Scalability**: Structure supports project growth
4. **CI/CD Ready**: Clean separation supports automation
5. **Professional**: Industry-standard organization
6. **Maintainable**: New developers can quickly understand the project

## Quick Commands

```bash
# Install for development
pip install -e .

# Run main application
python src/core/app.py

# Run tests
python tests/run_all_tests.py

# Update imports after reorganization
python tools/scripts/migrate_structure.py --update-imports

# Preview remaining migrations
python tools/scripts/migrate_structure.py --dry-run
```

## Next Steps

1. Complete migration using the migration script
2. Update all import statements
3. Test all functionality
4. Remove old empty directories
5. Update CI/CD pipelines for new structure

---

This reorganization transforms the Martin project into a well-structured, professional codebase ready for continued development and deployment.