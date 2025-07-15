# Project Organization Guide

This project has been reorganized into a logical folder structure based on functionality. Here's a breakdown of each directory:

## Core Components

### `/core`
- **config/** - Configuration management files (config managers, system configs)
- **setup/** - Setup and installation scripts
- **connectors/** - Connector and hivemind communication modules
- **base/** - Base initialization files (__init__.py files)

## Processing Workflows

### `/workflows`
- **main/** - Main numbered workflow files (0-10) that form the primary processing pipeline
- **demo/** - Demo and example files for testing and showcasing functionality

## AI & Machine Learning

### `/ai_ml`
- **neural_networks/** - Deep learning models (VAE, CNN, transformers)
- **computer_vision/** - Computer vision algorithms and tools
- **machine_learning/** - Traditional ML models and classifiers
- **training/** - Training scripts and utilities

## Image Processing

### `/image_processing`
- **preprocessing/** - Image preparation, noise reduction, enhancement
- **analysis/** - Feature extraction, statistical analysis
- **generation/** - Image generation and synthesis
- **visualization/** - Display and visualization tools

## Defect Detection

### `/defect_detection`
- **algorithms/** - Various defect detection algorithms (DO2MR, LEI, scratch detection)
- **models/** - Trained models for defect classification
- **tools/** - Supporting tools for defect analysis

## Fiber Optic Analysis

### `/fiber_analysis`
- **detection/** - Fiber core and cladding detection
- **segmentation/** - Fiber segmentation algorithms
- **measurement/** - Measurement and analysis tools

## Data Processing

### `/data_processing`
- **batch/** - Batch processing utilities
- **realtime/** - Real-time processing modules
- **gpu/** - GPU-accelerated processing
- **hpc/** - High-performance computing and parallel processing

## Support Utilities

### `/utilities`
- **helpers/** - General helper functions and utilities
- **monitoring/** - System monitoring tools
- **reporting/** - Report generation and data export

## Other Directories

### `/tests`
Test files and testing utilities

### `/integration`
- **api/** - API endpoints and interfaces
- **microservices/** - Microservice components
- **web/** - Web interfaces and socket servers

### `/docs`
Documentation, README files, and guides

### `/scripts`
Standalone utility scripts for various image processing tasks

### `/experimental`
Files that haven't been categorized yet or are experimental

## Key Files Locations

- Main configuration: `/core/config/0_config.py`
- Setup script: `/core/setup/setup.py`
- Primary workflow starts at: `/workflows/main/1_setup_directories.py`
- Main connector: `/core/connectors/connector.py`

## Usage

1. Start with the configuration files in `/core/config/`
2. Run setup scripts from `/core/setup/`
3. Follow the numbered workflow in `/workflows/main/`
4. Use specific modules from other directories as needed

## Note

Some files may still be in the `/experimental` directory pending further classification.