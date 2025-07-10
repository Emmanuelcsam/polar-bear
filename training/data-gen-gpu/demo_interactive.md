# Interactive Mode Demo

This document shows example interactions with the GPU-accelerated fiber optic analysis pipeline in interactive mode.

## Starting the Pipeline

```
$ python app_gpu_interactive.py

============================================================
GPU-Accelerated Fiber Optic Analysis Pipeline
Interactive Mode
============================================================

============================================================
GPU Configuration
============================================================

GPU detected: NVIDIA GeForce RTX 3080
GPU Memory: 10.00 GB

Would you like to use GPU acceleration? [Y/n]: y

============================================================
Pipeline Configuration
============================================================

Would you like to load configuration from a file? [y/N]: n

Select configuration mode:
  1. Use default settings (recommended) (default)
  2. Customize settings
  3. Minimal settings (fastest processing)
Enter choice [1-3]: 1

============================================================
Fiber Optic Analysis Pipeline
============================================================

Select processing mode:
  1. Process a single image (default)
  2. Process multiple images in a directory
  3. Exit program
Enter choice [1-3]: 1
```

## Single Image Processing

```
[Single Image Processing]
Enter path to input image: ~/Documents/fiber_samples/sample_001.png

Default output directory: /home/user/Documents/fiber_samples/sample_001_analysis

Use default output directory? [Y/n]: y

----------------------------------------
Processing image...
----------------------------------------
2024-01-15 10:23:45 - [FiberAnalysisPipelineGPU] - INFO - Starting analysis of /home/user/Documents/fiber_samples/sample_001.png
2024-01-15 10:23:45 - [FiberAnalysisPipelineGPU] - INFO - Stage 1: Processing image with GPU acceleration...
2024-01-15 10:23:46 - [FiberAnalysisPipelineGPU] - INFO - Stage 1 completed in 0.82s
2024-01-15 10:23:46 - [FiberAnalysisPipelineGPU] - INFO - Stage 2: Separating fiber zones...
2024-01-15 10:23:47 - [FiberAnalysisPipelineGPU] - INFO - Stage 2 completed in 0.95s
2024-01-15 10:23:47 - [FiberAnalysisPipelineGPU] - INFO - Stage 3: Detecting defects in regions...
2024-01-15 10:23:47 - [FiberAnalysisPipelineGPU] - INFO - Stage 3 completed in 0.43s
2024-01-15 10:23:47 - [FiberAnalysisPipelineGPU] - INFO - Stage 4: Aggregating results...
2024-01-15 10:23:48 - [FiberAnalysisPipelineGPU] - INFO - Stage 4 completed in 0.21s
2024-01-15 10:23:48 - [FiberAnalysisPipelineGPU] - INFO - Analysis completed in 2.41s

============================================================
Analysis Results
============================================================
Image: /home/user/Documents/fiber_samples/sample_001.png
Quality Score: 92.3%
Status: PASS - Good quality
Total Defects: 3
Critical Defects: 0
Processing Time: 2.41s
Results saved to: /home/user/Documents/fiber_samples/sample_001_analysis

Show detailed timing information? [y/N]: y

Detailed Timing:
  stage1_process: 0.82s
  stage2_separation: 0.95s
  stage3_detection: 0.43s
  stage4_acquisition: 0.21s

Would you like to process more images? [y/N]: n
```

## Batch Processing

```
[Batch Processing]
Enter directory containing images: ~/Documents/fiber_samples

Select file pattern:
  1. PNG files only (default)
  2. JPEG files only
  3. All image files
  4. Custom pattern
Enter choice [1-4]: 1

Found 25 files matching pattern '*.png'

Default output directory: /home/user/Documents/fiber_samples/analysis_results

Use default output directory? [Y/n]: n
Enter output directory path: ~/Documents/fiber_analysis_2024_01_15

Process 25 images? [Y/n]: y

----------------------------------------
Processing batch...
----------------------------------------

Processing image 1/25: sample_001.png
  ✓ Quality: 92.3% - PASS - Good quality

Processing image 2/25: sample_002.png
  ✓ Quality: 98.7% - PASS - Perfect quality

Processing image 3/25: sample_003.png
  ✓ Quality: 75.2% - PASS - Acceptable quality

...

Processing image 25/25: sample_025.png
  ✓ Quality: 88.9% - PASS - Good quality

============================================================
Batch Processing Results
============================================================
Total images: 25
Successful: 24
Failed: 1
Average quality score: 87.4%

Results saved to: /home/user/Documents/fiber_analysis_2024_01_15

Show failed images? [Y/n]: y

Failed images:
  - /home/user/Documents/fiber_samples/sample_019.png: Failed to achieve consensus on fiber zones

Would you like to process more images? [y/N]: n

Thank you for using the Fiber Optic Analysis Pipeline!
```

## Custom Configuration

```
Select configuration mode:
  1. Use default settings (recommended)
  2. Customize settings (default)
  3. Minimal settings (fastest processing)
Enter choice [1-3]: 2

----------------------------------------
Custom Configuration
----------------------------------------

[Image Processing Settings]
Enable all image filters? (slower but more thorough) [Y/n]: y

[Defect Detection Settings]
Minimum defect size in pixels (default: 10): 15
Anomaly threshold multiplier (higher = less sensitive) (default: 2.5): 3.0
Generate visualization images? [Y/n]: y

[Quality Thresholds]
Define quality score thresholds (0-100)
Perfect quality threshold (default: 95): 95
Good quality threshold (default: 85): 85
Acceptable quality threshold (default: 70): 75

Would you like to save this configuration for future use? [y/N]: y
Enter path to save configuration (e.g., my_config.json): strict_config.json
Configuration saved to: strict_config.json
```

## Key Features of Interactive Mode

1. **No command-line arguments needed** - Everything is prompted
2. **Smart defaults** - Press Enter to accept default values
3. **Input validation** - Ensures valid paths and values
4. **Progress feedback** - Clear status updates during processing
5. **Results summary** - Easy-to-read output format
6. **Continuous processing** - Option to process more images without restarting
7. **Configuration saving** - Save custom settings for reuse

## Tips

- Press Enter to accept default values (shown in brackets)
- Use Tab completion for file paths (in supported terminals)
- Paths starting with ~ are automatically expanded
- The program remembers your GPU choice throughout the session
- Saved configurations can be loaded in future sessions