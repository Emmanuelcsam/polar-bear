
QUICK START GUIDE - Fiber Optic Defect Detection Neural Network
================================================================

1. SETUP DIRECTORIES:
   project-directory/
   ├── dataset/chunk_1/, chunk_2/, ..., chunk_135/  (your images here)
   └── reference/subfolder1/, subfolder2/, ... subfolder40/  (.pt files here)

2. INSTALL REQUIREMENTS:
   pip install torch torchvision opencv-python numpy matplotlib tqdm

3. RUN THE SYSTEM:
   python fiber_optic_defect_detection.py

4. OUTPUTS:
   - defect_analysis_report.json  (detailed statistics)
   - Console output with training progress
   - Model performance metrics

5. KEY FEATURES:
   ✓ Multi-task learning (region + defect classification)
   ✓ OpenCV integration for traditional CV
   ✓ Based entirely on D2L examples and tutorials
   ✓ Automatic GPU detection and usage
   ✓ Comprehensive statistics and reporting
   ✓ Compatible with your exact directory structure

6. CUSTOMIZATION:
   - Modify network architecture in FiberOpticCNN class
   - Adjust training parameters in FiberOpticTrainer
   - Add new augmentations in dataset transforms
   - Extend analysis in FiberOpticAnalyzer class

All implementations follow Dive into Deep Learning examples from https://d2l.ai/
