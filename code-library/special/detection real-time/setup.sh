#!/bin/bash
# Setup and Installation Script for Real-Time Defect Detection System

echo "🎥 Real-Time Defect Detection System Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python found: $python_version"
else
    echo "❌ Python 3 not found. Please install Python 3.7 or higher"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv realtime_defection_env
source realtime_defection_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📥 Installing required packages..."

# Core dependencies
pip install numpy>=1.21.0
pip install opencv-python>=4.5.0
pip install scikit-image>=0.18.0
pip install matplotlib>=3.3.0
pip install Pillow>=8.0.0

# Scientific computing
pip install scipy>=1.7.0
pip install pandas>=1.3.0

# Optional but recommended
pip install imutils>=0.5.4
pip install tqdm>=4.60.0

# Try to install pypylon (may require additional setup)
echo "🎥 Installing Basler Pylon SDK..."
pip install pypylon

if [[ $? -ne 0 ]]; then
    echo "⚠️  Warning: pypylon installation failed"
    echo "   You may need to install Basler Pylon SDK manually"
    echo "   Visit: https://www.baslerweb.com/en/software/pylon/"
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p {realtime_output,reference_images,test_images,logs}

# Create a requirements.txt file
echo "📝 Creating requirements.txt..."
cat > requirements.txt << EOF
# Real-Time Defect Detection System Requirements
numpy>=1.21.0
opencv-python>=4.5.0
scikit-image>=0.18.0
matplotlib>=3.3.0
Pillow>=8.0.0
scipy>=1.7.0
pandas>=1.3.0
imutils>=0.5.4
tqdm>=4.60.0
pypylon>=1.9.0
EOF

# Create configuration file
echo "⚙️  Creating default configuration..."
cat > config.json << EOF
{
    "camera": {
        "exposure_time": 10000,
        "gain": 0,
        "buffer_size": 5,
        "grab_strategy": "LatestImageOnly"
    },
    "detection": {
        "anomaly_threshold": 2.0,
        "ssim_threshold": 0.85,
        "confidence_threshold": 0.5,
        "enable_fast_mode": true,
        "resize_factor": 1.0,
        "min_defect_area": 25,
        "max_defect_area": 5000
    },
    "processing": {
        "processing_fps": 10.0,
        "enable_visualization": true,
        "save_results": true,
        "output_dir": "realtime_output"
    },
    "logging": {
        "level": "INFO",
        "file": "logs/detection.log",
        "max_size_mb": 10,
        "backup_count": 5
    }
}
EOF

# Create startup script
echo "🚀 Creating startup scripts..."
cat > start_detection.sh << 'EOF'
#!/bin/bash
# Startup script for real-time defect detection

echo "🎥 Starting Real-Time Defect Detection System"
echo "============================================="

# Activate virtual environment
source realtime_defection_env/bin/activate

# Check if reference image is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./start_detection.sh <reference_image_path>"
    echo "Example: ./start_detection.sh reference_images/my_reference.jpg"
    exit 1
fi

REFERENCE_IMAGE=$1

if [ ! -f "$REFERENCE_IMAGE" ]; then
    echo "❌ Reference image not found: $REFERENCE_IMAGE"
    exit 1
fi

echo "📸 Using reference image: $REFERENCE_IMAGE"
echo "Press Ctrl+C to stop the system"
echo

# Start the detection system
python3 realtime_controller.py "$REFERENCE_IMAGE"
EOF

chmod +x start_detection.sh

# Create test script
cat > test_system.sh << 'EOF'
#!/bin/bash
# Test script for the detection system

echo "🧪 Testing Real-Time Defect Detection System"
echo "============================================"

# Activate virtual environment
source realtime_defection_env/bin/activate

echo "Running system tests..."
python3 examples_and_testing.py 4

echo "Test completed!"
EOF

chmod +x test_system.sh

# Create README
echo "📚 Creating README..."
cat > README.md << EOF
# Real-Time Defect Detection System

This system integrates Basler Pylon cameras with advanced defect detection algorithms for real-time quality control.

## Features

- **Real-time Processing**: Continuous frame capture and analysis
- **Specific Reference**: Uses your chosen reference image for comparison
- **Multi-threading**: Producer-consumer architecture for optimal performance
- **Live Visualization**: Real-time display of detection results
- **Flexible Configuration**: Adjustable sensitivity and processing parameters
- **Data Logging**: Automatic saving of results and defective frames

## Quick Start

1. **Setup Environment**:
   \`\`\`bash
   ./setup.sh
   \`\`\`

2. **Prepare Reference Image**:
   - Place your reference image in the \`reference_images/\` directory
   - Supported formats: .jpg, .png, .bmp, .tiff

3. **Start Detection**:
   \`\`\`bash
   ./start_detection.sh reference_images/your_reference.jpg
   \`\`\`

4. **Monitor Results**:
   - Live visualization window shows detection results
   - Results saved to \`realtime_output/\` directory
   - Logs available in \`logs/\` directory

## Configuration

Edit \`config.json\` to customize:

- **Camera Settings**: Exposure time, gain, buffer size
- **Detection Parameters**: Thresholds, sensitivity, processing mode
- **Performance Options**: Processing rate, visualization, logging

## File Structure

\`\`\`
realtime_defection_system/
├── enhanced_pylon_grabber.py    # Enhanced frame grabber
├── realtime_detector.py         # Detection engine adapter
├── realtime_controller.py       # Main system controller
├── examples_and_testing.py      # Usage examples and tests
├── detection.py                 # Your original detection module
├── pylon_grabber.py             # Your original grabber
├── config.json                  # System configuration
├── start_detection.sh           # Startup script
├── test_system.sh              # Test script
├── reference_images/            # Reference images directory
├── realtime_output/            # Detection results
└── logs/                       # System logs
\`\`\`

## System Architecture

\`\`\`
[Basler Camera] → [Enhanced Grabber] → [Frame Queue] → [Detector] → [Results]
       ↑                    ↓              ↓             ↓          ↓
   [Pylon SDK]         [Threading]    [Producer]   [Consumer]  [Visualization]
\`\`\`

## Performance Tips

1. **High-Speed Processing**: Disable visualization and saving
2. **Quality Focus**: Enable full detection mode with lower FPS
3. **Balanced Mode**: Use fast mode with selective saving
4. **Memory Optimization**: Adjust buffer sizes based on available RAM

## Troubleshooting

### Camera Issues
- Verify Pylon SDK installation
- Check camera connection and permissions
- Test with Pylon Viewer first

### Detection Issues
- Validate reference image format and quality
- Adjust detection thresholds in config
- Check lighting conditions consistency

### Performance Issues
- Reduce processing FPS
- Enable fast detection mode
- Optimize buffer sizes
- Check system resources

## Support

For issues and questions:
1. Check the log files in \`logs/\`
2. Run test examples: \`python3 examples_and_testing.py\`
3. Verify configuration in \`config.json\`
EOF

echo
echo "✅ Setup completed successfully!"
echo
echo "📁 Directory structure created:"
echo "   • realtime_output/     - Detection results"
echo "   • reference_images/    - Reference images"
echo "   • test_images/         - Test images"
echo "   • logs/               - System logs"
echo
echo "📜 Configuration files created:"
echo "   • config.json         - System settings"
echo "   • requirements.txt    - Python dependencies"
echo "   • README.md          - Documentation"
echo
echo "🚀 Startup scripts created:"
echo "   • start_detection.sh  - Main startup script"
echo "   • test_system.sh     - System test script"
echo
echo "📋 Next steps:"
echo "1. Place your reference image in reference_images/"
echo "2. Run: ./start_detection.sh reference_images/your_image.jpg"
echo "3. Or test with: ./test_system.sh"
echo
echo "💡 Tips:"
echo "• Edit config.json to customize settings"
echo "• Check README.md for detailed documentation"
echo "• Run examples_and_testing.py for usage examples"
echo
echo "🎉 Ready to detect defects in real-time!"