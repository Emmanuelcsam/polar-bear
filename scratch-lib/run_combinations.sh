#!/bin/bash
# Script to generate all image combinations

echo "Image Combination Generator"
echo "=========================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Install required packages if not already installed
echo "Checking/Installing required packages..."
pip3 install -q pillow numpy 2>/dev/null || {
    echo "Installing required packages..."
    pip3 install pillow numpy
}

# Display options
echo ""
echo "Select which script to run:"
echo "1) Standard version (generate_all_combinations.py)"
echo "2) Fast parallel version (generate_combinations_fast.py)"
echo ""
read -p "Enter choice (1 or 2): " choice

# Set default parameters
OPACITY=0.7
OUTPUT_DIR="combined_output"

# Ask for custom parameters
echo ""
read -p "Enter opacity (0.0-1.0) [default: 0.7]: " custom_opacity
if [[ ! -z "$custom_opacity" ]]; then
    OPACITY=$custom_opacity
fi

read -p "Enter output directory [default: combined_output]: " custom_output
if [[ ! -z "$custom_output" ]]; then
    OUTPUT_DIR=$custom_output
fi

# Run the selected script
echo ""
echo "Starting image combination generation..."
echo "Opacity: $OPACITY"
echo "Output directory: $OUTPUT_DIR"
echo ""

case $choice in
    1)
        echo "Running standard version..."
        python3 generate_all_combinations.py --opacity $OPACITY --output "$OUTPUT_DIR"
        ;;
    2)
        echo "Running fast parallel version..."
        python3 generate_combinations_fast.py --opacity $OPACITY --output "$OUTPUT_DIR"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Done! Check the $OUTPUT_DIR directory for results."