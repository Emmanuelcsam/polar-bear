#!/bin/bash
# Linux shell script to run the geometry detection applications

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if running on Wayland and fix Qt issues
check_wayland() {
    if [ ! -z "$WAYLAND_DISPLAY" ]; then
        echo -e "${YELLOW}Detected Wayland display system.${NC}"
        echo "Setting GDK_BACKEND=x11 for compatibility..."
        export GDK_BACKEND=x11
    fi
    
    # Qt fixes are now handled in run_with_system_qt.sh
}

# Function to display menu
show_menu() {
    echo -e "${GREEN}========================================"
    echo "Real-time Geometry Detection System"
    echo "========================================${NC}"
    echo
    echo "Available applications:"
    echo "1. Circle Detector (Optimized for Basler cameras)"
    echo "2. Geometry Demo (General shape detection)"
    echo "3. Calibration Tool (Interactive calibration)"
    echo "4. Run Tests"
    echo "5. Install Dependencies"
    echo "6. Exit"
    echo
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (1-6): " choice
    
    case $choice in
        1)
            echo -e "${GREEN}Starting Circle Detector...${NC}"
            check_wayland
            ./run_with_system_qt.sh run_circle_detector.py
            ;;
        2)
            echo -e "${GREEN}Starting Geometry Demo...${NC}"
            check_wayland
            python3 run_geometry_demo_fixed.py
            ;;
        3)
            echo -e "${GREEN}Starting Calibration Tool...${NC}"
            check_wayland
            python3 run_calibration.py
            ;;
        4)
            echo -e "${GREEN}Running tests...${NC}"
            python3 -m pytest src/tests/ -v
            ;;
        5)
            echo -e "${GREEN}Installing dependencies...${NC}"
            pip3 install -r requirements.txt
            echo
            echo -e "${GREEN}Installation complete!${NC}"
            ;;
        6)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
    clear
done