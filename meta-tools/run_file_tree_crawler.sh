#!/bin/bash

# File Tree Crawler Shell Script
# This script runs the Python file tree crawler with interactive prompts

echo "============================================"
echo "   INTERACTIVE FILE TREE CRAWLER"
echo "============================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.6 or higher"
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "Using Python command: $PYTHON_CMD"
echo

# Function to ask yes/no questions
ask_yes_no() {
    local prompt="$1"
    local default="$2"
    while true; do
        if [ "$default" = "y" ]; then
            read -p "$prompt [Y/n]: " answer
            answer=${answer:-y}
        else
            read -p "$prompt [y/N]: " answer
            answer=${answer:-n}
        fi
        
        case $answer in
            [Yy]* ) echo "y"; return;;
            [Nn]* ) echo "n"; return;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

# Ask interactive questions
echo "Let's configure your file tree crawler:"
echo

# Root directory
echo "1. Root Directory"
read -p "Enter the root directory to start crawling from (press Enter for current directory): " root_dir
root_dir=${root_dir:-.}

# Output file
echo
echo "2. Output File"
read -p "Enter the output file name (press Enter for 'file_tree_structure.txt'): " output_file
output_file=${output_file:-file_tree_structure.txt}

# Maximum depth
echo
echo "3. Maximum Depth"
read -p "Enter maximum depth to crawl (press Enter for unlimited): " max_depth

# Include files
echo
echo "4. File Inclusion"
include_files=$(ask_yes_no "Include files in the tree structure?" "y")

# Include hidden files
echo
echo "5. Hidden Files"
include_hidden=$(ask_yes_no "Include hidden files and directories?" "n")

# Include virtual environments
echo
echo "6. Virtual Environments"
include_venv=$(ask_yes_no "Include Python virtual environment directories?" "n")

# Include statistics
echo
echo "7. Statistics"
include_stats=$(ask_yes_no "Include directory statistics in the output?" "y")

# Build command arguments
ARGS="--root \"$root_dir\" --output \"$output_file\""

if [ ! -z "$max_depth" ]; then
    ARGS="$ARGS --max-depth $max_depth"
fi

if [ "$include_files" = "n" ]; then
    ARGS="$ARGS --no-files"
fi

if [ "$include_hidden" = "y" ]; then
    ARGS="$ARGS --include-hidden"
fi

if [ "$include_venv" = "y" ]; then
    ARGS="$ARGS --include-venv"
fi

if [ "$include_stats" = "y" ]; then
    ARGS="$ARGS --stats"
fi

# Show configuration summary
echo
echo "============================================"
echo "   CONFIGURATION SUMMARY"
echo "============================================"
echo "Root directory: $root_dir"
echo "Output file: $output_file"
echo "Max depth: ${max_depth:-unlimited}"
echo "Include files: $include_files"
echo "Include hidden: $include_hidden"
echo "Include virtual envs: $include_venv"
echo "Include statistics: $include_stats"
echo "============================================"
echo

# Confirm before running
confirm=$(ask_yes_no "Run the file tree crawler with these settings?" "y")

if [ "$confirm" = "y" ]; then
    echo "Running file tree crawler..."
    echo "Command: $PYTHON_CMD file_tree_crawler.py $ARGS"
    echo
    
    # Run the file tree crawler with collected settings
    eval "$PYTHON_CMD file_tree_crawler.py $ARGS"
    
    echo
    echo "============================================"
    echo "File tree generation complete!"
    echo "Check the '$output_file' file for results."
    echo "============================================"
else
    echo "Operation cancelled."
fi
