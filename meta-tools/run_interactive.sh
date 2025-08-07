#!/bin/bash

# Simple Interactive File Tree Crawler Shell Script
# This script runs the Python file tree crawler in interactive mode

echo "Starting Interactive File Tree Crawler..."
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

# Run the file tree crawler in interactive mode
$PYTHON_CMD file_tree_crawler.py --interactive
