#!/bin/bash

# Simple script to run the background removal tool with example paths
echo "Running auto-background-removal.py..."
echo "Input directory: C:\Users\Saem1001\Documents\GitHub\polar-bear\reference\masks\ferrule"
echo "Output directory: C:\Users\Saem1001\Documents\GitHub\polar-bear\reference\masks\ferrule-output"

# Create the output directory if it doesn't exist
mkdir -p "C:\Users\Saem1001\Documents\GitHub\polar-bear\reference\masks\ferrule-output"

# Run the script with input paths
echo -e "C:\Users\Saem1001\Documents\GitHub\polar-bear\reference\masks\ferrule\nC:\Users\Saem1001\Documents\GitHub\polar-bear\reference\masks\ferrule-output" | uv run python auto-background-removal.py
