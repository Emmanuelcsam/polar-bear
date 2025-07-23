#!/bin/bash
# Request an interactive GPU session for debugging

echo "Requesting interactive GPU session..."
echo "This will give you a shell with GPU access for testing"
echo ""

# Request interactive session with GPU
# Adjust time and resources as needed
srun --partition=gpu \
     --gres=gpu:1 \
     --cpus-per-task=8 \
     --mem=32G \
     --time=2:00:00 \
     --pty bash -i