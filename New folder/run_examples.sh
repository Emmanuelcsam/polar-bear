#!/bin/bash
# run_examples.sh
# Example commands for running the fiber optic analysis system

echo "=== Fiber Optic Analysis System - Example Commands ==="
echo "Note: All configuration is now done through config.yaml"
echo ""

# Training examples
echo "1. Basic Training:"
echo "python main.py"
echo "   (Set system.mode: 'train' in config.yaml)"
echo ""

echo "2. Evaluation:"
echo "python main.py"
echo "   (Set system.mode: 'eval' and system.checkpoint_path in config.yaml)"
echo ""

echo "3. Model optimization (pruning + distillation):"
echo "python main.py"
echo "   (Set system.mode: 'optimize' and system.checkpoint_path in config.yaml)"
echo ""

echo "4. Distributed training (2 GPUs):"
echo "torchrun --nproc_per_node=2 main.py"
echo "   (Set system.mode: 'train' in config.yaml)"
echo ""

# Web interface examples
echo "5. Launch web interface:"
echo "python app.py"
echo "   (Configure webapp settings in config.yaml)"
echo ""

echo "6. Launch base Gradio demo:"
echo "python base.py"
echo "   (Configure webapp settings in config.yaml)"
echo ""

echo ""
echo "=== Configuration Instructions ==="
echo "Edit config.yaml to set:"
echo "- system.mode: 'train', 'eval', or 'optimize'"
echo "- system.checkpoint_path: path to model checkpoint (for eval/optimize)"
echo "- system.verbose: true/false for debug logging"
echo "- system.seed: random seed for reproducibility"
echo "- webapp.host: host address for web interface"
echo "- webapp.port: port number for web interface"
echo "- webapp.share: true/false for public link"
echo "- webapp.default_checkpoint: default model to load"
echo ""

# Installation
echo "7. Install dependencies:"
echo "pip install -r requirements.txt"
echo ""

echo "=== All configuration is now centralized in config.yaml ==="
