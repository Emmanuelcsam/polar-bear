#!/bin/bash
# Quick test script for HPC environment

echo "Running quick HPC compatibility test..."

# Test 1: Python and PyTorch import
echo -e "\n1. Testing Python imports..."
python -c "
import torch
import torchvision
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

# Test 2: Test distributed imports
echo -e "\n2. Testing distributed imports..."
python -c "
import torch.distributed as dist
from distributed_utils import init_distributed, cleanup_distributed
print('Distributed imports successful')
"

# Test 3: Test main modules
echo -e "\n3. Testing main modules..."
python -c "
from main import UnifiedFiberOpticsSystem
from trainer import EnhancedTrainer
from data_loader import FiberOpticsDataLoader
print('Main module imports successful')
"

# Test 4: Check data paths
echo -e "\n4. Checking paths..."
python -c "
from config_loader import get_config
config = get_config()
print(f'Config loaded successfully')
print(f'Device: {config.get_device()}')
"

echo -e "\nQuick test completed!"