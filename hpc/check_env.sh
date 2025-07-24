#!/bin/bash
# Check environment type

ENV_PATH="/sciclone/scr-lst/$USER/polar-bear-env"

echo "Checking environment at: $ENV_PATH"
echo ""

if [ -f "$ENV_PATH/bin/activate" ]; then
    echo "Found: Python venv"
    echo "Activate with: source $ENV_PATH/bin/activate"
    
    # Try to activate and check
    source $ENV_PATH/bin/activate
    echo "Python: $(which python)"
    echo "Pip: $(which pip)"
    pip list | grep torch
    
elif [ -d "$ENV_PATH/conda-meta" ]; then
    echo "Found: Conda environment"
    echo "Activate with: conda activate $ENV_PATH"
    
else
    echo "ERROR: Cannot determine environment type"
    echo "Contents of $ENV_PATH:"
    ls -la $ENV_PATH | head -20
fi