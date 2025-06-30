#!/bin/bash
# Activate virtual environment and run analyzer

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run the analyzer with all arguments passed to this script
python3 gptscraper.py "$@"

# Deactivate virtual environment
deactivate
