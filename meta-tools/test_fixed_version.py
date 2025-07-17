#!/usr/bin/env python3
"""
Test the fixed fast background removal
"""

import os
import sys

# Test with automated input
test_input = """/media/jarvis/6E7A-FA6E/polar-bear/meta-tools/frontend
/tmp/test_fixed_output
n
y
"""

print("Testing fixed fast background removal...")
print("=" * 50)

# Run the script with automated input
import subprocess

proc = subprocess.Popen(
    [sys.executable, 'fast-background-removal-fixed.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

output, _ = proc.communicate(input=test_input)
print(output)

# Check if output was created
output_dir = '/tmp/test_fixed_output'
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    print(f"\nOutput directory contains {len(files)} files:")
    for f in files[:5]:  # Show first 5 files
        print(f"  - {f}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")
else:
    print("\nNo output directory created")

print("\nTest completed.")