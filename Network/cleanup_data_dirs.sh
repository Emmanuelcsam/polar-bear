#!/bin/bash
# Script to clean up data directories and create proper symlinks

echo "Checking current directory structure..."
echo ""

cd /sciclone/scr10/$USER/polar-bear

echo "=== Current dataset directory ==="
if [ -L dataset ]; then
    echo "dataset is a symlink pointing to: $(readlink dataset)"
elif [ -d dataset ]; then
    echo "dataset is a regular directory"
    echo "Contents:"
    ls -la dataset | head -10
    echo "Number of files: $(find dataset -type f | wc -l)"
else
    echo "dataset does not exist"
fi

echo ""
echo "=== Current reference directory ==="
if [ -L reference ]; then
    echo "reference is a symlink pointing to: $(readlink reference)"
elif [ -d reference ]; then
    echo "reference is a regular directory"
    echo "Contents:"
    ls -la reference | head -10
    echo "Number of files: $(find reference -type f | wc -l)"
else
    echo "reference does not exist"
fi

echo ""
echo "=== Actual data location ==="
echo "Dataset: /sciclone/data10/$USER/polar-bear/dataset"
ls /sciclone/data10/$USER/polar-bear/dataset | head -5
echo ""
echo "Reference: /sciclone/data10/$USER/polar-bear/reference"
ls /sciclone/data10/$USER/polar-bear/reference | head -5

echo ""
echo "To fix this, run:"
echo "  cd /sciclone/scr10/$USER/polar-bear"
echo "  rm -rf dataset reference"
echo "  ln -s /sciclone/data10/$USER/polar-bear/dataset dataset"
echo "  ln -s /sciclone/data10/$USER/polar-bear/reference reference"