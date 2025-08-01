import os
import pickle
import torch
import numpy as np
import json
import pandas as pd
import h5py
from pprint import pprint

def read_pkl(file_path):
    """Read and display contents of a .pkl file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Contents of .pkl file:")
    pprint(data)

def read_pth(file_path):
    """Read and display contents of a .pth file (PyTorch)."""
    data = torch.load(file_path, map_location='cpu')  # Load on CPU to avoid device issues
    print("Contents of .pth file:")
    if isinstance(data, dict):
        print("Dictionary keys:")
        pprint(list(data.keys()))
        print("\nSample values (first few items if large):")
        for key, value in list(data.items())[:5]:  # Limit to first 5 for large dicts
            print(f"{key}: {value}")
    else:
        pprint(data)

def read_npy(file_path):
    """Read and display contents of a .npy file (NumPy)."""
    data = np.load(file_path)
    print("Contents of .npy file:")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print("Sample data (first few elements):")
    print(data.flatten()[:10])  # Flatten and show first 10 for summary

def read_json(file_path):
    """Read and display contents of a .json file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("Contents of .json file:")
    pprint(data)

def read_csv(file_path):
    """Read and display contents of a .csv file using pandas."""
    data = pd.read_csv(file_path)
    print("Contents of .csv file:")
    print(data.head(10))  # Show first 10 rows
    print("\nSummary:")
    print(data.describe())

def read_h5(file_path):
    """Read and display contents of a .h5 or .hdf5 file."""
    with h5py.File(file_path, 'r') as f:
        print("Contents of .h5 file:")
        print("Keys:")
        pprint(list(f.keys()))
        for key in list(f.keys())[:5]:  # Limit to first 5 datasets
            dataset = f[key]
            print(f"\nDataset '{key}':")
            print(f"Shape: {dataset.shape}")
            print(f"Dtype: {dataset.dtype}")
            print("Sample data (first few elements):")
            print(dataset[:10])

def read_txt(file_path):
    """Read and display contents of a .txt file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    print("Contents of .txt file (first 20 lines):")
    for line in lines[:20]:
        print(line.strip())

def main():
    while True:
        file_path = input("Enter the path to the data file you want to read (or type 'exit' to quit): ").strip()
        if file_path.lower() == 'exit':
            print("Exiting the data reader.")
            break
        
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist. Please try again.")
            continue

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pkl':
            read_pkl(file_path)
        elif ext == '.pth':
            read_pth(file_path)
        elif ext == '.npy':
            read_npy(file_path)
        elif ext == '.json':
            read_json(file_path)
        elif ext == '.csv':
            read_csv(file_path)
        elif ext in ['.h5', '.hdf5']:
            read_h5(file_path)
        elif ext == '.txt':
            read_txt(file_path)
        else:
            print(f"Unsupported file extension: {ext}")
            print("Supported formats: .pkl, .pth, .npy, .json, .csv, .h5/.hdf5, .txt")

if __name__ == "__main__":
    print("Welcome to the interactive data reader. This tool allows you to view contents of various data files in a readable format.")
    main()
