import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D surface plot of pixel intensities (x, y positions vs. grayscale value)

# Load pixel data: skip metadata rows so the header aligns correctly
try:
    df = pd.read_csv('/mnt/data/img63_detailed.csv', skiprows=6)
except FileNotFoundError:
    print("Warning: The file '/mnt/data/img63_detailed.csv' was not found.")
    print("Creating an empty plot as a placeholder.")
    df = pd.DataFrame(columns=['X Coordinate', 'Y Coordinate', 'Grayscale Value (0-255)'])

# Keep only the X, Y, and intensity columns
if not all(col in df.columns for col in ['X Coordinate', 'Y Coordinate', 'Grayscale Value (0-255)']):
    print("Warning: CSV file does not contain the required columns. Skipping plot.")
    df = pd.DataFrame(columns=['X Coordinate', 'Y Coordinate', 'Grayscale Value (0-255)'])

if not df.empty:
    # Pivot to form a 2D array of intensities (rows = Y, columns = X)
    intensity_matrix = df.pivot(index='Y Coordinate', columns='X Coordinate', values='Grayscale Value (0-255)')
    Z = intensity_matrix.values
    x = intensity_matrix.columns.values
    y = intensity_matrix.index.values
    X, Y = np.meshgrid(x, y)

    # Downsample for performance (every 5th pixel)
    stride = 5
    X_ds = X[::stride, ::stride]
    Y_ds = Y[::stride, ::stride]
    Z_ds = Z[::stride, ::stride]
else:
    # Create empty data for the plot if the dataframe is empty
    X_ds, Y_ds, Z_ds = np.array([]), np.array([]), np.array([])


# Plot the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
if not df.empty:
    # Pivot to form a 2D array of intensities (rows = Y, columns = X)
    intensity_matrix = df.pivot(index='Y Coordinate', columns='X Coordinate', values='Grayscale Value (0-255)')
    Z = intensity_matrix.values
    x = intensity_matrix.columns.values
    y = intensity_matrix.index.values
    X, Y = np.meshgrid(x, y)

    # Downsample for performance (every 5th pixel)
    stride = 5
    X_ds = X[::stride, ::stride]
    Y_ds = Y[::stride, ::stride]
    Z_ds = Z[::stride, ::stride]
    ax.plot_surface(X_ds, Y_ds, Z_ds)
else:
    # Create empty data for the plot if the dataframe is empty
    X_ds, Y_ds, Z_ds = np.array([]), np.array([]), np.array([])
    ax.text2D(0.5, 0.5, "No data to display", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

ax.set_xlabel('X Pixel Position')
ax.set_ylabel('Y Pixel Position')
ax.set_zlabel('Pixel Intensity')
plt.tight_layout()
plt.show()
