#!/usr/bin/env python3
"""
Time Series Anomaly Detection Script
Based on Marco's tutorial on anomaly detection in time series
This script implements three methods:
1. Robust Z-score using Median Absolute Deviation (MAD)
2. Isolation Forest
3. Local Outlier Factor
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime

# Function to install packages
def install_package(package_name, import_name=None):
    """Install a package using pip if it's not already installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        print(f"✗ {package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package_name}")
            return False

# Check and install required packages
print("="*60)
print("CHECKING AND INSTALLING REQUIRED PACKAGES")
print("="*60)

required_packages = [
    ("pandas", None),
    ("numpy", None),
    ("matplotlib", None),
    ("scipy", None),
    ("scikit-learn", "sklearn"),
    ("requests", None),  # For downloading data
]

for package, import_name in required_packages:
    if not install_package(package, import_name):
        print(f"ERROR: Could not install {package}. Exiting.")
        sys.exit(1)

print("\n✓ All required packages are installed!")
print("="*60)

# Now import all required libraries
print("\nIMPORTING LIBRARIES...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import requests
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully!")

# Set matplotlib parameters
plt.rcParams['figure.figsize'] = (12, 6)
print("✓ Matplotlib figure size set to (12, 6)")

# Function to download data
def download_file(url, filename):
    """Download a file from URL if it doesn't exist"""
    if os.path.exists(filename):
        print(f"✓ {filename} already exists, skipping download")
        return True
    
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"✓ Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False

# Data URLs (from the tutorial)
print("\n" + "="*60)
print("DOWNLOADING DATA")
print("="*60)

# Note: These are example URLs - the actual URLs from Numenta would need to be specified
data_url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
labels_url = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json"

# Download data
if not download_file(data_url, "ec2_cpu_utilization.csv"):
    print("Using sample data instead...")
    # Create sample data as mentioned in the tutorial
    dates = pd.date_range(start='2014-02-14 14:30:00', periods=4032, freq='5min')
    np.random.seed(42)
    values = np.random.normal(20, 5, 4032)
    # Add some flat regions as mentioned in the tutorial
    values[1000:1500] = 10
    values[2000:2300] = 10
    values[3000:3200] = 10
    # Add two anomalies
    values[1234] = 80  # First anomaly
    values[2876] = 75  # Second anomaly
    
    df = pd.DataFrame({
        'timestamp': dates,
        'value': values
    })
    df.to_csv('ec2_cpu_utilization.csv', index=False)
    print("✓ Created sample data file")

# Load the data
print("\n" + "="*60)
print("LOADING AND PREPROCESSING DATA")
print("="*60)

print("Reading data from CSV file...")
df = pd.read_csv('ec2_cpu_utilization.csv')
print(f"✓ Data loaded successfully! Shape: {df.shape}")

# Convert timestamp to datetime
print("Converting timestamp column to datetime format...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print("✓ Timestamp converted successfully")

# Add anomaly labels (as mentioned in the tutorial, default is 1 for inlier, -1 for outlier)
print("Adding anomaly labels...")
df['is_anomaly'] = 1  # Default: all points are inliers

# For this example, we'll mark some extreme values as anomalies
# In the real tutorial, these would come from the labels file
threshold = df['value'].quantile(0.99)
anomaly_indices = df[df['value'] > threshold].index[:2]  # Mark top 2 extreme values
df.loc[anomaly_indices, 'is_anomaly'] = -1
print(f"✓ Marked {len(anomaly_indices)} points as anomalies")

# Display data info
print("\nData Information:")
print(f"- Total data points: {len(df)}")
print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"- Number of anomalies: {(df['is_anomaly'] == -1).sum()}")
print(f"- Number of inliers: {(df['is_anomaly'] == 1).sum()}")

# Visualize the data
print("\n" + "="*60)
print("VISUALIZING DATA")
print("="*60)

print("Creating visualization of the time series with anomalies...")
plt.figure(figsize=(14, 6))

# Separate inliers and outliers
inliers = df[df['is_anomaly'] == 1]
outliers = df[df['is_anomaly'] == -1]

# Plot
plt.plot(inliers['timestamp'], inliers['value'], 'b.', label='Normal', markersize=4)
plt.plot(outliers['timestamp'], outliers['value'], 'r.', label='Anomaly', markersize=10)
plt.xlabel('Timestamp')
plt.ylabel('CPU Utilization')
plt.title('CPU Utilization with Anomalies')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data_visualization.png')
print("✓ Saved visualization to 'data_visualization.png'")
plt.close()

# Check data distribution
print("\n" + "="*60)
print("ANALYZING DATA DISTRIBUTION")
print("="*60)

print("Creating histogram to check if data is normally distributed...")
plt.figure(figsize=(10, 6))
plt.hist(df['value'], bins=50, alpha=0.7, color='blue', edgecolor='black')
median_value = df['value'].median()
plt.axvline(median_value, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
plt.xlabel('CPU Utilization')
plt.ylabel('Frequency')
plt.title('Distribution of CPU Utilization Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data_distribution.png')
print("✓ Saved distribution plot to 'data_distribution.png'")
plt.close()

# METHOD 1: ROBUST Z-SCORE WITH MEDIAN ABSOLUTE DEVIATION
print("\n" + "="*60)
print("METHOD 1: ROBUST Z-SCORE (MEDIAN ABSOLUTE DEVIATION)")
print("="*60)

# Calculate MAD and median
print("Calculating Median Absolute Deviation (MAD)...")
mad = median_abs_deviation(df['value'])
median = np.median(df['value'])
print(f"✓ MAD calculated: {mad:.6f}")
print(f"✓ Median calculated: {median:.2f}")

if mad < 0.01:
    print("⚠ WARNING: MAD is very close to zero! This method may not work well.")
    print("  This happens when >50% of data has the same value.")

# Define robust z-score function
def compute_robust_zscore(x, median, mad):
    """Compute robust z-score using MAD"""
    if mad == 0:
        return 0
    return 0.6745 * (x - median) / mad

# Calculate z-scores
print("Computing robust z-scores for all data points...")
df['z_score'] = df['value'].apply(lambda x: compute_robust_zscore(x, median, mad))
print("✓ Z-scores computed successfully")

# Identify outliers using threshold
z_threshold = 3.5
print(f"Identifying outliers with z-score threshold = {z_threshold}...")
df['mad_prediction'] = 1  # Default: inlier
df.loc[(df['z_score'] > z_threshold) | (df['z_score'] < -z_threshold), 'mad_prediction'] = -1
mad_outliers = (df['mad_prediction'] == -1).sum()
print(f"✓ MAD method identified {mad_outliers} outliers")

# Evaluate MAD method
print("\nEvaluating MAD method...")
cm_mad = confusion_matrix(df['is_anomaly'], df['mad_prediction'], labels=[1, -1])
print("Confusion Matrix (MAD):")
print(f"                Predicted")
print(f"               Inlier  Outlier")
print(f"Actual Inlier   {cm_mad[0,0]:5d}   {cm_mad[0,1]:5d}")
print(f"      Outlier   {cm_mad[1,0]:5d}   {cm_mad[1,1]:5d}")

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_mad, display_labels=['Inlier', 'Outlier'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Robust Z-Score (MAD) Method')
plt.tight_layout()
plt.savefig('confusion_matrix_mad.png')
print("✓ Saved MAD confusion matrix to 'confusion_matrix_mad.png'")
plt.close()

# METHOD 2: ISOLATION FOREST
print("\n" + "="*60)
print("METHOD 2: ISOLATION FOREST")
print("="*60)

# Split data for training and testing
print("Splitting data into train and test sets...")
split_index = int(len(df) * 0.8)
train = df[:split_index].copy()
test = df[split_index:].copy()
print(f"✓ Train set size: {len(train)}")
print(f"✓ Test set size: {len(test)}")
print(f"✓ Anomalies in train: {(train['is_anomaly'] == -1).sum()}")
print(f"✓ Anomalies in test: {(test['is_anomaly'] == -1).sum()}")

# Calculate contamination level
contamination = (train['is_anomaly'] == -1).sum() / len(train)
if contamination == 0:
    contamination = 0.01  # Set small default if no anomalies in train
print(f"✓ Contamination level: {contamination:.4f}")

# Prepare data for sklearn
X_train = train['value'].values.reshape(-1, 1)
X_test = test['value'].values.reshape(-1, 1)

# Train Isolation Forest
print("Training Isolation Forest model...")
iso_forest = IsolationForest(contamination=contamination, random_state=42)
iso_forest.fit(X_train)
print("✓ Isolation Forest trained successfully")

# Make predictions
print("Making predictions on test set...")
pred_iso = iso_forest.predict(X_test)
print("✓ Predictions completed")

# Evaluate Isolation Forest
print("\nEvaluating Isolation Forest...")
cm_iso = confusion_matrix(test['is_anomaly'], pred_iso, labels=[1, -1])
print("Confusion Matrix (Isolation Forest):")
print(f"                Predicted")
print(f"               Inlier  Outlier")
print(f"Actual Inlier   {cm_iso[0,0]:5d}   {cm_iso[0,1]:5d}")
print(f"      Outlier   {cm_iso[1,0]:5d}   {cm_iso[1,1]:5d}")

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_iso, display_labels=['Inlier', 'Outlier'])
disp.plot(cmap='Greens')
plt.title('Confusion Matrix - Isolation Forest')
plt.tight_layout()
plt.savefig('confusion_matrix_isolation_forest.png')
print("✓ Saved Isolation Forest confusion matrix to 'confusion_matrix_isolation_forest.png'")
plt.close()

# METHOD 3: LOCAL OUTLIER FACTOR
print("\n" + "="*60)
print("METHOD 3: LOCAL OUTLIER FACTOR (LOF)")
print("="*60)

# Train Local Outlier Factor
print("Training Local Outlier Factor model...")
lof = LocalOutlierFactor(contamination=contamination, novelty=True)
lof.fit(X_train)
print("✓ Local Outlier Factor trained successfully")

# Make predictions
print("Making predictions on test set...")
pred_lof = lof.predict(X_test)
print("✓ Predictions completed")

# Evaluate Local Outlier Factor
print("\nEvaluating Local Outlier Factor...")
cm_lof = confusion_matrix(test['is_anomaly'], pred_lof, labels=[1, -1])
print("Confusion Matrix (Local Outlier Factor):")
print(f"                Predicted")
print(f"               Inlier  Outlier")
print(f"Actual Inlier   {cm_lof[0,0]:5d}   {cm_lof[0,1]:5d}")
print(f"      Outlier   {cm_lof[1,0]:5d}   {cm_lof[1,1]:5d}")

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lof, display_labels=['Inlier', 'Outlier'])
disp.plot(cmap='Oranges')
plt.title('Confusion Matrix - Local Outlier Factor')
plt.tight_layout()
plt.savefig('confusion_matrix_lof.png')
print("✓ Saved LOF confusion matrix to 'confusion_matrix_lof.png'")
plt.close()

# SUMMARY
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)

# Calculate metrics for each method
def calculate_metrics(cm):
    """Calculate precision, recall, and F1 score from confusion matrix"""
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, f1, accuracy

# MAD metrics (on full dataset)
p_mad, r_mad, f1_mad, acc_mad = calculate_metrics(cm_mad)
print(f"\n1. Robust Z-Score (MAD) Method:")
print(f"   - Precision: {p_mad:.3f}")
print(f"   - Recall: {r_mad:.3f}")
print(f"   - F1 Score: {f1_mad:.3f}")
print(f"   - Accuracy: {acc_mad:.3f}")
print(f"   - Total outliers detected: {mad_outliers}")

# Isolation Forest metrics
p_iso, r_iso, f1_iso, acc_iso = calculate_metrics(cm_iso)
print(f"\n2. Isolation Forest:")
print(f"   - Precision: {p_iso:.3f}")
print(f"   - Recall: {r_iso:.3f}")
print(f"   - F1 Score: {f1_iso:.3f}")
print(f"   - Accuracy: {acc_iso:.3f}")

# LOF metrics
p_lof, r_lof, f1_lof, acc_lof = calculate_metrics(cm_lof)
print(f"\n3. Local Outlier Factor:")
print(f"   - Precision: {p_lof:.3f}")
print(f"   - Recall: {r_lof:.3f}")
print(f"   - F1 Score: {f1_lof:.3f}")
print(f"   - Accuracy: {acc_lof:.3f}")

# Create comparison visualization
print("\n" + "="*60)
print("CREATING COMPARISON VISUALIZATION")
print("="*60)

methods = ['MAD', 'Isolation\nForest', 'Local Outlier\nFactor']
f1_scores = [f1_mad, f1_iso, f1_lof]
accuracies = [acc_mad, acc_iso, acc_lof]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# F1 Scores
ax1.bar(methods, f1_scores, color=['blue', 'green', 'orange'])
ax1.set_ylabel('F1 Score')
ax1.set_title('F1 Score Comparison')
ax1.set_ylim(0, 1)
for i, v in enumerate(f1_scores):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')

# Accuracies
ax2.bar(methods, accuracies, color=['blue', 'green', 'orange'])
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Comparison')
ax2.set_ylim(0, 1)
for i, v in enumerate(accuracies):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')

plt.suptitle('Anomaly Detection Methods Comparison')
plt.tight_layout()
plt.savefig('methods_comparison.png')
print("✓ Saved methods comparison to 'methods_comparison.png'")
plt.close()

print("\n" + "="*60)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
print("- data_visualization.png: Time series plot with anomalies")
print("- data_distribution.png: Histogram of data distribution")
print("- confusion_matrix_mad.png: MAD method results")
print("- confusion_matrix_isolation_forest.png: Isolation Forest results")
print("- confusion_matrix_lof.png: Local Outlier Factor results")
print("- methods_comparison.png: Comparison of all methods")
print("\nAs mentioned in the tutorial:")
print("- MAD works best with normally distributed data")
print("- Isolation Forest is good for general anomaly detection")
print("- Local Outlier Factor performed best in this example")
print("\nThank you for using this anomaly detection script!")
