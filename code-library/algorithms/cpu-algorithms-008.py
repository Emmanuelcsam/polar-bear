#!/usr/bin/env python3
"""
Enhanced Time Series Anomaly Detection Script with Connector Integration
Based on Marco's tutorial on anomaly detection in time series
This script implements three methods:
1. Robust Z-score using Median Absolute Deviation (MAD)
2. Isolation Forest
3. Local Outlier Factor

Now with full connector integration for parameter control and collaboration
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Try to import script interface
try:
    from script_interface import ScriptInterface, ConnectorClient, create_standalone_wrapper
    CONNECTOR_AVAILABLE = True
except ImportError:
    CONNECTOR_AVAILABLE = False
    print("Warning: Script interface not available, running in standalone mode")

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


class AnomalyDetectionSystem(ScriptInterface if CONNECTOR_AVAILABLE else object):
    """Enhanced Anomaly Detection System with Connector Integration"""
    
    def __init__(self):
        if CONNECTOR_AVAILABLE:
            super().__init__("anomaly_detection", "Time Series Anomaly Detection System")
            
            # Register parameters
            self.register_parameter("method", "zscore", ["zscore", "isolation_forest", "lof", "all"])
            self.register_parameter("threshold", 3.0, [2.0, 2.5, 3.0, 3.5, 4.0])
            self.register_parameter("contamination", 0.1, [0.01, 0.05, 0.1, 0.15, 0.2])
            self.register_parameter("n_neighbors", 20, range(5, 51))
            self.register_parameter("window_size", 100, range(10, 1001))
            self.register_parameter("save_plots", True, [True, False])
            self.register_parameter("data_source", "sample", ["sample", "file", "realtime"])
            
            # Register variables
            self.register_variable("total_anomalies", 0)
            self.register_variable("data_points_processed", 0)
            self.register_variable("last_anomaly_timestamp", None)
            self.register_variable("current_method", "zscore")
            self.register_variable("detection_rate", 0.0)
            
            # Register callback for method changes
            self.register_callback("method", self._on_method_change)
            
            # Initialize connector client
            self.client = ConnectorClient(self)
        else:
            # Standalone mode - use default values
            self.parameters = {
                "method": {"value": "zscore"},
                "threshold": {"value": 3.0},
                "contamination": {"value": 0.1},
                "n_neighbors": {"value": 20},
                "window_size": {"value": 100},
                "save_plots": {"value": True},
                "data_source": {"value": "sample"}
            }
            self.variables = {}
            self.results = {}
            
        self.data = None
        self.anomalies = {}
        
    def get_parameter(self, name):
        """Get parameter value (works in both modes)"""
        if CONNECTOR_AVAILABLE:
            return super().get_parameter(name)
        else:
            return self.parameters.get(name, {}).get("value")
            
    def _on_method_change(self, new_method):
        """Callback when detection method changes"""
        if CONNECTOR_AVAILABLE:
            self.logger.info(f"Detection method changed to: {new_method}")
            self.set_variable("current_method", new_method)
        
    def download_file(self, url, filename):
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
    
    def load_data(self):
        """Load or generate data based on data_source parameter"""
        data_source = self.get_parameter("data_source")
        
        if data_source == "sample":
            # Generate sample data with known anomalies
            print("\nGenerating sample time series data...")
            np.random.seed(42)
            n_samples = 1000
            
            # Normal data with trend and seasonality
            time = np.arange(n_samples)
            trend = 0.01 * time
            seasonal = 10 * np.sin(2 * np.pi * time / 100)
            noise = np.random.normal(0, 1, n_samples)
            
            # Base signal
            signal = trend + seasonal + noise
            
            # Add anomalies
            anomaly_indices = [150, 151, 350, 450, 451, 452, 750, 850, 950]
            for idx in anomaly_indices:
                signal[idx] += np.random.uniform(20, 30) * np.random.choice([-1, 1])
            
            # Create DataFrame
            self.data = pd.DataFrame({
                'time': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'),
                'value': signal,
                'is_anomaly': [1 if i in anomaly_indices else 0 for i in range(n_samples)]
            })
            
            print(f"✓ Generated {n_samples} data points with {len(anomaly_indices)} anomalies")
            
        elif data_source == "file":
            # Load from file
            filename = "anomaly_data.csv"
            if os.path.exists(filename):
                self.data = pd.read_csv(filename, parse_dates=['time'])
                print(f"✓ Loaded data from {filename}")
            else:
                print(f"✗ File {filename} not found, generating sample data instead")
                self.parameters["data_source"]["value"] = "sample"
                self.load_data()
                
        if CONNECTOR_AVAILABLE:
            self.set_variable("data_points_processed", len(self.data))
            
    def detect_anomalies_zscore(self):
        """Detect anomalies using Robust Z-score with MAD"""
        print("\n" + "="*60)
        print("METHOD 1: ROBUST Z-SCORE WITH MAD")
        print("="*60)
        
        threshold = self.get_parameter("threshold")
        
        # Calculate median and MAD
        median = self.data['value'].median()
        mad = median_abs_deviation(self.data['value'])
        
        # Calculate modified Z-score
        modified_z_scores = 0.6745 * (self.data['value'] - median) / mad
        
        # Identify anomalies
        anomalies = np.abs(modified_z_scores) > threshold
        self.anomalies['zscore'] = anomalies
        
        n_anomalies = anomalies.sum()
        print(f"✓ Detected {n_anomalies} anomalies using threshold {threshold}")
        
        if CONNECTOR_AVAILABLE:
            self.update_results("zscore_anomalies", int(n_anomalies))
            
        return anomalies
    
    def detect_anomalies_isolation_forest(self):
        """Detect anomalies using Isolation Forest"""
        print("\n" + "="*60)
        print("METHOD 2: ISOLATION FOREST")
        print("="*60)
        
        contamination = self.get_parameter("contamination")
        
        # Prepare data
        X = self.data['value'].values.reshape(-1, 1)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        predictions = iso_forest.fit_predict(X)
        anomalies = predictions == -1
        
        self.anomalies['isolation_forest'] = anomalies
        
        n_anomalies = anomalies.sum()
        print(f"✓ Detected {n_anomalies} anomalies with contamination={contamination}")
        
        if CONNECTOR_AVAILABLE:
            self.update_results("isolation_forest_anomalies", int(n_anomalies))
            
        return anomalies
    
    def detect_anomalies_lof(self):
        """Detect anomalies using Local Outlier Factor"""
        print("\n" + "="*60)
        print("METHOD 3: LOCAL OUTLIER FACTOR")
        print("="*60)
        
        contamination = self.get_parameter("contamination")
        n_neighbors = self.get_parameter("n_neighbors")
        
        # Prepare data
        X = self.data['value'].values.reshape(-1, 1)
        
        # Train LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        predictions = lof.fit_predict(X)
        anomalies = predictions == -1
        
        self.anomalies['lof'] = anomalies
        
        n_anomalies = anomalies.sum()
        print(f"✓ Detected {n_anomalies} anomalies with n_neighbors={n_neighbors}")
        
        if CONNECTOR_AVAILABLE:
            self.update_results("lof_anomalies", int(n_anomalies))
            
        return anomalies
    
    def visualize_results(self):
        """Visualize the anomaly detection results"""
        if not self.get_parameter("save_plots"):
            return
            
        print("\n" + "="*60)
        print("VISUALIZING RESULTS")
        print("="*60)
        
        # Create subplots for each method
        n_methods = len(self.anomalies)
        fig, axes = plt.subplots(n_methods, 1, figsize=(15, 5*n_methods))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method, anomalies) in enumerate(self.anomalies.items()):
            ax = axes[idx]
            
            # Plot normal points
            normal_mask = ~anomalies
            ax.scatter(self.data.loc[normal_mask, 'time'], 
                      self.data.loc[normal_mask, 'value'],
                      c='blue', label='Normal', alpha=0.6, s=20)
            
            # Plot anomalies
            ax.scatter(self.data.loc[anomalies, 'time'],
                      self.data.loc[anomalies, 'value'],
                      c='red', label='Anomaly', s=50, marker='x')
            
            ax.set_title(f'Anomaly Detection - {method.upper()}', fontsize=14)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"anomaly_detection_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Results saved to {filename}")
        plt.close()
        
        if CONNECTOR_AVAILABLE:
            self.update_results("visualization_saved", filename)
    
    def share_results_with_other_scripts(self):
        """Share detection results with other scripts via connector"""
        if not CONNECTOR_AVAILABLE or not hasattr(self, 'client'):
            return
            
        # Prepare summary data
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_data_points": len(self.data),
            "methods_used": list(self.anomalies.keys()),
            "anomalies_per_method": {
                method: int(anomalies.sum()) 
                for method, anomalies in self.anomalies.items()
            },
            "parameters": {
                "threshold": self.get_parameter("threshold"),
                "contamination": self.get_parameter("contamination"),
                "n_neighbors": self.get_parameter("n_neighbors")
            }
        }
        
        # Broadcast to other scripts
        self.client.broadcast_data({
            "type": "anomaly_detection_results",
            "summary": summary,
            "detailed_results": self.results
        })
        
        print("\n✓ Results shared with other scripts via connector")
    
    def run(self):
        """Main execution method"""
        print("\n" + "="*40)
        print("STARTING ANOMALY DETECTION")
        print("="*40)
        
        # Load data
        self.load_data()
        
        # Run selected detection method(s)
        method = self.get_parameter("method")
        
        if method == "all":
            # Run all methods
            self.detect_anomalies_zscore()
            self.detect_anomalies_isolation_forest()
            self.detect_anomalies_lof()
        elif method == "zscore":
            self.detect_anomalies_zscore()
        elif method == "isolation_forest":
            self.detect_anomalies_isolation_forest()
        elif method == "lof":
            self.detect_anomalies_lof()
        
        # Calculate total unique anomalies
        if self.anomalies:
            all_anomalies = np.logical_or.reduce(list(self.anomalies.values()))
            total_anomalies = int(all_anomalies.sum())
            
            if CONNECTOR_AVAILABLE:
                self.set_variable("total_anomalies", total_anomalies)
                self.set_variable("detection_rate", total_anomalies / len(self.data))
                
                # Find last anomaly
                anomaly_indices = np.where(all_anomalies)[0]
                if len(anomaly_indices) > 0:
                    last_idx = anomaly_indices[-1]
                    self.set_variable("last_anomaly_timestamp", 
                                    self.data.iloc[last_idx]['time'].isoformat())
        
        # Visualize results
        self.visualize_results()
        
        # Share results if connected
        if CONNECTOR_AVAILABLE:
            self.share_results_with_other_scripts()
        
        print("\n" + "="*40)
        print("ANOMALY DETECTION COMPLETE")
        print("="*40)
        
        # Print summary
        print("\nSUMMARY:")
        for method, anomalies in self.anomalies.items():
            print(f"- {method}: {anomalies.sum()} anomalies detected")
        
        if CONNECTOR_AVAILABLE:
            print(f"\nConnector Status: Connected")
            print(f"Results broadcasted: Yes")


# Main execution
def main():
    """Main function to run the anomaly detection system"""
    detector = AnomalyDetectionSystem()
    
    if CONNECTOR_AVAILABLE and "--with-connector" in sys.argv:
        # Run with connector integration
        print("\n✓ Running with connector integration")
        detector.run_with_connector()
    else:
        # Run standalone
        print("\n✓ Running in standalone mode")
        detector.run()


if __name__ == "__main__":
    main()