#!/usr/bin/env python3
"""
Machine Learning for Everyone - Comprehensive Tutorial Script
Based on Kylie Ying's tutorial covering supervised and unsupervised learning
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log(message, level="INFO"):
    """Log messages with timestamp and color coding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if level == "INFO":
        print(f"{Colors.BLUE}[{timestamp}] {Colors.GREEN}INFO: {message}{Colors.ENDC}")
    elif level == "WARNING":
        print(f"{Colors.BLUE}[{timestamp}] {Colors.YELLOW}WARNING: {message}{Colors.ENDC}")
    elif level == "ERROR":
        print(f"{Colors.BLUE}[{timestamp}] {Colors.RED}ERROR: {message}{Colors.ENDC}")
    elif level == "HEADER":
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
        print(f"{message}")
        print(f"{'='*60}{Colors.ENDC}\n")

def check_and_install_package(package_name, import_name=None):
    """Check if package is installed, install if not"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        log(f"Package '{package_name}' is already installed")
        return True
    except ImportError:
        log(f"Package '{package_name}' not found. Installing...", "WARNING")
        try:
            # Use --upgrade to ensure latest version
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            log(f"Successfully installed '{package_name}'", "INFO")
            return True
        except subprocess.CalledProcessError:
            log(f"Failed to install '{package_name}'", "ERROR")
            return False

def ensure_dependencies():
    """Ensure all required packages are installed"""
    log("CHECKING AND INSTALLING DEPENDENCIES", "HEADER")
    
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("tensorflow", "tensorflow"),
        ("imbalanced-learn", "imblearn"),
        ("requests", "requests"),  # For downloading datasets
    ]
    
    all_installed = True
    for package, import_name in required_packages:
        if not check_and_install_package(package, import_name):
            all_installed = False
    
    if not all_installed:
        log("Some packages failed to install. Please check your environment.", "ERROR")
        sys.exit(1)
    
    log("All dependencies successfully installed!", "INFO")
    return True

# Ensure dependencies before importing
ensure_dependencies()

# Now import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MLTutorial:
    """Main class for the Machine Learning tutorial"""
    
    def __init__(self):
        log("INITIALIZING MACHINE LEARNING TUTORIAL", "HEADER")
        self.data_dir = "ml_tutorial_data"
        self.create_data_directory()
    
    def create_data_directory(self):
        """Create directory for storing datasets"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            log(f"Created data directory: {self.data_dir}")
        else:
            log(f"Data directory already exists: {self.data_dir}")
    
    def download_dataset(self, url, filename):
        """Download dataset from URL"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            log(f"Dataset '{filename}' already exists")
            return filepath
        
        log(f"Downloading dataset from {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            log(f"Successfully downloaded '{filename}'")
            return filepath
        except Exception as e:
            log(f"Failed to download dataset: {str(e)}", "ERROR")
            return None
    
    def load_magic_dataset(self):
        """Load and preprocess the Magic Gamma Telescope dataset"""
        log("LOADING MAGIC GAMMA TELESCOPE DATASET", "HEADER")
        
        # Define column names based on UCI documentation
        cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 
                'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
        
        # Try to load from local file first
        filepath = os.path.join(self.data_dir, "magic04.data")
        if not os.path.exists(filepath):
            log("Dataset not found locally. Please download from UCI ML Repository.", "WARNING")
            log("Visit: https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope")
            # Create sample data for demonstration
            log("Creating sample data for demonstration purposes")
            np.random.seed(42)
            n_samples = 1000
            data = {col: np.random.randn(n_samples) for col in cols[:-1]}
            data['class'] = np.random.choice(['g', 'h'], n_samples)
            df = pd.DataFrame(data)
        else:
            log(f"Loading dataset from {filepath}")
            df = pd.read_csv(filepath, names=cols)
        
        log(f"Dataset shape: {df.shape}")
        log(f"First few rows:\n{df.head()}")
        
        # Convert class labels to binary
        log("Converting class labels from 'g'/'h' to 0/1")
        df['class'] = (df['class'] == 'g').astype(int)
        
        return df
    
    def visualize_features(self, df):
        """Visualize feature distributions"""
        log("VISUALIZING FEATURE DISTRIBUTIONS", "HEADER")
        
        # Plot histograms for each feature
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, label in enumerate(df.columns[:-1]):
            log(f"Plotting histogram for feature: {label}")
            axes[idx].hist(df[df['class'] == 1][label], bins=30, alpha=0.7, 
                          label='gamma', density=True, color='blue')
            axes[idx].hist(df[df['class'] == 0][label], bins=30, alpha=0.7, 
                          label='hadron', density=True, color='red')
            axes[idx].set_title(label)
            axes[idx].set_ylabel('Probability')
            axes[idx].set_xlabel(label)
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'feature_distributions.png'))
        log(f"Saved feature distributions plot to {os.path.join(self.data_dir, 'feature_distributions.png')}")
        plt.close()
    
    def prepare_data(self, df, oversample=True):
        """Prepare data for training"""
        log("PREPARING DATA FOR TRAINING", "HEADER")
        
        # Split into train, validation, and test sets
        log("Splitting data into train (60%), validation (20%), and test (20%)")
        train, temp = train_test_split(df, test_size=0.4, random_state=42, shuffle=True)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        log(f"Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}")
        
        # Function to scale dataset
        def scale_dataset(dataframe, oversample=False):
            X = dataframe[dataframe.columns[:-1]].values
            y = dataframe[dataframe.columns[-1]].values
            
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            if oversample:
                log("Oversampling minority class")
                ros = RandomOverSampler(random_state=42)
                X, y = ros.fit_resample(X, y)
                log(f"After oversampling - Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")
            
            return X, y
        
        log("Scaling features using StandardScaler")
        X_train, y_train = scale_dataset(train, oversample=oversample)
        X_val, y_val = scale_dataset(val, oversample=False)
        X_test, y_test = scale_dataset(test, oversample=False)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_knn(self, X_train, y_train, X_test, y_test):
        """Train K-Nearest Neighbors classifier"""
        log("TRAINING K-NEAREST NEIGHBORS CLASSIFIER", "HEADER")
        
        for k in [1, 3, 5]:
            log(f"Training KNN with k={k}")
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train, y_train)
            
            y_pred = knn_model.predict(X_test)
            log(f"KNN (k={k}) Classification Report:")
            print(classification_report(y_test, y_pred))
    
    def train_naive_bayes(self, X_train, y_train, X_test, y_test):
        """Train Naive Bayes classifier"""
        log("TRAINING NAIVE BAYES CLASSIFIER", "HEADER")
        
        nb_model = GaussianNB()
        log("Fitting Gaussian Naive Bayes model")
        nb_model.fit(X_train, y_train)
        
        y_pred = nb_model.predict(X_test)
        log("Naive Bayes Classification Report:")
        print(classification_report(y_test, y_pred))
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression classifier"""
        log("TRAINING LOGISTIC REGRESSION CLASSIFIER", "HEADER")
        
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        log("Fitting Logistic Regression model")
        lr_model.fit(X_train, y_train)
        
        y_pred = lr_model.predict(X_test)
        log("Logistic Regression Classification Report:")
        print(classification_report(y_test, y_pred))
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train Support Vector Machine classifier"""
        log("TRAINING SUPPORT VECTOR MACHINE CLASSIFIER", "HEADER")
        
        svm_model = SVC(random_state=42)
        log("Fitting SVM model (this may take a while...)")
        svm_model.fit(X_train, y_train)
        
        y_pred = svm_model.predict(X_test)
        log("SVM Classification Report:")
        print(classification_report(y_test, y_pred))
    
    def create_neural_network(self, input_shape):
        """Create neural network model for classification"""
        log("Creating neural network architecture")
        
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train Neural Network classifier"""
        log("TRAINING NEURAL NETWORK CLASSIFIER", "HEADER")
        
        # Grid search parameters
        node_options = [16, 32, 64]
        dropout_options = [0.0, 0.2]
        lr_options = [0.005, 0.001, 0.01]
        batch_options = [32, 64, 128]
        
        best_val_loss = float('inf')
        best_model = None
        
        log("Starting neural network hyperparameter search")
        
        # For demonstration, we'll just train one configuration
        # In practice, you'd want to try all combinations
        log("Training with nodes=32, dropout=0.2, lr=0.001, batch_size=32")
        
        model = self.create_neural_network(X_train.shape[1])
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            verbose=0
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        # Evaluate on test set
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)
        
        log("Neural Network Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return model
    
    def plot_training_history(self, history):
        """Plot neural network training history"""
        log("Plotting training history")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'nn_training_history.png'))
        log(f"Saved training history plot to {os.path.join(self.data_dir, 'nn_training_history.png')}")
        plt.close()
    
    def regression_example(self):
        """Demonstrate regression with bike sharing dataset"""
        log("REGRESSION EXAMPLE - BIKE SHARING DATASET", "HEADER")
        
        # Create synthetic bike sharing data for demonstration
        log("Creating synthetic bike sharing data")
        np.random.seed(42)
        n_samples = 500
        
        temperature = np.random.uniform(-20, 40, n_samples)
        humidity = np.random.uniform(0, 100, n_samples)
        hour = 12  # Fixed hour for simplicity
        
        # Create target with some relationship to features
        bike_count = (
            50 + 
            2 * temperature + 
            -0.5 * humidity + 
            np.random.normal(0, 20, n_samples)
        )
        bike_count = np.maximum(0, bike_count)  # No negative counts
        
        df = pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'bike_count': bike_count
        })
        
        log(f"Created dataset with {len(df)} samples")
        
        # Visualize relationships
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(df['temperature'], df['bike_count'], alpha=0.5)
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Bike Count')
        ax1.set_title('Bike Count vs Temperature')
        
        ax2.scatter(df['humidity'], df['bike_count'], alpha=0.5)
        ax2.set_xlabel('Humidity')
        ax2.set_ylabel('Bike Count')
        ax2.set_title('Bike Count vs Humidity')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'bike_data_scatter.png'))
        log(f"Saved scatter plots to {os.path.join(self.data_dir, 'bike_data_scatter.png')}")
        plt.close()
        
        # Prepare data
        X = df[['temperature', 'humidity']].values
        y = df['bike_count'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Linear Regression
        log("Training Linear Regression model")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        y_pred_lr = lr_model.predict(X_test)
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mse_lr)
        
        log(f"Linear Regression - RMSE: {rmse_lr:.2f}")
        log(f"Linear Regression - R² Score: {lr_model.score(X_test, y_test):.3f}")
        
        # Neural Network for Regression
        log("Training Neural Network for regression")
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train model
        nn_model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(2,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        nn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss='mean_squared_error'
        )
        
        history = nn_model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        y_pred_nn = nn_model.predict(X_test_scaled).flatten()
        mse_nn = mean_squared_error(y_test, y_pred_nn)
        rmse_nn = np.sqrt(mse_nn)
        
        log(f"Neural Network - RMSE: {rmse_nn:.2f}")
        
        # Plot predictions vs actual
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(y_test, y_pred_lr, alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predictions')
        ax1.set_title('Linear Regression: Predictions vs Actual')
        
        ax2.scatter(y_test, y_pred_nn, alpha=0.5)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax2.set_xlabel('True Values')
        ax2.set_ylabel('Predictions')
        ax2.set_title('Neural Network: Predictions vs Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'regression_predictions.png'))
        log(f"Saved prediction plots to {os.path.join(self.data_dir, 'regression_predictions.png')}")
        plt.close()
    
    def unsupervised_learning_example(self):
        """Demonstrate unsupervised learning with seeds dataset"""
        log("UNSUPERVISED LEARNING EXAMPLE - SEEDS DATASET", "HEADER")
        
        # Create synthetic seeds data for demonstration
        log("Creating synthetic seeds dataset")
        np.random.seed(42)
        n_samples_per_class = 70
        n_classes = 3
        
        # Generate data for 3 wheat varieties
        data = []
        true_labels = []
        
        for i in range(n_classes):
            # Each class has slightly different characteristics
            area = np.random.normal(15 + i*2, 1.5, n_samples_per_class)
            perimeter = np.random.normal(14 + i*1.5, 1, n_samples_per_class)
            compactness = np.random.normal(0.88 + i*0.02, 0.01, n_samples_per_class)
            length = np.random.normal(5.5 + i*0.3, 0.2, n_samples_per_class)
            width = np.random.normal(3 + i*0.2, 0.15, n_samples_per_class)
            asymmetry = np.random.normal(2.5 + i*0.5, 0.3, n_samples_per_class)
            groove = np.random.normal(5 + i*0.4, 0.2, n_samples_per_class)
            
            class_data = np.column_stack([
                area, perimeter, compactness, length, width, asymmetry, groove
            ])
            data.append(class_data)
            true_labels.extend([i+1] * n_samples_per_class)
        
        X = np.vstack(data)
        y_true = np.array(true_labels)
        
        columns = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove']
        df = pd.DataFrame(X, columns=columns)
        df['class'] = y_true
        
        log(f"Created dataset with {len(df)} samples and {n_classes} classes")
        
        # K-Means Clustering
        log("Performing K-Means clustering with k=3")
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # PCA for visualization
        log("Applying PCA for dimensionality reduction")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        log(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        # Visualize results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot K-Means results
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        ax1.set_title('K-Means Clustering Results')
        plt.colorbar(scatter1, ax=ax1)
        
        # Plot true labels
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.6)
        ax2.set_xlabel('First Principal Component')
        ax2.set_ylabel('Second Principal Component')
        ax2.set_title('True Class Labels')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'unsupervised_results.png'))
        log(f"Saved unsupervised learning results to {os.path.join(self.data_dir, 'unsupervised_results.png')}")
        plt.close()
        
        # Additional visualization - feature relationships
        log("Creating pairwise feature plots")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        feature_pairs = [
            ('area', 'perimeter'),
            ('area', 'compactness'),
            ('perimeter', 'asymmetry'),
            ('compactness', 'asymmetry'),
            ('length', 'width'),
            ('asymmetry', 'groove')
        ]
        
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            axes[idx].scatter(df[feat1], df[feat2], c=clusters, cmap='viridis', alpha=0.6)
            axes[idx].set_xlabel(feat1)
            axes[idx].set_ylabel(feat2)
            axes[idx].set_title(f'{feat1} vs {feat2}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'feature_pairs_clustering.png'))
        log(f"Saved feature pairs plot to {os.path.join(self.data_dir, 'feature_pairs_clustering.png')}")
        plt.close()
    
    def run_full_tutorial(self):
        """Run the complete machine learning tutorial"""
        log("STARTING COMPLETE MACHINE LEARNING TUTORIAL", "HEADER")
        
        # Part 1: Classification with Magic Gamma Telescope Dataset
        log("PART 1: SUPERVISED LEARNING - CLASSIFICATION", "HEADER")
        
        # Load and prepare data
        df = self.load_magic_dataset()
        self.visualize_features(df)
        
        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df)
        
        # Train various classifiers
        self.train_knn(X_train, y_train, X_test, y_test)
        self.train_naive_bayes(X_train, y_train, X_test, y_test)
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)
        self.train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Part 2: Regression
        log("\nPART 2: SUPERVISED LEARNING - REGRESSION", "HEADER")
        self.regression_example()
        
        # Part 3: Unsupervised Learning
        log("\nPART 3: UNSUPERVISED LEARNING", "HEADER")
        self.unsupervised_learning_example()
        
        log("\nTUTORIAL COMPLETED SUCCESSFULLY!", "HEADER")
        log(f"All results saved in: {os.path.abspath(self.data_dir)}")

def main():
    """Main function to run the tutorial"""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("="*60)
    print("MACHINE LEARNING FOR EVERYONE")
    print("Based on Kylie Ying's Tutorial")
    print("="*60)
    print(f"{Colors.ENDC}")
    
    # Create and run tutorial
    tutorial = MLTutorial()
    tutorial.run_full_tutorial()

if __name__ == "__main__":
    main()
