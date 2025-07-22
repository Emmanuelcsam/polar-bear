# Fiber Optic Anomaly Detection ML Model

This notebook converts the statistical anomaly detection system into a machine learning model.

## 1. Setup and Dependencies

```python
# Install required packages
!pip install firebase-admin pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
import os
from datetime import datetime

# Firebase imports
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib

# Import your detection module
import sys
sys.path.append('.')  # Add current directory to path
from detection import OmniFiberAnalyzer, OmniConfig

print("Dependencies loaded successfully!")
```

## 2. Initialize Firebase

```python
# Initialize Firebase Admin
cred = credentials.Certificate('service-account.json')
app = firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-project-id.appspot.com'
})
db = firestore.client()
bucket = storage.bucket()

print("Firebase initialized!")
```

## 3. Data Collection and Feature Extraction

```python
class FiberMLDataCollector:
    """Collect and prepare fiber optic data for ML training"""
    
    def __init__(self):
        self.config = OmniConfig()
        self.analyzer = OmniFiberAnalyzer(self.config)
        self.features_list = []
        self.labels_list = []
        
    def process_directory(self, directory_path, label):
        """Process all images in a directory with given label"""
        valid_extensions = ['.json', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        
        for filename in os.listdir(directory_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                filepath = os.path.join(directory_path, filename)
                print(f"Processing: {filename}")
                
                try:
                    # Load image
                    image = self.analyzer.load_image(filepath)
                    if image is None:
                        continue
                    
                    # Extract features
                    features, feature_names = self.analyzer.extract_ultra_comprehensive_features(image)
                    
                    # Add to lists
                    self.features_list.append(features)
                    self.labels_list.append(label)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
        return feature_names
    
    def create_dataframe(self, feature_names):
        """Create pandas DataFrame from collected features"""
        # Convert features to matrix
        feature_matrix = np.zeros((len(self.features_list), len(feature_names)))
        
        for i, features in enumerate(self.features_list):
            for j, fname in enumerate(feature_names):
                feature_matrix[i, j] = features.get(fname, 0)
        
        # Create DataFrame
        df = pd.DataFrame(feature_matrix, columns=feature_names)
        df['label'] = self.labels_list
        
        return df

# Collect data
collector = FiberMLDataCollector()

# Process normal fiber images (label = 0)
print("Processing normal fiber images...")
feature_names = collector.process_directory('data/normal_fibers', label=0)

# Process anomalous fiber images (label = 1) 
print("\nProcessing anomalous fiber images...")
collector.process_directory('data/anomalous_fibers', label=1)

# Create DataFrame
df = collector.create_dataframe(feature_names)
print(f"\nDataset created: {df.shape[0]} samples, {df.shape[1]} features")

# Save to CSV for future use
df.to_csv('fiber_anomaly_dataset.csv', index=False)
```

## 4. Exploratory Data Analysis

```python
# Load data if already saved
df = pd.read_csv('fiber_anomaly_dataset.csv')

# Basic statistics
print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(df['label'].value_counts())

# Visualize label distribution
plt.figure(figsize=(8, 6))
df['label'].value_counts().plot(kind='bar')
plt.title('Distribution of Normal vs Anomalous Samples')
plt.xlabel('Label (0=Normal, 1=Anomalous)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Feature importance preview (using a quick Random Forest)
X_temp = df.drop('label', axis=1)
y_temp = df['label']

rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_temp, y_temp)

# Top 20 most important features
feature_importance = pd.DataFrame({
    'feature': X_temp.columns,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Top 20 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

## 5. Data Preprocessing

```python
# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

## 6. Train Machine Learning Models

```python
# Try multiple algorithms
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Normal', 'Anomalous']))
```

## 7. Model Evaluation and Selection

```python
# Compare models
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print(f"Best model: {best_model_name}")
print(f"Best accuracy: {results[best_model_name]['accuracy']:.4f}")

# Confusion matrix for best model
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## 8. Add Anomaly Score Prediction

```python
# Train a regression model for anomaly scores
from sklearn.ensemble import RandomForestRegressor

# For regression, we need continuous anomaly scores
# You would need to generate these from your analyzer
# For demo, we'll create synthetic scores based on labels
y_scores = df['label'].apply(lambda x: np.random.uniform(0.7, 1.0) if x == 1 
                                      else np.random.uniform(0.0, 0.3))

# Train regression model
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_scores, test_size=0.2, random_state=42
)

regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

regressor.fit(scaler.fit_transform(X_train_reg), y_train_reg)

# Evaluate
from sklearn.metrics import mean_absolute_error, r2_score

y_pred_reg = regressor.predict(scaler.transform(X_test_reg))
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Regression MAE: {mae:.4f}")
print(f"Regression R²: {r2:.4f}")
```

## 9. Create Combined Model Pipeline

```python
class FiberAnomalyMLModel:
    """Combined classification and regression model for fiber anomaly detection"""
    
    def __init__(self, classifier, regressor, scaler, feature_names):
        self.classifier = classifier
        self.regressor = regressor
        self.scaler = scaler
        self.feature_names = feature_names
        
    def predict(self, features_dict):
        """Predict both class and anomaly score"""
        # Convert features dict to array
        feature_array = np.array([features_dict.get(fname, 0) 
                                 for fname in self.feature_names]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(feature_array)
        
        # Predictions
        is_anomalous = self.classifier.predict(features_scaled)[0]
        anomaly_score = self.regressor.predict(features_scaled)[0]
        
        # Get probability
        anomaly_probability = self.classifier.predict_proba(features_scaled)[0, 1]
        
        return {
            'is_anomalous': bool(is_anomalous),
            'anomaly_score': float(anomaly_score),
            'anomaly_probability': float(anomaly_probability),
            'severity': self._get_severity(anomaly_score)
        }
    
    def _get_severity(self, score):
        """Convert score to severity level"""
        if score < 0.3:
            return 'NORMAL'
        elif score < 0.5:
            return 'LOW'
        elif score < 0.7:
            return 'MEDIUM'
        elif score < 0.9:
            return 'HIGH'
        else:
            return 'CRITICAL'

# Create combined model
combined_model = FiberAnomalyMLModel(
    classifier=best_model,
    regressor=regressor,
    scaler=scaler,
    feature_names=list(X.columns)
)

# Test the combined model
test_features = X_test.iloc[0].to_dict()
prediction = combined_model.predict(test_features)
print("Sample prediction:", prediction)
```

## 10. Export Model for ML Engine

```python
# Save all model components
model_dir = 'fiber_anomaly_model_v1'
os.makedirs(model_dir, exist_ok=True)

# Save models
joblib.dump(best_model, f'{model_dir}/classifier.joblib')
joblib.dump(regressor, f'{model_dir}/regressor.joblib')
joblib.dump(scaler, f'{model_dir}/scaler.joblib')

# Save feature names
with open(f'{model_dir}/feature_names.json', 'w') as f:
    json.dump(list(X.columns), f)

# Save metadata
metadata = {
    'version': '1.0',
    'created_at': datetime.now().isoformat(),
    'classifier_type': type(best_model).__name__,
    'regressor_type': type(regressor).__name__,
    'n_features': len(X.columns),
    'accuracy': results[best_model_name]['accuracy'],
    'feature_extraction_version': '1.5'
}

with open(f'{model_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved to {model_dir}/")

# Upload to Firebase Storage
for filename in os.listdir(model_dir):
    local_path = os.path.join(model_dir, filename)
    blob_path = f'ml-models/fiber-anomaly-v1/{filename}'
    
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {filename} to {blob_path}")

print("\nModel uploaded to Firebase Storage!")
```

## 11. Create Inference Wrapper

```python
class FiberAnomalyPredictor:
    """Wrapper for making predictions on new fiber images"""
    
    def __init__(self, model_dir):
        # Load models
        self.classifier = joblib.load(f'{model_dir}/classifier.joblib')
        self.regressor = joblib.load(f'{model_dir}/regressor.joblib')
        self.scaler = joblib.load(f'{model_dir}/scaler.joblib')
        
        # Load feature names
        with open(f'{model_dir}/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Initialize feature extractor
        self.config = OmniConfig()
        self.analyzer = OmniFiberAnalyzer(self.config)
    
    def predict_image(self, image_path):
        """Make prediction on a single image"""
        # Load image
        image = self.analyzer.load_image(image_path)
        if image is None:
            return None
        
        # Extract features
        features, _ = self.analyzer.extract_ultra_comprehensive_features(image)
        
        # Prepare feature array
        feature_array = np.array([features.get(fname, 0) 
                                 for fname in self.feature_names]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(feature_array)
        
        # Make predictions
        is_anomalous = self.classifier.predict(features_scaled)[0]
        anomaly_score = self.regressor.predict(features_scaled)[0]
        anomaly_probability = self.classifier.predict_proba(features_scaled)[0, 1]
        
        # Get top contributing features
        if hasattr(self.classifier, 'feature_importances_'):
            importances = self.classifier.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            top_features = [(self.feature_names[i], importances[i]) 
                           for i in top_indices]
        else:
            top_features = []
        
        return {
            'image_path': image_path,
            'is_anomalous': bool(is_anomalous),
            'anomaly_score': float(anomaly_score),
            'anomaly_probability': float(anomaly_probability),
            'severity': self._get_severity(anomaly_score),
            'top_features': top_features,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_severity(self, score):
        if score < 0.3:
            return 'NORMAL'
        elif score < 0.5:
            return 'LOW'
        elif score < 0.7:
            return 'MEDIUM'
        elif score < 0.9:
            return 'HIGH'
        else:
            return 'CRITICAL'

# Test the predictor
predictor = FiberAnomalyPredictor(model_dir)
test_result = predictor.predict_image('test_fiber.png')
print("Test prediction:", json.dumps(test_result, indent=2))
```

## 12. Save Training History to Firestore

```python
# Save model training history
training_record = {
    'model_id': 'fiber-anomaly-v1',
    'created_at': firestore.SERVER_TIMESTAMP,
    'metrics': {
        'accuracy': float(results[best_model_name]['accuracy']),
        'cv_mean': float(results[best_model_name]['cv_mean']),
        'cv_std': float(results[best_model_name]['cv_std']),
        'n_samples': len(df),
        'n_features': len(X.columns)
    },
    'model_type': best_model_name,
    'storage_path': 'ml-models/fiber-anomaly-v1/'
}

# Add to Firestore
db.collection('ml_models').add(training_record)
print("Training record saved to Firestore")
```