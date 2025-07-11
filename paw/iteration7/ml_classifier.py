import json
import time
import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

def extract_features_from_images():
    """Extract features from all available images"""
    
    print("[ML] Extracting features from images...")
    
    features = []
    labels = []
    filenames = []
    
    # Load from various sources
    sources = [
        ('pixel_data.json', 'single'),
        ('batch_results.json', 'batch'),
        ('parallel_batch.json', 'parallel'),
        ('gpu_batch_results.json', 'gpu')
    ]
    
    for source_file, source_type in sources:
        if os.path.exists(source_file):
            with open(source_file, 'r') as f:
                data = json.load(f)
            
            if source_type == 'single' and 'pixels' in data:
                # Single image features
                pixels = np.array(data['pixels'])
                feature_vec = extract_pixel_features(pixels)
                features.append(feature_vec)
                labels.append('original')
                filenames.append('pixel_data')
                
            elif 'results' in data:
                # Batch results
                for result in data['results']:
                    if 'statistics' in result:
                        stats = result['statistics']
                        feature_vec = [
                            stats.get('mean', 0),
                            stats.get('std', 0),
                            stats.get('entropy', 0),
                            stats.get('edge_strength', 0)
                        ]
                        features.append(feature_vec)
                        labels.append(source_type)
                        filenames.append(result.get('filename', 'unknown'))
    
    # Also check for generated images
    for file in os.listdir('.'):
        if file.startswith('generated_') and file.endswith('.jpg'):
            labels.append('generated')
        elif file.startswith('neural_generated_') and file.endswith('.jpg'):
            labels.append('neural')
        elif file.startswith('gpu_') and file.endswith('.jpg'):
            labels.append('gpu')
    
    print(f"[ML] Extracted features from {len(features)} images")
    
    return np.array(features), labels, filenames

def extract_pixel_features(pixels):
    """Extract statistical features from pixel array"""
    
    pixels = np.array(pixels)
    
    features = [
        np.mean(pixels),
        np.std(pixels),
        np.min(pixels),
        np.max(pixels),
        np.median(pixels),
        np.percentile(pixels, 25),
        np.percentile(pixels, 75),
        len(np.unique(pixels))
    ]
    
    # Add histogram features
    hist, _ = np.histogram(pixels, bins=10, range=(0, 256))
    features.extend(hist / len(pixels))  # Normalized histogram
    
    return features

def perform_clustering(features, labels):
    """Perform clustering analysis"""
    
    print("\n[ML] Performing clustering analysis...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Save scaler
    joblib.dump(scaler, 'ml_scaler.pkl')
    
    # K-means clustering
    n_clusters = min(5, len(features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(features_scaled)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    dbscan_labels = dbscan.fit_predict(features_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    print(f"[ML] K-means found {n_clusters} clusters")
    print(f"[ML] DBSCAN found {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)} clusters")
    print(f"[ML] PCA explained variance: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f}")
    
    # Save clustering results
    clustering_results = {
        'n_samples': len(features),
        'n_features': features.shape[1] if len(features) > 0 else 0,
        'kmeans': {
            'n_clusters': n_clusters,
            'labels': kmeans_labels.tolist(),
            'inertia': float(kmeans.inertia_),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        },
        'dbscan': {
            'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'labels': dbscan_labels.tolist(),
            'n_noise': int(np.sum(dbscan_labels == -1))
        },
        'pca': {
            'components': features_pca.tolist(),
            'explained_variance': pca.explained_variance_ratio_.tolist()
        }
    }
    
    with open('ml_clustering.json', 'w') as f:
        json.dump(clustering_results, f)
    
    return kmeans, dbscan, pca, features_pca

def train_classifiers(features, labels):
    """Train multiple classifiers"""
    
    if len(features) < 10:
        print("[ML] Not enough samples for classification")
        return
    
    print("\n[ML] Training classifiers...")
    
    # Convert labels to numeric
    unique_labels = list(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_map[label] for label in labels]
    
    # Split data (simple split, no train/test for small datasets)
    X = features
    y = np.array(numeric_labels)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    rf_score = rf.score(X, y)
    
    # Train SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X, y)
    svm_score = svm.score(X, y)
    
    print(f"[ML] Random Forest accuracy: {rf_score:.2f}")
    print(f"[ML] SVM accuracy: {svm_score:.2f}")
    
    # Feature importance
    feature_importance = rf.feature_importances_
    
    # Save models
    joblib.dump(rf, 'ml_rf_model.pkl')
    joblib.dump(svm, 'ml_svm_model.pkl')
    
    # Save classification results
    classification_results = {
        'label_map': label_map,
        'n_classes': len(unique_labels),
        'n_samples': len(features),
        'random_forest': {
            'accuracy': float(rf_score),
            'feature_importance': feature_importance.tolist()
        },
        'svm': {
            'accuracy': float(svm_score)
        }
    }
    
    with open('ml_classification.json', 'w') as f:
        json.dump(classification_results, f)
    
    return rf, svm

def predict_new_image():
    """Predict class of new image if available"""
    
    print("\n[ML] Looking for new images to classify...")
    
    # Check if models exist
    if not os.path.exists('ml_rf_model.pkl'):
        print("[ML] No trained model found")
        return
    
    # Load models
    rf = joblib.load('ml_rf_model.pkl')
    scaler = joblib.load('ml_scaler.pkl')
    
    # Find a recent image
    test_image = None
    for file in sorted(os.listdir('.'), reverse=True):
        if file.endswith('.jpg') and 'test' not in file:
            test_image = file
            break
    
    if not test_image:
        print("[ML] No test image found")
        return
    
    # Extract features (simplified)
    from PIL import Image
    img = Image.open(test_image).convert('L')
    pixels = np.array(img.getdata())
    features = extract_pixel_features(pixels)
    
    # Pad features if needed
    if len(features) < rf.n_features_in_:
        features = np.pad(features, (0, rf.n_features_in_ - len(features)))
    elif len(features) > rf.n_features_in_:
        features = features[:rf.n_features_in_]
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    prediction = rf.predict(features_scaled)[0]
    probability = rf.predict_proba(features_scaled)[0]
    
    # Load label map
    with open('ml_classification.json', 'r') as f:
        class_data = json.load(f)
        label_map = class_data['label_map']
        reverse_map = {v: k for k, v in label_map.items()}
    
    predicted_label = reverse_map.get(prediction, 'unknown')
    
    print(f"[ML] Prediction for {test_image}:")
    print(f"[ML] Class: {predicted_label}")
    print(f"[ML] Confidence: {max(probability):.2f}")
    
    # Save prediction
    with open('ml_prediction.json', 'w') as f:
        json.dump({
            'image': test_image,
            'predicted_class': predicted_label,
            'confidence': float(max(probability)),
            'probabilities': {
                reverse_map.get(i, f'class_{i}'): float(prob)
                for i, prob in enumerate(probability)
            }
        }, f)

def anomaly_detection_ml():
    """Detect anomalies using machine learning"""
    
    print("\n[ML] Performing anomaly detection...")
    
    features, labels, filenames = extract_features_from_images()
    
    if len(features) < 2:
        print("[ML] Not enough data for anomaly detection")
        return
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Isolation Forest
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(features_scaled)
    
    # Local Outlier Factor
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=min(5, len(features)-1))
    lof_anomalies = lof.fit_predict(features_scaled)
    
    # Find anomalous images
    anomalous_images = []
    for i, (iso, lof_score) in enumerate(zip(anomalies, lof_anomalies)):
        if iso == -1 or lof_score == -1:
            anomalous_images.append({
                'index': i,
                'filename': filenames[i] if i < len(filenames) else f'image_{i}',
                'isolation_forest': int(iso),
                'lof': int(lof_score),
                'features': features[i].tolist()[:5]  # First 5 features
            })
    
    print(f"[ML] Found {len(anomalous_images)} anomalous images")
    
    # Save results
    with open('ml_anomalies.json', 'w') as f:
        json.dump({
            'n_images': len(features),
            'n_anomalies': len(anomalous_images),
            'anomalous_images': anomalous_images[:10]  # Limit to 10
        }, f)

def generate_ml_report():
    """Generate comprehensive ML analysis report"""
    
    print("\n[ML] Generating analysis report...")
    
    report = {
        'timestamp': time.time(),
        'analyses_performed': []
    }
    
    # Check what analyses were done
    if os.path.exists('ml_clustering.json'):
        report['analyses_performed'].append('clustering')
        with open('ml_clustering.json', 'r') as f:
            clustering = json.load(f)
            report['clustering_summary'] = {
                'n_samples': clustering['n_samples'],
                'kmeans_clusters': clustering['kmeans']['n_clusters'],
                'dbscan_clusters': clustering['dbscan']['n_clusters']
            }
    
    if os.path.exists('ml_classification.json'):
        report['analyses_performed'].append('classification')
        with open('ml_classification.json', 'r') as f:
            classification = json.load(f)
            report['classification_summary'] = {
                'n_classes': classification['n_classes'],
                'rf_accuracy': classification['random_forest']['accuracy'],
                'svm_accuracy': classification['svm']['accuracy']
            }
    
    if os.path.exists('ml_anomalies.json'):
        report['analyses_performed'].append('anomaly_detection')
        with open('ml_anomalies.json', 'r') as f:
            anomalies = json.load(f)
            report['anomaly_summary'] = {
                'n_anomalies': anomalies['n_anomalies'],
                'anomaly_rate': anomalies['n_anomalies'] / anomalies['n_images'] if anomalies['n_images'] > 0 else 0
            }
    
    # Save report
    with open('ml_report.json', 'w') as f:
        json.dump(report, f)
    
    print("[ML] Report generated")
    
    # Print summary
    print("\n[ML] === ANALYSIS SUMMARY ===")
    for analysis in report['analyses_performed']:
        print(f"  ✓ {analysis}")
    
    if 'clustering_summary' in report:
        print(f"\n  Clustering: {report['clustering_summary']['kmeans_clusters']} groups found")
    
    if 'classification_summary' in report:
        print(f"  Classification: {report['classification_summary']['rf_accuracy']:.0%} accuracy")
    
    if 'anomaly_summary' in report:
        print(f"  Anomalies: {report['anomaly_summary']['n_anomalies']} detected")

def main():
    """Run complete ML analysis pipeline"""
    
    print("=== MACHINE LEARNING ANALYSIS ===\n")
    
    # Extract features
    features, labels, filenames = extract_features_from_images()
    
    if len(features) == 0:
        print("[ML] No image data found. Please run other modules first.")
        return
    
    # Clustering
    if len(features) >= 2:
        kmeans, dbscan, pca, features_pca = perform_clustering(features, labels)
    
    # Classification
    if len(features) >= 10 and len(set(labels)) >= 2:
        rf, svm = train_classifiers(features, labels)
        
        # Predict on new image
        predict_new_image()
    
    # Anomaly detection
    anomaly_detection_ml()
    
    # Generate report
    generate_ml_report()
    
    print("\n[ML] Analysis complete!")
    print("[ML] Results saved to:")
    print("  - ml_clustering.json")
    print("  - ml_classification.json")
    print("  - ml_anomalies.json")
    print("  - ml_prediction.json")
    print("  - ml_report.json")

if __name__ == "__main__":
    main()