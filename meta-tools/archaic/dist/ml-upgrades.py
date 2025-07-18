# Add these imports at the top
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import joblib  # for saving models

class MLDefectClassifier:
    """Machine Learning based defect classifier"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'svm': SVC(kernel='rbf', probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'neural_net': MLPClassifier(hidden_layer_sizes=(64, 32), 
                                       activation='relu', 
                                       learning_rate_init=0.001)
        }
        self.best_model = None
        self.feature_names = None
        
    def prepare_training_data(self, feature_list, labels):
        """Prepare data for training with train/val/test split"""
        # Convert to numpy arrays
        X = np.array(feature_list)
        y = np.array(labels)
        
        # Split data (60% train, 20% val, 20% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and select the best one"""
        best_score = 0
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_score = model.score(X_val, y_val)
            print(f"{name} validation accuracy: {val_score:.3f}")
            
            if val_score > best_score:
                best_score = val_score
                self.best_model = model
                self.best_model_name = name
        
        return self.best_model
    
    def predict_defect_type(self, features):
        """Predict defect type from features"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get prediction and probability
        prediction = self.best_model.predict(features_scaled)[0]
        probabilities = self.best_model.predict_proba(features_scaled)[0]
        
        return prediction, probabilities
    
    
##########################################################################################################

from sklearn.decomposition import PCA

class FeatureDimensionalityReducer:
    """Use PCA to reduce feature dimensions"""
    
    def __init__(self, n_components=0.95):  # Keep 95% variance
        self.pca = PCA(n_components=n_components)
        self.original_features = None
        self.reduced_features = None
        
    def fit_transform(self, features_matrix):
        """Fit PCA and transform features"""
        self.original_features = features_matrix
        self.reduced_features = self.pca.fit_transform(features_matrix)
        
        print(f"Reduced features from {features_matrix.shape[1]} to {self.reduced_features.shape[1]}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return self.reduced_features
    
    def visualize_components(self):
        """Visualize principal components"""
        if self.pca.n_components_ >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.reduced_features[:, 0], 
                       self.reduced_features[:, 1], 
                       alpha=0.5)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('PCA Visualization of Features')
            plt.show()
            
#####################################################################################################


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class UnsupervisedAnomalyDetector:
    """Unsupervised anomaly detection using clustering"""
    
    def __init__(self):
        self.kmeans = None
        self.cluster_centers = None
        self.threshold_distances = None
        
    def fit(self, features, n_clusters=None):
        """Fit K-means clustering"""
        if n_clusters is None:
            # Find optimal number of clusters using elbow method
            n_clusters = self._find_optimal_clusters(features)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.kmeans.fit_predict(features)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Calculate threshold distances for each cluster
        self._calculate_thresholds(features, labels)
        
        return labels
    
    def _find_optimal_clusters(self, features, max_k=10):
        """Find optimal number of clusters using silhouette score"""
        scores = []
        K = range(2, min(max_k, len(features)))
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            scores.append(score)
        
        # Return k with highest silhouette score
        optimal_k = K[np.argmax(scores)]
        return optimal_k
    
    def _calculate_thresholds(self, features, labels):
        """Calculate anomaly thresholds for each cluster"""
        self.threshold_distances = {}
        
        for cluster_id in np.unique(labels):
            # Get points in this cluster
            cluster_mask = labels == cluster_id
            cluster_points = features[cluster_mask]
            
            # Calculate distances to cluster center
            center = self.cluster_centers[cluster_id]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            
            # Set threshold as 95th percentile of distances
            self.threshold_distances[cluster_id] = np.percentile(distances, 95)
    
    def detect_anomalies(self, features):
        """Detect anomalies based on distance to nearest cluster center"""
        if self.kmeans is None:
            raise ValueError("Model not fitted yet!")
        
        # Predict cluster for each point
        labels = self.kmeans.predict(features)
        
        # Calculate distances to assigned cluster centers
        anomaly_scores = []
        is_anomaly = []
        
        for i, (feature, label) in enumerate(zip(features, labels)):
            center = self.cluster_centers[label]
            distance = np.linalg.norm(feature - center)
            anomaly_scores.append(distance)
            
            # Check if distance exceeds threshold
            is_anomaly.append(distance > self.threshold_distances[label])
        
        return np.array(is_anomaly), np.array(anomaly_scores)
    
    
    
    
class OmniFiberAnalyzer:
    def __init__(self, config: OmniConfig):
        # ... existing init code ...
        
        # Add ML components
        self.ml_classifier = MLDefectClassifier()
        self.pca_reducer = FeatureDimensionalityReducer()
        self.unsupervised_detector = UnsupervisedAnomalyDetector()
        self.use_ml_models = True  # Flag to enable/disable ML
        
        # Load pre-trained models if available
        self._load_ml_models()
    
    def _load_ml_models(self):
        """Load pre-trained ML models if they exist"""
        model_path = "ml_models/"
        if os.path.exists(f"{model_path}defect_classifier.pkl"):
            self.ml_classifier.best_model = joblib.load(f"{model_path}defect_classifier.pkl")
            self.ml_classifier.scaler = joblib.load(f"{model_path}scaler.pkl")
            self.logger.info("Loaded pre-trained ML models")
    
    def train_ml_models(self, training_data_dir):
        """Train ML models on labeled training data"""
        self.logger.info("Training ML models...")
        
        # Collect features and labels from training data
        all_features = []
        all_labels = []
        
        for img_path in glob.glob(f"{training_data_dir}/*.json"):
            # Load image and extract features
            image = self.load_image(img_path)
            features, _ = self.extract_ultra_comprehensive_features(image)
            feature_vector = [features[fname] for fname in sorted(features.keys())]
            
            # Get label from filename or metadata
            # Assuming format: image_name_LABEL.json
            label = self._extract_label_from_filename(img_path)
            
            all_features.append(feature_vector)
            all_labels.append(label)
        
        # Apply PCA
        features_reduced = self.pca_reducer.fit_transform(np.array(all_features))
        
        # Prepare data and train models
        train_data, val_data, test_data = self.ml_classifier.prepare_training_data(
            features_reduced, all_labels
        )
        
        best_model = self.ml_classifier.train_models(*train_data[:2], *val_data[:2])
        
        # Evaluate on test set
        X_test, y_test = test_data
        y_pred = best_model.predict(X_test)
        print("\nTest Set Performance:")
        print(classification_report(y_test, y_pred))
        
        # Train unsupervised detector on normal samples
        normal_features = features_reduced[np.array(all_labels) == 'normal']
        self.unsupervised_detector.fit(normal_features)
        
        # Save models
        self._save_ml_models()
    
    def detect_anomalies_with_ml(self, test_features):
        """Use ML models for anomaly detection"""
        # Reduce dimensions
        features_vector = [test_features[fname] for fname in sorted(test_features.keys())]
        features_reduced = self.pca_reducer.pca.transform([features_vector])
        
        # Supervised classification
        if self.ml_classifier.best_model is not None:
            defect_type, probabilities = self.ml_classifier.predict_defect_type(features_reduced[0])
            confidence = np.max(probabilities)
        else:
            defect_type = "unknown"
            confidence = 0.0
        
        # Unsupervised anomaly detection
        is_anomaly, anomaly_score = self.unsupervised_detector.detect_anomalies(features_reduced)
        
        return {
            'defect_type': defect_type,
            'confidence': confidence,
            'is_anomaly': is_anomaly[0],
            'anomaly_score': anomaly_score[0],
            'probabilities': probabilities if self.ml_classifier.best_model else None
        }
        
import tensorflow as tf
from tensorflow.keras import layers, models

class CNNDefectDetector:
    """Convolutional Neural Network for defect detection"""
    
    def __init__(self, input_shape=(256, 256, 1), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """Build CNN architecture"""
        model = models.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the CNN model"""
        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, self.num_classes)
        
        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )
        
        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train_cat, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val_cat),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5)
            ]
        )
        
        return history
    
    def predict(self, image):
        """Predict defect type for a single image"""
        # Preprocess image
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Resize if necessary
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image, self.input_shape[:2])
        
        # Normalize and add batch dimension
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        return predicted_class, confidence, predictions
    
    