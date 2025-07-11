# Deployment Configuration Files

## 1. ML Engine Model Deployment (`ml_engine_config.yaml`)

```yaml
# ml_engine_config.yaml
# Configuration for deploying model to ML Engine

deploymentUri: gs://YOUR_BUCKET/ml-models/fiber-anomaly-v1/
framework: SCIKIT_LEARN
pythonVersion: '3.7'
runtimeVersion: '2.8'
autoScaling:
  minNodes: 1
  maxNodes: 10
  metrics:
  - name: CPU_USAGE
    target: 0.6
```

## 2. Cloud Run Dockerfile (`Dockerfile`)

```dockerfile
# Dockerfile for feature extraction service
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY detection.py .
COPY feature_extraction_service.py .

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "feature_extraction_service.py"]
```

## 3. Requirements File (`requirements.txt`)

```txt
# requirements.txt
numpy==1.21.0
opencv-python==4.5.3.56
pandas==1.3.0
scikit-learn==0.24.2
flask==2.0.1
google-cloud-storage==1.42.0
google-cloud-firestore==2.3.0
firebase-admin==5.0.1
requests==2.26.0
gunicorn==20.1.0
matplotlib==3.4.2
scipy==1.7.0
```

## 4. Cloud Function Package.json (`functions/package.json`)

```json
{
  "name": "fiber-anomaly-functions",
  "version": "1.0.0",
  "description": "Cloud Functions for Fiber Anomaly Detection",
  "scripts": {
    "lint": "eslint --ext .js,.ts .",
    "build": "tsc",
    "serve": "npm run build && firebase emulators:start --only functions",
    "shell": "npm run build && firebase functions:shell",
    "start": "npm run shell",
    "deploy": "firebase deploy --only functions",
    "logs": "firebase functions:log"
  },
  "engines": {
    "node": "14"
  },
  "main": "lib/index.js",
  "dependencies": {
    "firebase-admin": "^9.8.0",
    "firebase-functions": "^3.14.1",
    "googleapis": "^82.0.0"
  },
  "devDependencies": {
    "@typescript-eslint/eslint-plugin": "^4.26.0",
    "@typescript-eslint/parser": "^4.26.0",
    "eslint": "^7.28.0",
    "eslint-config-google": "^0.14.0",
    "firebase-functions-test": "^0.2.3",
    "typescript": "^4.3.2"
  },
  "private": true
}
```

## 5. Deployment Script (`deploy.sh`)

```bash
#!/bin/bash
# deploy.sh - Complete deployment script

# Set your project ID
PROJECT_ID="your-project-id"
BUCKET_NAME="$PROJECT_ID-ml-models"
MODEL_NAME="fiber_anomaly_detector"
VERSION_NAME="v1"
REGION="us-central1"

echo "🚀 Starting Fiber Anomaly ML Deployment..."

# 1. Create storage bucket if it doesn't exist
echo "📦 Creating storage bucket..."
gsutil mb -p $PROJECT_ID gs://$BUCKET_NAME 2>/dev/null || echo "Bucket already exists"

# 2. Upload model files to storage
echo "📤 Uploading model files..."
gsutil -m cp -r fiber_anomaly_model_v1/* gs://$BUCKET_NAME/fiber-anomaly-v1/

# 3. Create ML Engine model
echo "🤖 Creating ML Engine model..."
gcloud ml-engine models create $MODEL_NAME \
    --project=$PROJECT_ID \
    --regions=$REGION \
    2>/dev/null || echo "Model already exists"

# 4. Deploy model version
echo "📊 Deploying model version..."
gcloud ml-engine versions create $VERSION_NAME \
    --project=$PROJECT_ID \
    --model=$MODEL_NAME \
    --origin=gs://$BUCKET_NAME/fiber-anomaly-v1/ \
    --runtime-version=2.8 \
    --framework=scikit-learn \
    --python-version=3.7

# 5. Build and deploy feature extraction service to Cloud Run
echo "🏃 Deploying feature extraction service..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/fiber-feature-extraction
gcloud run deploy fiber-feature-extraction \
    --image gcr.io/$PROJECT_ID/fiber-feature-extraction \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2

# 6. Deploy Cloud Functions
echo "☁️  Deploying Cloud Functions..."
cd functions
npm install
npm run build
firebase deploy --only functions --project $PROJECT_ID

echo "✅ Deployment complete!"
echo "📍 Endpoints:"
echo "   - ML Engine Model: projects/$PROJECT_ID/models/$MODEL_NAME/versions/$VERSION_NAME"
echo "   - Feature Extraction: https://fiber-feature-extraction-xxxx-$REGION.a.run.app"
echo "   - Cloud Functions: https://$REGION-$PROJECT_ID.cloudfunctions.net/predictFiberAnomaly"
```

## 6. Testing Script (`test_deployment.py`)

```python
# test_deployment.py
import requests
import json
import base64

# Configuration
PROJECT_ID = "your-project-id"
REGION = "us-central1"
FUNCTION_URL = f"https://{REGION}-{PROJECT_ID}.cloudfunctions.net/predictFiberAnomaly"

def test_with_features():
    """Test with pre-extracted features"""
    print("Testing with features...")
    
    # Sample features (replace with actual)
    features = {
        "stat_mean": 128.5,
        "stat_std": 45.2,
        "stat_variance": 2043.04,
        "norm_frobenius": 1250.8,
        # ... add more features
    }
    
    response = requests.post(FUNCTION_URL, json={
        "features": features,
        "model": "fiber_anomaly_detector",
        "version": "v1"
    })
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_with_image():
    """Test with image URL"""
    print("\nTesting with image URL...")
    
    response = requests.post(FUNCTION_URL, json={
        "imageUrl": "https://example.com/fiber-image.png",
        "model": "fiber_anomaly_detector",
        "version": "v1"
    })
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_batch():
    """Test batch prediction"""
    print("\nTesting batch prediction...")
    
    batch_url = f"https://{REGION}-{PROJECT_ID}.cloudfunctions.net/batchPredictFiberAnomalies"
    
    response = requests.post(batch_url, json={
        "images": [
            "https://example.com/fiber1.png",
            "https://example.com/fiber2.png",
            "https://example.com/fiber3.png"
        ],
        "model": "fiber_anomaly_detector",
        "version": "v1"
    })
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    test_with_features()
    test_with_image()
    test_batch()
```

## 7. Local Development Setup (`setup_local.sh`)

```bash
#!/bin/bash
# setup_local.sh - Setup local development environment

# Create virtual environment
python3 -m venv fiber_ml_env
source fiber_ml_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download service account key
echo "📥 Please download your service account key from Firebase Console"
echo "   Save it as 'service-account.json' in the project root"

# Create necessary directories
mkdir -p data/normal_fibers
mkdir -p data/anomalous_fibers
mkdir -p fiber_anomaly_model_v1

# Install Firebase CLI
npm install -g firebase-tools

echo "✅ Local setup complete!"
echo "📝 Next steps:"
echo "   1. Add training images to data/normal_fibers and data/anomalous_fibers"
echo "   2. Run the Jupyter notebook to train the model"
echo "   3. Deploy using ./deploy.sh"
```