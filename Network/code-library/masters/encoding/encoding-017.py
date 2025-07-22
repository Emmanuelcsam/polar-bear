# feature_extraction_service.py
"""
Cloud Run service for extracting features from fiber optic images.
This bridges the gap between your image processing and ML Engine.
"""

import os
import json
import base64
import numpy as np
from flask import Flask, request, jsonify
import cv2
from google.cloud import storage
import requests
from io import BytesIO
import tempfile

# Import your detection module
from detection import OmniFiberAnalyzer, OmniConfig

app = Flask(__name__)

# Initialize analyzer
config = OmniConfig()
analyzer = OmniFiberAnalyzer(config)

# Initialize Google Cloud Storage
storage_client = storage.Client()


@app.route('/extract_features', methods=['POST'])
def extract_features():
    """Extract features from a fiber optic image"""
    try:
        data = request.get_json()
        
        # Get image from various sources
        image = None
        
        if 'image_url' in data:
            # Download from URL
            image = download_image(data['image_url'])
        elif 'image_base64' in data:
            # Decode from base64
            image = decode_base64_image(data['image_base64'])
        elif 'gcs_path' in data:
            # Download from Google Cloud Storage
            image = download_from_gcs(data['gcs_path'])
        else:
            return jsonify({'error': 'No image source provided'}), 400
        
        if image is None:
            return jsonify({'error': 'Failed to load image'}), 400
        
        # Extract features
        features, feature_names = analyzer.extract_ultra_comprehensive_features(image)
        
        # Convert to list format for ML Engine
        feature_list = [float(features.get(fname, 0)) for fname in feature_names]
        
        # Optionally return feature dictionary
        include_dict = data.get('include_feature_dict', False)
        
        response = {
            'success': True,
            'features': feature_list,
            'feature_count': len(feature_list),
            'feature_names': feature_names if include_dict else None,
            'feature_dict': features if include_dict else None
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/extract_features_batch', methods=['POST'])
def extract_features_batch():
    """Extract features from multiple images"""
    try:
        data = request.get_json()
        images = data.get('images', [])
        
        if not images:
            return jsonify({'error': 'No images provided'}), 400
        
        results = []
        feature_names = None
        
        for img_data in images:
            try:
                # Load image
                if 'url' in img_data:
                    image = download_image(img_data['url'])
                elif 'base64' in img_data:
                    image = decode_base64_image(img_data['base64'])
                else:
                    continue
                
                # Extract features
                features, fnames = analyzer.extract_ultra_comprehensive_features(image)
                
                if feature_names is None:
                    feature_names = fnames
                
                # Convert to list
                feature_list = [float(features.get(fname, 0)) for fname in feature_names]
                
                results.append({
                    'id': img_data.get('id', len(results)),
                    'features': feature_list,
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'id': img_data.get('id', len(results)),
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'feature_names': feature_names,
            'processed': len(results)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Full analysis including anomaly detection"""
    try:
        data = request.get_json()
        
        # Load image
        image_path = None
        if 'image_url' in data:
            # Download to temp file
            image = download_image(data['image_url'])
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                image_path = tmp.name
        else:
            return jsonify({'error': 'No image source provided'}), 400
        
        # Run full analysis
        output_dir = tempfile.mkdtemp()
        results = analyzer.analyze_end_face(image_path, output_dir)
        
        # Clean up temp files
        if image_path:
            os.unlink(image_path)
        
        # Extract key information
        response = {
            'success': results.get('success', False),
            'overall_quality_score': results.get('overall_quality_score'),
            'defect_count': len(results.get('defects', [])),
            'defects': results.get('defects', []),
            'summary': results.get('summary', {}),
            'timestamp': results.get('timestamp')
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'fiber-feature-extraction'}), 200


def download_image(url):
    """Download image from URL"""
    response = requests.get(url)
    response.raise_for_status()
    
    # Convert to numpy array
    image = cv2.imdecode(
        np.frombuffer(response.content, np.uint8), 
        cv2.IMREAD_COLOR
    )
    return image


def decode_base64_image(base64_string):
    """Decode base64 image"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode
    img_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    image = cv2.imdecode(
        np.frombuffer(img_data, np.uint8),
        cv2.IMREAD_COLOR
    )
    return image


def download_from_gcs(gcs_path):
    """Download image from Google Cloud Storage"""
    # Parse bucket and blob name
    parts = gcs_path.replace('gs://', '').split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    # Download
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    img_data = blob.download_as_bytes()
    
    # Convert to numpy array
    image = cv2.imdecode(
        np.frombuffer(img_data, np.uint8),
        cv2.IMREAD_COLOR
    )
    return image


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)