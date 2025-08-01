// functions/src/index.ts
import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import { google } from 'googleapis';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';

// Initialize Firebase Admin
admin.initializeApp();
const storage = admin.storage();
const firestore = admin.firestore();

// ML Engine setup
const ml = google.ml('v1');

/**
 * Cloud Function to predict fiber optic anomalies using ML Engine
 */
export const predictFiberAnomaly = functions.https.onRequest(async (req, res) => {
  // Enable CORS
  res.set('Access-Control-Allow-Origin', '*');
  
  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'POST');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.set('Access-Control-Max-Age', '3600');
    res.status(204).send('');
    return;
  }

  try {
    // Extract request data
    const { 
      model = 'fiber_anomaly_detector',
      version = 'v1',
      instances,
      imageUrl,
      features
    } = req.body;

    // Validate input
    if (!instances && !imageUrl && !features) {
      res.status(400).json({
        error: 'Must provide either instances, imageUrl, or features'
      });
      return;
    }

    let predictionInstances = instances;

    // If imageUrl provided, download and extract features
    if (imageUrl) {
      const extractedFeatures = await extractFeaturesFromImage(imageUrl);
      predictionInstances = [extractedFeatures];
    } else if (features) {
      // Convert feature dictionary to array format expected by ML Engine
      predictionInstances = [convertFeaturesToArray(features)];
    }

    // Get authentication
    const authClient = await google.auth.getClient({
      scopes: ['https://www.googleapis.com/auth/cloud-platform']
    });

    // Format model name
    const projectId = process.env.GCLOUD_PROJECT;
    const modelName = `projects/${projectId}/models/${model}/versions/${version}`;

    // Make prediction request to ML Engine
    const prediction = await ml.projects.predict({
      auth: authClient,
      name: modelName,
      requestBody: {
        instances: predictionInstances
      }
    } as any);

    // Process results
    const results = prediction.data.predictions.map((pred: any, idx: number) => ({
      instance: idx,
      is_anomalous: pred[0] > 0.5,
      anomaly_probability: pred[0],
      severity: getSeverityLevel(pred[0]),
      confidence: Math.abs(pred[0] - 0.5) * 2 // Convert to confidence score
    }));

    // Log prediction to Firestore
    await logPrediction(results, req.body);

    // Send response
    res.status(200).json({
      success: true,
      model: model,
      version: version,
      predictions: results,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({
      error: 'Prediction failed',
      message: error.message
    });
  }
});

/**
 * Extract features from an image URL
 */
async function extractFeaturesFromImage(imageUrl: string): Promise<number[]> {
  // This would integrate with your Python feature extraction service
  // For now, returning mock features
  console.log('Extracting features from:', imageUrl);
  
  // In production, you would:
  // 1. Download the image
  // 2. Call a Cloud Function or Cloud Run service running your Python feature extractor
  // 3. Return the extracted features
  
  // Mock 100+ features
  return Array(150).fill(0).map(() => Math.random());
}

/**
 * Convert feature dictionary to array format
 */
function convertFeaturesToArray(features: Record<string, number>): number[] {
  // Load feature names from configuration
  const featureNames = getFeatureNames();
  
  // Convert to array in correct order
  return featureNames.map(name => features[name] || 0);
}

/**
 * Get ordered feature names
 */
function getFeatureNames(): string[] {
  // In production, load from Cloud Storage or Firestore
  // For now, return sample feature names
  return [
    'stat_mean', 'stat_std', 'stat_variance', 'stat_skew', 'stat_kurtosis',
    'norm_frobenius', 'norm_l1', 'norm_l2', 'norm_linf',
    'lbp_r1_mean', 'lbp_r1_std', 'lbp_r2_mean', 'lbp_r2_std',
    'glcm_d1_a0_contrast', 'glcm_d1_a0_energy', 'glcm_d1_a0_homogeneity',
    'fft_mean_magnitude', 'fft_std_magnitude', 'fft_max_magnitude',
    // ... add all your feature names
  ];
}

/**
 * Convert probability to severity level
 */
function getSeverityLevel(probability: number): string {
  if (probability < 0.3) return 'NORMAL';
  if (probability < 0.5) return 'LOW';
  if (probability < 0.7) return 'MEDIUM';
  if (probability < 0.9) return 'HIGH';
  return 'CRITICAL';
}

/**
 * Log prediction to Firestore for monitoring
 */
async function logPrediction(results: any[], request: any): Promise<void> {
  try {
    await firestore.collection('ml_predictions').add({
      model: request.model || 'fiber_anomaly_detector',
      version: request.version || 'v1',
      results: results,
      request_type: request.imageUrl ? 'image' : request.features ? 'features' : 'instances',
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
      metadata: {
        imageUrl: request.imageUrl,
        hasFeatures: !!request.features,
        instanceCount: request.instances ? request.instances.length : 1
      }
    });
  } catch (error) {
    console.error('Failed to log prediction:', error);
  }
}

/**
 * Batch prediction function for multiple fiber images
 */
export const batchPredictFiberAnomalies = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');
  
  try {
    const { images, model = 'fiber_anomaly_detector', version = 'v1' } = req.body;
    
    if (!images || !Array.isArray(images)) {
      res.status(400).json({ error: 'Images array required' });
      return;
    }
    
    // Process images in parallel
    const featurePromises = images.map(img => 
      extractFeaturesFromImage(img.url || img)
    );
    const allFeatures = await Promise.all(featurePromises);
    
    // Get authentication
    const authClient = await google.auth.getClient({
      scopes: ['https://www.googleapis.com/auth/cloud-platform']
    });
    
    // Make batch prediction
    const projectId = process.env.GCLOUD_PROJECT;
    const modelName = `projects/${projectId}/models/${model}/versions/${version}`;
    
    const prediction = await ml.projects.predict({
      auth: authClient,
      name: modelName,
      requestBody: {
        instances: allFeatures
      }
    } as any);
    
    // Process results
    const results = prediction.data.predictions.map((pred: any, idx: number) => ({
      image: images[idx],
      is_anomalous: pred[0] > 0.5,
      anomaly_probability: pred[0],
      severity: getSeverityLevel(pred[0])
    }));
    
    res.status(200).json({
      success: true,
      results: results,
      summary: {
        total: results.length,
        anomalous: results.filter((r: any) => r.is_anomalous).length,
        normal: results.filter((r: any) => !r.is_anomalous).length
      }
    });
    
  } catch (error) {
    console.error('Batch prediction error:', error);
    res.status(500).json({ error: 'Batch prediction failed' });
  }
});

/**
 * Get model information and metrics
 */
export const getFiberModelInfo = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');
  
  try {
    const { model = 'fiber_anomaly_detector', version = 'v1' } = req.query;
    
    // Get model metadata from Firestore
    const modelDocs = await firestore
      .collection('ml_models')
      .where('model_id', '==', `${model}-${version}`)
      .orderBy('created_at', 'desc')
      .limit(1)
      .get();
    
    if (modelDocs.empty) {
      res.status(404).json({ error: 'Model not found' });
      return;
    }
    
    const modelData = modelDocs.docs[0].data();
    
    // Get recent prediction statistics
    const recentPredictions = await firestore
      .collection('ml_predictions')
      .where('model', '==', model)
      .where('version', '==', version)
      .orderBy('timestamp', 'desc')
      .limit(100)
      .get();
    
    const predictions = recentPredictions.docs.map(doc => doc.data());
    const anomalyCount = predictions.filter(p => 
      p.results && p.results[0] && p.results[0].is_anomalous
    ).length;
    
    res.status(200).json({
      model: {
        name: model,
        version: version,
        created_at: modelData.created_at,
        metrics: modelData.metrics,
        model_type: modelData.model_type
      },
      usage: {
        total_predictions: predictions.length,
        anomaly_rate: anomalyCount / predictions.length,
        last_prediction: predictions[0]?.timestamp || null
      }
    });
    
  } catch (error) {
    console.error('Error getting model info:', error);
    res.status(500).json({ error: 'Failed to get model info' });
  }
});