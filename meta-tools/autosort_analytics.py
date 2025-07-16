#!/usr/bin/env python3
"""
Enhanced Image Sorter with Comprehensive Analytics and Statistics
This version saves exhaustive statistics, correlations, and analysis data
"""

import sys
import os
import shutil
import subprocess
import importlib
import logging
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import re
import json
import pickle
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
import sqlite3
import csv
from pathlib import Path
import hashlib
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import h5py

# Function to install missing packages
def install_if_missing(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure all required packages are installed
required_packages = ['pygame', 'torch', 'torchvision', 'pillow', 'numpy', 
                    'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'scipy', 'h5py']
for pkg in required_packages:
    install_if_missing(pkg)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analytics_image_sorter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveAnalytics:
    """Tracks and saves comprehensive statistics and correlations"""
    
    def __init__(self, analytics_dir='analytics_data'):
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(exist_ok=True)
        
        # Initialize data structures
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'decisions': [],
            'predictions': [],
            'feature_statistics': defaultdict(list),
            'keyword_correlations': defaultdict(dict),
            'folder_statistics': defaultdict(dict),
            'time_series_data': [],
            'model_performance': [],
            'user_behavior': defaultdict(list),
            'image_features': [],
            'confusion_data': defaultdict(dict)
        }
        
        # Initialize database
        self.db_path = self.analytics_dir / 'analytics.db'
        self._init_database()
        
        # Initialize HDF5 for large numerical data
        self.h5_path = self.analytics_dir / f'features_{self.session_id}.h5'
        self.h5_file = h5py.File(self.h5_path, 'w')
        
    def _init_database(self):
        """Initialize SQLite database for structured data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                filename TEXT,
                file_hash TEXT,
                source_folder TEXT,
                target_folder TEXT,
                is_manual BOOLEAN,
                confidence REAL,
                decision_time REAL,
                keywords TEXT,
                file_size INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                dominant_colors TEXT,
                brightness REAL,
                contrast REAL,
                entropy REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keyword_folder_correlations (
                keyword TEXT,
                folder TEXT,
                count INTEGER,
                confidence REAL,
                PRIMARY KEY (keyword, folder)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                session_id TEXT,
                feature_name TEXT,
                importance_score REAL,
                correlation_with_accuracy REAL,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                session_id TEXT,
                timestamp TEXT,
                accuracy REAL,
                precision_json TEXT,
                recall_json TEXT,
                f1_score_json TEXT,
                loss REAL,
                learning_rate REAL,
                epoch INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_image_statistics(self, image_path: str) -> Dict[str, Any]:
        """Calculate comprehensive image statistics"""
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Basic properties
            stats = {
                'file_size': os.path.getsize(image_path),
                'width': img.width,
                'height': img.height,
                'aspect_ratio': img.width / img.height,
                'mode': img.mode,
                'format': img.format,
            }
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img_array = np.array(img)
            
            # Color statistics
            if len(img_array.shape) == 3:
                stats['mean_color'] = img_array.mean(axis=(0, 1)).tolist()
                stats['std_color'] = img_array.std(axis=(0, 1)).tolist()
                
                # Dominant colors using histogram
                hist_r = np.histogram(img_array[:,:,0], bins=256)[0]
                hist_g = np.histogram(img_array[:,:,1], bins=256)[0]
                hist_b = np.histogram(img_array[:,:,2], bins=256)[0]
                
                stats['dominant_r'] = np.argmax(hist_r)
                stats['dominant_g'] = np.argmax(hist_g)
                stats['dominant_b'] = np.argmax(hist_b)
                
                # Brightness and contrast
                gray = img.convert('L')
                gray_array = np.array(gray)
                stats['brightness'] = gray_array.mean()
                stats['contrast'] = gray_array.std()
                
                # Entropy (measure of information content)
                def calculate_entropy(hist):
                    hist = hist / np.sum(hist)  # Normalize
                    hist = hist[hist > 0]  # Remove zeros
                    return -np.sum(hist * np.log2(hist))
                
                stats['entropy'] = calculate_entropy(hist_r) + calculate_entropy(hist_g) + calculate_entropy(hist_b)
            
            # Edge detection statistics
            edges = self._detect_edges(gray_array)
            stats['edge_density'] = np.sum(edges) / (img.width * img.height)
            
            # File hash for deduplication
            with open(image_path, 'rb') as f:
                stats['file_hash'] = hashlib.md5(f.read()).hexdigest()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating image statistics: {e}")
            return {}
    
    def _detect_edges(self, gray_array: np.ndarray) -> np.ndarray:
        """Simple edge detection using Sobel operator"""
        try:
            from scipy import ndimage
            sx = ndimage.sobel(gray_array, axis=0)
            sy = ndimage.sobel(gray_array, axis=1)
            edges = np.hypot(sx, sy)
            return edges > np.mean(edges)
        except:
            return np.zeros_like(gray_array, dtype=bool)
    
    def record_decision(self, filename: str, source_path: str, target_folder: str, 
                       keywords: List[str], is_manual: bool, confidence: float,
                       decision_time: float, model_output: Optional[torch.Tensor] = None):
        """Record a comprehensive decision with all metadata"""
        
        # Calculate image statistics
        img_stats = self.calculate_image_statistics(source_path)
        
        # Create decision record
        decision = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'file_hash': img_stats.get('file_hash', ''),
            'source_folder': os.path.dirname(source_path),
            'target_folder': target_folder,
            'is_manual': is_manual,
            'confidence': confidence,
            'decision_time': decision_time,
            'keywords': keywords,
            'image_stats': img_stats,
            'model_output': model_output.detach().cpu().numpy() if model_output is not None else None
        }
        
        # Store in memory
        self.session_data['decisions'].append(decision)
        
        # Update correlations
        for keyword in keywords:
            if keyword not in self.session_data['keyword_correlations']:
                self.session_data['keyword_correlations'][keyword] = {}
            if target_folder not in self.session_data['keyword_correlations'][keyword]:
                self.session_data['keyword_correlations'][keyword][target_folder] = 0
            self.session_data['keyword_correlations'][keyword][target_folder] += 1
        
        # Update folder statistics
        if target_folder not in self.session_data['folder_statistics']:
            self.session_data['folder_statistics'][target_folder] = {'count': 0, 'total_confidence': 0}
        self.session_data['folder_statistics'][target_folder]['count'] += 1
        self.session_data['folder_statistics'][target_folder]['total_confidence'] += confidence
        
        # Store in database
        self._save_decision_to_db(decision, img_stats)
        
        # Update time series
        self.session_data['time_series_data'].append({
            'timestamp': datetime.now().timestamp(),
            'cumulative_decisions': len(self.session_data['decisions']),
            'accuracy': self._calculate_running_accuracy()
        })
    
    def _save_decision_to_db(self, decision: Dict, img_stats: Dict):
        """Save decision to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO decisions (
                session_id, timestamp, filename, file_hash, source_folder,
                target_folder, is_manual, confidence, decision_time, keywords,
                file_size, image_width, image_height, dominant_colors,
                brightness, contrast, entropy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision['session_id'],
            decision['timestamp'],
            decision['filename'],
            decision['file_hash'],
            decision['source_folder'],
            decision['target_folder'],
            decision['is_manual'],
            decision['confidence'],
            decision['decision_time'],
            json.dumps(decision['keywords']),
            img_stats.get('file_size', 0),
            img_stats.get('width', 0),
            img_stats.get('height', 0),
            json.dumps([img_stats.get('dominant_r', 0), 
                       img_stats.get('dominant_g', 0), 
                       img_stats.get('dominant_b', 0)]),
            img_stats.get('brightness', 0),
            img_stats.get('contrast', 0),
            img_stats.get('entropy', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def save_model_features(self, image_path: str, features: torch.Tensor, 
                           text_features: torch.Tensor, combined_features: torch.Tensor):
        """Save neural network features to HDF5"""
        try:
            # Create unique key for this image
            key = hashlib.md5(image_path.encode()).hexdigest()
            
            # Create group for this image
            if key not in self.h5_file:
                grp = self.h5_file.create_group(key)
                grp.attrs['filename'] = image_path
                grp.attrs['timestamp'] = datetime.now().isoformat()
                
                # Save features
                grp.create_dataset('image_features', data=features.cpu().numpy())
                grp.create_dataset('text_features', data=text_features.cpu().numpy())
                grp.create_dataset('combined_features', data=combined_features.cpu().numpy())
                
        except Exception as e:
            logger.error(f"Error saving model features: {e}")
    
    def record_model_performance(self, accuracy: float, precision: Dict[str, float],
                               recall: Dict[str, float], f1_score: Dict[str, float],
                               loss: float, learning_rate: float, epoch: int):
        """Record model performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics (
                session_id, timestamp, accuracy, precision_json,
                recall_json, f1_score_json, loss, learning_rate, epoch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.session_id,
            datetime.now().isoformat(),
            accuracy,
            json.dumps(precision),
            json.dumps(recall),
            json.dumps(f1_score),
            loss,
            learning_rate,
            epoch
        ))
        
        conn.commit()
        conn.close()
    
    def calculate_correlations(self) -> Dict[str, Any]:
        """Calculate comprehensive correlations between features"""
        if len(self.session_data['decisions']) < 2:
            return {}
        
        # Extract features for correlation analysis
        features_df = pd.DataFrame([
            {
                'brightness': d['image_stats'].get('brightness', 0),
                'contrast': d['image_stats'].get('contrast', 0),
                'edge_density': d['image_stats'].get('edge_density', 0),
                'aspect_ratio': d['image_stats'].get('aspect_ratio', 1),
                'file_size': d['image_stats'].get('file_size', 0),
                'keyword_count': len(d['keywords']),
                'confidence': d['confidence'],
                'is_manual': int(d['is_manual']),
                'target_folder': d['target_folder']
            }
            for d in self.session_data['decisions']
        ])
        
        # Calculate correlations
        numeric_features = features_df.select_dtypes(include=[np.number])
        correlations = numeric_features.corr()
        
        # Feature importance using mutual information
        from sklearn.feature_selection import mutual_info_classif
        X = numeric_features.drop(['confidence', 'is_manual'], axis=1, errors='ignore')
        y = features_df['target_folder']
        
        if len(y.unique()) > 1:
            mi_scores = mutual_info_classif(X, y)
            feature_importance = dict(zip(X.columns, mi_scores))
        else:
            feature_importance = {}
        
        return {
            'correlation_matrix': correlations.to_dict(),
            'feature_importance': feature_importance,
            'summary_statistics': features_df.describe().to_dict()
        }
    
    def generate_reports(self):
        """Generate comprehensive reports in multiple formats"""
        logger.info("Generating analytics reports...")
        
        # Calculate final statistics
        correlations = self.calculate_correlations()
        
        # 1. JSON Report
        json_report = {
            'session_id': self.session_id,
            'session_duration': (datetime.now() - datetime.fromisoformat(
                self.session_data['start_time'])).total_seconds(),
            'total_decisions': len(self.session_data['decisions']),
            'manual_decisions': sum(1 for d in self.session_data['decisions'] if d['is_manual']),
            'auto_decisions': sum(1 for d in self.session_data['decisions'] if not d['is_manual']),
            'keyword_correlations': dict(self.session_data['keyword_correlations']),
            'folder_statistics': dict(self.session_data['folder_statistics']),
            'correlations': correlations,
            'time_series_data': self.session_data['time_series_data']
        }
        
        with open(self.analytics_dir / f'report_{self.session_id}.json', 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # 2. CSV Reports
        # Decisions CSV
        decisions_df = pd.DataFrame([
            {
                'timestamp': d['timestamp'],
                'filename': d['filename'],
                'target_folder': d['target_folder'],
                'is_manual': d['is_manual'],
                'confidence': d['confidence'],
                'keywords': ','.join(d['keywords']),
                'brightness': d['image_stats'].get('brightness', 0),
                'contrast': d['image_stats'].get('contrast', 0),
                'file_size': d['image_stats'].get('file_size', 0)
            }
            for d in self.session_data['decisions']
        ])
        decisions_df.to_csv(self.analytics_dir / f'decisions_{self.session_id}.csv', index=False)
        
        # Keyword correlations CSV
        keyword_data = []
        for keyword, folders in self.session_data['keyword_correlations'].items():
            for folder, count in folders.items():
                keyword_data.append({
                    'keyword': keyword,
                    'folder': folder,
                    'count': count,
                    'percentage': count / sum(folders.values()) * 100
                })
        
        if keyword_data:
            keyword_df = pd.DataFrame(keyword_data)
            keyword_df.to_csv(self.analytics_dir / f'keyword_correlations_{self.session_id}.csv', 
                            index=False)
        
        # 3. Visualizations
        self._generate_visualizations()
        
        # 4. SQLite summary
        self._generate_db_summary()
        
        # 5. Pickle for Python reuse
        with open(self.analytics_dir / f'session_data_{self.session_id}.pkl', 'wb') as f:
            pickle.dump(self.session_data, f)
        
        logger.info(f"Reports saved to {self.analytics_dir}")
    
    def _generate_visualizations(self):
        """Generate visualization plots"""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8-darkgrid')
            fig_dir = self.analytics_dir / 'visualizations'
            fig_dir.mkdir(exist_ok=True)
            
            # 1. Decision timeline
            if self.session_data['time_series_data']:
                plt.figure(figsize=(10, 6))
                ts_data = pd.DataFrame(self.session_data['time_series_data'])
                ts_data['time'] = pd.to_datetime(ts_data['timestamp'], unit='s')
                plt.plot(ts_data['time'], ts_data['cumulative_decisions'])
                plt.xlabel('Time')
                plt.ylabel('Cumulative Decisions')
                plt.title('Decision Timeline')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(fig_dir / f'timeline_{self.session_id}.png')
                plt.close()
            
            # 2. Folder distribution
            if self.session_data['folder_statistics']:
                plt.figure(figsize=(10, 6))
                folders = list(self.session_data['folder_statistics'].keys())
                counts = [stats['count'] for stats in self.session_data['folder_statistics'].values()]
                plt.bar(folders, counts)
                plt.xlabel('Folder')
                plt.ylabel('Count')
                plt.title('Image Distribution by Folder')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(fig_dir / f'folder_distribution_{self.session_id}.png')
                plt.close()
            
            # 3. Keyword heatmap
            if self.session_data['keyword_correlations']:
                # Create matrix for heatmap
                keywords = list(self.session_data['keyword_correlations'].keys())[:20]  # Top 20
                folders = list(set(folder for kw_data in self.session_data['keyword_correlations'].values() 
                                 for folder in kw_data.keys()))
                
                matrix = np.zeros((len(keywords), len(folders)))
                for i, kw in enumerate(keywords):
                    for j, folder in enumerate(folders):
                        matrix[i, j] = self.session_data['keyword_correlations'][kw].get(folder, 0)
                
                if matrix.size > 0:
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(matrix, xticklabels=folders, yticklabels=keywords, 
                              cmap='YlOrRd', annot=True, fmt='.0f')
                    plt.title('Keyword-Folder Correlation Heatmap')
                    plt.tight_layout()
                    plt.savefig(fig_dir / f'keyword_heatmap_{self.session_id}.png')
                    plt.close()
            
            # 4. Confidence distribution
            if self.session_data['decisions']:
                plt.figure(figsize=(10, 6))
                confidences = [d['confidence'] for d in self.session_data['decisions']]
                plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Confidence')
                plt.ylabel('Frequency')
                plt.title('Decision Confidence Distribution')
                plt.tight_layout()
                plt.savefig(fig_dir / f'confidence_dist_{self.session_id}.png')
                plt.close()
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _generate_db_summary(self):
        """Generate database summary statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get summary statistics
        summary = {}
        
        # Total decisions
        cursor = conn.execute("SELECT COUNT(*) FROM decisions")
        summary['total_decisions'] = cursor.fetchone()[0]
        
        # Decisions by folder
        cursor = conn.execute("""
            SELECT target_folder, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM decisions
            GROUP BY target_folder
        """)
        summary['folder_summary'] = cursor.fetchall()
        
        # Top keywords
        cursor = conn.execute("""
            SELECT keyword, folder, count
            FROM keyword_folder_correlations
            ORDER BY count DESC
            LIMIT 20
        """)
        summary['top_keyword_correlations'] = cursor.fetchall()
        
        conn.close()
        
        # Save summary
        with open(self.analytics_dir / f'db_summary_{self.session_id}.txt', 'w') as f:
            f.write(f"Database Summary for Session {self.session_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Decisions: {summary['total_decisions']}\n\n")
            
            f.write("Folder Summary:\n")
            for folder, count, avg_conf in summary['folder_summary']:
                f.write(f"  {folder}: {count} images (avg confidence: {avg_conf:.3f})\n")
            
            f.write("\nTop Keyword-Folder Correlations:\n")
            for kw, folder, count in summary['top_keyword_correlations']:
                f.write(f"  '{kw}' -> {folder}: {count} times\n")
    
    def _calculate_running_accuracy(self) -> float:
        """Calculate running accuracy based on confidence"""
        if not self.session_data['decisions']:
            return 0.0
        
        # Use confidence as proxy for accuracy
        manual_decisions = [d for d in self.session_data['decisions'] if d['is_manual']]
        if not manual_decisions:
            return 0.0
        
        return np.mean([d['confidence'] for d in manual_decisions])
    
    def export_for_ml(self):
        """Export data in formats suitable for machine learning"""
        # 1. Feature matrix as NPY
        if self.session_data['decisions']:
            features = []
            labels = []
            
            for decision in self.session_data['decisions']:
                if decision['image_stats']:
                    feature_vec = [
                        decision['image_stats'].get('brightness', 0),
                        decision['image_stats'].get('contrast', 0),
                        decision['image_stats'].get('edge_density', 0),
                        decision['image_stats'].get('aspect_ratio', 1),
                        decision['image_stats'].get('file_size', 0),
                        len(decision['keywords']),
                        decision['confidence']
                    ]
                    features.append(feature_vec)
                    labels.append(decision['target_folder'])
            
            if features:
                np.save(self.analytics_dir / f'features_{self.session_id}.npy', np.array(features))
                np.save(self.analytics_dir / f'labels_{self.session_id}.npy', np.array(labels))
        
        # 2. TensorFlow/Keras compatible HDF5
        # Already saved in self.h5_file
        
        # 3. PyTorch dataset format
        torch_data = {
            'features': features if 'features' in locals() else [],
            'labels': labels if 'labels' in locals() else [],
            'metadata': {
                'session_id': self.session_id,
                'feature_names': ['brightness', 'contrast', 'edge_density', 
                                'aspect_ratio', 'file_size', 'keyword_count', 'confidence'],
                'label_mapping': {folder: i for i, folder in 
                                enumerate(set(d['target_folder'] for d in self.session_data['decisions']))}
            }
        }
        
        torch.save(torch_data, self.analytics_dir / f'torch_dataset_{self.session_id}.pt')
    
    def close(self):
        """Clean up resources"""
        try:
            # Save final reports
            self.generate_reports()
            self.export_for_ml()
            
            # Close HDF5 file
            if hasattr(self, 'h5_file'):
                self.h5_file.close()
                
            # Update keyword correlations in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for keyword, folders in self.session_data['keyword_correlations'].items():
                for folder, count in folders.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO keyword_folder_correlations 
                        (keyword, folder, count, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (keyword, folder, count, count / sum(folders.values())))
            
            conn.commit()
            conn.close()
            
            logger.info("Analytics saved and closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing analytics: {e}")


# Integrate analytics into the existing classes

class FilenameAnalyzer:
    """Enhanced filename analyzer with analytics tracking"""
    
    def __init__(self, analytics: Optional[ComprehensiveAnalytics] = None):
        self.analytics = analytics
        self.separators = ['-', '_', ' ', '.', '(', ')', '[', ']']
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                          'img', 'image', 'photo', 'pic', 'picture', 'file', 'copy', 'new'}
        self.date_pattern = re.compile(r'\d{4}[-_]?\d{2}[-_]?\d{2}|\d{2}[-_]?\d{2}[-_]?\d{4}')
        self.number_pattern = re.compile(r'\d+')
    
    def extract_keywords(self, filename):
        """Extract meaningful keywords from a filename"""
        name_without_ext = os.path.splitext(filename)[0]
        
        if not name_without_ext or name_without_ext == '':
            return [], []
        
        name_lower = name_without_ext.lower()
        
        for sep in self.separators:
            name_lower = name_lower.replace(sep, ' ')
        
        dates = self.date_pattern.findall(name_without_ext)
        
        for date in dates:
            name_lower = name_lower.replace(date.lower(), '')
        
        words = name_lower.split()
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        keywords = [k for k in keywords if not k.isdigit()]
        
        # Track keyword extraction in analytics
        if self.analytics:
            self.analytics.session_data['feature_statistics']['keywords_extracted'].extend(keywords)
        
        return keywords, dates


class MultiModalImageClassifier(nn.Module):
    """Enhanced classifier with analytics tracking"""
    
    def __init__(self, num_classes, analytics: Optional[ComprehensiveAnalytics] = None):
        super(MultiModalImageClassifier, self).__init__()
        self.analytics = analytics
        
        # Image feature extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()
        
        # Text feature processor
        self.text_embedding = nn.Embedding(10000, 64)
        self.text_lstm = nn.LSTM(64, 128, batch_first=True)
        
        # Fusion layers
        self.fusion_layer = nn.Linear(512 + 128, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, image, text_indices, save_features=False, image_path=None):
        # Extract image features
        image_features = self.resnet(image)
        
        # Process text features
        text_embedded = self.text_embedding(text_indices)
        _, (text_features, _) = self.text_lstm(text_embedded)
        text_features = text_features.squeeze(0)
        
        # Combine features
        combined = torch.cat([image_features, text_features], dim=1)
        
        # Save features if requested
        if save_features and self.analytics and image_path:
            self.analytics.save_model_features(
                image_path, image_features, text_features, combined
            )
        
        # Final classification
        fused = self.relu(self.fusion_layer(combined))
        fused = self.dropout(fused)
        output = self.classifier(fused)
        
        return output


# The main function would be updated to use ComprehensiveAnalytics
# I'll show the key integration points:

def main():
    """Enhanced main function with comprehensive analytics"""
    logger.info("Starting analytics-enhanced image sorter.")
    
    # Initialize analytics
    analytics = ComprehensiveAnalytics()
    
    # Initialize components with analytics
    filename_analyzer = FilenameAnalyzer(analytics=analytics)
    
   
    # When creating the model:
    # model = MultiModalImageClassifier(num_classes, analytics=analytics)
    
    # When making decisions, track them:
    # analytics.record_decision(
    #     filename=filename,
    #     source_path=source_path,
    #     target_folder=target_folder,
    #     keywords=keywords,
    #     is_manual=is_manual,
    #     confidence=confidence,
    #     decision_time=decision_time,
    #     model_output=output
    # )
    
    # At the end:
    # analytics.close()
    
    logger.info("Analytics-enhanced image sorter completed.")

if __name__ == "__main__":
    # This is a framework - integrate with your existing main() function
    print("Analytics module loaded. Integrate with autosort_enhanced.py")
    
    
    analytics = ComprehensiveAnalytics()
    
    # Simulate some decisions
    for i in range(5):
        analytics.record_decision(
            filename=f"test_image_{i}.jpg",
            source_path=f"/path/to/test_image_{i}.jpg",
            target_folder=f"Folder_{i % 2}",
            keywords=[f"keyword_{i}", "test"],
            is_manual=True,
            confidence=0.85 + i * 0.02,
            decision_time=1.5 + i * 0.1
        )
    
    analytics.close()
    print(f"Analytics saved to {analytics.analytics_dir}")