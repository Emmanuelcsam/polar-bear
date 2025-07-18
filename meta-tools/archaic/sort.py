#!/usr/bin/env python3
"""
Ultimate Image Sorter - A comprehensive image organization tool with multiple sorting modes.
Supports clustering, resolution-based sorting, reference image sorting, and manual GUI sorting.
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
from scipy import stats, ndimage
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import mutual_info_classif
import h5py
import cv2
import imagehash
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_sorter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveAnalytics:
    """Handles all analytics, metrics tracking, and reporting for the image sorter."""
    
    def __init__(self, analytics_dir='analytics_data'):
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(exist_ok=True)
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
        
        self.db_path = self.analytics_dir / 'analytics.db'
        self._init_database()
        
        self.h5_path = self.analytics_dir / f'features_{self.session_id}.h5'
        self.h5_file = h5py.File(self.h5_path, 'w')
        
    def _init_database(self):
        """Initialize SQLite database for analytics storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Decisions table
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
            
            # Keyword-folder correlations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS keyword_folder_correlations (
                    keyword TEXT,
                    folder TEXT,
                    count INTEGER,
                    confidence REAL,
                    PRIMARY KEY (keyword, folder)
                )
            ''')
            
            # Feature importance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_importance (
                    session_id TEXT,
                    feature_name TEXT,
                    importance_score REAL,
                    correlation_with_accuracy REAL,
                    timestamp TEXT
                )
            ''')
            
            # Model metrics table
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
    
    def calculate_image_statistics(self, image_path: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for an image."""
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            stats = {
                'file_size': os.path.getsize(image_path),
                'width': img.width,
                'height': img.height,
                'aspect_ratio': img.width / img.height if img.height != 0 else 0,
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
                
                # Dominant colors
                hist_r = np.histogram(img_array[:,:,0], bins=256)[0]
                hist_g = np.histogram(img_array[:,:,1], bins=256)[0]
                hist_b = np.histogram(img_array[:,:,2], bins=256)[0]
                
                stats['dominant_r'] = int(np.argmax(hist_r))
                stats['dominant_g'] = int(np.argmax(hist_g))
                stats['dominant_b'] = int(np.argmax(hist_b))
                
                # Brightness and contrast
                gray = img.convert('L')
                gray_array = np.array(gray)
                stats['brightness'] = float(gray_array.mean())
                stats['contrast'] = float(gray_array.std())
                
                # Entropy
                def calculate_entropy(hist):
                    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
                    hist = hist[hist > 0]
                    return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
                
                stats['entropy'] = float(calculate_entropy(hist_r) + 
                                       calculate_entropy(hist_g) + 
                                       calculate_entropy(hist_b))
                
                # Edge density
                edges = self._detect_edges(gray_array)
                stats['edge_density'] = float(np.sum(edges) / (img.width * img.height)) if img.width * img.height > 0 else 0
            
            # File hash
            with open(image_path, 'rb') as f:
                stats['file_hash'] = hashlib.md5(f.read()).hexdigest()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating image statistics for {image_path}: {e}")
            return {}
    
    def _detect_edges(self, gray_array: np.ndarray) -> np.ndarray:
        """Detect edges using Sobel operator."""
        try:
            sx = ndimage.sobel(gray_array, axis=0)
            sy = ndimage.sobel(gray_array, axis=1)
            edges = np.hypot(sx, sy)
            return edges > np.mean(edges)
        except Exception as e:
            logger.error(f"Error detecting edges: {e}")
            return np.zeros_like(gray_array, dtype=bool)
    
    def record_decision(self, filename: str, source_path: str, target_folder: str, 
                       keywords: List[str], is_manual: bool, confidence: float,
                       decision_time: float, model_output: Optional[torch.Tensor] = None):
        """Record a sorting decision with all associated metrics."""
        img_stats = self.calculate_image_statistics(source_path)
        
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
        
        self.session_data['decisions'].append(decision)
        
        # Update keyword correlations
        for keyword in keywords:
            if target_folder not in self.session_data['keyword_correlations'][keyword]:
                self.session_data['keyword_correlations'][keyword][target_folder] = 0
            self.session_data['keyword_correlations'][keyword][target_folder] += 1
        
        # Update folder statistics
        if target_folder not in self.session_data['folder_statistics']:
            self.session_data['folder_statistics'][target_folder] = {
                'count': 0, 
                'total_confidence': 0
            }
        self.session_data['folder_statistics'][target_folder]['count'] += 1
        self.session_data['folder_statistics'][target_folder]['total_confidence'] += confidence
        
        # Save to database
        self._save_decision_to_db(decision, img_stats)
        
        # Update time series data
        self.session_data['time_series_data'].append({
            'timestamp': datetime.now().timestamp(),
            'cumulative_decisions': len(self.session_data['decisions']),
            'accuracy': self._calculate_running_accuracy()
        })
    
    def _save_decision_to_db(self, decision: Dict, img_stats: Dict):
        """Save decision to SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
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
                json.dumps([
                    img_stats.get('dominant_r', 0),
                    img_stats.get('dominant_g', 0),
                    img_stats.get('dominant_b', 0)
                ]),
                img_stats.get('brightness', 0),
                img_stats.get('contrast', 0),
                img_stats.get('entropy', 0)
            ))
            conn.commit()
    
    def save_model_features(self, image_path: str, features: torch.Tensor, 
                           text_features: torch.Tensor, combined_features: torch.Tensor):
        """Save model features to HDF5 file."""
        try:
            key = hashlib.md5(image_path.encode()).hexdigest()
            if key not in self.h5_file:
                grp = self.h5_file.create_group(key)
                grp.attrs['filename'] = image_path
                grp.attrs['timestamp'] = datetime.now().isoformat()
                grp.create_dataset('image_features', data=features.cpu().numpy())
                grp.create_dataset('text_features', data=text_features.cpu().numpy())
                grp.create_dataset('combined_features', data=combined_features.cpu().numpy())
        except Exception as e:
            logger.error(f"Error saving model features for {image_path}: {e}")
    
    def record_model_performance(self, accuracy: float, precision: Dict[str, float],
                                recall: Dict[str, float], f1_score: Dict[str, float],
                                loss: float, learning_rate: float, epoch: int):
        """Record model performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
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
    
    def calculate_correlations(self) -> Dict[str, Any]:
        """Calculate feature correlations and importance scores."""
        if len(self.session_data['decisions']) < 2:
            return {}
        
        # Create DataFrame from decisions
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
        
        # Calculate feature importance
        X = numeric_features.drop(['confidence', 'is_manual'], axis=1, errors='ignore')
        y = LabelEncoder().fit_transform(features_df['target_folder'])
        
        feature_importance = {}
        if len(np.unique(y)) > 1:
            mi_scores = mutual_info_classif(X, y)
            feature_importance = dict(zip(X.columns, mi_scores))
        
        return {
            'correlation_matrix': correlations.to_dict(),
            'feature_importance': feature_importance,
            'summary_statistics': features_df.describe().to_dict()
        }
    
    def generate_reports(self):
        """Generate comprehensive analytics reports."""
        logger.info("Generating analytics reports...")
        
        correlations = self.calculate_correlations()
        
        # JSON report
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
        
        # CSV reports
        if self.session_data['decisions']:
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
                    'percentage': count / sum(folders.values()) * 100 if sum(folders.values()) > 0 else 0
                })
        
        if keyword_data:
            keyword_df = pd.DataFrame(keyword_data)
            keyword_df.to_csv(self.analytics_dir / f'keyword_correlations_{self.session_id}.csv', 
                            index=False)
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate database summary
        self._generate_db_summary()
        
        # Save session data
        with open(self.analytics_dir / f'session_data_{self.session_id}.pkl', 'wb') as f:
            pickle.dump(self.session_data, f)
        
        logger.info(f"Reports saved to {self.analytics_dir}")
    
    def _generate_visualizations(self):
        """Generate visualization plots."""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig_dir = self.analytics_dir / 'visualizations'
            fig_dir.mkdir(exist_ok=True)
            
            # Timeline plot
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
            
            # Folder distribution
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
            
            # Keyword-folder heatmap
            if self.session_data['keyword_correlations']:
                keywords = list(self.session_data['keyword_correlations'].keys())[:20]
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
            
            # Confidence distribution
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
        """Generate database summary report."""
        with sqlite3.connect(self.db_path) as conn:
            summary = {}
            
            # Total decisions
            cursor = conn.execute("SELECT COUNT(*) FROM decisions")
            summary['total_decisions'] = cursor.fetchone()[0]
            
            # Folder summary
            cursor = conn.execute("""
                SELECT target_folder, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM decisions
                GROUP BY target_folder
            """)
            summary['folder_summary'] = cursor.fetchall()
            
            # Top keyword correlations
            cursor = conn.execute("""
                SELECT keyword, folder, count
                FROM keyword_folder_correlations
                ORDER BY count DESC
                LIMIT 20
            """)
            summary['top_keyword_correlations'] = cursor.fetchall()
        
        # Write summary
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
        """Calculate running accuracy based on manual decisions."""
        if not self.session_data['decisions']:
            return 0.0
        
        manual_decisions = [d for d in self.session_data['decisions'] if d['is_manual']]
        if not manual_decisions:
            return 0.0
        
        return np.mean([d['confidence'] for d in manual_decisions])
    
    def export_for_ml(self):
        """Export data for machine learning training."""
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
            # Save numpy arrays
            np.save(self.analytics_dir / f'features_{self.session_id}.npy', np.array(features))
            np.save(self.analytics_dir / f'labels_{self.session_id}.npy', np.array(labels))
            
            # Save PyTorch dataset
            torch_data = {
                'features': features,
                'labels': labels,
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
        """Close analytics and save all data."""
        try:
            self.generate_reports()
            self.export_for_ml()
            
            # Close HDF5 file
            if hasattr(self, 'h5_file'):
                self.h5_file.close()
            
            # Update keyword correlations in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for keyword, folders in self.session_data['keyword_correlations'].items():
                    for folder, count in folders.items():
                        cursor.execute('''
                            INSERT OR REPLACE INTO keyword_folder_correlations 
                            (keyword, folder, count, confidence)
                            VALUES (?, ?, ?, ?)
                        ''', (keyword, folder, count, 
                            count / sum(folders.values()) if sum(folders.values()) > 0 else 0))
                conn.commit()
            
            logger.info("Analytics saved and closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing analytics: {e}")


class FilenameAnalyzer:
    """Analyzes filenames to extract meaningful keywords and metadata."""
    
    def __init__(self, analytics: Optional[ComprehensiveAnalytics] = None):
        self.analytics = analytics
        self.separators = ['-', '_', ' ', '.', '(', ')', '[', ']']
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'img', 'image', 'photo', 'pic', 'picture', 'file', 'copy', 'new'
        }
        self.date_pattern = re.compile(r'\d{4}[-_]?\d{2}[-_]?\d{2}|\d{2}[-_]?\d{2}[-_]?\d{4}')
        self.number_pattern = re.compile(r'\d+')
    
    def extract_keywords(self, filename: str) -> Tuple[List[str], List[str]]:
        """Extract keywords and dates from filename."""
        name_without_ext = os.path.splitext(filename)[0]
        if not name_without_ext:
            return [], []
        
        name_lower = name_without_ext.lower()
        
        # Replace separators with spaces
        for sep in self.separators:
            name_lower = name_lower.replace(sep, ' ')
        
        # Extract dates
        dates = self.date_pattern.findall(name_without_ext)
        for date in dates:
            name_lower = name_lower.replace(date.lower(), '')
        
        # Extract words
        words = name_lower.split()
        
        # Filter keywords
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        keywords = [k for k in keywords if not k.isdigit()]
        
        # Record in analytics
        if self.analytics:
            self.analytics.session_data['feature_statistics']['keywords_extracted'].extend(keywords)
        
        return keywords, dates
    
    def get_filename_features(self, filename: str) -> Dict[str, Any]:
        """Get comprehensive features from filename."""
        keywords, dates = self.extract_keywords(filename)
        
        features = {
            'keywords': keywords,
            'has_date': len(dates) > 0,
            'date_count': len(dates),
            'keyword_count': len(keywords),
            'filename_length': len(filename),
            'has_numbers': bool(self.number_pattern.search(filename))
        }
        
        return features


class MultiModalImageClassifier(nn.Module):
    """Multi-modal neural network for image classification using both visual and text features."""
    
    def __init__(self, num_classes: int, analytics: Optional[ComprehensiveAnalytics] = None):
        super(MultiModalImageClassifier, self).__init__()
        self.analytics = analytics
        
        # Visual feature extractor (ResNet18)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Remove final classification layer
        
        # Text feature extractor
        self.text_embedding = nn.Embedding(10000, 64)
        self.text_lstm = nn.LSTM(64, 128, batch_first=True)
        
        # Fusion and classification layers
        self.fusion_layer = nn.Linear(512 + 128, 256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, image: torch.Tensor, text_indices: torch.Tensor, 
                save_features: bool = False, image_path: Optional[str] = None) -> torch.Tensor:
        """Forward pass through the network."""
        # Extract image features
        image_features = self.resnet(image)
        
        # Extract text features
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
        
        # Fusion and classification
        fused = self.relu(self.fusion_layer(combined))
        fused = self.dropout(fused)
        output = self.classifier(fused)
        
        return output


class LearningHistory:
    """Manages learning history and keyword-folder correlations."""
    
    def __init__(self, history_file: str = 'learning_history.pkl', 
                 analytics: Optional[ComprehensiveAnalytics] = None):
        self.history_file = history_file
        self.analytics = analytics
        self.keyword_to_folder = defaultdict(lambda: defaultdict(int))
        self.folder_keywords = defaultdict(set)
        self.decisions = []
        self.load_history()
    
    def load_history(self):
        """Load history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'rb') as f:
                    data = pickle.load(f)
                    self.keyword_to_folder = defaultdict(
                        lambda: defaultdict(int), 
                        data.get('keyword_to_folder', {})
                    )
                    self.folder_keywords = defaultdict(
                        set, 
                        {k: set(v) for k, v in data.get('folder_keywords', {}).items()}
                    )
                    self.decisions = data.get('decisions', [])
                logger.info(f"Loaded {len(self.decisions)} decisions from history")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
    
    def save_history(self):
        """Save history to file."""
        try:
            data = {
                'keyword_to_folder': dict(self.keyword_to_folder),
                'folder_keywords': {k: list(v) for k, v in self.folder_keywords.items()},
                'decisions': self.decisions[-10000:]  # Keep only last 10k decisions
            }
            with open(self.history_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info("Saved learning history")
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def record_decision(self, filename: str, keywords: List[str], folder_name: str):
        """Record a sorting decision."""
        decision = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'keywords': keywords,
            'folder': folder_name
        }
        self.decisions.append(decision)
        
        # Update correlations
        for keyword in keywords:
            self.keyword_to_folder[keyword][folder_name] += 1
            self.folder_keywords[folder_name].add(keyword)
        
        # Keep history size manageable
        if len(self.decisions) > 10000:
            self.decisions = self.decisions[-10000:]
        
        # Update analytics
        if self.analytics:
            self.analytics.session_data['user_behavior']['learning_history_size'].append(
                len(self.decisions)
            )
    
    def get_keyword_predictions(self, keywords: List[str]) -> Dict[str, float]:
        """Get folder predictions based on keywords."""
        folder_scores = defaultdict(float)
        
        for keyword in keywords:
            if keyword in self.keyword_to_folder:
                for folder, count in self.keyword_to_folder[keyword].items():
                    folder_scores[folder] += count
        
        return dict(folder_scores)


class Vocabulary:
    """Manages vocabulary for text encoding."""
    
    def __init__(self, vocab_file: str = 'vocabulary.json', 
                 analytics: Optional[ComprehensiveAnalytics] = None):
        self.vocab_file = vocab_file
        self.analytics = analytics
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.load_vocabulary()
    
    def load_vocabulary(self):
        """Load vocabulary from file."""
        if os.path.exists(self.vocab_file):
            try:
                with open(self.vocab_file, 'r') as f:
                    self.word_to_idx = json.load(f)
                    self.idx_to_word = {int(idx): word for word, idx in self.word_to_idx.items()}
                logger.info(f"Loaded vocabulary with {len(self.word_to_idx)} words")
            except Exception as e:
                logger.error(f"Error loading vocabulary: {e}")
    
    def save_vocabulary(self):
        """Save vocabulary to file."""
        try:
            with open(self.vocab_file, 'w') as f:
                json.dump(self.word_to_idx, f, indent=2)
            
            if self.analytics:
                self.analytics.session_data['feature_statistics']['vocabulary_size'].append(
                    len(self.word_to_idx)
                )
            logger.info("Saved vocabulary")
        except Exception as e:
            logger.error(f"Error saving vocabulary: {e}")
    
    def add_word(self, word: str):
        """Add a word to vocabulary."""
        if word not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def encode_keywords(self, keywords: List[str], max_len: int = 20) -> List[int]:
        """Encode keywords to indices."""
        indices = []
        
        for word in keywords[:max_len]:
            indices.append(self.word_to_idx.get(word, self.word_to_idx['<UNK>']))
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(self.word_to_idx['<PAD>'])
        
        return indices[:max_len]


class FeatureCache:
    """SQLite-based cache for image features."""
    
    def __init__(self, db_path: str = 'features.sqlite'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS feats
                (path TEXT PRIMARY KEY, vec BLOB)
            """)
    
    def get(self, fpath: str) -> Optional[np.ndarray]:
        """Get cached features for a file."""
        cur = self.conn.execute("SELECT vec FROM feats WHERE path=?", (fpath,))
        row = cur.fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None
    
    def put(self, fpath: str, vec: np.ndarray):
        """Cache features for a file."""
        self.conn.execute(
            "INSERT OR REPLACE INTO feats VALUES (?,?)",
            (fpath, vec.astype(np.float32).tobytes())
        )
        self.conn.commit()


# Feature extraction functions
def perceptual_bits(img: Image.Image, hash_size: int = 16) -> np.ndarray:
    """Extract perceptual hash features."""
    ph = imagehash.phash(img, hash_size=hash_size).hash.astype(np.uint8).flatten()
    dh = imagehash.dhash(img, hash_size=hash_size).hash.astype(np.uint8).flatten()
    return np.concatenate([ph, dh])


def colour_hist(img_bgr: np.ndarray, bins: Tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    """Extract color histogram features."""
    hist = cv2.calcHist([img_bgr], [0, 1, 2], None, bins, [0, 256] * 3).flatten()
    return hist / (hist.sum() + 1e-8)


def simple_stats(img: np.ndarray) -> np.ndarray:
    """Extract simple statistical features."""
    resized = cv2.resize(img, (32, 32))
    return np.array([resized.mean(), resized.std(), np.median(resized)])


def ensure_dir(p: Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def load_class_rules(path: Path) -> Dict[str, str]:
    """Load classification rules from file."""
    mapping = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.lower()
            mapping[token] = token.replace(" ", "_")
    return mapping


def extract_features_worker(fpath: str, cache: FeatureCache) -> Tuple[str, Optional[np.ndarray]]:
    """Worker function for parallel feature extraction."""
    # Check cache first
    vec = cache.get(fpath)
    if vec is not None:
        return fpath, vec
    
    try:
        # Read image
        img_cv = cv2.imread(fpath)
        if img_cv is None:
            return fpath, None
        
        # Convert to PIL
        pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # Extract features
        hist = colour_hist(img_cv)
        hash_bits = perceptual_bits(pil)
        stats = simple_stats(img_cv)
        
        # Combine features
        vec = np.concatenate([hist, hash_bits, stats])
        
        # Cache features
        cache.put(fpath, vec)
        
        return fpath, vec
        
    except Exception as e:
        logger.error(f"Error extracting features for {fpath}: {e}")
        return fpath, None


class UltimateImageSorter:
    """Main class for the Ultimate Image Sorter application."""
    
    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp")
    
    def __init__(self, input_dir: str, output_dir: str, analytics_dir: str, mode: str,
                 reference_image: Optional[str] = None, target_dirs: Optional[List[str]] = None):
        """Initialize the Ultimate Image Sorter."""
        self.in_dir = Path(input_dir).resolve()
        self.out_dir = Path(output_dir).resolve()
        ensure_dir(self.out_dir)
        
        # Initialize components
        self.analytics = ComprehensiveAnalytics(analytics_dir)
        self.cache = FeatureCache()
        self.class_rules = load_class_rules(self.in_dir / "classification.txt")
        self.filename_analyzer = FilenameAnalyzer(self.analytics)
        self.learning_history = LearningHistory(analytics=self.analytics)
        self.vocabulary = Vocabulary(analytics=self.analytics)
        
        # Image data
        self.images = []
        self.features = []
        self.cluster_labels = []
        
        # Sorting parameters
        self.mode = mode
        self.reference_image = reference_image
        self.target_dirs = target_dirs or []
        
        # PyTorch setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
    
    def gather(self):
        """Gather all images and extract features."""
        logger.info("Scanning for images...")
        
        # Find all images
        for fp in self.in_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                self.images.append(str(fp))
                # Extract keywords for vocabulary
                keywords, _ = self.filename_analyzer.extract_keywords(fp.name)
                for keyword in keywords:
                    self.vocabulary.add_word(keyword)
        
        self.vocabulary.save_vocabulary()
        
        if not self.images:
            logger.error("No images found in input directory")
            sys.exit(1)
        
        logger.info(f"Found {len(self.images)} images")
        
        # Extract features in parallel
        logger.info("Extracting features...")
        num_workers = max(1, mp.cpu_count() - 1)
        
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(partial(extract_features_worker, cache=self.cache), self.images),
                total=len(self.images),
                desc="Extracting features"
            ))
        
        # Filter valid results
        valid_results = [r for r in results if r[1] is not None]
        self.images = [r[0] for r in valid_results]
        self.features = np.vstack([r[1] for r in valid_results])
        
        # Normalize features
        self.features = StandardScaler().fit_transform(self.features)
        
        logger.info(f"Successfully extracted features for {len(self.images)} images")
    
    def adaptive_cluster(self):
        """Perform adaptive clustering with multiple passes."""
        logger.info("Performing adaptive clustering...")
        
        remaining_idx = np.arange(len(self.images))
        eps = 0.45
        min_samples = 2
        label_counter = 0
        
        self.cluster_labels = np.full(len(self.images), -99, dtype=int)
        
        for pass_no in range(1, 6):
            if not remaining_idx.size:
                break
            
            # Cluster remaining images
            feats_pass = self.features[remaining_idx]
            labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(feats_pass)
            
            # Assign global labels
            for local_lbl in set(labels):
                if local_lbl == -1:
                    continue
                global_lbl = label_counter
                label_counter += 1
                self.cluster_labels[remaining_idx[labels == local_lbl]] = global_lbl
            
            # Update remaining indices
            remaining_idx = remaining_idx[labels == -1]
            
            # Report progress
            pct_left = 100 * remaining_idx.size / len(self.images)
            logger.info(f"Pass {pass_no}: eps={eps:.2f}, unsorted={pct_left:.1f}%")
            
            if pct_left <= 5:
                break
            
            # Adjust parameters for next pass
            eps += 0.25
            min_samples = max(2, min_samples - 1)
        
        # Mark remaining as unique
        self.cluster_labels[self.cluster_labels == -99] = -1
        
        logger.info(f"Clustering complete: {label_counter} clusters found")
    
    def sort_by_resolution(self):
        """Sort images by resolution."""
        logger.info("Sorting images by resolution...")
        
        dst_map = defaultdict(list)
        
        for img_path in tqdm(self.images, desc="Sorting by resolution"):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    folder = f"{width}x{height}"
                    dst_map[folder].append(img_path)
            except Exception as e:
                logger.error(f"Error sorting {img_path} by resolution: {e}")
                continue
        
        # Move files
        for folder, files in dst_map.items():
            tgt_dir = self.out_dir / folder
            ensure_dir(tgt_dir)
            
            for src in files:
                dst = tgt_dir / Path(src).name
                dup_idx = 1
                while dst.exists():
                    dst = tgt_dir / f"{dst.stem}_{dup_idx}{dst.suffix}"
                    dup_idx += 1
                shutil.move(src, dst)
        
        logger.info(f"Sorted {len(self.images)} images into {len(dst_map)} resolution folders")
    
    def sort_by_reference(self):
        """Sort images by similarity to reference image."""
        if not self.reference_image:
            logger.error("No reference image provided")
            return
        
        logger.info(f"Sorting by similarity to reference: {self.reference_image}")
        
        try:
            ref_img = Image.open(self.reference_image)
            ref_hash = imagehash.dhash(ref_img)
            
            dst_map = {'similar': [], 'dissimilar': []}
            threshold = 10
            
            for img_path in tqdm(self.images, desc="Comparing to reference"):
                try:
                    img = Image.open(img_path)
                    img_hash = imagehash.dhash(img)
                    distance = ref_hash - img_hash
                    
                    if distance < threshold:
                        dst_map['similar'].append(img_path)
                    else:
                        dst_map['dissimilar'].append(img_path)
                except Exception as e:
                    logger.error(f"Error comparing {img_path}: {e}")
                    dst_map['dissimilar'].append(img_path)
            
            # Move files
            for folder, files in dst_map.items():
                tgt_dir = self.out_dir / folder
                ensure_dir(tgt_dir)
                
                for src in files:
                    dst = tgt_dir / Path(src).name
                    dup_idx = 1
                    while dst.exists():
                        dst = tgt_dir / f"{dst.stem}_{dup_idx}{dst.suffix}"
                        dup_idx += 1
                    shutil.move(src, dst)
            
            logger.info(f"Sorted {len(dst_map['similar'])} similar and "
                       f"{len(dst_map['dissimilar'])} dissimilar images")
            
        except Exception as e:
            logger.error(f"Error in reference sorting: {e}")
    
    def manual_sort_gui(self):
        """Manual sorting with GUI."""
        if not self.target_dirs:
            logger.error("No target directories provided for manual mode")
            return
        
        # Initialize Pygame
        pygame.init()
        screen_width = 1200
        screen_height = 600
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Ultimate Image Sorter - Manual Mode")
        
        # Fonts
        font = pygame.font.SysFont(None, 30)
        small_font = pygame.font.SysFont(None, 20)
        
        # Layout calculations
        col_width = screen_width // 3
        button_height = 50
        button_width = screen_width // len(self.target_dirs)
        
        # Create buttons for target directories
        buttons = []
        for idx, tdir in enumerate(self.target_dirs):
            rect = pygame.Rect(
                idx * button_width, 
                screen_height - button_height, 
                button_width, 
                button_height
            )
            label = os.path.basename(tdir)
            buttons.append((rect, tdir, label))
        
        # Key mappings
        dir_keys = [pygame.K_1 + i for i in range(len(self.target_dirs))]
        image_keys = [pygame.K_a, pygame.K_s, pygame.K_d]
        key_labels = ['A', 'S', 'D']
        
        # Auto mode button
        auto_mode = False
        auto_rect = pygame.Rect(screen_width - 150, 0, 150, 50)
        
        # Initialize model
        num_classes = len(self.target_dirs)
        self.model = MultiModalImageClassifier(num_classes, self.analytics).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Load model if exists
        model_path = 'sorter_model.pth'
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['state_dict'])
                logger.info("Loaded existing model")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Starting with fresh model.")
        
        # Image queue and slots
        image_queue = list(self.images)
        current_slots = [None] * 3
        current_keywords = [[] for _ in range(3)]
        
        # Fill initial slots
        for slot in range(3):
            if image_queue:
                img_path = image_queue.pop(0)
                current_slots[slot] = img_path
                current_keywords[slot] = self.filename_analyzer.extract_keywords(
                    os.path.basename(img_path)
                )[0]
        
        selected_slot = None
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # Check if done
            if not any(current_slots) and not image_queue:
                break
            
            # Clear screen
            screen.fill((0, 0, 0))
            
            # Draw images
            for slot in range(3):
                if current_slots[slot]:
                    try:
                        img = pygame.image.load(current_slots[slot])
                        img_rect = img.get_rect()
                        
                        # Scale image to fit
                        scale = min(
                            col_width * 0.9 / img_rect.w, 
                            (screen_height - button_height - 30) / img_rect.h, 
                            1
                        )
                        scaled_size = (int(img_rect.w * scale), int(img_rect.h * scale))
                        scaled_image = pygame.transform.scale(img, scaled_size)
                        
                        # Position image
                        x = slot * col_width + (col_width - scaled_size[0]) // 2
                        y = 10
                        screen.blit(scaled_image, (x, y))
                        
                        # Draw key label
                        text_color = (255, 0, 0) if selected_slot == slot else (255, 255, 255)
                        text = font.render(key_labels[slot], True, text_color)
                        text_rect = text.get_rect(
                            center=(x + col_width // 2, y + scaled_size[1] + 20)
                        )
                        screen.blit(text, text_rect)
                        
                    except Exception as e:
                        logger.error(f"Error displaying image {current_slots[slot]}: {e}")
            
            # Draw buttons
            for rect, _, label in buttons:
                pygame.draw.rect(screen, (0, 255, 0), rect)
                text = font.render(label, True, (0, 0, 0))
                screen.blit(text, text.get_rect(center=rect.center))
            
            # Draw auto mode button
            auto_color = (255, 0, 0) if auto_mode else (0, 0, 255)
            pygame.draw.rect(screen, auto_color, auto_rect)
            auto_text = "Auto: ON" if auto_mode else "Auto: OFF"
            text = font.render(auto_text, True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=auto_rect.center))
            
            # Update display
            pygame.display.flip()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if auto_mode:
                        continue
                    
                    # Select image
                    if selected_slot is None:
                        for idx, key in enumerate(image_keys):
                            if event.key == key and current_slots[idx]:
                                selected_slot = idx
                    
                    # Move selected image
                    else:
                        if event.key in dir_keys:
                            dir_idx = dir_keys.index(event.key)
                            self.move_slot_to_dir(
                                selected_slot, dir_idx, current_slots, 
                                current_keywords, image_queue, is_manual=True
                            )
                            selected_slot = None
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    
                    # Toggle auto mode
                    if auto_rect.collidepoint(pos):
                        auto_mode = not auto_mode
                        continue
                    
                    if auto_mode:
                        continue
                    
                    # Select image by click
                    if selected_slot is None:
                        for slot in range(3):
                            slot_x_start = slot * col_width
                            slot_x_end = (slot + 1) * col_width
                            if (slot_x_start <= pos[0] < slot_x_end and 
                                pos[1] < screen_height - button_height and
                                current_slots[slot]):
                                selected_slot = slot
                    
                    # Move to folder by click
                    else:
                        for btn_idx, (rect, _, _) in enumerate(buttons):
                            if rect.collidepoint(pos):
                                self.move_slot_to_dir(
                                    selected_slot, btn_idx, current_slots,
                                    current_keywords, image_queue, is_manual=True
                                )
                                selected_slot = None
            
            # Auto mode processing
            if auto_mode:
                for slot in range(3):
                    if current_slots[slot]:
                        self.move_slot_to_dir(
                            slot, None, current_slots, current_keywords, 
                            image_queue, is_manual=False
                        )
                time.sleep(0.1)  # Small delay for auto mode
            
            clock.tick(30)  # 30 FPS
        
        # Cleanup
        pygame.quit()
        
        # Save model
        if self.model:
            torch.save({'state_dict': self.model.state_dict()}, model_path)
            logger.info("Saved model")
    
    def move_slot_to_dir(self, slot_idx: int, dir_idx: Optional[int], 
                        current_slots: List[Optional[str]], 
                        current_keywords: List[List[str]],
                        image_queue: List[str], is_manual: bool = True):
        """Move image from slot to directory."""
        path = current_slots[slot_idx]
        keywords = current_keywords[slot_idx]
        
        if not path:
            return
        
        start_time = time.time()
        confidence = 0.0
        output = None
        
        # Use model for prediction or training
        if self.model:
            try:
                # Prepare inputs
                img = Image.open(path).convert('RGB')
                input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                
                text_indices = self.vocabulary.encode_keywords(keywords)
                text_tensor = torch.tensor([text_indices]).to(self.device)
                
                # Forward pass
                output = self.model(input_tensor, text_tensor, save_features=True, image_path=path)
                probs = torch.softmax(output, dim=1)
                
                if is_manual:
                    # Training mode
                    confidence = probs[0, dir_idx].item()
                    label = torch.tensor([dir_idx]).to(self.device)
                    loss = self.criterion(output, label)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    # Prediction mode
                    dir_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, dir_idx].item()
                    
            except Exception as e:
                logger.error(f"Error in model inference/training for {path}: {e}")
                if not is_manual:
                    dir_idx = 0  # Default to first directory
        
        # Ensure we have a valid directory index
        if dir_idx is None:
            dir_idx = 0
        
        # Record decision
        folder_name = os.path.basename(self.target_dirs[dir_idx])
        self.learning_history.record_decision(os.path.basename(path), keywords, folder_name)
        
        decision_time = time.time() - start_time
        self.analytics.record_decision(
            os.path.basename(path), path, folder_name, keywords, 
            is_manual, confidence, decision_time, output
        )
        
        # Move file
        tdir = self.target_dirs[dir_idx]
        ensure_dir(Path(tdir))
        
        target_path = os.path.join(tdir, os.path.basename(path))
        dup_idx = 1
        while os.path.exists(target_path):
            stem, suffix = os.path.splitext(os.path.basename(path))
            target_path = os.path.join(tdir, f"{stem}_{dup_idx}{suffix}")
            dup_idx += 1
        
        shutil.move(path, target_path)
        
        # Clear slot and fill with next image
        current_slots[slot_idx] = None
        current_keywords[slot_idx] = []
        
        if image_queue:
            next_path = image_queue.pop(0)
            current_slots[slot_idx] = next_path
            current_keywords[slot_idx] = self.filename_analyzer.extract_keywords(
                os.path.basename(next_path)
            )[0]
    
    def organise(self):
        """Organize images based on clustering results."""
        logger.info("Organizing images...")
        
        dst_map = defaultdict(list)
        
        for fp, lbl in zip(self.images, self.cluster_labels):
            # Check classification rules
            rule_dst = None
            lower = Path(fp).name.lower()
            for token, folder in self.class_rules.items():
                if token in lower:
                    rule_dst = folder
                    break
            
            # Determine destination folder
            if lbl == -1:
                dst_folder = rule_dst or "unique_images"
            else:
                dst_folder = rule_dst or f"group_{lbl+1:03d}"
            
            dst_map[dst_folder].append(fp)
        
        # Move files
        for folder, files in tqdm(dst_map.items(), desc="Moving files"):
            tgt_dir = self.out_dir / folder
            ensure_dir(tgt_dir)
            
            for src in files:
                dst = tgt_dir / Path(src).name
                dup_idx = 1
                while dst.exists():
                    dst = tgt_dir / f"{dst.stem}_{dup_idx}{dst.suffix}"
                    dup_idx += 1
                shutil.move(src, dst)
        
        # Save summary
        summary = {
            "total_images": len(self.images),
            "clusters": int(max(self.cluster_labels) + 1) if self.cluster_labels.size > 0 else 0,
            "unique_images": int((self.cluster_labels == -1).sum()) if self.cluster_labels.size > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "folders": {folder: len(files) for folder, files in dst_map.items()}
        }
        
        with open(self.out_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Organized {len(self.images)} images into {len(dst_map)} folders")
    
    def run(self):
        """Run the image sorter."""
        logger.info(f"Starting Ultimate Image Sorter in {self.mode} mode")
        
        # Gather images and extract features
        self.gather()
        
        # Execute sorting based on mode
        if self.mode == "cluster":
            self.adaptive_cluster()
            self.organise()
        elif self.mode == "resolution":
            self.sort_by_resolution()
        elif self.mode == "reference":
            self.sort_by_reference()
        elif self.mode == "manual":
            self.manual_sort_gui()
        
        # Cleanup
        self.analytics.close()
        self.learning_history.save_history()
        
        logger.info("Sorting complete!")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Ultimate Image Sorter")
    print("=" * 60)
    
    # Get input directory
    input_dir = input("Enter input directory path: ").strip()
    while not input_dir or not os.path.exists(input_dir):
        print("Invalid input directory. Please enter a valid path.")
        input_dir = input("Enter input directory path: ").strip()
    
    # Get output directory
    output_dir = input("Enter output directory path: ").strip()
    while not output_dir:
        print("Output directory cannot be empty.")
        output_dir = input("Enter output directory path: ").strip()
    
    # Get analytics directory
    analytics_dir = input("Enter analytics directory path (default: analytics_data): ").strip()
    if not analytics_dir:
        analytics_dir = "analytics_data"
    
    # Get mode
    print("\nAvailable modes:")
    print("  cluster    - Group similar images using adaptive clustering")
    print("  resolution - Sort images by their resolution")
    print("  reference  - Sort images by similarity to a reference image")
    print("  manual     - Manually sort images with ML assistance")
    
    mode = input("\nEnter mode: ").strip().lower()
    while mode not in ["cluster", "resolution", "reference", "manual"]:
        print("Invalid mode. Choose from: cluster, resolution, reference, manual")
        mode = input("Enter mode: ").strip().lower()
    
    # Mode-specific parameters
    reference_image = None
    target_dirs = None
    
    if mode == "reference":
        reference_image = input("Enter reference image path: ").strip()
        while not reference_image or not os.path.exists(reference_image):
            print("Invalid reference image path.")
            reference_image = input("Enter reference image path: ").strip()
    
    elif mode == "manual":
        print("\nEnter target directories for manual sorting.")
        print("You can specify existing directories or new ones will be created.")
        target_dirs_input = input("Enter directories (comma-separated): ").strip()
        target_dirs = [d.strip() for d in target_dirs_input.split(',') if d.strip()]
        
        while not target_dirs:
            print("At least one target directory is required.")
            target_dirs_input = input("Enter directories (comma-separated): ").strip()
            target_dirs = [d.strip() for d in target_dirs_input.split(',') if d.strip()]
        
        # Ensure target directories are absolute paths
        target_dirs = [os.path.abspath(d) for d in target_dirs]
    
    # Create and run sorter
    try:
        sorter = UltimateImageSorter(
            input_dir, output_dir, analytics_dir, mode, 
            reference_image, target_dirs
        )
        sorter.run()
        
        print("\n" + "=" * 60)
        print("Sorting completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Analytics saved to: {analytics_dir}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Check ultimate_sorter.log for details.")


if __name__ == "__main__":
    main()