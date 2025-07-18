import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import numpy as np
import hashlib
import json
from collections import defaultdict
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartImageTensorizer:
    def __init__(self, 
                 resize=(224, 224), 
                 normalize=True, 
                 save_format='pt',
                 compute_stats=True):
        """
        Initialize the smart image tensorizer with statistical analysis
        
        Args:
            resize: Tuple (height, width) to resize images for consistent analysis
            normalize: Whether to normalize images to [0, 1] range
            save_format: Format to save tensors ('pt' for PyTorch, 'npy' for NumPy)
            compute_stats: Whether to compute comprehensive statistics
        """
        self.resize = resize
        self.normalize = normalize
        self.save_format = save_format
        self.compute_stats = compute_stats
        
        # Define transforms
        transform_list = []
        if resize:
            transform_list.append(transforms.Resize(resize))
        transform_list.append(transforms.ToTensor())
        if not normalize:
            transform_list.append(transforms.Lambda(lambda x: x * 255))
        
        self.transform = transforms.Compose(transform_list)
        
        # Supported image extensions
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', 
                               '.tiff', '.tif', '.ico', '.jfif')
        
        # Initialize data structures for analysis
        self.class_data = defaultdict(lambda: {
            'tensors': [],
            'paths': [],
            'metadata': [],
            'statistics': {}
        })
        
        self.global_statistics = {
            'total_images': 0,
            'class_distribution': {},
            'cross_class_correlations': {},
            'pca_analysis': {},
            'neural_network_insights': {}
        }
        
        logger.info(f"Initialized smart tensorizer with analysis capabilities")
    
    def extract_class_from_path(self, file_path):
        """Extract class label from folder structure"""
        parts = Path(file_path).parts
        
        # Find batch or category folder
        for part in parts:
            if 'batch' in part.lower() or any(keyword in part.lower() for keyword in 
                ['cladding', 'core', 'clean', 'defect', 'ferrule', 'scratch', 'zone']):
                return part
        
        # Default to parent folder name
        return Path(file_path).parent.name
    
    def compute_tensor_statistics(self, tensor):
        """Compute comprehensive statistics for a single tensor"""
        stats = {}
        
        # Flatten tensor for certain computations
        flat_tensor = tensor.flatten()
        
        # First order statistics
        stats['mean'] = tensor.mean().item()
        stats['std'] = tensor.std().item()
        stats['min'] = tensor.min().item()
        stats['max'] = tensor.max().item()
        stats['median'] = tensor.median().item()
        
        # Second order statistics
        stats['variance'] = tensor.var().item()
        stats['skewness'] = scipy_stats.skew(flat_tensor.numpy())
        stats['kurtosis'] = scipy_stats.kurtosis(flat_tensor.numpy())
        
        # Channel-wise statistics (if multi-channel)
        if tensor.dim() > 2:
            stats['channel_means'] = [tensor[i].mean().item() for i in range(tensor.shape[0])]
            stats['channel_stds'] = [tensor[i].std().item() for i in range(tensor.shape[0])]
            stats['channel_correlations'] = self.compute_channel_correlations(tensor)
        
        # Gradient information
        stats['gradients'] = self.compute_gradient_stats(tensor)
        
        # Frequency domain analysis
        stats['frequency_features'] = self.compute_frequency_features(tensor)
        
        # Texture features
        stats['texture_features'] = self.compute_texture_features(tensor)
        
        return stats
    
    def compute_channel_correlations(self, tensor):
        """Compute correlations between color channels"""
        if tensor.shape[0] < 2:
            return None
        
        correlations = {}
        for i in range(tensor.shape[0]):
            for j in range(i+1, tensor.shape[0]):
                corr = torch.corrcoef(torch.stack([
                    tensor[i].flatten(),
                    tensor[j].flatten()
                ]))[0, 1].item()
                correlations[f'channel_{i}_vs_{j}'] = corr
        
        return correlations
    
    def compute_gradient_stats(self, tensor):
        """Compute gradient statistics (edge information)"""
        if tensor.dim() < 3:
            tensor = tensor.unsqueeze(0)
        
        # Sobel filters for gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        gradient_stats = {}
        
        for i in range(tensor.shape[0]):
            channel = tensor[i:i+1].unsqueeze(0)
            
            # Apply Sobel filters
            grad_x = F.conv2d(channel, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
            grad_y = F.conv2d(channel, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
            
            # Gradient magnitude
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
            
            gradient_stats[f'channel_{i}_gradient_mean'] = grad_mag.mean().item()
            gradient_stats[f'channel_{i}_gradient_std'] = grad_mag.std().item()
            gradient_stats[f'channel_{i}_gradient_max'] = grad_mag.max().item()
        
        return gradient_stats
    
    def compute_frequency_features(self, tensor):
        """Compute frequency domain features using FFT"""
        freq_features = {}
        
        # 2D FFT for each channel
        for i in range(tensor.shape[0]):
            fft = torch.fft.fft2(tensor[i])
            magnitude = torch.abs(fft)
            
            # Radial average of magnitude spectrum
            center = (magnitude.shape[0] // 2, magnitude.shape[1] // 2)
            y, x = torch.meshgrid(torch.arange(magnitude.shape[0]), torch.arange(magnitude.shape[1]))
            r = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # Compute energy in different frequency bands
            low_freq_mask = r < min(center) * 0.1
            mid_freq_mask = (r >= min(center) * 0.1) & (r < min(center) * 0.5)
            high_freq_mask = r >= min(center) * 0.5
            
            freq_features[f'channel_{i}_low_freq_energy'] = magnitude[low_freq_mask].mean().item()
            freq_features[f'channel_{i}_mid_freq_energy'] = magnitude[mid_freq_mask].mean().item()
            freq_features[f'channel_{i}_high_freq_energy'] = magnitude[high_freq_mask].mean().item()
        
        return freq_features
    
    def compute_texture_features(self, tensor):
        """Compute texture features (simplified GLCM-like features)"""
        texture_features = {}
        
        for i in range(tensor.shape[0]):
            channel = tensor[i]
            
            # Local variance (texture roughness)
            kernel_size = 5
            unfold = F.unfold(channel.unsqueeze(0).unsqueeze(0), kernel_size, padding=kernel_size//2)
            local_vars = unfold.var(dim=1)
            
            texture_features[f'channel_{i}_texture_variance'] = local_vars.mean().item()
            texture_features[f'channel_{i}_texture_contrast'] = local_vars.std().item()
        
        return texture_features
    
    def find_all_images(self, root_dir):
        """Recursively find all image files and organize by class"""
        root_path = Path(root_dir)
        image_files_by_class = defaultdict(list)
        
        logger.info(f"Scanning for images in {root_dir} and organizing by class...")
        
        for path in root_path.rglob('*'):
            if path.is_file() and path.suffix.lower() in self.image_extensions:
                class_label = self.extract_class_from_path(path)
                image_files_by_class[class_label].append(path)
        
        total_images = sum(len(files) for files in image_files_by_class.values())
        logger.info(f"Found {total_images} images across {len(image_files_by_class)} classes")
        
        for class_name, files in image_files_by_class.items():
            logger.info(f"  {class_name}: {len(files)} images")
        
        return image_files_by_class
    
    def tensorize_and_analyze_image(self, img_path, class_label):
        """Convert image to tensor and perform analysis"""
        try:
            # Load and tensorize image
            img = Image.open(img_path)
            
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            tensor = self.transform(img)
            
            # Compute statistics if enabled
            stats = None
            if self.compute_stats:
                stats = self.compute_tensor_statistics(tensor)
            
            metadata = {
                'original_size': img.size,
                'original_mode': img.mode,
                'tensor_shape': tensor.shape,
                'source_path': str(img_path),
                'class_label': class_label,
                'statistics': stats
            }
            
            return tensor, metadata
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return None, None
    
    def compute_class_level_statistics(self, class_name, tensors):
        """Compute comprehensive statistics for an entire class"""
        if not tensors:
            return {}
        
        logger.info(f"Computing class-level statistics for {class_name}...")
        
        # Stack all tensors
        stacked = torch.stack(tensors)
        
        class_stats = {
            'sample_count': len(tensors),
            'mean_tensor': stacked.mean(dim=0),
            'std_tensor': stacked.std(dim=0),
            'covariance_matrix': self.compute_covariance_matrix(stacked),
            'intra_class_variance': stacked.var(dim=0).mean().item(),
            'representative_indices': self.find_representative_samples(stacked),
            'outlier_indices': self.find_outliers(stacked),
            'dimensionality_reduction': self.perform_pca(stacked)
        }
        
        # Third order statistics
        flat_stacked = stacked.reshape(stacked.shape[0], -1)
        class_stats['third_order'] = {
            'skewness': scipy_stats.skew(flat_stacked.numpy(), axis=1).mean(),
            'kurtosis': scipy_stats.kurtosis(flat_stacked.numpy(), axis=1).mean()
        }
        
        return class_stats
    
    def compute_covariance_matrix(self, stacked_tensors):
        """Compute covariance matrix for feature analysis"""
        # Flatten tensors
        flat = stacked_tensors.reshape(stacked_tensors.shape[0], -1)
        
        # Compute covariance (subsample if too large)
        if flat.shape[1] > 1000:
            # Random subsample of features
            indices = torch.randperm(flat.shape[1])[:1000]
            flat = flat[:, indices]
        
        cov = torch.cov(flat.T)
        
        return {
            'shape': list(cov.shape),
            'trace': cov.trace().item(),
            'determinant': torch.linalg.det(cov).item() if cov.shape[0] < 100 else 'too_large',
            'top_eigenvalues': torch.linalg.eigvals(cov).real[:10].tolist() if cov.shape[0] < 500 else []
        }
    
    def find_representative_samples(self, stacked_tensors, n_samples=5):
        """Find most representative samples of the class"""
        mean_tensor = stacked_tensors.mean(dim=0)
        
        # Compute distances to mean
        distances = torch.norm(
            stacked_tensors.reshape(stacked_tensors.shape[0], -1) - 
            mean_tensor.reshape(-1), 
            dim=1
        )
        
        # Get indices of closest samples
        _, indices = torch.topk(distances, k=min(n_samples, len(distances)), largest=False)
        
        return indices.tolist()
    
    def find_outliers(self, stacked_tensors, n_outliers=5):
        """Find outlier samples using statistical methods"""
        flat = stacked_tensors.reshape(stacked_tensors.shape[0], -1)
        
        # Use Mahalanobis distance or simple z-score
        mean = flat.mean(dim=0)
        std = flat.std(dim=0)
        
        # Z-scores
        z_scores = torch.abs((flat - mean) / (std + 1e-8))
        max_z_scores = z_scores.max(dim=1)[0]
        
        # Get indices of highest z-scores
        _, indices = torch.topk(max_z_scores, k=min(n_outliers, len(max_z_scores)))
        
        return indices.tolist()
    
    def perform_pca(self, stacked_tensors, n_components=10):
        """Perform PCA for dimensionality reduction"""
        flat = stacked_tensors.reshape(stacked_tensors.shape[0], -1).numpy()
        
        # Subsample features if too many
        if flat.shape[1] > 5000:
            indices = np.random.choice(flat.shape[1], 5000, replace=False)
            flat = flat[:, indices]
        
        # Standardize
        scaler = StandardScaler()
        flat_scaled = scaler.fit_transform(flat)
        
        # PCA
        pca = PCA(n_components=min(n_components, flat.shape[0]-1, flat.shape[1]))
        pca_result = pca.fit_transform(flat_scaled)
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_explained': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'n_components_90_variance': np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9) + 1
        }
    
    def compute_cross_class_analysis(self):
        """Analyze relationships between different classes"""
        logger.info("Computing cross-class analysis...")
        
        class_names = list(self.class_data.keys())
        n_classes = len(class_names)
        
        if n_classes < 2:
            return {}
        
        # Compute mean tensors for each class
        mean_tensors = {}
        for class_name in class_names:
            if self.class_data[class_name]['tensors']:
                mean_tensors[class_name] = torch.stack(
                    self.class_data[class_name]['tensors']
                ).mean(dim=0)
        
        # Cross-class correlations
        correlation_matrix = np.zeros((n_classes, n_classes))
        
        for i, class1 in enumerate(class_names):
            for j, class2 in enumerate(class_names):
                if class1 in mean_tensors and class2 in mean_tensors:
                    corr = torch.corrcoef(torch.stack([
                        mean_tensors[class1].flatten(),
                        mean_tensors[class2].flatten()
                    ]))[0, 1].item()
                    correlation_matrix[i, j] = corr
        
        # Inter-class distances
        distance_matrix = np.zeros((n_classes, n_classes))
        
        for i, class1 in enumerate(class_names):
            for j, class2 in enumerate(class_names):
                if class1 in mean_tensors and class2 in mean_tensors:
                    dist = torch.norm(
                        mean_tensors[class1].flatten() - mean_tensors[class2].flatten()
                    ).item()
                    distance_matrix[i, j] = dist
        
        return {
            'class_names': class_names,
            'correlation_matrix': correlation_matrix.tolist(),
            'distance_matrix': distance_matrix.tolist(),
            'most_similar_pairs': self.find_most_similar_classes(correlation_matrix, class_names),
            'most_different_pairs': self.find_most_different_classes(distance_matrix, class_names)
        }
    
    def find_most_similar_classes(self, corr_matrix, class_names, n_pairs=5):
        """Find most similar class pairs based on correlation"""
        pairs = []
        n = len(class_names)
        
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((corr_matrix[i, j], class_names[i], class_names[j]))
        
        pairs.sort(reverse=True)
        return [(pair[1], pair[2], pair[0]) for pair in pairs[:n_pairs]]
    
    def find_most_different_classes(self, dist_matrix, class_names, n_pairs=5):
        """Find most different class pairs based on distance"""
        pairs = []
        n = len(class_names)
        
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((dist_matrix[i, j], class_names[i], class_names[j]))
        
        pairs.sort(reverse=True)
        return [(pair[1], pair[2], pair[0]) for pair in pairs[:n_pairs]]
    
    def generate_neural_network_insights(self):
        """Generate insights for neural network training"""
        logger.info("Generating neural network training insights...")
        
        insights = {
            'recommended_architecture': {},
            'data_characteristics': {},
            'training_recommendations': {},
            'preprocessing_suggestions': {}
        }
        
        # Analyze data characteristics
        total_samples = sum(len(data['tensors']) for data in self.class_data.values())
        n_classes = len(self.class_data)
        
        # Class balance analysis
        class_sizes = {name: len(data['tensors']) for name, data in self.class_data.items()}
        max_class_size = max(class_sizes.values()) if class_sizes else 0
        min_class_size = min(class_sizes.values()) if class_sizes else 0
        imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
        
        insights['data_characteristics'] = {
            'total_samples': total_samples,
            'num_classes': n_classes,
            'class_distribution': class_sizes,
            'imbalance_ratio': imbalance_ratio,
            'balanced': imbalance_ratio < 2.0
        }
        
        # Architecture recommendations
        if n_classes == 2:
            insights['recommended_architecture']['task_type'] = 'binary_classification'
            insights['recommended_architecture']['output_activation'] = 'sigmoid'
            insights['recommended_architecture']['loss_function'] = 'binary_crossentropy'
        else:
            insights['recommended_architecture']['task_type'] = 'multiclass_classification'
            insights['recommended_architecture']['output_activation'] = 'softmax'
            insights['recommended_architecture']['loss_function'] = 'categorical_crossentropy'
        
        # Model complexity recommendation
        if total_samples < 1000:
            insights['recommended_architecture']['model_size'] = 'small'
            insights['recommended_architecture']['suggested_layers'] = [64, 32]
            insights['training_recommendations']['use_augmentation'] = True
        elif total_samples < 10000:
            insights['recommended_architecture']['model_size'] = 'medium'
            insights['recommended_architecture']['suggested_layers'] = [128, 64, 32]
        else:
            insights['recommended_architecture']['model_size'] = 'large'
            insights['recommended_architecture']['suggested_layers'] = [256, 128, 64]
        
        # Training recommendations
        insights['training_recommendations']['batch_size'] = min(32, total_samples // 10)
        insights['training_recommendations']['learning_rate'] = 0.001 if total_samples > 5000 else 0.0001
        
        if imbalance_ratio > 3.0:
            insights['training_recommendations']['use_class_weights'] = True
            insights['training_recommendations']['use_oversampling'] = True
        
        # Preprocessing suggestions based on statistics
        all_stats = []
        for data in self.class_data.values():
            for metadata in data['metadata']:
                if metadata and 'statistics' in metadata and metadata['statistics']:
                    all_stats.append(metadata['statistics'])
        
        if all_stats:
            mean_stds = [s['std'] for s in all_stats if 'std' in s]
            avg_std = np.mean(mean_stds) if mean_stds else 0
            
            insights['preprocessing_suggestions'] = {
                'normalization_recommended': avg_std > 50,
                'standardization_recommended': avg_std > 10,
                'histogram_equalization_recommended': avg_std < 10
            }
        
        return insights
    
    def save_analysis_results(self, output_dir):
        """Save all analysis results and visualizations"""
        analysis_dir = Path(output_dir) / 'neural_network_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save class statistics
        class_stats_path = analysis_dir / 'class_statistics.json'
        class_stats_data = {}
        
        for class_name, data in self.class_data.items():
            if 'statistics' in data and data['statistics']:
                # Convert tensors to lists for JSON serialization
                stats_copy = data['statistics'].copy()
                if 'mean_tensor' in stats_copy:
                    stats_copy['mean_tensor_shape'] = list(stats_copy['mean_tensor'].shape)
                    del stats_copy['mean_tensor']
                if 'std_tensor' in stats_copy:
                    stats_copy['std_tensor_shape'] = list(stats_copy['std_tensor'].shape)
                    del stats_copy['std_tensor']
                
                class_stats_data[class_name] = stats_copy
        
        with open(class_stats_path, 'w') as f:
            json.dump(class_stats_data, f, indent=2)
        
        # Save global statistics
        global_stats_path = analysis_dir / 'global_statistics.json'
        with open(global_stats_path, 'w') as f:
            json.dump(self.global_statistics, f, indent=2)
        
        # Generate visualizations
        self.generate_visualizations(analysis_dir)
        
        # Save training configuration
        nn_insights = self.generate_neural_network_insights()
        training_config_path = analysis_dir / 'neural_network_training_config.json'
        with open(training_config_path, 'w') as f:
            json.dump(nn_insights, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(analysis_dir)
        
        logger.info(f"Analysis results saved to {analysis_dir}")
    
    def generate_visualizations(self, output_dir):
        """Generate visualizations of the analysis"""
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Class distribution
        if self.global_statistics['class_distribution']:
            plt.figure(figsize=(12, 6))
            classes = list(self.global_statistics['class_distribution'].keys())
            counts = list(self.global_statistics['class_distribution'].values())
            
            plt.bar(classes, counts)
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.title('Class Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(viz_dir / 'class_distribution.png')
            plt.close()
        
        # Correlation heatmap
        if 'correlation_matrix' in self.global_statistics['cross_class_correlations']:
            corr_matrix = np.array(self.global_statistics['cross_class_correlations']['correlation_matrix'])
            class_names = self.global_statistics['cross_class_correlations']['class_names']
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, 
                       xticklabels=class_names, 
                       yticklabels=class_names, 
                       cmap='coolwarm', 
                       center=0, 
                       annot=True, 
                       fmt='.2f')
            plt.title('Cross-Class Correlation Matrix')
            plt.tight_layout()
            plt.savefig(viz_dir / 'correlation_matrix.png')
            plt.close()
    
    def generate_summary_report(self, output_dir):
        """Generate a comprehensive summary report"""
        report_path = output_dir / 'analysis_summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("NEURAL NETWORK DATA ANALYSIS SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Images: {self.global_statistics['total_images']}\n")
            f.write(f"Number of Classes: {len(self.class_data)}\n\n")
            
            # Class Distribution
            f.write("CLASS DISTRIBUTION\n")
            f.write("-"*40 + "\n")
            for class_name, count in self.global_statistics['class_distribution'].items():
                percentage = (count / self.global_statistics['total_images']) * 100
                f.write(f"{class_name}: {count} ({percentage:.1f}%)\n")
            
            # Key Insights
            nn_insights = self.generate_neural_network_insights()
            f.write("\n\nNEURAL NETWORK TRAINING RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            f.write(f"Recommended Task Type: {nn_insights['recommended_architecture']['task_type']}\n")
            f.write(f"Model Size: {nn_insights['recommended_architecture']['model_size']}\n")
            f.write(f"Suggested Architecture: {nn_insights['recommended_architecture']['suggested_layers']}\n")
            f.write(f"Loss Function: {nn_insights['recommended_architecture']['loss_function']}\n")
            f.write(f"Batch Size: {nn_insights['training_recommendations']['batch_size']}\n")
            f.write(f"Learning Rate: {nn_insights['training_recommendations']['learning_rate']}\n")
            
            # Data Characteristics
            f.write("\n\nDATA CHARACTERISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Class Imbalance Ratio: {nn_insights['data_characteristics']['imbalance_ratio']:.2f}\n")
            f.write(f"Dataset Balanced: {'Yes' if nn_insights['data_characteristics']['balanced'] else 'No'}\n")
            
            # Cross-class Analysis
            if 'most_similar_pairs' in self.global_statistics['cross_class_correlations']:
                f.write("\n\nMOST SIMILAR CLASS PAIRS\n")
                f.write("-"*40 + "\n")
                for pair in self.global_statistics['cross_class_correlations']['most_similar_pairs']:
                    f.write(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}\n")
            
            f.write("\n" + "="*60 + "\n")
    
    def process_directory(self, input_dir, output_dir, analysis_output_dir, num_workers=4):
        """Process all images with comprehensive analysis"""
        # Find all images organized by class
        image_files_by_class = self.find_all_images(input_dir)
        
        if not image_files_by_class:
            logger.error("No images found in directory tree")
            return
        
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(analysis_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process each class
        for class_name, image_paths in image_files_by_class.items():
            logger.info(f"\nProcessing class: {class_name}")
            
            class_output_dir = Path(output_dir) / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            successful = 0
            failed = 0
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_file = {
                    executor.submit(
                        self.process_single_image_with_analysis,
                        img_path,
                        class_name,
                        input_dir,
                        class_output_dir
                    ): img_path
                    for img_path in image_paths
                }
                
                with tqdm(total=len(image_paths), desc=f"Processing {class_name}") as pbar:
                    for future in as_completed(future_to_file):
                        img_path = future_to_file[future]
                        try:
                            result = future.result()
                            if result:
                                successful += 1
                            else:
                                failed += 1
                        except Exception as e:
                            failed += 1
                            logger.error(f"Error processing {img_path}: {e}")
                        pbar.update(1)
            
            logger.info(f"{class_name}: {successful} successful, {failed} failed")
            
            # Compute class-level statistics
            if self.compute_stats and self.class_data[class_name]['tensors']:
                self.class_data[class_name]['statistics'] = self.compute_class_level_statistics(
                    class_name, 
                    self.class_data[class_name]['tensors']
                )
        
        # Update global statistics
        self.global_statistics['total_images'] = sum(
            len(data['tensors']) for data in self.class_data.values()
        )
        self.global_statistics['class_distribution'] = {
            name: len(data['tensors']) for name, data in self.class_data.items()
        }
        
        # Perform cross-class analysis
        self.global_statistics['cross_class_correlations'] = self.compute_cross_class_analysis()
        
        # Generate neural network insights
        self.global_statistics['neural_network_insights'] = self.generate_neural_network_insights()
        
        # Save all analysis results
        self.save_analysis_results(analysis_output_dir)
        
        logger.info("\nProcessing and analysis complete!")
    
    def process_single_image_with_analysis(self, img_path, class_label, input_root, output_dir):
        """Process single image with analysis and store results"""
        try:
            # Tensorize and analyze
            tensor, metadata = self.tensorize_and_analyze_image(img_path, class_label)
            
            if tensor is None:
                return False
            
            # Store in class data for later analysis
            if self.compute_stats:
                self.class_data[class_label]['tensors'].append(tensor)
                self.class_data[class_label]['paths'].append(str(img_path))
                self.class_data[class_label]['metadata'].append(metadata)
            
            # Save tensor
            output_path = output_dir / f"{img_path.stem}.pt"
            
            if self.save_format == 'pt':
                torch.save({
                    'tensor': tensor,
                    'metadata': metadata
                }, output_path)
            else:
                np_array = tensor.numpy()
                np.save(output_path.with_suffix('.npy'), np_array)
                
                # Save metadata
                metadata_path = output_path.with_suffix('.json')
                import json
                with open(metadata_path, 'w') as f:
                    # Make metadata JSON serializable
                    json_metadata = {k: v for k, v in metadata.items() 
                                   if k != 'statistics' or not isinstance(v, torch.Tensor)}
                    json.dump(json_metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return False

def main():
    print("=== Smart Image Tensorizer with Neural Network Analysis ===\n")
    
    # Get input directory
    print("Enter the path to the root directory containing images:")
    input_dir = input().strip()
    
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print("Error: Invalid input directory")
        sys.exit(1)
    
    # Get output directory for tensors
    print("\nEnter the path to the output directory for tensors:")
    tensor_output_dir = input().strip()
    
    # Get output directory for analysis
    print("\nEnter the path to the output directory for analysis results:")
    analysis_output_dir = input().strip()
    
    # Configuration
    print("\nUsing optimized settings for neural network preparation:")
    print("- Resize to 224x224 for consistency")
    print("- Normalize to [0,1] range")
    print("- Comprehensive statistical analysis")
    print("- Cross-class correlation computation")
    print("- Neural network training recommendations")
    
    # Number of workers
    print("\nEnter number of parallel workers (default: 4):")
    workers_input = input().strip()
    num_workers = int(workers_input) if workers_input.isdigit() else 4
    
    # Create smart tensorizer
    tensorizer = SmartImageTensorizer(
        resize=(224, 224),
        normalize=True,
        save_format='pt',
        compute_stats=True
    )
    
    # Process the directory
    print(f"\nProcessing images with {num_workers} workers...")
    print("This will:")
    print("1. Recursively find all images")
    print("2. Organize by folder/batch names")
    print("3. Tensorize all images")
    print("4. Compute comprehensive statistics")
    print("5. Find correlations and patterns")
    print("6. Generate neural network insights")
    print("7. Save analysis results and visualizations\n")
    
    tensorizer.process_directory(
        input_dir, 
        tensor_output_dir, 
        analysis_output_dir, 
        num_workers
    )
    
    print("\nDone! Check the output directories for:")
    print(f"- Tensorized images: {tensor_output_dir}")
    print(f"- Analysis results: {analysis_output_dir}")
    print("\nThe analysis directory contains:")
    print("- class_statistics.json: Detailed statistics per class")
    print("- global_statistics.json: Overall dataset analysis")
    print("- neural_network_training_config.json: Training recommendations")
    print("- visualizations/: Plots and heatmaps")
    print("- analysis_summary_report.txt: Human-readable summary")

if __name__ == "__main__":
    main()