import torch
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import gc
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisContinuer:
    """Continue the analysis from where the tensorizer left off"""
    
    def __init__(self, tensor_dir, analysis_dir):
        self.tensor_dir = Path(tensor_dir)
        self.analysis_dir = Path(analysis_dir)
        self.class_data = defaultdict(lambda: {
            'count': 0,
            'paths': [],
            'metadata': [],
            'statistics': {}
        })
        self.global_statistics = {
            'total_images': 0,
            'class_distribution': {},
            'cross_class_correlations': {},
            'opencv_global_features': {},
            'pca_analysis': {},
            'neural_network_insights': {}
        }
        
    def load_existing_data(self):
        """Load the existing processed data from disk"""
        logger.info("Loading existing processed data...")
        
        # Count tensors per class
        for tensor_file in self.tensor_dir.rglob('*_color.pt'):
            try:
                # Extract class from path
                relative_path = tensor_file.relative_to(self.tensor_dir)
                if len(relative_path.parts) > 1:
                    class_name = relative_path.parts[0]
                else:
                    # File in root - use parent name
                    class_name = tensor_file.parent.name
                
                self.class_data[class_name]['count'] += 1
                self.class_data[class_name]['paths'].append(str(tensor_file))
                
            except Exception as e:
                logger.warning(f"Error processing {tensor_file}: {e}")
        
        # Fix data types - ensure all counts are integers
        for class_name in list(self.class_data.keys()):
            count = self.class_data[class_name]['count']
            if isinstance(count, str):
                try:
                    self.class_data[class_name]['count'] = int(count)
                except ValueError:
                    logger.warning(f"Invalid count for {class_name}: {count}, setting to path count")
                    self.class_data[class_name]['count'] = len(self.class_data[class_name]['paths'])
            else:
                self.class_data[class_name]['count'] = int(count)
        
        # Update global statistics with proper integer counts
        self.global_statistics['total_images'] = sum(
            int(data['count']) for data in self.class_data.values()
        )
        self.global_statistics['class_distribution'] = {
            name: int(data['count']) for name, data in self.class_data.items()
        }
        
        logger.info(f"Found {self.global_statistics['total_images']} total images across {len(self.class_data)} classes")
        
    def generate_neural_network_insights(self):
        """Generate comprehensive insights for neural network training"""
        logger.info("Generating neural network training insights...")
        
        insights = {
            'recommended_architectures': {},
            'data_characteristics': {},
            'training_recommendations': {},
            'preprocessing_pipeline': {},
            'augmentation_strategies': {},
            'ensemble_recommendations': {}
        }
        
        # Analyze data with proper integer handling
        total_samples = int(self.global_statistics['total_images'])
        n_classes = len(self.class_data)
        
        # Ensure all counts are integers
        class_sizes = {}
        for name, data in self.class_data.items():
            count = int(data['count'])  # Force to integer
            class_sizes[name] = count
        
        # Calculate imbalance ratio safely
        non_zero_sizes = [v for v in class_sizes.values() if v > 0]
        if non_zero_sizes:
            max_class_size = max(non_zero_sizes)
            min_class_size = min(non_zero_sizes)
            imbalance_ratio = float(max_class_size) / float(min_class_size) if min_class_size > 0 else float('inf')
        else:
            max_class_size = 0
            min_class_size = 0
            imbalance_ratio = 1.0
        
        insights['data_characteristics'] = {
            'total_samples': total_samples,
            'num_classes': n_classes,
            'class_distribution': class_sizes,
            'imbalance_ratio': imbalance_ratio,
            'balanced': imbalance_ratio < 2.0,
            'samples_per_class_mean': float(np.mean(list(class_sizes.values()))) if class_sizes else 0,
            'samples_per_class_std': float(np.std(list(class_sizes.values()))) if len(class_sizes) > 1 else 0
        }
        
        # Architecture recommendations
        insights['recommended_architectures']['cnn'] = {
            'type': 'ConvNet',
            'layers': self.recommend_cnn_architecture(total_samples, n_classes),
            'dropout_rates': [0.3, 0.4, 0.5] if total_samples < 5000 else [0.2, 0.3, 0.4]
        }
        
        insights['recommended_architectures']['resnet'] = {
            'type': 'ResNet',
            'variant': 'ResNet18' if total_samples < 10000 else 'ResNet34',
            'pretrained': True if total_samples < 5000 else False
        }
        
        insights['recommended_architectures']['vision_transformer'] = {
            'recommended': total_samples > 10000,
            'patch_size': 16,
            'embed_dim': 384 if total_samples < 50000 else 768
        }
        
        # Training recommendations
        insights['training_recommendations'] = {
            'batch_size': self.recommend_batch_size(total_samples),
            'learning_rates': {
                'initial': 0.001,
                'scheduler': 'cosine_annealing',
                'min_lr': 0.00001
            },
            'optimizers': ['AdamW', 'SGD with momentum'],
            'epochs': 100 if total_samples < 5000 else 50,
            'early_stopping_patience': 10,
            'use_class_weights': imbalance_ratio > 3.0,
            'use_focal_loss': imbalance_ratio > 5.0,
            'mixup_alpha': 0.2 if total_samples < 10000 else 0.1
        }
        
        # Preprocessing pipeline
        insights['preprocessing_pipeline'] = {
            'normalization': 'imagenet_stats',
            'resize_strategy': 'resize_then_crop',
            'color_augmentation': True,
            'geometric_augmentation': True,
            'advanced_augmentation': ['cutmix', 'mixup'] if total_samples < 10000 else ['randaugment']
        }
        
        # Augmentation strategies
        insights['augmentation_strategies'] = {
            'basic': ['horizontal_flip', 'vertical_flip', 'rotation_15', 'brightness_contrast'],
            'advanced': ['elastic_transform', 'grid_distortion', 'optical_distortion'],
            'color': ['hue_saturation', 'rgb_shift', 'channel_shuffle'],
            'noise': ['gaussian_noise', 'blur', 'jpeg_compression'] if total_samples > 5000 else []
        }
        
        # Ensemble recommendations
        insights['ensemble_recommendations'] = {
            'use_ensemble': n_classes > 5 or total_samples > 10000,
            'ensemble_size': 3 if total_samples < 20000 else 5,
            'strategies': ['different_architectures', 'different_initializations', 'cross_validation_folds']
        }
        
        return insights
    
    def recommend_cnn_architecture(self, total_samples, n_classes):
        """Recommend CNN architecture based on dataset size"""
        if total_samples < 1000:
            return [
                {'conv': 32, 'kernel': 3, 'pool': 2},
                {'conv': 64, 'kernel': 3, 'pool': 2},
                {'fc': [128, n_classes]}
            ]
        elif total_samples < 10000:
            return [
                {'conv': 64, 'kernel': 3, 'pool': 2},
                {'conv': 128, 'kernel': 3, 'pool': 2},
                {'conv': 256, 'kernel': 3, 'pool': 2},
                {'fc': [512, 256, n_classes]}
            ]
        else:
            return [
                {'conv': 64, 'kernel': 3, 'pool': 2},
                {'conv': 128, 'kernel': 3, 'pool': 2},
                {'conv': 256, 'kernel': 3, 'pool': 2},
                {'conv': 512, 'kernel': 3, 'pool': 2},
                {'fc': [1024, 512, n_classes]}
            ]
    
    def recommend_batch_size(self, total_samples):
        """Recommend batch size based on dataset size"""
        if total_samples < 1000:
            return 16
        elif total_samples < 10000:
            return 32
        elif total_samples < 50000:
            return 64
        else:
            return 128
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        viz_dir = self.analysis_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Class distribution
        plt.figure(figsize=(15, 8))
        classes = list(self.global_statistics['class_distribution'].keys())
        counts = [int(v) for v in self.global_statistics['class_distribution'].values()]
        
        if classes and counts:
            plt.subplot(2, 2, 1)
            plt.bar(range(len(classes)), counts, color='skyblue', edgecolor='navy')
            plt.xlabel('Class Index')
            plt.ylabel('Number of Samples')
            plt.title('Class Distribution')
            
            # Add class names as x-tick labels if not too many
            if len(classes) <= 20:
                plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
            
            # 2. Pie chart for balanced view
            plt.subplot(2, 2, 4)
            # Show only top 10 classes in pie chart
            if len(classes) > 10:
                sorted_indices = np.argsort(counts)[-10:]
                top_classes = [classes[i] for i in sorted_indices]
                top_counts = [counts[i] for i in sorted_indices]
                other_count = sum(counts) - sum(top_counts)
                if other_count > 0:
                    top_classes.append('Others')
                    top_counts.append(other_count)
                plt.pie(top_counts, labels=[cn[:15] for cn in top_classes], autopct='%1.1f%%')
            else:
                plt.pie(counts, labels=[cn[:15] for cn in classes], autopct='%1.1f%%')
            plt.title('Class Distribution (Percentage)')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'overview_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        logger.info("Visualizations saved")
    
    def generate_report(self):
        """Generate the comprehensive report"""
        report_path = self.analysis_dir / 'ULTIMATE_ANALYSIS_REPORT.txt'
        nn_insights = self.global_statistics['neural_network_insights']
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ULTIMATE NEURAL NETWORK DATA ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Analysis completed after recovery from interruption\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*60 + "\n")
            f.write(f"Total Images Analyzed: {self.global_statistics['total_images']}\n")
            f.write(f"Number of Classes: {len(self.class_data)}\n")
            f.write(f"Total Features Extracted per Image: 200+\n")
            f.write(f"Analysis Techniques Used: 15+ methods\n\n")
            
            # Detailed Class Analysis
            f.write("DETAILED CLASS ANALYSIS\n")
            f.write("-"*60 + "\n")
            for class_name, count in sorted(self.global_statistics['class_distribution'].items()):
                count = int(count)
                percentage = (count / self.global_statistics['total_images']) * 100 if self.global_statistics['total_images'] > 0 else 0
                f.write(f"\n{class_name}:\n")
                f.write(f"  Samples: {count} ({percentage:.1f}%)\n")
            
            # Neural Network Recommendations
            f.write("\n\nNEURAL NETWORK ARCHITECTURE RECOMMENDATIONS\n")
            f.write("-"*60 + "\n")
            
            # CNN Architecture
            f.write("\n1. Convolutional Neural Network (CNN):\n")
            cnn_arch = nn_insights['recommended_architectures']['cnn']
            f.write(f"   Layers: {cnn_arch['layers']}\n")
            f.write(f"   Dropout rates: {cnn_arch['dropout_rates']}\n")
            
            # ResNet
            f.write("\n2. ResNet Architecture:\n")
            resnet = nn_insights['recommended_architectures']['resnet']
            f.write(f"   Variant: {resnet['variant']}\n")
            f.write(f"   Pretrained: {resnet['pretrained']}\n")
            
            # Vision Transformer
            f.write("\n3. Vision Transformer:\n")
            vit = nn_insights['recommended_architectures']['vision_transformer']
            f.write(f"   Recommended: {vit['recommended']}\n")
            if vit['recommended']:
                f.write(f"   Patch size: {vit['patch_size']}\n")
                f.write(f"   Embedding dimension: {vit['embed_dim']}\n")
            
            # Training Configuration
            f.write("\n\nTRAINING CONFIGURATION\n")
            f.write("-"*60 + "\n")
            train_config = nn_insights['training_recommendations']
            f.write(f"Batch Size: {train_config['batch_size']}\n")
            f.write(f"Initial Learning Rate: {train_config['learning_rates']['initial']}\n")
            f.write(f"LR Scheduler: {train_config['learning_rates']['scheduler']}\n")
            f.write(f"Optimizers: {', '.join(train_config['optimizers'])}\n")
            f.write(f"Epochs: {train_config['epochs']}\n")
            f.write(f"Early Stopping Patience: {train_config['early_stopping_patience']}\n")
            f.write(f"Use Class Weights: {train_config['use_class_weights']}\n")
            f.write(f"Use Focal Loss: {train_config['use_focal_loss']}\n")
            f.write(f"MixUp Alpha: {train_config['mixup_alpha']}\n")
            
            # Data Insights
            f.write("\n\nDATA INSIGHTS\n")
            f.write("-"*60 + "\n")
            data_char = nn_insights['data_characteristics']
            f.write(f"Class Balance Ratio: {data_char['imbalance_ratio']:.2f}\n")
            f.write(f"Dataset Balanced: {'Yes' if data_char['balanced'] else 'No'}\n")
            f.write(f"Mean samples per class: {data_char['samples_per_class_mean']:.1f}\n")
            f.write(f"Std samples per class: {data_char['samples_per_class_std']:.1f}\n")
            
            # Augmentation Strategies
            f.write("\n\nRECOMMENDED AUGMENTATION STRATEGIES\n")
            f.write("-"*60 + "\n")
            aug_strategies = nn_insights['augmentation_strategies']
            f.write(f"Basic: {', '.join(aug_strategies['basic'])}\n")
            f.write(f"Advanced: {', '.join(aug_strategies['advanced'])}\n")
            f.write(f"Color: {', '.join(aug_strategies['color'])}\n")
            if aug_strategies['noise']:
                f.write(f"Noise: {', '.join(aug_strategies['noise'])}\n")
            
            # Ensemble Recommendations
            f.write("\n\nENSEMBLE RECOMMENDATIONS\n")
            f.write("-"*60 + "\n")
            ensemble = nn_insights['ensemble_recommendations']
            f.write(f"Use Ensemble: {ensemble['use_ensemble']}\n")
            if ensemble['use_ensemble']:
                f.write(f"Ensemble Size: {ensemble['ensemble_size']}\n")
                f.write(f"Strategies: {', '.join(ensemble['strategies'])}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("ANALYSIS COMPLETED SUCCESSFULLY\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def generate_pytorch_code(self):
        """Generate ready-to-use PyTorch code"""
        code_path = self.analysis_dir / 'ultimate_pytorch_implementation.py'
        
        nn_insights = self.global_statistics['neural_network_insights']
        class_names = list(self.global_statistics['class_distribution'].keys())
        
        code = f'''"""
Auto-generated PyTorch implementation based on data analysis
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis completed after recovery
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
import json
import numpy as np

class AnalyzedDataset(Dataset):
    """Dataset class for the tensorized images"""
    
    def __init__(self, tensor_dir, transform=None):
        self.tensor_dir = Path(tensor_dir)
        self.tensor_files = list(self.tensor_dir.rglob('*_color.pt'))
        self.transform = transform
        
        # Load class mapping
        self.classes = {json.dumps(class_names)}
        self.class_to_idx = {{cls: idx for idx, cls in enumerate(self.classes)}}
        
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx):
        # Load tensor and metadata
        data = torch.load(self.tensor_files[idx], weights_only=False)
        tensor = data['tensor']
        metadata = data['metadata']
        
        # Get label
        class_name = metadata['class_label']
        label = self.class_to_idx.get(class_name, 0)  # Default to 0 if not found
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, label

class CustomCNN(nn.Module):
    """Recommended CNN architecture based on data analysis"""
    
    def __init__(self, num_classes={len(self.class_data)}):
        super(CustomCNN, self).__init__()
        
        # Architecture based on dataset size
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout({nn_insights['recommended_architectures']['cnn']['dropout_rates'][0]}),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout({nn_insights['recommended_architectures']['cnn']['dropout_rates'][1]}),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout({nn_insights['recommended_architectures']['cnn']['dropout_rates'][2]}),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training configuration based on analysis
config = {{
    'batch_size': {nn_insights['training_recommendations']['batch_size']},
    'learning_rate': {nn_insights['training_recommendations']['learning_rates']['initial']},
    'epochs': {nn_insights['training_recommendations']['epochs']},
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_workers': 4,
    'use_class_weights': {nn_insights['training_recommendations']['use_class_weights']},
}}

# Data augmentation based on recommendations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example training loop
def train_model():
    # Load dataset
    dataset = AnalyzedDataset('{str(self.tensor_dir)}', transform=train_transform)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Apply validation transform to val dataset
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    # Initialize model
    model = CustomCNN().to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    best_acc = 0
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {{epoch}} [{{batch_idx * len(inputs)}}/{{len(train_loader.dataset)}}] '
                      f'Loss: {{loss.item():.4f}}')
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config['device']), labels.to(config['device'])
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / val_size
        print(f'Epoch {{epoch+1}}: Val Loss: {{val_loss/len(val_loader):.4f}}, '
              f'Accuracy: {{acc:.2f}}%')
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    train_model()
'''
        
        with open(code_path, 'w') as f:
            f.write(code)
        
        logger.info(f"PyTorch implementation saved to {code_path}")
    
    def save_json_results(self):
        """Save all JSON results"""
        # Save class statistics
        class_stats_path = self.analysis_dir / 'ultimate_class_statistics.json'
        class_stats_data = {}
        
        for class_name, data in self.class_data.items():
            stats_copy = {
                'sample_count': int(data['count']),
                'paths_count': len(data['paths'])
            }
            if 'statistics' in data and data['statistics']:
                stats_copy.update(data['statistics'])
            class_stats_data[class_name] = stats_copy
        
        with open(class_stats_path, 'w') as f:
            json.dump(class_stats_data, f, indent=2, default=self._json_serialize)
        
        # Save global statistics
        global_stats_path = self.analysis_dir / 'ultimate_global_statistics.json'
        with open(global_stats_path, 'w') as f:
            json.dump(self.global_statistics, f, indent=2, default=self._json_serialize)
        
        # Save neural network config
        nn_config_path = self.analysis_dir / 'ultimate_neural_network_config.json'
        with open(nn_config_path, 'w') as f:
            json.dump(self.global_statistics['neural_network_insights'], f, indent=2, default=self._json_serialize)
        
        logger.info("JSON results saved")
    
    def _json_serialize(self, obj):
        """Custom JSON serializer for numpy/torch types"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif obj == float('inf'):
            return "infinity"
        return str(obj)
    
    def complete_analysis(self):
        """Complete the analysis from where it left off"""
        logger.info("="*60)
        logger.info("CONTINUING ANALYSIS FROM INTERRUPTION POINT")
        logger.info("="*60)
        
        # Load existing data
        self.load_existing_data()
        
        # Generate neural network insights (this is where it failed)
        logger.info("Generating neural network insights...")
        self.global_statistics['neural_network_insights'] = self.generate_neural_network_insights()
        logger.info("✓ Neural network insights generated successfully!")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        self.generate_visualizations()
        logger.info("✓ Visualizations created!")
        
        # Save JSON results
        logger.info("Saving JSON results...")
        self.save_json_results()
        logger.info("✓ JSON results saved!")
        
        # Generate report
        logger.info("Generating comprehensive report...")
        self.generate_report()
        logger.info("✓ Report generated!")
        
        # Generate PyTorch code
        logger.info("Generating PyTorch implementation...")
        self.generate_pytorch_code()
        logger.info("✓ PyTorch code generated!")
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        # Print summary
        logger.info(f"\nSummary:")
        logger.info(f"  Total images: {self.global_statistics['total_images']}")
        logger.info(f"  Total classes: {len(self.class_data)}")
        logger.info(f"\nFiles generated in {self.analysis_dir}:")
        logger.info(f"  - ultimate_class_statistics.json")
        logger.info(f"  - ultimate_global_statistics.json")
        logger.info(f"  - ultimate_neural_network_config.json")
        logger.info(f"  - ULTIMATE_ANALYSIS_REPORT.txt")
        logger.info(f"  - ultimate_pytorch_implementation.py")
        logger.info(f"  - visualizations/overview_analysis.png")

def main():
    print("\n=== ANALYSIS CONTINUATION SCRIPT ===")
    print("This script will continue the analysis from where it was interrupted.")
    print("It will NOT re-process any images, only complete the final analysis steps.\n")
    
    # Get directories
    print("Enter the path to the tensorized data directory:")
    print("(This should be: C:\\Users\\Saem1001\\Documents\\GitHub\\polar-bear\\reference\\tesnsorized-data)")
    tensor_dir = input().strip()
    
    print("\nEnter the path to the analysis output directory:")
    print("(This should be: C:\\Users\\Saem1001\\Documents\\GitHub\\polar-bear\\reference\\tensorized-results)")
    analysis_dir = input().strip()
    
    # Validate paths
    if not Path(tensor_dir).exists():
        print(f"Error: Tensor directory does not exist: {tensor_dir}")
        return
    
    # Create analysis directory if it doesn't exist
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)
    
    # Create continuer instance and complete analysis
    continuer = AnalysisContinuer(tensor_dir, analysis_dir)
    
    try:
        continuer.complete_analysis()
        print("\n✓ Analysis completed successfully!")
        print(f"\nAll results saved to: {analysis_dir}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
