# evaluator.py
# Evaluation utilities for the fiber optic analysis system

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import logging
from typing import Dict, Any
import numpy as np

class Evaluator:
    """Handles model evaluation and metric calculation."""
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def evaluate(self, model, val_loader):
        """
        Evaluate model on validation set.
        
        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        
        total_similarity = 0.0
        correct_predictions = 0
        total_samples = 0
        class_correct = np.zeros(len(self.config.data.class_names))
        class_total = np.zeros(len(self.config.data.class_names))
        
        all_predictions = []
        all_labels = []
        all_similarities = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.config.training.use_amp):
                    outputs = model(
                        images, 
                        ref_image=None, 
                        equation_coeffs=self.config.equation.coefficients
                    )
                
                # Calculate metrics
                similarity_scores = outputs['final_similarity_score']
                total_similarity += similarity_scores.sum().item()
                
                # Classification accuracy
                _, predicted = torch.max(outputs['region_logits'], 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1
                
                # Store for additional analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_similarities.extend(similarity_scores.cpu().numpy())
        
        # Calculate final metrics
        avg_similarity = total_similarity / total_samples
        accuracy = correct_predictions / total_samples
        
        # Per-class accuracies
        class_accuracies = {}
        for i, class_name in enumerate(self.config.data.class_names):
            if class_total[i] > 0:
                class_accuracies[class_name] = class_correct[i] / class_total[i]
            else:
                class_accuracies[class_name] = 0.0
        
        # Pass/fail rate based on similarity threshold
        pass_count = sum(1 for sim in all_similarities if sim >= self.config.similarity.threshold)
        pass_rate = pass_count / len(all_similarities)
        
        metrics = {
            'accuracy': accuracy,
            'avg_similarity': avg_similarity,
            'class_accuracies': class_accuracies,
            'pass_rate': pass_rate,
            'total_samples': total_samples,
            'predictions': all_predictions,
            'labels': all_labels,
            'similarities': all_similarities
        }
        
        return metrics
    
    def evaluate_single_sample(self, model, image_tensor, ref_tensor=None):
        """
        Evaluate a single sample.
        
        Args:
            model: Model to use for evaluation
            image_tensor: Input image tensor
            ref_tensor: Optional reference image tensor
            
        Returns:
            Dictionary containing predictions and scores
        """
        model.eval()
        
        with torch.no_grad():
            # Ensure tensors are on correct device and have batch dimension
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            if ref_tensor is not None:
                if ref_tensor.dim() == 3:
                    ref_tensor = ref_tensor.unsqueeze(0)
                ref_tensor = ref_tensor.to(self.device)
            
            with autocast(enabled=self.config.training.use_amp):
                outputs = model(
                    image_tensor,
                    ref_image=ref_tensor,
                    equation_coeffs=self.config.equation.coefficients
                )
            
            # Process outputs
            region_probs = F.softmax(outputs['region_logits'], dim=1)
            predicted_class = torch.argmax(region_probs, dim=1).item()
            confidence = torch.max(region_probs, dim=1)[0].item()
            
            similarity_score = outputs['final_similarity_score'].item()
            anomaly_score = outputs['anomaly_score'].item()
            
            # Determine pass/fail
            passes_threshold = similarity_score >= self.config.similarity.threshold
            
            result = {
                'predicted_class': predicted_class,
                'predicted_class_name': self.config.data.class_names[predicted_class],
                'confidence': confidence,
                'class_probabilities': region_probs.squeeze().cpu().numpy(),
                'similarity_score': similarity_score,
                'anomaly_score': anomaly_score,
                'passes_threshold': passes_threshold,
                'status': 'PASS' if passes_threshold else 'FAIL',
                'anomaly_map': outputs['anomaly_map'].squeeze().cpu().numpy(),
                'embedding': outputs['embedding'].squeeze().cpu().numpy()
            }
            
            return result
    
    def analyze_failure_cases(self, model, val_loader, threshold=None):
        """
        Analyze samples that fail the similarity threshold.
        
        Args:
            model: Model to analyze
            val_loader: Validation data loader
            threshold: Custom threshold (uses config default if None)
            
        Returns:
            Dictionary containing failure analysis
        """
        if threshold is None:
            threshold = self.config.similarity.threshold
            
        model.eval()
        
        failures = []
        passes = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.config.training.use_amp):
                    outputs = model(images, equation_coeffs=self.config.equation.coefficients)
                
                similarities = outputs['final_similarity_score']
                _, predicted = torch.max(outputs['region_logits'], 1)
                
                for i in range(images.size(0)):
                    sample_info = {
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'similarity': similarities[i].item(),
                        'predicted_class': predicted[i].item(),
                        'true_class': labels[i].item(),
                        'correct_prediction': predicted[i].item() == labels[i].item(),
                        'anomaly_score': outputs['anomaly_score'][i].item(),
                        'classification_confidence': outputs['classification_confidence'][i].item()
                    }
                    
                    if similarities[i].item() < threshold:
                        failures.append(sample_info)
                    else:
                        passes.append(sample_info)
        
        analysis = {
            'total_failures': len(failures),
            'total_passes': len(passes),
            'failure_rate': len(failures) / (len(failures) + len(passes)),
            'failures': failures,
            'passes': passes
        }
        
        if failures:
            # Analyze failure patterns
            failure_by_class = {}
            for failure in failures:
                class_name = self.config.data.class_names[failure['true_class']]
                if class_name not in failure_by_class:
                    failure_by_class[class_name] = []
                failure_by_class[class_name].append(failure)
            
            analysis['failure_by_class'] = failure_by_class
            
            # Calculate average scores for failures
            avg_failure_similarity = np.mean([f['similarity'] for f in failures])
            avg_failure_anomaly = np.mean([f['anomaly_score'] for f in failures])
            avg_failure_confidence = np.mean([f['classification_confidence'] for f in failures])
            
            analysis['failure_statistics'] = {
                'avg_similarity': avg_failure_similarity,
                'avg_anomaly_score': avg_failure_anomaly,
                'avg_classification_confidence': avg_failure_confidence
            }
        
        return analysis

def create_evaluator(config, device='cpu'):
    """Factory function to create evaluator instance."""
    return Evaluator(config, device)
