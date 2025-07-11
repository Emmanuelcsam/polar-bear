#!/usr/bin/env python3
"""
Integration script to combine CNN with existing detection.py pipeline
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import json
import logging

# Import your existing detection module
from detection import OmniFiberAnalyzer, OmniConfig

# Import the new CNN module
from cnn_fiber_detector import FiberOpticCNN, CNNConfig, enhance_detection_with_cnn


class IntegratedFiberAnalyzer:
    """Combined traditional and CNN-based fiber optic analyzer"""
    
    def __init__(self, omni_config: OmniConfig, cnn_config: CNNConfig, cnn_model_path: Optional[str] = None):
        # Initialize traditional analyzer
        self.omni_analyzer = OmniFiberAnalyzer(omni_config)
        
        # Initialize CNN analyzer
        self.cnn_analyzer = FiberOpticCNN(cnn_config)
        if cnn_model_path and os.path.exists(cnn_model_path):
            self.cnn_analyzer.load_model(cnn_model_path)
        else:
            self.cnn_analyzer.build_model()
            logging.warning("No pre-trained CNN model found. Using untrained model.")
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_comprehensive(self, image_path: str, output_dir: str) -> Dict:
        """Perform both traditional and CNN analysis"""
        self.logger.info(f"Starting comprehensive analysis of {image_path}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 1: Traditional detection
        self.logger.info("Running traditional anomaly detection...")
        traditional_results = self.omni_analyzer.detect_anomalies_comprehensive(image_path)
        
        if traditional_results is None:
            self.logger.error("Traditional detection failed")
            return None
        
        # Step 2: CNN analysis
        self.logger.info("Running CNN analysis...")
        test_image = traditional_results['test_image']
        cnn_results = self.cnn_analyzer.predict_single(test_image)
        
        # Step 3: Combine results
        combined_results = traditional_results.copy()
        combined_results['cnn_analysis'] = cnn_results
        
        # Step 4: Enhanced verdict combining both approaches
        combined_verdict = self._combine_verdicts(
            traditional_results['verdict'],
            cnn_results
        )
        combined_results['combined_verdict'] = combined_verdict
        
        # Step 5: Generate enhanced visualizations
        self._generate_combined_visualization(combined_results, output_dir)
        
        # Step 6: Save combined report
        report_path = os.path.join(output_dir, f"{Path(image_path).stem}_combined_report.json")
        self._save_combined_report(combined_results, report_path)
        
        return combined_results
    
    def _combine_verdicts(self, traditional_verdict: Dict, cnn_results: Dict) -> Dict:
        """Combine traditional and CNN verdicts intelligently"""
        # Weight factors for combining predictions
        traditional_weight = 0.6
        cnn_weight = 0.4
        
        # Calculate combined confidence
        if cnn_results['is_anomaly']:
            cnn_anomaly_confidence = cnn_results['confidence']
        else:
            # If CNN says normal, use 1 - max(anomaly_probabilities)
            anomaly_probs = [prob for cls, prob in cnn_results['all_probabilities'].items() 
                           if cls != 'Normal']
            cnn_anomaly_confidence = 1 - max(anomaly_probs) if anomaly_probs else 0
        
        combined_confidence = (
            traditional_verdict['confidence'] * traditional_weight +
            cnn_anomaly_confidence * cnn_weight
        )
        
        # Determine if anomalous based on combined analysis
        is_anomalous = (
            traditional_verdict['is_anomalous'] or 
            cnn_results['is_anomaly'] or
            combined_confidence > 0.7
        )
        
        # Identify specific defect type
        if cnn_results['is_anomaly']:
            defect_type = cnn_results['predicted_class']
        elif traditional_verdict['is_anomalous']:
            # Infer from traditional detection
            specific_defects = traditional_verdict.get('specific_defects', {})
            if specific_defects.get('scratches', []):
                defect_type = 'Scratch'
            elif specific_defects.get('digs', []):
                defect_type = 'Dig'
            elif specific_defects.get('blobs', []):
                defect_type = 'Contamination'
            else:
                defect_type = 'Unknown Anomaly'
        else:
            defect_type = 'Normal'
        
        return {
            'is_anomalous': is_anomalous,
            'confidence': float(combined_confidence),
            'defect_type': defect_type,
            'traditional_weight': traditional_weight,
            'cnn_weight': cnn_weight,
            'agreement': traditional_verdict['is_anomalous'] == cnn_results['is_anomaly']
        }
    
    def _generate_combined_visualization(self, results: Dict, output_dir: str):
        """Generate visualization combining both analyses"""
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(20, 12))
        
        # Original image
        ax1 = plt.subplot(3, 3, 1)
        test_img = results['test_image']
        if len(test_img.shape) == 3:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        else:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
        ax1.imshow(test_img_rgb)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # Traditional anomaly map
        ax2 = plt.subplot(3, 3, 2)
        anomaly_map = results['local_analysis']['anomaly_map']
        if anomaly_map.shape != test_img_rgb.shape[:2]:
            anomaly_map = cv2.resize(anomaly_map, (test_img_rgb.shape[1], test_img_rgb.shape[0]))
        im2 = ax2.imshow(anomaly_map, cmap='hot')
        ax2.set_title('Traditional Anomaly Heatmap', fontsize=14)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Traditional detection overlay
        ax3 = plt.subplot(3, 3, 3)
        overlay = test_img_rgb.copy()
        for region in results['local_analysis']['anomaly_regions']:
            x, y, w, h = region['bbox']
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 3)
        ax3.imshow(overlay)
        ax3.set_title(f"Traditional: {'ANOMALY' if results['verdict']['is_anomalous'] else 'NORMAL'}", 
                     fontsize=14, color='red' if results['verdict']['is_anomalous'] else 'green')
        ax3.axis('off')
        
        # CNN conv1 visualization
        ax4 = plt.subplot(3, 3, 4)
        conv1_outputs = self.cnn_analyzer.visualize_layer_activations(test_img, 'conv1')
        ax4.imshow(conv1_outputs[0, :, :, 0], cmap='gray')
        ax4.set_title('CNN Conv1 Filter 1', fontsize=12)
        ax4.axis('off')
        
        # CNN conv2 visualization
        ax5 = plt.subplot(3, 3, 5)
        conv2_outputs = self.cnn_analyzer.visualize_layer_activations(test_img, 'conv2')
        ax5.imshow(conv2_outputs[0, :, :, 0], cmap='gray')
        ax5.set_title('CNN Conv2 Filter 1', fontsize=12)
        ax5.axis('off')
        
        # CNN prediction bar chart
        ax6 = plt.subplot(3, 3, 6)
        cnn_results = results['cnn_analysis']
        classes = list(cnn_results['all_probabilities'].keys())
        probs = list(cnn_results['all_probabilities'].values())
        bars = ax6.bar(classes, probs)
        # Color the predicted class differently
        pred_idx = classes.index(cnn_results['predicted_class'])
        bars[pred_idx].set_color('red')
        ax6.set_title(f"CNN: {cnn_results['predicted_class']}", fontsize=14,
                     color='red' if cnn_results['is_anomaly'] else 'green')
        ax6.set_ylabel('Probability')
        plt.xticks(rotation=45, ha='right')
        
        # Combined verdict visualization
        ax7 = plt.subplot(3, 3, 7)
        combined = results['combined_verdict']
        ax7.text(0.5, 0.5, f"COMBINED VERDICT\n\n{'ANOMALY' if combined['is_anomalous'] else 'NORMAL'}\n"
                          f"Type: {combined['defect_type']}\n"
                          f"Confidence: {combined['confidence']:.1%}\n"
                          f"Agreement: {'YES' if combined['agreement'] else 'NO'}",
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='red' if combined['is_anomalous'] else 'green', 
                         alpha=0.3))
        ax7.axis('off')
        
        # Comparison metrics
        ax8 = plt.subplot(3, 3, 8)
        metrics = [
            ('Traditional\nConfidence', results['verdict']['confidence']),
            ('CNN\nConfidence', cnn_results['confidence']),
            ('Combined\nConfidence', combined['confidence'])
        ]
        names, values = zip(*metrics)
        bars = ax8.bar(names, values)
        bars[0].set_color('blue')
        bars[1].set_color('orange')
        bars[2].set_color('purple')
        ax8.set_ylim(0, 1)
        ax8.set_ylabel('Confidence Score')
        ax8.set_title('Confidence Comparison', fontsize=14)
        
        # Summary text
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        summary_text = f"""
Analysis Summary:
-----------------
Traditional Detection:
  • Verdict: {'ANOMALY' if results['verdict']['is_anomalous'] else 'NORMAL'}
  • Confidence: {results['verdict']['confidence']:.1%}
  • Regions: {len(results['local_analysis']['anomaly_regions'])}
  
CNN Detection:
  • Class: {cnn_results['predicted_class']}
  • Confidence: {cnn_results['confidence']:.1%}
  
Combined Analysis:
  • Final Verdict: {'ANOMALY' if combined['is_anomalous'] else 'NORMAL'}
  • Defect Type: {combined['defect_type']}
  • Combined Confidence: {combined['confidence']:.1%}
  • Methods Agree: {'YES' if combined['agreement'] else 'NO'}
"""
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Integrated Traditional + CNN Analysis', fontsize=18)
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(output_dir, f"{Path(results['metadata']['filename']).stem}_combined_analysis.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Combined visualization saved to {viz_path}")
    
    def _save_combined_report(self, results: Dict, report_path: str):
        """Save combined analysis report"""
        # Create clean report structure
        report = {
            'metadata': results['metadata'],
            'timestamp': results.get('timestamp', self.omni_analyzer._get_timestamp()),
            
            'traditional_analysis': {
                'verdict': results['verdict'],
                'mahalanobis_distance': results['global_analysis']['mahalanobis_distance'],
                'anomaly_regions': len(results['local_analysis']['anomaly_regions']),
                'ssim_score': results['structural_analysis']['ssim'],
                'specific_defects': {
                    'scratches': len(results['specific_defects']['scratches']),
                    'digs': len(results['specific_defects']['digs']),
                    'blobs': len(results['specific_defects']['blobs']),
                    'edges': len(results['specific_defects']['edges'])
                }
            },
            
            'cnn_analysis': results['cnn_analysis'],
            
            'combined_verdict': results['combined_verdict'],
            
            'recommendations': self._generate_recommendations(results)
        }
        
        # Save to JSON
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Combined report saved to {report_path}")
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on combined analysis"""
        recommendations = []
        combined = results['combined_verdict']
        
        if combined['is_anomalous']:
            if combined['defect_type'] == 'Scratch':
                recommendations.append("Clean fiber end face with appropriate cleaning tools")
                recommendations.append("Re-cleave fiber if scratches persist")
            elif combined['defect_type'] == 'Dig':
                recommendations.append("Inspect for physical damage to fiber end")
                recommendations.append("Consider replacing fiber if digs are severe")
            elif combined['defect_type'] == 'Contamination':
                recommendations.append("Clean with approved fiber cleaning solution")
                recommendations.append("Use lint-free wipes and proper technique")
            
            if not combined['agreement']:
                recommendations.append("Consider manual inspection due to disagreement between methods")
        else:
            recommendations.append("Fiber appears to be in good condition")
            recommendations.append("Proceed with connection/testing")
        
        return recommendations


# Helper class for JSON encoding
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Example usage
def example_integration():
    """Example of how to use the integrated analyzer"""
    # Configure both analyzers
    omni_config = OmniConfig()
    cnn_config = CNNConfig()
    
    # Create integrated analyzer
    analyzer = IntegratedFiberAnalyzer(
        omni_config=omni_config,
        cnn_config=cnn_config,
        cnn_model_path="fiber_cnn_model.h5"  # Path to pre-trained model
    )
    
    # Analyze an image
    test_image = "path/to/your/fiber/image.png"
    output_dir = "integrated_results"
    
    results = analyzer.analyze_comprehensive(test_image, output_dir)
    
    if results:
        print(f"\nAnalysis complete!")
        print(f"Combined verdict: {results['combined_verdict']['defect_type']}")
        print(f"Confidence: {results['combined_verdict']['confidence']:.1%}")
        print(f"Results saved to: {output_dir}")


# Training data generator for CNN
def generate_training_data_from_detections(detection_results_dir: str, output_dir: str):
    """Generate training data for CNN from detection results"""
    import shutil
    
    # Create output directories
    for class_name in ['Normal', 'Scratch', 'Dig', 'Contamination', 'Edge']:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # Process detection results
    for result_file in os.listdir(detection_results_dir):
        if result_file.endswith('_report.json'):
            with open(os.path.join(detection_results_dir, result_file), 'r') as f:
                report = json.load(f)
            
            # Determine class based on detection results
            if not report.get('verdict', {}).get('is_anomalous', False):
                class_name = 'Normal'
            else:
                # Classify based on dominant defect type
                defects = report.get('specific_defects', {})
                if defects.get('scratches', []):
                    class_name = 'Scratch'
                elif defects.get('digs', []):
                    class_name = 'Dig'
                elif defects.get('blobs', []):
                    class_name = 'Contamination'
                elif defects.get('edges', []):
                    class_name = 'Edge'
                else:
                    class_name = 'Contamination'  # Default anomaly
            
            # Copy image to appropriate class directory
            image_name = result_file.replace('_report.json', '.png')
            src_path = os.path.join(detection_results_dir, image_name)
            dst_path = os.path.join(output_dir, class_name, image_name)
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                print(f"Copied {image_name} to {class_name}/")


if __name__ == "__main__":
    # Run example
    example_integration()
