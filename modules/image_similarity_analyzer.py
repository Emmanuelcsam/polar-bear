#!/usr/bin/env python3
"""
Image Similarity Analyzer - SSIM and Structural Comparison Functions
Extracted from detection.py - Standalone modular script
"""

import cv2
import numpy as np
import json
import sys
import os
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageSimilarityAnalyzer:
    """Advanced image similarity and structural comparison toolkit."""
    
    def __init__(self):
        self.logger = logger
    
    def load_image(self, image_path):
        """Load image from file path."""
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Could not read image: {image_path}")
            return None
        return img
    
    def compute_ssim(self, img1, img2):
        """Compute Structural Similarity Index (SSIM) between two images."""
        # Ensure images have same dimensions
        if img1.shape != img2.shape:
            h, w = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # SSIM constants
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        # Create Gaussian window
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        # Compute local means
        mu1 = cv2.filter2D(img1.astype(float), -1, window)
        mu2 = cv2.filter2D(img2.astype(float), -1, window)
        
        # Compute local statistics
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = cv2.filter2D(img1.astype(float)**2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2.astype(float)**2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1.astype(float) * img2.astype(float), -1, window) - mu1_mu2
        
        # SSIM components
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast = (2 * np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2) / (sigma1_sq + sigma2_sq + C2)
        structure = (sigma12 + C2/2) / (np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2/2)
        
        # Combine components
        ssim_map = luminance * contrast * structure
        ssim_index = np.mean(ssim_map)
        
        return {
            'ssim': float(ssim_index),
            'ssim_map': ssim_map,
            'luminance_map': luminance,
            'contrast_map': contrast,
            'structure_map': structure,
            'mean_luminance': float(np.mean(luminance)),
            'mean_contrast': float(np.mean(contrast)),
            'mean_structure': float(np.mean(structure)),
        }
    
    def compute_multiscale_ssim(self, img1, img2, scales=[1, 2, 4]):
        """Compute multi-scale SSIM at different resolutions."""
        ms_ssim_values = []
        
        for scale in scales:
            # Downsample images
            if scale > 1:
                img1_scaled = cv2.resize(img1, (img1.shape[1]//scale, img1.shape[0]//scale))
                img2_scaled = cv2.resize(img2, (img2.shape[1]//scale, img2.shape[0]//scale))
            else:
                img1_scaled = img1
                img2_scaled = img2
            
            # Compute SSIM for this scale
            ssim_result = self.compute_ssim(img1_scaled, img2_scaled)
            ms_ssim_values.append(ssim_result['ssim'])
        
        return {
            'scales': scales,
            'ms_ssim_values': ms_ssim_values,
            'mean_ms_ssim': float(np.mean(ms_ssim_values)),
            'weighted_ms_ssim': float(np.average(ms_ssim_values, weights=[0.5, 0.3, 0.2][:len(scales)]))
        }
    
    def compute_histogram_comparison(self, img1, img2):
        """Compare images using histogram-based metrics."""
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
        
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Compute histograms
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)
        
        # Compute histogram comparison metrics
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        
        # Earth Mover's Distance (EMD)
        try:
            # Prepare for EMD calculation
            sig1 = np.column_stack((np.arange(256).astype(np.float32), hist1.flatten().astype(np.float32)))
            sig2 = np.column_stack((np.arange(256).astype(np.float32), hist2.flatten().astype(np.float32)))
            
            # Filter out zero weights
            sig1 = sig1[sig1[:, 1] > 0]
            sig2 = sig2[sig2[:, 1] > 0]
            
            if len(sig1) > 0 and len(sig2) > 0:
                emd = cv2.EMD(sig1, sig2, cv2.DIST_L2)[0]
            else:
                emd = float('inf')
        except:
            emd = float('inf')
        
        return {
            'histogram_correlation': float(correlation),
            'chi_square_distance': float(chi_square),
            'histogram_intersection': float(intersection),
            'bhattacharyya_distance': float(bhattacharyya),
            'earth_movers_distance': float(emd),
        }
    
    def compute_feature_based_similarity(self, img1, img2):
        """Compare images using feature-based methods."""
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
        
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Detect keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return {
                'keypoints_img1': len(kp1) if kp1 else 0,
                'keypoints_img2': len(kp2) if kp2 else 0,
                'matches': 0,
                'match_ratio': 0.0,
                'feature_similarity': 0.0,
            }
        
        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Calculate feature similarity metrics
        good_matches = [m for m in matches if m.distance < 50]  # Threshold for good matches
        
        match_ratio = len(good_matches) / max(len(des1), len(des2)) if max(len(des1), len(des2)) > 0 else 0
        feature_similarity = len(good_matches) / len(matches) if len(matches) > 0 else 0
        
        return {
            'keypoints_img1': len(kp1),
            'keypoints_img2': len(kp2),
            'total_matches': len(matches),
            'good_matches': len(good_matches),
            'match_ratio': float(match_ratio),
            'feature_similarity': float(feature_similarity),
            'avg_match_distance': float(np.mean([m.distance for m in matches])) if matches else float('inf'),
        }
    
    def compute_pixel_based_metrics(self, img1, img2):
        """Compute pixel-level similarity metrics."""
        # Ensure same dimensions
        if img1.shape != img2.shape:
            h, w = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to float for calculations
        img1_f = img1.astype(np.float64)
        img2_f = img2.astype(np.float64)
        
        # Mean Squared Error
        mse = np.mean((img1_f - img2_f) ** 2)
        
        # Peak Signal-to-Noise Ratio
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Mean Absolute Error
        mae = np.mean(np.abs(img1_f - img2_f))
        
        # Normalized Cross-Correlation
        ncc = np.corrcoef(img1_f.flatten(), img2_f.flatten())[0, 1]
        if np.isnan(ncc):
            ncc = 0.0
        
        # Structural Content
        sc = np.sum(img1_f ** 2) / (np.sum(img2_f ** 2) + 1e-10)
        
        # Average Difference
        ad = np.mean(img1_f - img2_f)
        
        # Normalized Absolute Error
        nae = np.sum(np.abs(img1_f - img2_f)) / (np.sum(np.abs(img1_f)) + 1e-10)
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'mae': float(mae),
            'ncc': float(ncc),
            'structural_content': float(sc),
            'average_difference': float(ad),
            'normalized_absolute_error': float(nae),
        }
    
    def compute_gradient_similarity(self, img1, img2):
        """Compare images based on gradient information."""
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
        
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Ensure same dimensions
        if gray1.shape != gray2.shape:
            h, w = max(gray1.shape[0], gray2.shape[0]), max(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (w, h))
            gray2 = cv2.resize(gray2, (w, h))
        
        # Compute gradients
        grad1_x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        
        grad2_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Gradient magnitude correlation
        grad_correlation = np.corrcoef(grad1_mag.flatten(), grad2_mag.flatten())[0, 1]
        if np.isnan(grad_correlation):
            grad_correlation = 0.0
        
        # Gradient direction similarity
        grad1_dir = np.arctan2(grad1_y, grad1_x)
        grad2_dir = np.arctan2(grad2_y, grad2_x)
        
        # Circular correlation for angles
        dir_diff = np.abs(grad1_dir - grad2_dir)
        dir_diff = np.minimum(dir_diff, 2*np.pi - dir_diff)  # Handle wraparound
        dir_similarity = 1 - np.mean(dir_diff) / np.pi
        
        # Edge density similarity
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        edge_density1 = np.sum(edges1) / edges1.size
        edge_density2 = np.sum(edges2) / edges2.size
        edge_density_ratio = min(edge_density1, edge_density2) / (max(edge_density1, edge_density2) + 1e-10)
        
        return {
            'gradient_magnitude_correlation': float(grad_correlation),
            'gradient_direction_similarity': float(dir_similarity),
            'edge_density_ratio': float(edge_density_ratio),
            'gradient_mse': float(np.mean((grad1_mag - grad2_mag)**2)),
        }
    
    def analyze_image_similarity(self, image1_path, image2_path, output_path=None):
        """Comprehensive similarity analysis between two images."""
        # Load images
        img1 = self.load_image(image1_path)
        img2 = self.load_image(image2_path)
        
        if img1 is None or img2 is None:
            return None
        
        self.logger.info(f"Analyzing similarity between {image1_path} and {image2_path}")
        
        # Perform all similarity analyses
        ssim_result = self.compute_ssim(img1, img2)
        ms_ssim_result = self.compute_multiscale_ssim(img1, img2)
        hist_result = self.compute_histogram_comparison(img1, img2)
        feature_result = self.compute_feature_based_similarity(img1, img2)
        pixel_result = self.compute_pixel_based_metrics(img1, img2)
        gradient_result = self.compute_gradient_similarity(img1, img2)
        
        # Combine all results
        result = {
            'image1_path': image1_path,
            'image2_path': image2_path,
            'image1_shape': img1.shape,
            'image2_shape': img2.shape,
            'structural_similarity': ssim_result,
            'multiscale_ssim': ms_ssim_result,
            'histogram_comparison': hist_result,
            'feature_similarity': feature_result,
            'pixel_metrics': pixel_result,
            'gradient_similarity': gradient_result,
            'analyzed_at': str(np.datetime64('now')),
        }
        
        # Calculate overall similarity score (weighted combination)
        overall_score = (
            ssim_result['ssim'] * 0.3 +
            ms_ssim_result['mean_ms_ssim'] * 0.2 +
            hist_result['histogram_correlation'] * 0.15 +
            feature_result['feature_similarity'] * 0.15 +
            pixel_result['ncc'] * 0.1 +
            gradient_result['gradient_magnitude_correlation'] * 0.1
        )
        
        result['overall_similarity_score'] = float(max(0, min(1, overall_score)))
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=self._json_serialize)
            self.logger.info(f"Similarity analysis saved to: {output_path}")
        
        return result
    
    def _json_serialize(self, obj):
        """JSON serialization helper for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    def create_similarity_visualization(self, image1_path, image2_path, similarity_data, output_path=None):
        """Create visualization comparing two images and their similarity metrics."""
        import matplotlib.pyplot as plt
        
        # Load images
        img1 = self.load_image(image1_path)
        img2 = self.load_image(image2_path)
        
        if img1 is None or img2 is None:
            return None
        
        # Convert BGR to RGB for matplotlib
        if len(img1.shape) == 3:
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        else:
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        
        if len(img2.shape) == 3:
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        else:
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original images
        axes[0, 0].imshow(img1_rgb)
        axes[0, 0].set_title(f'Image 1\n{Path(image1_path).name}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2_rgb)
        axes[0, 1].set_title(f'Image 2\n{Path(image2_path).name}')
        axes[0, 1].axis('off')
        
        # SSIM map
        ssim_map = similarity_data['structural_similarity']['ssim_map']
        if img1.shape != img2.shape:
            h, w = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
            ssim_map = cv2.resize(ssim_map.astype(np.float32), (w, h))
        
        im = axes[0, 2].imshow(ssim_map, cmap='RdYlBu', vmin=0, vmax=1)
        axes[0, 2].set_title(f'SSIM Map\n(Index: {similarity_data["structural_similarity"]["ssim"]:.3f})')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        # Similarity metrics bar chart
        metrics = {
            'SSIM': similarity_data['structural_similarity']['ssim'],
            'MS-SSIM': similarity_data['multiscale_ssim']['mean_ms_ssim'],
            'Hist Corr': similarity_data['histogram_comparison']['histogram_correlation'],
            'Feature Sim': similarity_data['feature_similarity']['feature_similarity'],
            'NCC': similarity_data['pixel_metrics']['ncc'],
            'Overall': similarity_data['overall_similarity_score']
        }
        
        bars = axes[1, 0].bar(metrics.keys(), metrics.values())
        axes[1, 0].set_title('Similarity Metrics')
        axes[1, 0].set_ylabel('Similarity Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Color code bars
        for bar, value in zip(bars, metrics.values()):
            if value > 0.8:
                bar.set_color('green')
            elif value > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Text summary
        axes[1, 1].axis('off')
        summary_text = f"""SIMILARITY ANALYSIS SUMMARY

Overall Score: {similarity_data['overall_similarity_score']:.3f}

Structural Analysis:
• SSIM: {similarity_data['structural_similarity']['ssim']:.3f}
• Mean Luminance: {similarity_data['structural_similarity']['mean_luminance']:.3f}
• Mean Contrast: {similarity_data['structural_similarity']['mean_contrast']:.3f}

Pixel Metrics:
• MSE: {similarity_data['pixel_metrics']['mse']:.2f}
• PSNR: {similarity_data['pixel_metrics']['psnr']:.2f} dB
• MAE: {similarity_data['pixel_metrics']['mae']:.2f}

Feature Analysis:
• Keypoints 1: {similarity_data['feature_similarity']['keypoints_img1']}
• Keypoints 2: {similarity_data['feature_similarity']['keypoints_img2']}
• Good Matches: {similarity_data['feature_similarity']['good_matches']}"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Difference image
        if img1.shape == img2.shape:
            diff_img = cv2.absdiff(img1, img2)
            if len(diff_img.shape) == 3:
                diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
            axes[1, 2].imshow(diff_img, cmap='hot')
            axes[1, 2].set_title('Absolute Difference')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, 'Images have\ndifferent dimensions', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')
        
        plt.suptitle(f'Image Similarity Analysis\nOverall Score: {similarity_data["overall_similarity_score"]:.3f}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Similarity visualization saved to: {output_path}")
        
        plt.show()


def main():
    """Command line interface for image similarity analysis."""
    parser = argparse.ArgumentParser(description='Advanced image similarity analyzer')
    parser.add_argument('image1', help='Path to first image')
    parser.add_argument('image2', help='Path to second image')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--visualize', help='Create visualization plot (provide output path)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.image1):
        print(f"Error: Image file not found: {args.image1}")
        sys.exit(1)
    
    if not os.path.exists(args.image2):
        print(f"Error: Image file not found: {args.image2}")
        sys.exit(1)
    
    analyzer = ImageSimilarityAnalyzer()
    
    # Generate output path if not provided
    output_path = args.output
    if not output_path:
        path1 = Path(args.image1)
        path2 = Path(args.image2)
        output_path = path1.parent / f"{path1.stem}_vs_{path2.stem}_similarity.json"
    
    result = analyzer.analyze_image_similarity(args.image1, args.image2, output_path)
    
    if result:
        print(f"Similarity analysis complete:")
        print(f"Overall similarity score: {result['overall_similarity_score']:.3f}")
        print(f"SSIM: {result['structural_similarity']['ssim']:.3f}")
        print(f"Results saved to: {output_path}")
        
        if args.visualize:
            analyzer.create_similarity_visualization(args.image1, args.image2, result, args.visualize)
    else:
        print("Similarity analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
