#!/usr/bin/env python3

import logging
from utils import get_timestamp


def generate_detailed_report(results, output_path):
    """Generate a detailed text report of the analysis."""
    # Open file for writing
    with open(output_path, 'w') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write("ULTRA-COMPREHENSIVE ANOMALY DETECTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # File information section
        f.write("FILE INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Test File: {results['metadata'].get('filename', 'Unknown')}\n")
        f.write(f"Analysis Date: {get_timestamp()}\n")
        f.write(f"Image Dimensions: {results['test_gray'].shape}\n")
        f.write("\n")
        
        # Overall verdict section
        f.write("OVERALL VERDICT\n")
        f.write("-"*40 + "\n")
        verdict = results['verdict']
        f.write(f"Status: {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'}\n")
        f.write(f"Confidence: {verdict['confidence']:.1%}\n")
        f.write("\n")
        
        # Global analysis section
        f.write("GLOBAL STATISTICAL ANALYSIS\n")
        f.write("-"*40 + "\n")
        global_stats = results['global_analysis']
        f.write(f"Mahalanobis Distance: {global_stats['mahalanobis_distance']:.4f}\n")
        f.write(f"Comparison Scores:\n")
        f.write(f"  - Mean: {global_stats['comparison_stats']['mean']:.4f}\n")
        f.write(f"  - Std: {global_stats['comparison_stats']['std']:.4f}\n")
        f.write(f"  - Min: {global_stats['comparison_stats']['min']:.4f}\n")
        f.write(f"  - Max: {global_stats['comparison_stats']['max']:.4f}\n")
        f.write("\n")
        
        # Top deviant features section
        f.write("TOP DEVIANT FEATURES (Z-Score > 2)\n")
        f.write("-"*40 + "\n")
        # Iterate through top 10 deviant features
        for fname, z_score, test_val, ref_val in global_stats['deviant_features'][:10]:
            # Only show features with significant deviation
            if z_score > 2:
                f.write(f"{fname:30} Z={z_score:6.2f}  Test={test_val:10.4f}  Ref={ref_val:10.4f}\n")
        f.write("\n")
        
        # Structural analysis section
        f.write("STRUCTURAL ANALYSIS\n")
        f.write("-"*40 + "\n")
        structural = results['structural_analysis']
        f.write(f"SSIM Index: {structural['ssim']:.4f}\n")
        f.write(f"Luminance Similarity: {structural['mean_luminance']:.4f}\n")
        f.write(f"Contrast Similarity: {structural['mean_contrast']:.4f}\n")
        f.write(f"Structure Similarity: {structural['mean_structure']:.4f}\n")
        f.write("\n")
        
        # Local anomalies section
        f.write("LOCAL ANOMALY REGIONS\n")
        f.write("-"*40 + "\n")
        regions = results['local_analysis']['anomaly_regions']
        f.write(f"Total Regions Found: {len(regions)}\n")
        # Detail first 5 regions
        for i, region in enumerate(regions[:5], 1):
            f.write(f"\nRegion {i}:\n")
            f.write(f"  - Location: {region['bbox']}\n")
            f.write(f"  - Area: {region['area']} pixels\n")
            f.write(f"  - Confidence: {region['confidence']:.3f}\n")
            f.write(f"  - Centroid: {region['centroid']}\n")
        # Note if more regions exist
        if len(regions) > 5:
            f.write(f"\n... and {len(regions) - 5} more regions\n")
        f.write("\n")
        
        # Specific defects section
        f.write("SPECIFIC DEFECTS DETECTED\n")
        f.write("-"*40 + "\n")
        defects = results['specific_defects']
        f.write(f"Scratches: {len(defects['scratches'])}\n")
        f.write(f"Digs: {len(defects['digs'])}\n")
        f.write(f"Blobs: {len(defects['blobs'])}\n")
        f.write(f"Edge Irregularities: {len(defects['edges'])}\n")
        f.write("\n")
        
        # Criteria summary section
        f.write("ANOMALY CRITERIA SUMMARY\n")
        f.write("-"*40 + "\n")
        criteria = verdict['criteria_triggered']
        f.write(f"Mahalanobis Threshold Exceeded: {'Yes' if criteria['mahalanobis'] else 'No'}\n")
        f.write(f"Comparison Threshold Exceeded: {'Yes' if criteria['comparison'] else 'No'}\n")
        f.write(f"Low Structural Similarity: {'Yes' if criteria['structural'] else 'No'}\n")
        f.write(f"Multiple Local Anomalies: {'Yes' if criteria['local'] else 'No'}\n")
        
        # Footer
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    # Log report location
    logging.info(f"Detailed report saved to: {output_path}") 