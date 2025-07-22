from typing import Dict, Any
from datetime import datetime

def generate_detailed_report(results: Dict[str, Any], output_path: str):
    """Generate a detailed text report of the analysis."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\nANOMALY DETECTION REPORT\n" + "="*80 + "\n\n")
        f.write(f"File: {results['metadata'].get('filename', 'Unknown')}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        verdict = results['verdict']
        f.write(f"OVERALL VERDICT: {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'} "
                f"(Confidence: {verdict['confidence']:.1%})\n\n")
        
        f.write("-- Global Analysis --\n")
        f.write(f"Mahalanobis Distance: {results['global_analysis']['mahalanobis_distance']:.4f}\n\n")
        
        f.write("-- Structural Analysis --\n")
        f.write(f"SSIM Index vs Archetype: {results['structural_analysis']['ssim']:.4f}\n\n")
        
        f.write("-- Local Analysis --\n")
        regions = results['local_analysis']['anomaly_regions']
        f.write(f"Found {len(regions)} anomaly region(s).\n")
        for i, r in enumerate(regions[:5]):
            f.write(f"  Region {i+1}: BBox={r['bbox']}, Confidence={r['confidence']:.3f}\n")
        if len(regions) > 5: f.write("  ... and more.\n")
        f.write("\n")
        
        f.write("-- Specific Defects --\n")
        defects = results['specific_defects']
        f.write(f"Scratches: {len(defects['scratches'])}\n")
        f.write(f"Digs: {len(defects['digs'])}\n")
        f.write(f"Blobs: {len(defects['blobs'])}\n\n")
        
        f.write("="*80 + "\nEND OF REPORT\n" + "="*80 + "\n")
        
    print(f"✓ Detailed report saved to: {output_path}")

if __name__ == '__main__':
    # Create dummy results for testing
    results = {
        'metadata': {'filename': 'test.png'},
        'verdict': {'is_anomalous': True, 'confidence': 0.85},
        'global_analysis': {'mahalanobis_distance': 3.5},
        'structural_analysis': {'ssim': 0.65},
        'local_analysis': {'anomaly_regions': [{'bbox': (10,10,20,20), 'confidence': 0.9}]},
        'specific_defects': {'scratches': [], 'digs': [{'center':(15,15),'area':10}], 'blobs': []}
    }
    report_path = "dummy_report.txt"
    
    print(f"Generating a dummy report to '{report_path}'...")
    generate_detailed_report(results, report_path)
    print("Done.")