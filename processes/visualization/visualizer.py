import json
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_results():
    """Create visualizations of analysis results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Image Analysis Results', fontsize=16)
    
    # 1. Intensity Histogram
    if os.path.exists('intensity_analysis.json'):
        with open('intensity_analysis.json', 'r') as f:
            intensity = json.load(f)
            
        if 'histogram' in intensity:
            bins = intensity['histogram']['bins']
            counts = intensity['histogram']['counts']
            
            ax1 = axes[0, 0]
            ax1.bar(bins, counts, width=1, edgecolor='none')
            ax1.set_title('Intensity Histogram')
            ax1.set_xlabel('Pixel Value')
            ax1.set_ylabel('Count')
            ax1.set_xlim(0, 255)
            
            print("[VISUALIZER] Plotted intensity histogram")
    
    # 2. Pattern Frequencies
    if os.path.exists('patterns.json'):
        with open('patterns.json', 'r') as f:
            patterns = json.load(f)
            
        if 'frequency' in patterns:
            freq = patterns['frequency']
            top_values = sorted(freq.items(), key=lambda x: int(x[1]), reverse=True)[:20]
            
            if top_values:
                values = [int(v[0]) for v in top_values]
                counts = [int(v[1]) for v in top_values]
                
                ax2 = axes[0, 1]
                ax2.bar(range(len(values)), counts)
                ax2.set_xticks(range(len(values)))
                ax2.set_xticklabels(values, rotation=45)
                ax2.set_title('Top 20 Most Frequent Values')
                ax2.set_xlabel('Pixel Value')
                ax2.set_ylabel('Frequency')
                
                print("[VISUALIZER] Plotted frequency patterns")
    
    # 3. Anomaly Distribution
    if os.path.exists('anomalies.json'):
        with open('anomalies.json', 'r') as f:
            anomalies = json.load(f)
            
        if 'z_score_anomalies' in anomalies:
            anom_values = [a['value'] for a in anomalies['z_score_anomalies']]
            
            if anom_values:
                ax3 = axes[1, 0]
                ax3.hist(anom_values, bins=20, alpha=0.7, color='red')
                ax3.set_title('Anomaly Value Distribution')
                ax3.set_xlabel('Pixel Value')
                ax3.set_ylabel('Count')
                
                print("[VISUALIZER] Plotted anomaly distribution")
    
    # 4. Calculation Results
    if os.path.exists('calculations.json'):
        with open('calculations.json', 'r') as f:
            calc = json.load(f)
            
        ax4 = axes[1, 1]
        
        if 'entropy' in calc:
            metrics = {
                'Shannon Entropy': calc['entropy']['shannon'],
                'Normalized Entropy': calc['entropy']['normalized']
            }
            
            if 'texture' in calc:
                metrics['Contrast'] = calc['texture']['contrast'] / 100  # Scale down
                metrics['Homogeneity'] = calc['texture']['homogeneity']
            
            names = list(metrics.keys())
            values = list(metrics.values())
            
            bars = ax4.bar(names, values)
            ax4.set_title('Analysis Metrics')
            ax4.set_ylabel('Value')
            ax4.set_ylim(0, max(values) * 1.2)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            print("[VISUALIZER] Plotted analysis metrics")
    
    plt.tight_layout()
    plt.savefig('analysis_visualization.png', dpi=150, bbox_inches='tight')
    print("[VISUALIZER] Saved visualization to analysis_visualization.png")
    
    plt.show()

if __name__ == "__main__":
    visualize_results()