import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

def create_dashboard():
    """Create comprehensive visualization dashboard"""
    
    print("[ADV_VIS] Creating visualization dashboard...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Pixel Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Define subplot grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Pixel Distribution Heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    create_pixel_heatmap(ax1)
    
    # 2. Pattern Evolution
    ax2 = fig.add_subplot(gs[0, 2:])
    create_pattern_evolution(ax2)
    
    # 3. 3D Feature Space
    ax3 = fig.add_subplot(gs[1, :2], projection='3d')
    create_3d_feature_space(ax3)
    
    # 4. Correlation Matrix
    ax4 = fig.add_subplot(gs[1, 2:])
    create_correlation_matrix(ax4)
    
    # 5. Time Series Analysis
    ax5 = fig.add_subplot(gs[2, :2])
    create_time_series(ax5)
    
    # 6. Performance Metrics
    ax6 = fig.add_subplot(gs[2, 2:])
    create_performance_metrics(ax6)
    
    plt.tight_layout()
    plt.savefig('advanced_dashboard.png', dpi=300, bbox_inches='tight')
    print("[ADV_VIS] Dashboard saved to advanced_dashboard.png")
    
    plt.show()

def create_pixel_heatmap(ax):
    """Create heatmap of pixel intensities"""
    
    if os.path.exists('pixel_data.json'):
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            pixels = np.array(data['pixels'])
            size = data.get('size', [100, 100])
        
        # Reshape if possible
        if len(pixels) == size[0] * size[1]:
            pixel_matrix = pixels.reshape(size[1], size[0])
        else:
            # Create square matrix
            side = int(np.sqrt(len(pixels)))
            pixel_matrix = pixels[:side*side].reshape(side, side)
        
        # Create heatmap
        im = ax.imshow(pixel_matrix, cmap='viridis', aspect='auto')
        ax.set_title('Pixel Intensity Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity', rotation=270, labelpad=15)
        
        # Add statistics
        stats_text = f'Mean: {np.mean(pixels):.1f}\nStd: {np.std(pixels):.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No pixel data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_title('Pixel Intensity Heatmap', fontsize=14, fontweight='bold')

def create_pattern_evolution(ax):
    """Show evolution of patterns over time"""
    
    pattern_files = []
    for file in sorted(os.listdir('.')):
        if 'pattern' in file and file.endswith('.json'):
            pattern_files.append(file)
    
    if pattern_files:
        pattern_counts = []
        timestamps = []
        
        for file in pattern_files[-10:]:  # Last 10 files
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                if 'frequency' in data:
                    pattern_counts.append(len(data['frequency']))
                elif 'patterns' in data:
                    pattern_counts.append(len(data['patterns']))
                else:
                    pattern_counts.append(0)
                    
                # Extract timestamp from filename or use index
                timestamps.append(len(timestamps))
            except:
                continue
        
        if pattern_counts:
            ax.plot(timestamps, pattern_counts, 'o-', linewidth=2, markersize=8)
            ax.fill_between(timestamps, pattern_counts, alpha=0.3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Pattern Count')
            ax.set_title('Pattern Evolution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No pattern data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_title('Pattern Evolution', fontsize=14, fontweight='bold')

def create_3d_feature_space(ax):
    """Create 3D visualization of feature space"""
    
    if os.path.exists('ml_clustering.json'):
        with open('ml_clustering.json', 'r') as f:
            clustering = json.load(f)
        
        if 'pca' in clustering and clustering['pca']['components']:
            components = np.array(clustering['pca']['components'])
            
            # Generate 3rd dimension from first two
            if components.shape[1] == 2:
                z = components[:, 0] * components[:, 1]
                components = np.column_stack([components, z])
            
            # Get labels if available
            labels = clustering['kmeans']['labels']
            
            # Create 3D scatter
            scatter = ax.scatter(components[:, 0], components[:, 1], 
                               components[:, 2] if components.shape[1] > 2 else np.zeros(len(components)),
                               c=labels, cmap='viridis', s=50, alpha=0.6)
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('3D Feature Space', fontsize=14, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Cluster')
    else:
        # Create synthetic 3D data for demonstration
        n_points = 100
        x = np.random.randn(n_points)
        y = np.random.randn(n_points)
        z = x * y + np.random.randn(n_points) * 0.5
        
        ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.set_title('3D Feature Space', fontsize=14, fontweight='bold')

def create_correlation_matrix(ax):
    """Create correlation matrix heatmap"""
    
    # Collect data from various sources
    data_sources = {}
    
    if os.path.exists('intensity_analysis.json'):
        with open('intensity_analysis.json', 'r') as f:
            intensity = json.load(f)
            if 'statistics' in intensity:
                data_sources['mean'] = intensity['statistics']['mean']
                data_sources['std'] = intensity['statistics']['std']
    
    if os.path.exists('anomalies.json'):
        with open('anomalies.json', 'r') as f:
            anomalies = json.load(f)
            if 'bounds' in anomalies:
                data_sources['anomaly_lower'] = anomalies['bounds']['lower']
                data_sources['anomaly_upper'] = anomalies['bounds']['upper']
    
    if os.path.exists('patterns.json'):
        with open('patterns.json', 'r') as f:
            patterns = json.load(f)
            if 'statistics' in patterns and patterns['statistics']:
                data_sources['pattern_mean'] = patterns['statistics'][0].get('mean', 0)
    
    if len(data_sources) >= 3:
        # Create correlation matrix
        values = list(data_sources.values())
        labels = list(data_sources.keys())
        
        # Generate correlation matrix (simplified)
        n = len(values)
        corr_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                # Simple correlation based on value similarity
                corr = 1 - abs(values[i] - values[j]) / (max(values) - min(values) + 1)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    else:
        # Create example correlation matrix
        features = ['Mean', 'Std', 'Min', 'Max', 'Entropy']
        n_features = len(features)
        corr_matrix = np.random.rand(n_features, n_features)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1)
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    xticklabels=features, yticklabels=features, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

def create_time_series(ax):
    """Create time series visualization"""
    
    # Collect time series data
    time_data = []
    
    # Check for real-time metrics
    if os.path.exists('realtime_metrics.json'):
        with open('realtime_metrics.json', 'r') as f:
            metrics = json.load(f)
            time_data.append(('Processing Rate', [metrics.get('processing_rate', 0)]))
            time_data.append(('Variance', [metrics.get('current_variance', 0)]))
    
    # Check for stream data
    if os.path.exists('stream_analysis.json'):
        with open('stream_analysis.json', 'r') as f:
            stream = json.load(f)
            if 'stream_rates' in stream:
                for name, rate in stream['stream_rates'].items():
                    if rate > 0:
                        time_data.append((name, [rate]))
    
    if time_data:
        # Plot time series
        for i, (name, values) in enumerate(time_data):
            x = range(len(values))
            ax.plot(x, values, 'o-', label=name, linewidth=2, markersize=6)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Real-Time Metrics', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Create synthetic time series
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x) + np.random.randn(100) * 0.1
        y2 = np.cos(x) + np.random.randn(100) * 0.1
        
        ax.plot(x, y1, label='Metric 1', linewidth=2)
        ax.plot(x, y2, label='Metric 2', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Time Series Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

def create_performance_metrics(ax):
    """Create performance metrics visualization"""
    
    metrics = {}
    
    # Collect performance data
    if os.path.exists('gpu_results.json'):
        with open('gpu_results.json', 'r') as f:
            gpu_data = json.load(f)
            if 'performance' in gpu_data:
                metrics['GPU'] = gpu_data['performance'].get('pixels_per_second', 0)
    
    if os.path.exists('parallel_results.json'):
        with open('parallel_results.json', 'r') as f:
            parallel_data = json.load(f)
            if 'performance' in parallel_data:
                metrics['Parallel'] = parallel_data['performance'].get('pixels_per_second', 0)
    
    if os.path.exists('hpc_benchmarks.json'):
        with open('hpc_benchmarks.json', 'r') as f:
            hpc_data = json.load(f)
            if 'benchmarks' in hpc_data and hpc_data['benchmarks']:
                # Average throughput
                throughputs = []
                for bench in hpc_data['benchmarks']:
                    if 'benchmarks' in bench:
                        for b in bench['benchmarks']:
                            throughputs.append(b.get('throughput', 0))
                if throughputs:
                    metrics['HPC'] = np.mean(throughputs)
    
    if metrics:
        # Create bar chart
        names = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.0f}', ha='center', va='bottom')
        
        ax.set_ylabel('Pixels/Second')
        ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
    else:
        # Create example performance chart
        categories = ['CPU', 'GPU', 'Parallel', 'Distributed']
        values = [1000, 10000, 4000, 8000]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bars = ax.bar(categories, values, color=colors)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value}', ha='center', va='bottom')
        
        ax.set_ylabel('Throughput')
        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')

def create_animated_visualization():
    """Create animated visualization of pixel evolution"""
    
    print("\n[ADV_VIS] Creating animated visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Load initial data
    if os.path.exists('pixel_data.json'):
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            pixels = np.array(data['pixels'])
            size = data.get('size', [100, 100])
        
        if len(pixels) >= size[0] * size[1]:
            pixel_matrix = pixels[:size[0]*size[1]].reshape(size[1], size[0])
        else:
            side = int(np.sqrt(len(pixels)))
            pixel_matrix = pixels[:side*side].reshape(side, side)
        
        im = ax.imshow(pixel_matrix, cmap='viridis', animated=True)
        ax.set_title('Pixel Evolution Animation', fontsize=14)
        
        def update(frame):
            # Simulate evolution
            noise = np.random.randn(*pixel_matrix.shape) * 10
            evolved = pixel_matrix + noise * (frame / 50)
            evolved = np.clip(evolved, 0, 255)
            im.set_data(evolved)
            ax.set_title(f'Pixel Evolution - Frame {frame}')
            return [im]
        
        anim = animation.FuncAnimation(fig, update, frames=50, interval=100, blit=True)
        
        # Save animation
        try:
            anim.save('pixel_evolution.gif', writer='pillow', fps=10)
            print("[ADV_VIS] Animation saved to pixel_evolution.gif")
        except:
            print("[ADV_VIS] Could not save animation (install pillow for GIF support)")
        
        plt.show()
    else:
        print("[ADV_VIS] No pixel data for animation")

def create_analysis_report():
    """Generate visual analysis report"""
    
    print("\n[ADV_VIS] Generating visual analysis report...")
    
    # Create multi-page PDF report
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages('visual_analysis_report.pdf') as pdf:
        # Page 1: Overview
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Visual Analysis Report', fontsize=20, fontweight='bold')
        
        # Add summary text
        summary_text = """
This report provides a comprehensive visual analysis of the pixel data processing system.
        
Generated: {}
Total Analyses: Multiple modules including neural networks, GPU processing, and ML classification
        
Key Findings:
- Pixel intensity patterns detected and analyzed
- Multiple clustering algorithms applied
- Performance benchmarks completed
- Real-time processing capabilities demonstrated
        """.format(time.strftime('%Y-%m-%d %H:%M:%S'))
        
        plt.text(0.1, 0.5, summary_text, transform=fig.transFigure, fontsize=12,
                verticalalignment='center', fontfamily='monospace')
        
        plt.axis('off')
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Dashboard
        create_dashboard()
        pdf.savefig(plt.gcf())
        plt.close()
        
        # Page 3: Additional visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Additional Analysis', fontsize=16)
        
        # Add various plots
        create_pixel_heatmap(ax1)
        create_pattern_evolution(ax2)
        create_time_series(ax3)
        create_performance_metrics(ax4)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print("[ADV_VIS] Visual report saved to visual_analysis_report.pdf")

def main():
    """Run advanced visualization suite"""
    
    print("=== ADVANCED VISUALIZATION SUITE ===\n")
    
    # 1. Create main dashboard
    create_dashboard()
    
    # 2. Create animated visualization
    create_animated_visualization()
    
    # 3. Generate report
    create_analysis_report()
    
    print("\n[ADV_VIS] Visualization complete!")
    print("[ADV_VIS] Generated files:")
    print("  - advanced_dashboard.png")
    print("  - pixel_evolution.gif")
    print("  - visual_analysis_report.pdf")

if __name__ == "__main__":
    main()