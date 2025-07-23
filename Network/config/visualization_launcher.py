#!/usr/bin/env python3
"""
Unified Visualization Launcher for Fiber Optics Neural Network System

This script helps users choose and launch the appropriate visualization tool.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def print_menu():
    """Display the visualization tools menu"""
    print("\n" + "="*60)
    print("FIBER OPTICS NEURAL NETWORK - VISUALIZATION LAUNCHER")
    print("="*60)
    print("\nAvailable Visualization Tools:\n")
    print("1. Configuration Editor (config_visualizer.py)")
    print("   - Edit system configuration with GUI")
    print("   - View theoretical performance predictions")
    print("   - Visualize system architecture")
    print()
    print("2. Real-time Monitor (visualization_ui.py)")
    print("   - Monitor neural network processing in real-time")
    print("   - Adjust parameters during runtime")
    print("   - View live statistics and results")
    print()
    print("3. Generate Reports (using visualizer.py)")
    print("   - Create configuration overview plots")
    print("   - Generate HTML configuration report")
    print("   - Export all visualization assets")
    print()
    print("4. View Documentation")
    print("   - Open visualization guide")
    print()
    print("0. Exit")
    print("\n" + "-"*60)

def run_config_editor():
    """Launch the configuration editor"""
    print("\nLaunching Configuration Editor...")
    print("Note: Changes will be saved to config.yaml")
    try:
        subprocess.run([sys.executable, str(Path(__file__).parent / "config_visualizer.py")])
    except Exception as e:
        print(f"Error launching config editor: {e}")
        print("Make sure PyQt5 is installed: pip install PyQt5")

def run_realtime_monitor():
    """Launch the real-time monitor"""
    print("\nLaunching Real-time Monitor...")
    print("Note: Make sure the neural network is running")
    try:
        subprocess.run([sys.executable, str(Path(__file__).parent / "visualization_ui.py")])
    except Exception as e:
        print(f"Error launching monitor: {e}")
        print("Make sure GTK3 is installed: sudo apt-get install python3-gi")

def generate_reports():
    """Generate visualization reports"""
    print("\nGenerating Visualization Reports...")
    
    try:
        # Import the visualizer
        from config.visualizer import FiberOpticsVisualizer
        from datetime import datetime
        
        # Create output directory
        output_dir = Path(__file__).parent.parent / "visualization_reports"
        output_dir.mkdir(exist_ok=True)
        
        # Initialize visualizer
        print("Initializing visualizer...")
        visualizer = FiberOpticsVisualizer()
        
        # Generate reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Generating configuration overview...")
        config_overview_path = output_dir / f"config_overview_{timestamp}.png"
        visualizer.visualize_config_overview(str(config_overview_path))
        print(f"  ✓ Saved to: {config_overview_path}")
        
        print("Generating HTML report...")
        html_report_path = output_dir / f"config_report_{timestamp}.html"
        visualizer.generate_config_report(str(html_report_path))
        print(f"  ✓ Saved to: {html_report_path}")
        
        # Test with dummy data
        print("Generating example visualizations...")
        import numpy as np
        
        # Create dummy results
        dummy_results = {
            'region_probs': np.random.rand(1, 3, 256, 256),
            'anomaly_map': np.random.rand(1, 256, 256) * 0.5,
            'summary': {
                'final_similarity_score': 0.85,
                'meets_threshold': True,
                'primary_region': 'core',
                'anomaly_score': 0.12,
            },
            'equation_info': {
                'components': {
                    'reference_similarity': 0.88,
                    'trend_adherence': 0.91,
                    'anomaly_inverse': 0.88,
                    'segmentation_confidence': 0.95,
                    'reconstruction': 0.82
                },
                'final_score': 0.85
            }
        }
        
        # Equation components visualization
        equation_viz_path = output_dir / f"equation_components_{timestamp}.png"
        visualizer.visualize_equation_components(dummy_results, str(equation_viz_path))
        print(f"  ✓ Saved to: {equation_viz_path}")
        
        print(f"\nAll reports saved to: {output_dir}/")
        print("\nYou can open the HTML report in your browser:")
        print(f"  firefox {html_report_path}")
        
    except Exception as e:
        print(f"Error generating reports: {e}")
        import traceback
        traceback.print_exc()

def view_documentation():
    """Open the visualization guide"""
    guide_path = Path(__file__).parent.parent / "VISUALIZATION_GUIDE.md"
    if guide_path.exists():
        print("\nOpening visualization guide...")
        try:
            # Try different methods to open the file
            if sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', str(guide_path)])
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(guide_path)])
            elif sys.platform == 'win32':
                os.startfile(str(guide_path))
            else:
                # Fallback: print to console
                with open(guide_path, 'r') as f:
                    print(f.read())
        except Exception as e:
            print(f"Error opening guide: {e}")
            print(f"You can manually open: {guide_path}")
    else:
        print("Visualization guide not found!")
        print("Expected location: VISUALIZATION_GUIDE.md")

def main():
    """Main launcher loop"""
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '0':
                print("\nExiting visualization launcher...")
                break
            elif choice == '1':
                run_config_editor()
            elif choice == '2':
                run_realtime_monitor()
            elif choice == '3':
                generate_reports()
            elif choice == '4':
                view_documentation()
            else:
                print("\nInvalid choice! Please enter a number between 0-4.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("Starting Fiber Optics Neural Network Visualization Launcher...")
    
    # Check if we're in the correct directory
    config_dir = Path(__file__).parent
    required_files = ['visualizer.py', 'config_visualizer.py', 'visualization_ui.py']
    missing_files = [f for f in required_files if not (config_dir / f).exists()]
    
    if missing_files:
        print("\nError: The following required files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nMake sure you're running this script from the Network directory.")
        sys.exit(1)
    
    main()