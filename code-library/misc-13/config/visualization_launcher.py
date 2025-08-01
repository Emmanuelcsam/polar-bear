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
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
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
        # Using sys.executable ensures we use the same python interpreter
        script_path = Path(__file__).parent / "config_visualizer.py"
        subprocess.run([sys.executable, str(script_path)], check=True)
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error launching config editor: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Make sure PyQt5 is installed: pip install PyQt5")

def run_realtime_monitor():
    """Launch the real-time monitor"""
    print("\nLaunching Real-time Monitor...")
    print("Note: Make sure the neural network is running")
    try:
        script_path = Path(__file__).parent / "visualization_ui.py"
        subprocess.run([sys.executable, str(script_path)], check=True)
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error launching monitor: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Make sure GTK3 is installed on your system.")

def generate_reports():
    """Generate visualization reports"""
    print("\nGenerating Visualization Reports...")
    
    try:
        # Import the visualizer
        from visualizer import FiberOpticsVisualizer
        from datetime import datetime
        
        # Create output directory
        output_dir = project_root / "visualization_reports"
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
        print("Generating example visualizations with dummy data...")
        import numpy as np
        
        dummy_results = {
            'region_probs': np.random.rand(1, 3, 256, 256),
            'anomaly_map': np.random.rand(1, 256, 256) * 0.5,
            'summary': {
                'final_similarity_score': 0.85, 'meets_threshold': True,
                'primary_region': 'core', 'anomaly_score': 0.12,
            },
            'equation_info': {
                'components': {
                    'reference_similarity': 0.88, 'trend_adherence': 0.91,
                    'anomaly_inverse': 0.88, 'segmentation_confidence': 0.95,
                    'reconstruction': 0.82
                },
                'final_score': 0.85
            }
        }
        
        equation_viz_path = output_dir / f"equation_components_{timestamp}.png"
        visualizer.visualize_equation_components(dummy_results, str(equation_viz_path))
        print(f"  ✓ Saved to: {equation_viz_path}")
        
        print(f"\nAll reports saved to: {output_dir.resolve()}/")
        
    except ImportError as e:
        print(f"Error: Failed to import a required module: {e}")
        print("Please ensure all dependencies are installed and core modules are in the python path.")
    except Exception as e:
        print(f"An error occurred while generating reports: {e}")
        import traceback
        traceback.print_exc()

def view_documentation():
    """Open the visualization guide"""
    guide_path = project_root / "VISUALIZATION_GUIDE.md"
    if guide_path.exists():
        print(f"\nAttempting to open documentation: {guide_path.resolve()}")
        try:
            if sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', str(guide_path)], check=True)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(guide_path)], check=True)
            elif sys.platform == 'win32':
                os.startfile(str(guide_path))
            else:
                print("Could not automatically open file. Please open it manually.")
        except Exception as e:
            print(f"Error opening guide: {e}")
            print(f"You can manually open: {guide_path}")
    else:
        print(f"Error: Visualization guide not found at expected location: {guide_path.resolve()}")

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
                print("\nInvalid choice! Please enter a number between 0 and 4.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("Starting Fiber Optics Neural Network Visualization Launcher...")
    
    # Check if we're in the correct directory
    config_dir = Path(__file__).parent
    required_files = ['visualizer.py', 'config_visualizer.py', 'visualization_ui.py']
    missing_files = [f for f in required_files if not (config_dir / f).exists()]
    
    if missing_files:
        print("\nError: The following required files are missing in the current directory:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure you are running this script from the correct directory.")
        sys.exit(1)
    
    main()