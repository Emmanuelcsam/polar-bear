import os
import time
import pickle
from pathlib import Path
from datetime import datetime

from jack_functions.build_comprehensive_reference_model import build_comprehensive_reference_model
from jack_functions.detect_anomalies_comprehensive import detect_anomalies_comprehensive
from jack_functions.visualize_comprehensive_results import visualize_comprehensive_results
from jack_functions.generate_detailed_report import generate_detailed_report

def main():
    """Main execution function for the anomaly detection system."""
    print("\n" + "="*80 + "\nULTRA-COMPREHENSIVE MATRIX ANOMALY DETECTION SYSTEM\n" + "="*80)
    
    kb_path = "anomaly_knowledge_base.pkl"
    
    # Step 1: Build or load reference model
    if not os.path.exists(kb_path):
        print(f"Knowledge base '{kb_path}' not found.")
        ref_dir = input("Enter path to folder containing reference images/JSONs to build one: ").strip()
        if os.path.isdir(ref_dir):
            if not build_comprehensive_reference_model(ref_dir, kb_path):
                print("✗ Failed to build reference model. Exiting.")
                return
        else:
            print(f"✗ Directory not found: {ref_dir}. Exiting.")
            return
    else:
        print(f"✓ Using existing knowledge base: {kb_path}")

    try:
        with open(kb_path, 'rb') as f:
            reference_model = pickle.load(f)
    except Exception as e:
        print(f"✗ Could not load knowledge base: {e}. Exiting.")
        return

    # Step 2: Analyze test images
    while True:
        print("\n" + "-"*80)
        test_path = input("Enter path to test image/JSON file (or 'quit' to exit): ").strip()
        if test_path.lower() == 'quit': break
        if not os.path.isfile(test_path):
            print(f"✗ File not found: {test_path}"); continue
        
        start_time = time.time()
        results = detect_anomalies_comprehensive(test_path, kb_path)
        
        if results:
            print(f"\n✓ Analysis completed in {time.time() - start_time:.2f} seconds")
            
            # Generate outputs
            base_name = Path(test_path).stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_prefix = f"analysis_{base_name}_{timestamp}"
            
            visualize_comprehensive_results(results, reference_model, f"{output_prefix}.png")
            generate_detailed_report(results, f"{output_prefix}.txt")
        else:
            print("✗ Analysis failed.")
    
    print("\n" + "="*80 + "\nSystem finished.\n" + "="*80 + "\n")

if __name__ == "__main__":
    # This script is intended to be run interactively from the command line.
    # The `input()` calls will not work in this environment.
    # To run the system, execute `python jack_main.py` in your terminal.
    print("This is the main entry point for the Jack anomaly detection system.")
    print("To run it, execute 'python jack_main.py' in your terminal.")
    
    # Create a dummy environment to showcase functionality without user input.
    print("\n--- Running a non-interactive demo ---")
    kb_path = "demo_kb.pkl"
    ref_dir = Path("demo_reference_images")
    test_dir = Path("demo_test_images")
    
    if not ref_dir.exists():
        ref_dir.mkdir()
        for i in range(3):
            cv2.imwrite(str(ref_dir / f"ref_{i}.png"), np.random.randint(120, 140, (50, 50), dtype=np.uint8))
    
    if not test_dir.exists():
        test_dir.mkdir()
        # Normal-like image
        cv2.imwrite(str(test_dir / "test_normal.png"), np.random.randint(125, 135, (50, 50), dtype=np.uint8))
        # Anomalous image
        anom_img = np.random.randint(125, 135, (50, 50), dtype=np.uint8)
        cv2.rectangle(anom_img, (10, 10), (20, 20), 255, -1)
        cv2.imwrite(str(test_dir / "test_anomalous.png"), anom_img)

    print("\n1. Building reference model for demo...")
    build_comprehensive_reference_model(str(ref_dir), kb_path)
    
    with open(kb_path, 'rb') as f:
        reference_model = pickle.load(f)

    print("\n2. Analyzing a 'normal' test image...")
    results_normal = detect_anomalies_comprehensive(str(test_dir / "test_normal.png"), kb_path)
    if results_normal:
        visualize_comprehensive_results(results_normal, reference_model, "demo_normal_results.png")
        generate_detailed_report(results_normal, "demo_normal_report.txt")

    print("\n3. Analyzing an 'anomalous' test image...")
    results_anom = detect_anomalies_comprehensive(str(test_dir / "test_anomalous.png"), kb_path)
    if results_anom:
        visualize_comprehensive_results(results_anom, reference_model, "demo_anomalous_results.png")
        generate_detailed_report(results_anom, "demo_anomalous_report.txt")
        
    print("\n--- Demo finished ---")