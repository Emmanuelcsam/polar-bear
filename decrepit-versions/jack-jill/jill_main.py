import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from jill_functions.defect_detection_config import DefectDetectionConfig
from jill_functions.detect_defects import detect_defects
from jill_functions.visualize_results import visualize_results

def main():
    """Main function to run the unified defect detector"""
    parser = argparse.ArgumentParser(description='Unified Fiber Optic Defect Detection System')
    parser.add_argument('image_path', type=str, help='Path to fiber optic image')
    parser.add_argument('--cladding-diameter', type=float, default=125.0, help='Cladding diameter in microns (e.g., 125)')
    parser.add_argument('--core-diameter', type=float, default=9.0, help='Core diameter in microns (e.g., 9 for SMF, 50/62.5 for MMF)')
    parser.add_argument('--output', type=str, default=None, help='Output path for visualization PNG')
    parser.add_argument('--save-masks', action='store_true', help='Save individual detection masks')
    
    args = parser.parse_args()
    
    print("Initializing Unified Fiber Defect Detector...")
    config = DefectDetectionConfig()
    
    try:
        results = detect_defects(
            args.image_path,
            config,
            cladding_diameter_um=args.cladding_diameter,
            core_diameter_um=args.core_diameter
        )
        
        # Print summary
        print("\n" + "="*50 + "\nDETECTION RESULTS\n" + "="*50)
        print(f"Status: {results['pass_fail']['status']}")
        print(f"Total Defects: {len(results['defects'])}")
        
        # Save visualization
        output_path = args.output or f"{Path(args.image_path).stem}_results.png"
        visualize_results(args.image_path, results, output_path)
        
        # Save masks if requested
        if args.save_masks and 'detection_masks' in results:
            mask_dir = Path(args.image_path).parent / f"{Path(args.image_path).stem}_masks"
            mask_dir.mkdir(exist_ok=True)
            for name, mask in results['detection_masks'].items():
                cv2.imwrite(str(mask_dir / f"{name}.png"), mask)
            print(f"\nMasks saved to: {mask_dir}")
        
        # Save detailed results as JSON
        json_results = {
            'image_path': results['image_path'],
            'status': results['pass_fail']['status'],
            'total_defects': len(results['defects']),
            'defects': [d.__dict__ for d in results['defects']]
        }
        json_path = f"{Path(args.image_path).stem}_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nDetailed results saved to: {json_path}")
        
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # To run this, you would use the command line:
    # python jill_main.py path/to/your/image.png
    #
    # Since we cannot provide command line arguments here, this block will not execute.
    # The purpose of this file is to be the main entry point for the user.
    print("This is the main entry point for the Jill defect detection system.")
    print("Please run from the command line, e.g.:")
    print("python jill_main.py <image_path> --cladding-diameter 125")
    # As a demonstration, we can try to run with a placeholder.
    # Create a dummy image to run on.
    dummy_image = np.full((480, 480), 128, dtype=np.uint8)
    cv2.circle(dummy_image, (240, 240), 150, 200, -1) # Cladding
    cv2.circle(dummy_image, (240, 240), 15, 220, -1) # Core
    cv2.line(dummy_image, (100, 100), (300, 120), 64, 2) # Scratch
    cv2.circle(dummy_image, (300, 300), 5, 32, -1) # Dig
    dummy_path = "dummy_fiber_image.png"
    cv2.imwrite(dummy_path, dummy_image)
    
    print(f"\nCreated a dummy image '{dummy_path}' to demonstrate.")
    print("Running detection on the dummy image...")
    
    # Simulate command line arguments
    class Args:
        image_path = dummy_path
        cladding_diameter = 125.0
        core_diameter = 9.0
        output = None
        save_masks = False
    
    args = Args()
    main.__globals__['args'] = args
    
    # Re-running main with simulated args
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, nargs='?', default=args.image_path)
    parser.add_argument('--cladding-diameter', type=float, default=args.cladding_diameter)
    parser.add_argument('--core-diameter', type=float, default=args.core_diameter)
    parser.add_argument('--output', type=str, default=args.output)
    parser.add_argument('--save-masks', action='store_true', default=args.save_masks)
    
    # This is a bit of a hack to make it runnable without user interaction
    try:
        parsed_args = parser.parse_args([args.image_path])
        # Manually set other args
        parsed_args.cladding_diameter = args.cladding_diameter
        parsed_args.core_diameter = args.core_diameter
        parsed_args.output = args.output
        parsed_args.save_masks = args.save_masks
        
        config = DefectDetectionConfig()
        results = detect_defects(
            parsed_args.image_path,
            config,
            cladding_diameter_um=parsed_args.cladding_diameter,
            core_diameter_um=parsed_args.core_diameter
        )
        output_path = parsed_args.output or f"{Path(parsed_args.image_path).stem}_results.png"
        visualize_results(parsed_args.image_path, results, output_path)
    except SystemExit:
        # Prevent argparse from exiting the script
        pass
    except Exception as e:
        print(f"Error during demo run: {e}")
