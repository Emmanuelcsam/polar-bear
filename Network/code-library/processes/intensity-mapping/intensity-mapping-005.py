from PIL import Image
import random
import os
import sys
import time

def match_intensities(image_path="image.jpg", target_intensity=None, max_iterations=None):
    """Match pixel intensities - can be called independently or via connector"""
    
    # Check if file exists or get from shared memory
    if not os.path.exists(image_path):
        if '_get_shared' in globals():
            shared_img = _get_shared('image_data')
            if shared_img is not None:
                if isinstance(shared_img, list):
                    img_data = shared_img
                else:
                    img_data = list(shared_img)
            else:
                return {"error": f"Image file {image_path} not found"}
        else:
            return {"error": f"Image file {image_path} not found"}
    else:
        img_data = list(Image.open(image_path).convert('L').getdata())
    
    # Get parameters from shared memory if available
    if '_get_shared' in globals():
        if target_intensity is None:
            target_intensity = _get_shared('target_intensity')
        if max_iterations is None:
            max_iterations = _get_shared('max_iterations')
    
    # Default values
    if max_iterations is None:
        max_iterations = 100 if target_intensity else float('inf')
    
    matches = []
    iterations = 0
    
    # If target intensity specified, find matching pixels
    if target_intensity is not None:
        count = img_data.count(target_intensity)
        percentage = count / len(img_data) * 100
        matches.append({
            "intensity": target_intensity,
            "count": count,
            "percentage": percentage
        })
        
        # Share result
        if '_set_shared' in globals():
            _set_shared('intensity_match_result', matches[0])
            _send_message('all', f"Found {count} pixels ({percentage:.2f}%) with intensity {target_intensity}")
    else:
        # Random matching mode
        while iterations < max_iterations:
            r = random.randint(0, 255)
            count = img_data.count(r)
            percentage = count / len(img_data) * 100
            
            match_info = {
                "intensity": r,
                "count": count,
                "percentage": percentage
            }
            matches.append(match_info)
            
            # In connector mode, share interesting findings
            if '_set_shared' in globals() and percentage > 5:
                _set_shared('high_frequency_intensity', match_info)
                _send_message('all', f"High frequency intensity found: {r} ({percentage:.2f}%)")
            
            iterations += 1
            
            # In independent mode with infinite loop, print and sleep
            if max_iterations == float('inf') and __name__ == "__main__":
                if r in img_data:
                    print(f"{r} {percentage:.2f}%")
                else:
                    print(f"{r} 0%")
                time.sleep(0.1)  # Small delay to not flood output
    
    results = {
        "total_pixels": len(img_data),
        "unique_intensities": len(set(img_data)),
        "matches": matches,
        "iterations": iterations
    }
    
    # Share final results if running under connector
    if '_set_shared' in globals():
        _set_shared('intensity_matcher_results', results)
    
    return results

def main():
    """Main function for independent execution"""
    # Parse command line arguments
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    
    # Check if target intensity specified
    if len(sys.argv) > 2:
        target = int(sys.argv[2])
        results = match_intensities(image_path, target_intensity=target)
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            match = results['matches'][0]
            print(f"{match['intensity']} {match['percentage']:.2f}%")
    else:
        # Infinite loop mode
        try:
            match_intensities(image_path)
        except KeyboardInterrupt:
            print("\nIntensity matching stopped")

# Support both independent and connector-based execution
if __name__ == "__main__":
    main()
elif '_connector_control' in globals():
    # Running under connector control
    print("Intensity matcher loaded under connector control")