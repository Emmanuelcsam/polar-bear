from PIL import Image
import numpy as np
import os
import sys

def detect_anomalies(image1_path="image1.jpg", image2_path="image2.jpg", threshold_percentile=95):
    """Detect anomalies between two images - can be called independently or via connector"""
    
    # Check if files exist
    if not os.path.exists(image1_path):
        # Try to get from shared memory if available
        if '_get_shared' in globals():
            shared_img1 = _get_shared('image1_data')
            if shared_img1 is not None:
                img1 = np.array(shared_img1)
            else:
                return {"error": f"Image file {image1_path} not found"}
        else:
            return {"error": f"Image file {image1_path} not found"}
    else:
        img1 = np.array(Image.open(image1_path).convert('L'))
    
    if not os.path.exists(image2_path):
        if '_get_shared' in globals():
            shared_img2 = _get_shared('image2_data')
            if shared_img2 is not None:
                img2 = np.array(shared_img2)
            else:
                return {"error": f"Image file {image2_path} not found"}
        else:
            return {"error": f"Image file {image2_path} not found"}
    else:
        img2 = np.array(Image.open(image2_path).convert('L'))
    
    # Calculate differences
    diff = np.abs(img1.astype(int) - img2.astype(int))
    threshold = np.percentile(diff, threshold_percentile)
    anomalies = np.where(diff > threshold)
    
    # Calculate KL divergence
    hist1, _ = np.histogram(img1, 256, [0, 256])
    hist2, _ = np.histogram(img2, 256, [0, 256])
    hist1 = hist1 + 1e-10  # Avoid log(0)
    hist2 = hist2 + 1e-10
    kl_div = np.sum(hist1 * np.log(hist1/hist2))
    
    # Prepare results
    anomaly_details = []
    for i in range(min(10, len(anomalies[0]))):
        anomaly_details.append({
            "x": int(anomalies[1][i]),
            "y": int(anomalies[0][i]),
            "difference": int(diff[anomalies[0][i], anomalies[1][i]])
        })
    
    results = {
        "anomaly_count": len(anomalies[0]),
        "kl_divergence": float(kl_div),
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "anomaly_details": anomaly_details,
        "image1_shape": img1.shape,
        "image2_shape": img2.shape
    }
    
    # Share results if running under connector
    if '_set_shared' in globals():
        _set_shared('anomaly_results', results)
        _send_message('all', f"Anomaly detection complete: {results['anomaly_count']} anomalies found")
    
    return results

def main():
    """Main function for independent execution"""
    # Parse command line arguments if provided
    image1 = sys.argv[1] if len(sys.argv) > 1 else "image1.jpg"
    image2 = sys.argv[2] if len(sys.argv) > 2 else "image2.jpg"
    
    results = detect_anomalies(image1, image2)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Anomalies: {results['anomaly_count']} pixels, KL-div: {results['kl_divergence']:.4f}")
        for detail in results['anomaly_details']:
            print(f"({detail['x']},{detail['y']}): {detail['difference']}")

# Support both independent and connector-based execution
if __name__ == "__main__":
    main()
elif '_connector_control' in globals():
    # Running under connector control
    print("Image anomaly detector loaded under connector control")