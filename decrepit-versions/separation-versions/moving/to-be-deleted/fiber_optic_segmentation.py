import cv2
import numpy as np
import os
import argparse
from scipy.signal import find_peaks


class FiberOpticSegmenter:
    """
    A comprehensive fiber optic endface image segmentation tool that combines:
    - Manual intensity-based segmentation
    - Automatic peak-based segmentation
    - Fiber-specific segmentation (core, cladding, ferrule)
    """
    
    def __init__(self, image_path, output_dir="segmented_output"):
        """Initialize the segmenter with an image."""
        self.image_path = image_path
        self.output_dir = output_dir
        self.original_image = None
        self.gray_image = None
        self.blurred_image = None
        self.center_x = None
        self.center_y = None
        self.core_radius = None
        self.cladding_radius = None
        
        # Load and prepare the image
        self._load_image()
        
    def _load_image(self):
        """Load and prepare the image for processing."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"The file '{self.image_path}' was not found.")
        
        # Load original image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read the image from '{self.image_path}'.")
        
        # Convert to grayscale
        if len(self.original_image.shape) == 3:
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_image = self.original_image
            self.original_image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        
        # Apply blur for better processing
        self.blurred_image = cv2.GaussianBlur(self.gray_image, (9, 9), 2)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Successfully loaded image: {self.image_path}")
        print(f"Image dimensions: {self.gray_image.shape}")
        
    def manual_segment_by_intensity(self, intensity_ranges):
        """
        Segment the image based on user-defined intensity ranges.
        From sam.py functionality.
        """
        print("\n--- Manual Intensity-Based Segmentation ---")
        print(f"Processing {len(intensity_ranges)} intensity ranges...")
        
        manual_dir = os.path.join(self.output_dir, "manual_intensity")
        os.makedirs(manual_dir, exist_ok=True)
        
        for i, (min_val, max_val) in enumerate(intensity_ranges):
            print(f"Processing range {i+1}: Intensity {min_val} to {max_val}...")
            
            # Create mask and segment
            mask = cv2.inRange(self.gray_image, min_val, max_val)
            segmented_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
            
            # Save the result
            output_filename = f"manual_region_{i+1}_intensity_{min_val}-{max_val}.png"
            output_path = os.path.join(manual_dir, output_filename)
            cv2.imwrite(output_path, segmented_image)
            print(f"-> Saved to '{output_path}'")
            
    def adaptive_segment_by_peaks(self, peak_prominence=1000):
        """
        Automatically segment the image by finding intensity peaks.
        From sedrik.py functionality.
        """
        print("\n--- Adaptive Peak-Based Segmentation ---")
        
        # Calculate histogram
        histogram = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        histogram = histogram.flatten()
        
        # Find peaks
        peaks, properties = find_peaks(histogram, prominence=peak_prominence, 
                                     width=(None, None), rel_height=1.0)
        
        if len(peaks) == 0:
            print(f"No significant peaks found with prominence={peak_prominence}.")
            return
        
        # Extract peak boundaries
        left_bases = [int(x) for x in properties['left_ips']]
        right_bases = [int(x) for x in properties['right_ips']]
        intensity_ranges = list(zip(left_bases, right_bases))
        
        print(f"Found {len(peaks)} significant intensity regions.")
        print(f"Detected ranges: {intensity_ranges}")
        
        adaptive_dir = os.path.join(self.output_dir, "adaptive_peaks")
        os.makedirs(adaptive_dir, exist_ok=True)
        
        for i, (min_val, max_val) in enumerate(intensity_ranges):
            peak_intensity = peaks[i]
            print(f"Processing region {i+1}: Peak at {peak_intensity} (Range: {min_val}-{max_val})...")
            
            # Create mask and segment
            mask = cv2.inRange(self.gray_image, min_val, max_val)
            segmented_image = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
            
            # Save the result
            output_filename = f"adaptive_region_{i+1}_peak_{peak_intensity}_range_{min_val}-{max_val}.png"
            output_path = os.path.join(adaptive_dir, output_filename)
            cv2.imwrite(output_path, segmented_image)
            print(f"-> Saved to '{output_path}'")
            
    def detect_fiber_center(self):
        """Detect the center of the fiber using Hough Circle Transform."""
        circles = cv2.HoughCircles(
            self.blurred_image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=int(self.gray_image.shape[0] / 3)
        )
        
        if circles is None:
            print("Could not detect fiber center automatically.")
            # Use image center as fallback
            self.center_y, self.center_x = self.gray_image.shape[0] // 2, self.gray_image.shape[1] // 2
            print(f"Using image center as fallback: ({self.center_x}, {self.center_y})")
        else:
            circles = np.uint16(np.around(circles))
            self.center_x, self.center_y, _ = circles[0, 0]
            print(f"Detected fiber center at: ({self.center_x}, {self.center_y})")
            
    def fiber_specific_segment(self):
        """
        Segment the fiber into core, cladding, and ferrule regions.
        From sergio.py functionality.
        """
        print("\n--- Fiber-Specific Segmentation ---")
        
        # Detect center if not already done
        if self.center_x is None:
            self.detect_fiber_center()
        
        # Calculate gradient magnitude
        sobel_x = cv2.Sobel(self.blurred_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(self.blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        change_magnitude = cv2.magnitude(sobel_x, sobel_y)
        
        # Analyze radial profile
        height, width = self.gray_image.shape
        max_radius = int(np.sqrt(height**2 + width**2) / 2)
        
        radial_profile = np.zeros(max_radius)
        radial_counts = np.zeros(max_radius, dtype=int)
        
        for y in range(height):
            for x in range(width):
                radius = int(np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2))
                if radius < max_radius:
                    radial_profile[radius] += change_magnitude[y, x]
                    radial_counts[radius] += 1
        
        # Avoid division by zero
        radial_counts[radial_counts == 0] = 1
        radial_profile /= radial_counts
        
        # Find peaks in radial profile
        peaks = []
        for r in range(1, len(radial_profile) - 1):
            if (radial_profile[r] > radial_profile[r-1] and 
                radial_profile[r] > radial_profile[r+1] and 
                radial_profile[r] > np.mean(radial_profile)):
                peaks.append((r, radial_profile[r]))
        
        if len(peaks) < 2:
            print("Using default radii for core and cladding boundaries.")
            # Use default values based on typical fiber dimensions
            self.core_radius = int(0.1 * min(height, width))
            self.cladding_radius = int(0.3 * min(height, width))
        else:
            # Sort peaks and get the two strongest
            peaks.sort(key=lambda p: p[1], reverse=True)
            radii = sorted([p[0] for p in peaks[:2]])
            self.core_radius = radii[0]
            self.cladding_radius = radii[1]
        
        print(f"Core radius: {self.core_radius} pixels")
        print(f"Cladding radius: {self.cladding_radius} pixels")
        
        fiber_dir = os.path.join(self.output_dir, "fiber_regions")
        os.makedirs(fiber_dir, exist_ok=True)
        
        # Create masks
        mask = np.zeros_like(self.gray_image)
        
        # Core region
        core_mask = cv2.circle(mask.copy(), (self.center_x, self.center_y), 
                              self.core_radius, 255, -1)
        core_region = cv2.bitwise_and(self.original_image, self.original_image, mask=core_mask)
        
        # Cladding region
        cladding_mask_outer = cv2.circle(mask.copy(), (self.center_x, self.center_y), 
                                        self.cladding_radius, 255, -1)
        cladding_mask = cv2.subtract(cladding_mask_outer, core_mask)
        cladding_region = cv2.bitwise_and(self.original_image, self.original_image, 
                                         mask=cladding_mask)
        
        # Ferrule region
        ferrule_mask = cv2.bitwise_not(cladding_mask_outer)
        ferrule_region = cv2.bitwise_and(self.original_image, self.original_image, 
                                        mask=ferrule_mask)
        
        # Diagnostic image
        diagnostic_image = self.original_image.copy()
        cv2.circle(diagnostic_image, (self.center_x, self.center_y), 
                  self.core_radius, (0, 255, 0), 2)  # Green for core
        cv2.circle(diagnostic_image, (self.center_x, self.center_y), 
                  self.cladding_radius, (0, 0, 255), 2)  # Red for cladding
        cv2.circle(diagnostic_image, (self.center_x, self.center_y), 
                  3, (255, 255, 0), -1)  # Yellow dot at center
        
        # Save all outputs
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        cv2.imwrite(os.path.join(fiber_dir, f'{base_name}_core_region.png'), core_region)
        cv2.imwrite(os.path.join(fiber_dir, f'{base_name}_cladding_region.png'), cladding_region)
        cv2.imwrite(os.path.join(fiber_dir, f'{base_name}_ferrule_region.png'), ferrule_region)
        cv2.imwrite(os.path.join(fiber_dir, f'{base_name}_boundaries_detected.png'), 
                   diagnostic_image)
        
        print(f"Fiber regions saved to '{fiber_dir}'")
        
    def generate_combined_analysis(self):
        """Generate a comprehensive analysis combining all methods."""
        print("\n--- Generating Combined Analysis ---")
        
        analysis_dir = os.path.join(self.output_dir, "combined_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Create intensity map
        intensity_map = cv2.applyColorMap(self.gray_image, cv2.COLORMAP_JET)
        
        # Create gradient magnitude visualization
        sobel_x = cv2.Sobel(self.blurred_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(self.blurred_image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_mag = cv2.magnitude(sobel_x, sobel_y)
        gradient_normalized = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        gradient_colored = cv2.applyColorMap(gradient_normalized, cv2.COLORMAP_HOT)
        
        # Create histogram visualization
        histogram = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        hist_height = 300
        hist_width = 512
        hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
        
        # Normalize histogram for display
        hist_normalized = cv2.normalize(histogram, None, 0, hist_height, cv2.NORM_MINMAX)
        
        # Draw histogram
        for i in range(256):
            x = int(i * hist_width / 256)
            y = hist_height - int(hist_normalized[i])
            cv2.line(hist_img, (x, hist_height), (x, y), (255, 255, 255), 2)
        
        # Save all analysis images
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        cv2.imwrite(os.path.join(analysis_dir, f'{base_name}_intensity_map.png'), intensity_map)
        cv2.imwrite(os.path.join(analysis_dir, f'{base_name}_gradient_magnitude.png'), gradient_colored)
        cv2.imwrite(os.path.join(analysis_dir, f'{base_name}_histogram.png'), hist_img)
        
        print(f"Combined analysis saved to '{analysis_dir}'")
        
    def run_all_segmentations(self, manual_ranges=None, peak_prominence=1000):
        """Run all segmentation methods."""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE FIBER OPTIC SEGMENTATION")
        print(f"{'='*60}")
        
        # Run manual segmentation if ranges provided
        if manual_ranges:
            self.manual_segment_by_intensity(manual_ranges)
        
        # Run adaptive segmentation
        self.adaptive_segment_by_peaks(peak_prominence)
        
        # Run fiber-specific segmentation
        self.fiber_specific_segment()
        
        # Generate combined analysis
        self.generate_combined_analysis()
        
        print(f"\n{'='*60}")
        print(f"All segmentation complete! Results saved to: {self.output_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="""Comprehensive Fiber Optic Endface Segmentation Tool.
        Combines manual intensity-based, automatic peak-based, and fiber-specific segmentation.""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--image', type=str, required=True, 
                       help='Path to the input fiber optic image file.')
    parser.add_argument('-o', '--output', type=str, default='segmented_output', 
                       help='Directory to save all output files.')
    parser.add_argument('-p', '--prominence', type=int, default=1000,
                       help='Peak prominence threshold for adaptive segmentation (default: 1000)')
    parser.add_argument('-m', '--manual-ranges', type=str, nargs='*',
                       help='Manual intensity ranges in format "min-max" (e.g., "80-130 170-200 200-230")')
    parser.add_argument('--all', action='store_true',
                       help='Run all segmentation methods with default settings')
    
    args = parser.parse_args()
    
    # Parse manual ranges if provided
    manual_ranges = None
    if args.manual_ranges:
        manual_ranges = []
        for range_str in args.manual_ranges:
            min_val, max_val = map(int, range_str.split('-'))
            manual_ranges.append((min_val, max_val))
    
    # Create segmenter instance
    segmenter = FiberOpticSegmenter(args.image, args.output)
    
    # Run segmentations
    if args.all:
        # Use default manual ranges if none provided
        if manual_ranges is None:
            manual_ranges = [(80, 130), (170, 200), (200, 230)]
        segmenter.run_all_segmentations(manual_ranges, args.prominence)
    else:
        # Run individual methods as requested
        if manual_ranges:
            segmenter.manual_segment_by_intensity(manual_ranges)
        segmenter.adaptive_segment_by_peaks(args.prominence)
        segmenter.fiber_specific_segment()
        segmenter.generate_combined_analysis()


if __name__ == '__main__':
    # Example usage without command line arguments
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python fiber_segmenter.py -i 'path/to/image.png' --all")
        print("\nRunning with default test parameters...")
        
        # Default test parameters
        test_image = r"C:\Users\Saem1001\Documents\GitHub\IPPS\processing\output\img (210)_intensity_map.png"
        test_ranges = [(80, 130), (170, 200), (200, 230)]
        
        if os.path.exists(test_image):
            segmenter = FiberOpticSegmenter(test_image)
            segmenter.run_all_segmentations(test_ranges, peak_prominence=1000)
        else:
            print(f"Test image not found: {test_image}")
            print("Please run with command line arguments.")
    else:
        main()