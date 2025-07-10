import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')
import json
import os

class XYScanFiberAnalyzer:
    """
    Fiber optic region detection by performing second derivative analysis on each
    horizontal (X) and vertical (Y) line profile of the image. The resulting
    boundary points are then used with a Hough Circle Transform to identify the
    fiber's structure.
    """
    
    def __init__(self, image_path):
        """Initialize with image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        print(f"Loaded image: {self.width}x{self.height} pixels")
    
    def analyze(self):
        """Run the full X/Y scanning analysis"""
        print("\nStarting X/Y Scan-Based Fiber Analysis...")
        print("=" * 60)
        
        # Step 1: Scan all rows and columns, find boundary points using 2nd derivative
        self.boundary_points = self._scan_and_find_points()
        
        # Step 2: Create a 'voting' image from the detected points
        self.voting_image = self._create_voting_image()
        
        # Step 3: Align and compare data by finding circles using Hough Transform
        self.detected_circles = self._find_circles_with_hough()
        
        # Step 4: Define core and cladding regions from the detected circles
        self.define_regions()
        
        # Step 5: Create masks from the final identified regions
        self.create_masks()
        
        # Step 6: Generate visualizations and report
        self.generate_output()
        
        print("\nAnalysis Complete!")
        return self.results

    def _analyze_1d_profile(self, profile):
        """Analyzes a single 1D profile to find boundary points."""
        if len(profile) < 50: return []
        
        # Strong smoothing is needed for noisy single-pixel lines
        profile_smooth = gaussian_filter1d(profile.astype(float), sigma=3)
        
        # First derivative (gradient)
        first_derivative = np.gradient(profile_smooth)
        
        # Find significant peaks in the gradient magnitude
        grad_mag = np.abs(first_derivative)
        
        # Height threshold should be adaptive to the line's own contrast
        height_thresh = np.std(grad_mag) * 2.0
        if height_thresh < 1.0: # Set a minimum sensitivity
            height_thresh = 1.0

        peaks, _ = find_peaks(grad_mag, distance=20, height=height_thresh)
        
        return peaks

    def _scan_and_find_points(self):
        """Iterate through every row and column, analyzing each as a 1D profile."""
        print("\nStep 1: Scanning all rows and columns to find boundary points...")
        points = []
        
        # Scan all rows (horizontal scan)
        for r in range(self.height):
            profile = self.gray[r, :]
            boundary_indices = self._analyze_1d_profile(profile)
            for c in boundary_indices:
                points.append((c, r)) # Append as (x, y)
        
        # Scan all columns (vertical scan)
        for c in range(self.width):
            profile = self.gray[:, c]
            boundary_indices = self._analyze_1d_profile(profile)
            for r in boundary_indices:
                points.append((c, r)) # Append as (x, y)
        
        print(f"  Detected {len(points)} potential boundary points.")
        return points

    def _create_voting_image(self):
        """Create a binary image where detected boundary points are white."""
        print("\nStep 2: Creating a voting image from detected points...")
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        for p in self.boundary_points:
            cv2.circle(image, p, 1, 255, -1) # Draw a small circle at each point
        
        # Apply a Gaussian blur to help the Hough Transform by consolidating nearby points
        image = cv2.GaussianBlur(image, (5, 5), 1)
        print("  Voting image created.")
        return image

    def _find_circles_with_hough(self):
        """Use Hough Circle Transform to find the best-fit circles from the voting image."""
        print("\nStep 3: Aligning data with Hough Circle Transform...")
        
        # Parameters for HoughCircles need to be tuned.
        # minDist: Minimum distance between the centers of detected circles.
        # param1: Upper threshold for the internal Canny edge detector.
        # param2: Accumulator threshold for circle centers at the detection stage.
        # minRadius, maxRadius: Self-explanatory.
        
        circles = cv2.HoughCircles(
            self.voting_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=15,
            minRadius=int(self.width * 0.05),
            maxRadius=int(self.width * 0.45)
        )
        
        if circles is None or len(circles[0]) < 2:
            raise RuntimeError("Hough Transform failed to detect at least two distinct circles. "
                               "The image may be low contrast or have unclear boundaries.")
        
        # Sort circles by radius
        circles = np.uint16(np.around(circles[0, :]))
        sorted_circles = sorted(circles, key=lambda c: c[2])
        
        # We assume the smallest is the core, the next is the cladding
        detected = {
            'core': sorted_circles[0],
            'cladding': sorted_circles[1]
        }
        
        print(f"  Detected Core at ({detected['core'][0]}, {detected['core'][1]}) with radius {detected['core'][2]}px")
        print(f"  Detected Cladding at ({detected['cladding'][0]}, {detected['cladding'][1]}) with radius {detected['cladding'][2]}px")
        
        return detected

    def define_regions(self):
        """Define regions based on the circles found by the Hough Transform."""
        print("\nStep 4: Defining regions from detected circles...")
        
        # Use the cladding circle's center as the definitive center for consistency
        self.center_x, self.center_y = self.detected_circles['cladding'][:2]
        
        self.regions = {
            'core': {'radius': self.detected_circles['core'][2]},
            'cladding': {'inner_radius': self.detected_circles['core'][2], 'outer_radius': self.detected_circles['cladding'][2]}
        }
        print("  Regions defined.")

    def create_masks(self):
        """Create circular masks based on the identified regions."""
        print("\nStep 5: Creating region masks...")
        self.masks = {}
        h, w = self.height, self.width
        cx, cy = int(self.center_x), int(self.center_y)
        
        # Core mask
        self.masks['core'] = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(self.masks['core'], (cx, cy), int(self.regions['core']['radius']), 255, -1)
        
        # Cladding mask
        cladding_full_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(cladding_full_mask, (cx, cy), int(self.regions['cladding']['outer_radius']), 255, -1)
        self.masks['cladding'] = cv2.subtract(cladding_full_mask, self.masks['core'])
        
        # Ferrule mask
        self.masks['ferrule'] = cv2.bitwise_not(cladding_full_mask)
        
        for name, mask in self.masks.items():
            print(f"  {name} mask created with {np.sum(mask > 0)} pixels.")
            
    def generate_output(self, output_dir='xy_scan_analysis_results'):
        """Generate visualizations and a JSON report."""
        print("\nStep 6: Generating output...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save voting image for debugging/visualization
        cv2.imwrite(os.path.join(output_dir, 'boundary_points_voting_image.png'), self.voting_image)
        
        # Create an image showing the detected circles on the original
        overlay_image = self.original.copy()
        core_c = self.detected_circles['core']
        clad_c = self.detected_circles['cladding']
        # Draw core circle (cyan)
        cv2.circle(overlay_image, (core_c[0], core_c[1]), core_c[2], (255, 255, 0), 2)
        # Draw cladding circle (red)
        cv2.circle(overlay_image, (clad_c[0], clad_c[1]), clad_c[2], (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_dir, 'detected_circles_overlay.png'), overlay_image)
        
        # Generate JSON report
        self.results = {
            'center': {'x': float(self.center_x), 'y': float(self.center_y)},
            'regions': {
                'core': {
                    'radius_px': int(self.regions['core']['radius']),
                    'pixel_count': int(np.sum(self.masks['core'] > 0))
                },
                'cladding': {
                    'inner_radius_px': int(self.regions['cladding']['inner_radius']),
                    'outer_radius_px': int(self.regions['cladding']['outer_radius']),
                    'pixel_count': int(np.sum(self.masks['cladding'] > 0))
                },
                'ferrule': {
                     'pixel_count': int(np.sum(self.masks['ferrule'] > 0))
                }
            }
        }
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}/")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use the image path provided from the command line
        image_path = sys.argv[1]
    else:
        # Fallback to creating a test image if no path is provided
        image_path = 'test_fiber_for_xy.jpg'
        if not os.path.exists(image_path):
            print("Creating a test fiber image...")
            size = 512
            img = np.full((size, size, 3), 170, dtype=np.uint8)
            cx, cy = size // 2, size // 2
            
            # Create Core
            cv2.circle(img, (cx, cy), 100, (220, 220, 220), -1)
            # Create Cladding
            cv2.circle(img, (cx, cy), 150, (110, 110, 110), -1)
            
            # Override core on top of cladding
            cv2.circle(img, (cx, cy), 100, (220, 220, 220), -1)

            # Apply blur and noise to make it realistic
            img = cv2.GaussianBlur(img, (7, 7), 2)
            noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            cv2.imwrite(image_path, img)
            print(f"Test image saved as {image_path}")

    try:
        analyzer = XYScanFiberAnalyzer(image_path)
        results = analyzer.analyze()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        core = results['regions']['core']
        cladding = results['regions']['cladding']
        
        print(f"Core -> Radius: {core['radius_px']} px")
        print(f"Cladding -> Inner Radius: {cladding['inner_radius_px']} px, Outer Radius: {cladding['outer_radius_px']} px")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
