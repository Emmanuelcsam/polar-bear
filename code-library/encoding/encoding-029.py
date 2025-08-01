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

class DirectionalCurvatureFiberAnalyzer:
    """
    Fiber optic region detection by performing curvature analysis along multiple radial directions independently.
    This allows for the detection of non-circular or irregular fiber cross-sections.
    """
    
    def __init__(self, image_path):
        """Initialize with image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        print(f"Loaded image: {self.width}x{self.height} pixels")
    
    def analyze(self, num_angles=360):
        """Run the full directional analysis"""
        print("\nStarting Directional Curvature-Based Fiber Analysis...")
        print("=" * 60)
        
        self.num_angles = num_angles
        
        # Step 1: Find a stable center point for the analysis
        self.find_center()
        
        # Step 2: Compute intensity profiles for all radial directions
        self.compute_all_radial_profiles()
        
        # Step 3: Find boundaries (inflection points) for each direction
        self.find_per_angle_boundaries()
        
        # Step 4: Define regions based on the per-angle boundaries
        self.define_regions()
        
        # Step 5: Create precise, non-circular masks from boundary points
        self.create_polygon_masks()
        
        # Step 6: Extract regions using the new masks
        self.extract_regions()
        
        # Step 7: Generate visualizations and a detailed report
        self.generate_output()
        
        print("\nAnalysis Complete!")
        return self.results
    
    def find_center(self):
        """Find image center using the brightest region"""
        print("\nStep 1: Finding center...")
        smoothed = cv2.GaussianBlur(self.gray, (11, 11), 3)
        _, max_val, _, max_loc = cv2.minMaxLoc(smoothed)
        
        # For more stability, calculate centroid of the brightest area
        threshold = np.percentile(smoothed, 98)
        bright_mask = (smoothed > threshold).astype(np.uint8)
        moments = cv2.moments(bright_mask)
        
        if moments['m00'] > 0:
            self.center_x = moments['m10'] / moments['m00']
            self.center_y = moments['m01'] / moments['m00']
        else:
            # Fallback to the absolute brightest point
            self.center_x, self.center_y = max_loc
            
        print(f"  Center: ({self.center_x:.1f}, {self.center_y:.1f})")

    def compute_all_radial_profiles(self):
        """Compute radial intensity profiles for every angle"""
        print(f"\nStep 2: Computing {self.num_angles} radial intensity profiles...")
        
        max_radius = int(min(self.center_x, self.center_y, 
                            self.width - self.center_x, 
                            self.height - self.center_y))
        
        self.radii = np.arange(max_radius)
        self.angles = np.linspace(0, 2*np.pi, self.num_angles, endpoint=False)
        
        self.intensity_profiles = []
        
        for angle in self.angles:
            x_coords = self.center_x + self.radii * np.cos(angle)
            y_coords = self.center_y + self.radii * np.sin(angle)
            
            # Use bilinear interpolation for smoother profiles
            intensities = cv2.remap(self.gray, x_coords.astype(np.float32), y_coords.astype(np.float32), 
                                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).flatten()
            
            self.intensity_profiles.append(intensities)
            
        print(f"  {len(self.intensity_profiles)} profiles computed for {len(self.radii)} radial positions each.")

    def find_per_angle_boundaries(self):
        """Find boundaries for each angle by analyzing each radial profile independently."""
        print("\nStep 3: Finding boundaries for each of the {} directions...".format(self.num_angles))
        
        self.per_angle_boundaries = []
        
        for i, profile in enumerate(self.intensity_profiles):
            # Smooth the individual profile
            profile_smooth = gaussian_filter1d(profile, sigma=2)
            
            # First derivative (gradient)
            first_derivative = np.gradient(profile_smooth)
            
            # Second derivative (curvature)
            second_derivative = np.gradient(first_derivative)
            
            # Find the two largest gradient peaks (one negative, one positive)
            grad_mag = np.abs(first_derivative)
            peaks, _ = find_peaks(grad_mag, distance=10, height=np.std(grad_mag))
            
            if len(peaks) < 2:
                self.per_angle_boundaries.append(None) # Mark as failed for this angle
                continue
            
            # Find the boundary inflection points around the main gradient peaks
            sorted_peaks = sorted(peaks[np.argsort(grad_mag[peaks])[-2:]])
            
            # Find where curve starts to cave in (inflection)
            boundary1 = self._find_inflection_point(second_derivative, sorted_peaks[0], 'before')
            boundary2 = self._find_inflection_point(second_derivative, sorted_peaks[1], 'after')

            if boundary1 is not None and boundary2 is not None:
                self.per_angle_boundaries.append(sorted([boundary1, boundary2]))
            else:
                self.per_angle_boundaries.append(None) # Mark as failed

        # Post-process to fix failed angles
        self._clean_boundaries()
        print("  Finished boundary analysis for all directions.")

    def _find_inflection_point(self, second_derivative, peak_loc, direction, search_window=30):
        """Finds the 'start' or 'end' of a feature by looking for where the curvature settles."""
        threshold = np.std(second_derivative) * 0.15
        
        if direction == 'before':
            start = max(0, peak_loc - search_window)
            search_region = second_derivative[start:peak_loc]
            # Find the last point BEFORE the peak where curvature is near zero
            indices = np.where(np.abs(search_region) < threshold)[0]
            if len(indices) > 0:
                return start + indices[-1]
        
        elif direction == 'after':
            end = min(len(second_derivative), peak_loc + search_window)
            search_region = second_derivative[peak_loc:end]
            # Find the first point AFTER the peak where curvature is near zero
            indices = np.where(np.abs(search_region) < threshold)[0]
            if len(indices) > 0:
                return peak_loc + indices[0]
                
        return None # Could not find a stable inflection point

    def _clean_boundaries(self):
        """Replaces failed boundary detections with the median of successful ones."""
        successful_boundaries = [b for b in self.per_angle_boundaries if b is not None]
        if not successful_boundaries:
            raise RuntimeError("Could not detect any boundaries in any direction. Image may be unsuitable.")
            
        median_boundary = np.median(successful_boundaries, axis=0).astype(int)
        
        failed_count = 0
        for i in range(len(self.per_angle_boundaries)):
            if self.per_angle_boundaries[i] is None:
                self.per_angle_boundaries[i] = median_boundary
                failed_count += 1
        
        if failed_count > 0:
            print(f"  Corrected {failed_count} angles using median boundaries: {median_boundary}px")

    def define_regions(self):
        """Define regions based on the now complete per-angle boundaries."""
        print("\nStep 4: Defining regions from per-angle data...")
        self.boundaries = np.array(self.per_angle_boundaries)
        
        self.regions = {}
        
        # Core: From center to the first boundary
        self.regions['core'] = {'end_radii': self.boundaries[:, 0]}
        
        # Cladding: Between the first and second boundaries
        self.regions['cladding'] = {
            'start_radii': self.boundaries[:, 0],
            'end_radii': self.boundaries[:, 1]
        }
        
        # Calculate average stats for reporting
        avg_core_radius = np.mean(self.regions['core']['end_radii'])
        avg_cladding_outer_radius = np.mean(self.regions['cladding']['end_radii'])
        
        print(f"  Average Core Radius: {avg_core_radius:.1f}px")
        print(f"  Average Cladding Outer Radius: {avg_cladding_outer_radius:.1f}px")

    def create_polygon_masks(self):
        """Create masks for each region by generating polygons from the boundary points."""
        print("\nStep 5: Creating polygon region masks...")
        
        self.masks = {}
        h, w = self.height, self.width
        
        # Create core mask
        core_points = self._get_cartesian_points(self.regions['core']['end_radii'])
        self.masks['core'] = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self.masks['core'], [core_points], 255)
        
        # Create combined core+cladding mask first
        cladding_outer_points = self._get_cartesian_points(self.regions['cladding']['end_radii'])
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(combined_mask, [cladding_outer_points], 255)
        
        # Cladding mask is the combined mask minus the core mask
        self.masks['cladding'] = cv2.subtract(combined_mask, self.masks['core'])
        
        # Ferrule mask is everything outside the cladding
        self.masks['ferrule'] = cv2.bitwise_not(combined_mask)
        
        for name, mask in self.masks.items():
            print(f"  {name}: {np.sum(mask > 0)} pixels")
            
    def _get_cartesian_points(self, radii_list):
        """Converts a list of radii at different angles to (x, y) coordinates."""
        points = []
        for r, angle in zip(radii_list, self.angles):
            x = self.center_x + r * np.cos(angle)
            y = self.center_y + r * np.sin(angle)
            points.append([x, y])
        return np.array(points, dtype=np.int32)

    def extract_regions(self):
        """Extract regions using the generated polygon masks."""
        print("\nStep 6: Extracting regions...")
        self.extracted_regions = {}
        for region_name, mask in self.masks.items():
            self.extracted_regions[region_name] = cv2.bitwise_and(self.original, self.original, mask=mask)

    def generate_output(self, output_dir='directional_analysis_results'):
        """Generate visualizations and save results."""
        print("\nStep 7: Generating output...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save extracted regions and masks
        for name in self.masks.keys():
            cv2.imwrite(os.path.join(output_dir, f'{name}.png'), self.extracted_regions[name])
            cv2.imwrite(os.path.join(output_dir, f'{name}_mask.png'), self.masks[name])
        
        self._create_visualizations(output_dir)
        
        # Generate JSON report
        self.results = {
            'center': {'x': float(self.center_x), 'y': float(self.center_y)},
            'regions': {
                'core': {
                    'avg_radius': np.mean(self.regions['core']['end_radii']),
                    'std_dev_radius': np.std(self.regions['core']['end_radii']),
                    'pixel_count': int(np.sum(self.masks['core'] > 0)),
                    'radii_per_angle': self.regions['core']['end_radii'].tolist()
                },
                'cladding': {
                    'avg_inner_radius': np.mean(self.regions['cladding']['start_radii']),
                    'avg_outer_radius': np.mean(self.regions['cladding']['end_radii']),
                    'avg_thickness': np.mean(self.regions['cladding']['end_radii'] - self.regions['cladding']['start_radii']),
                    'pixel_count': int(np.sum(self.masks['cladding'] > 0)),
                    'inner_radii_per_angle': self.regions['cladding']['start_radii'].tolist(),
                    'outer_radii_per_angle': self.regions['cladding']['end_radii'].tolist()
                },
                'ferrule': {
                    'pixel_count': int(np.sum(self.masks['ferrule'] > 0))
                }
            }
        }
        
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}/")

    def _create_visualizations(self, output_dir):
        """Create visualization plots showing the detected shapes."""
        # Plot showing the detected boundaries on the original image
        overlay_image = self.original.copy()
        
        # Draw core boundary
        core_pts = self._get_cartesian_points(self.regions['core']['end_radii'])
        cv2.polylines(overlay_image, [core_pts], isClosed=True, color=(0, 255, 255), thickness=1) # Cyan

        # Draw cladding boundary
        cladding_pts = self._get_cartesian_points(self.regions['cladding']['end_radii'])
        cv2.polylines(overlay_image, [cladding_pts], isClosed=True, color=(0, 0, 255), thickness=1) # Red
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Boundaries on Image (Core=Cyan, Cladding=Red)')
        plt.plot(self.center_x, self.center_y, 'g+', markersize=10, markeredgewidth=2, label='Detected Center')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'boundaries_on_image.png'), dpi=150)
        plt.close()
        
        # Results grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Directional Analysis Results', fontsize=16)

        axes[0, 0].imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].plot(self.center_x, self.center_y, 'r+', markersize=10)
        axes[0, 0].axis('off')
        
        mask_overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_overlay[self.masks['core'] > 0] = [255, 0, 0]    # Red
        mask_overlay[self.masks['cladding'] > 0] = [0, 255, 0] # Green
        mask_overlay[self.masks['ferrule'] > 0] = [0, 0, 255]   # Blue
        axes[0, 1].imshow(mask_overlay)
        axes[0, 1].set_title('Region Masks (R=Core, G=Clad, B=Ferrule)')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(cv2.cvtColor(self.extracted_regions['cladding'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Extracted Cladding (Irregular Shape)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cv2.cvtColor(self.extracted_regions['core'], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Extracted Core')
        axes[1, 1].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, 'results_grid.png'), dpi=150)
        plt.close()

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'test_fiber_irregular.jpg'
        if not os.path.exists(image_path):
            print("Creating a test fiber image with some irregularity...")
            size = 400
            img = np.zeros((size, size, 3), dtype=np.uint8)
            center_x, center_y = size // 2, size // 2
            
            Y, X = np.ogrid[:size, :size]
            
            # Create an irregular shape by modulating radius with angle
            angle = np.arctan2(Y - center_y, X - center_x)
            dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            core_rad = 40 * (1 + 0.05 * np.sin(3 * angle)) # Slightly triangular
            clad_rad = 90 * (1 - 0.08 * np.sin(4 * angle + 1)) # Irregular outer
            
            intensity = np.ones((size, size)) * 140
            intensity[dist < core_rad] = 160
            intensity[(dist >= core_rad) & (dist < clad_rad)] = 120
            
            img[:, :, 0] = intensity
            img[:, :, 1] = intensity
            img[:, :, 2] = intensity
            
            img = cv2.GaussianBlur(img, (5,5), 1.5).astype(np.uint8)
            noise = np.random.normal(0, 1, img.shape)
            img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite(image_path, img)
            print(f"Test image saved as {image_path}")

    try:
        analyzer = DirectionalCurvatureFiberAnalyzer(image_path)
        results = analyzer.analyze()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        core = results['regions']['core']
        cladding = results['regions']['cladding']
        
        print(f"Core -> Avg Radius: {core['avg_radius']:.2f}px (StdDev: {core['std_dev_radius']:.2f}px)")
        print(f"Cladding -> Avg Outer Radius: {cladding['avg_outer_radius']:.2f}px, Avg Thickness: {cladding['avg_thickness']:.2f}px")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()