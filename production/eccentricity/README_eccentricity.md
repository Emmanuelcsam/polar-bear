# Eccentricity Tester

A comprehensive real-time image analysis tool that measures how circular objects are by combining Hough circle detection with intensity profile and gradient analysis.

## Overview

The eccentricity tester goes beyond simple circle fitting by analyzing:
- **Radial edge uniformity**: How consistent the object's edge is at different angles
- **Intensity profiles**: How uniform the intensity is from center to edge
- **Gradient analysis**: How well gradients align with expected radial directions
- **Shape metrics**: Roundness, eccentricity, and solidity measurements

## Components

### 1. `eccentricity_tester.py`
The core analysis module containing:
- `EccentricityTester`: Main class that performs multi-factor circularity analysis
- `EccentricityProcessor`: High-level processor combining Hough detection with analysis

### 2. `eccentricity_gui.py`
Full-featured GUI application with:
- Real-time video display with analysis overlay
- Live metrics dashboard
- Interactive plots showing:
  - Radial edge profile
  - Score history over time
  - Polar plot of edge variations
  - Component score breakdown
- Parameter controls for Hough circle detection
- Results export functionality

## How It Works

### Analysis Process
1. **Hough Circle Detection**: Finds circular objects in the image
2. **Radial Profile Analysis**: 
   - Samples edge positions at 360 angles
   - Calculates deviation from perfect circle
   - Detects periodicity (indicates polygons vs circles)
3. **Intensity Profile Analysis**:
   - Measures intensity variations along radial lines
   - Checks symmetry by comparing opposite angles
4. **Gradient Analysis**:
   - Examines gradient orientations in annulus around circle
   - Verifies gradients point radially (expected for circles)
5. **Shape Metrics**:
   - Uses contour analysis for roundness
   - Fits ellipse to measure eccentricity
   - Calculates solidity (convexity measure)

### Scoring System
The overall eccentricity score (0-100%) is a weighted combination of:
- Radial uniformity (25%)
- Intensity uniformity (15%)
- Intensity symmetry (15%)
- Gradient consistency (15%)
- Gradient circularity (10%)
- Shape roundness (15%)
- Shape eccentricity (5%)

## Usage

### GUI Application
```bash
python eccentricity_gui.py
```

Features:
- Load BMP/PNG/JPG images for analysis
- Adjust frame rate for emulated video
- Fine-tune Hough circle detection parameters
- View real-time analysis with visual overlays
- Monitor score history and component breakdowns
- Export results and plots

### Programmatic Usage
```python
from eccentricity_tester import EccentricityTester
import cv2

# Create tester
tester = EccentricityTester()

# Load image
image = cv2.imread('circular_object.bmp')

# Detect circle (x, y, radius)
circle = (250, 250, 100)  # Or use Hough detection

# Analyze eccentricity
results = tester.analyze_eccentricity(image, circle)

# Display results
print(f"Eccentricity Score: {results['eccentricity_score']:.1f}%")
print(f"Radial Uniformity: {results['radial_uniformity']:.3f}")
print(f"Shape Roundness: {results['shape_roundness']:.3f}")

# Visualize
output = tester.visualize_analysis(image, results)
cv2.imshow('Analysis', output)

# Show detailed plots
tester.plot_detailed_analysis(results)
```

### Integration with Existing Code
```python
from eccentricity_tester import EccentricityProcessor
from hough_circles import HoughCirclesDetector

# Create processor with custom detector
detector = HoughCirclesDetector(min_radius=10, max_radius=300)
processor = EccentricityProcessor(hough_detector=detector)

# Process video frame
processed_frame, results = processor.process_frame(frame)
```

## Interpretation of Results

### Score Ranges
- **90-100%**: Excellent circularity (nearly perfect circle)
- **75-90%**: Good circularity (minor deviations)
- **60-75%**: Fair circularity (noticeable deviations)
- **Below 60%**: Poor circularity (significant deviations)

### Key Metrics
- **Radial Deviation**: Standard deviation of edge distances (lower is better)
- **Intensity Symmetry**: Correlation between opposite sides (higher is better)
- **Gradient Consistency**: How well gradients align radially (higher is better)
- **Shape Eccentricity**: Deviation from circular to elliptical (lower is better)

### Common Issues Detected
- **Low radial uniformity**: Irregular or damaged edges
- **Poor intensity symmetry**: Uneven lighting or material defects
- **Low gradient consistency**: Surface texture irregularities
- **High eccentricity**: Elliptical rather than circular shape
- **Periodicity detected**: Object is polygonal (e.g., hexagon) rather than circular

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- SciPy
- Matplotlib
- Tkinter (for GUI)
- Pillow (PIL)

## Installation

```bash
pip install opencv-python numpy scipy matplotlib pillow
```

## Examples

### Testing with Perfect Circle
```python
# Create test image with perfect circle
test_image = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.circle(test_image, (250, 250), 100, (255, 255, 255), -1)

# Should score close to 100%
```

### Testing with Ellipse
```python
# Create elliptical shape
test_image = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.ellipse(test_image, (250, 250), (100, 80), 0, 0, 360, (255, 255, 255), -1)

# Will show high eccentricity, lower overall score
```

### Testing with Polygon
```python
# Create hexagon
center = (250, 250)
radius = 100
points = []
for i in range(6):
    angle = i * np.pi / 3
    x = int(center[0] + radius * np.cos(angle))
    y = int(center[1] + radius * np.sin(angle))
    points.append([x, y])
points = np.array(points)
cv2.fillPoly(test_image, [points], (255, 255, 255))

# Will detect periodicity, lower radial uniformity
```

## Advanced Configuration

### Adjusting Analysis Parameters
```python
tester = EccentricityTester(
    num_radial_samples=720,      # More angular resolution
    num_radius_samples=100,      # More radial resolution
    gradient_threshold=15,       # Lower threshold for gradients
    intensity_smoothing=3        # Less smoothing
)
```

### Custom Weighting
Modify the weights in `_calculate_eccentricity_score()` to emphasize different aspects:
```python
weights = {
    'radial_uniformity': 0.40,    # Emphasize edge uniformity
    'intensity_uniformity': 0.10,
    'intensity_symmetry': 0.10,
    'gradient_consistency': 0.20,
    'gradient_circularity': 0.05,
    'shape_roundness': 0.10,
    'shape_eccentricity': 0.05
}
```

## Performance Tips

1. **Optimize Hough parameters** first to ensure good circle detection
2. **Reduce sampling** for faster processing:
   - `num_radial_samples=180` (every 2 degrees)
   - `num_radius_samples=25`
3. **Adjust gradient threshold** based on image contrast
4. **Use ROI** to focus on specific image regions

## Troubleshooting

### No circles detected
- Adjust Hough parameters (especially param1 and param2)
- Check image contrast and lighting
- Verify min/max radius settings

### Low scores on circular objects
- Check for shadows or uneven lighting
- Ensure object is fully visible in frame
- Verify gradient threshold isn't too high

### Inconsistent results
- Increase blur to reduce noise effects
- Adjust intensity smoothing parameter
- Check for reflections or glare
