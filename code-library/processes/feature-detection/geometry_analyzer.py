import json
import numpy as np
from PIL import Image

def analyze_geometry():
    try:
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            pixels = data['pixels']
            size = data['size']
        
        # Reshape to 2D
        img_array = np.array(pixels).reshape(size[1], size[0])
        
        # Edge detection (simple gradient)
        horizontal_edges = np.abs(np.diff(img_array, axis=1))
        vertical_edges = np.abs(np.diff(img_array, axis=0))
        
        # Find lines (high gradient areas)
        h_lines = np.where(horizontal_edges > 50)
        v_lines = np.where(vertical_edges > 50)
        
        print(f"[GEOMETRY] Found {len(h_lines[0])} horizontal edge pixels")
        print(f"[GEOMETRY] Found {len(v_lines[0])} vertical edge pixels")
        
        # Detect patterns in rows/columns
        row_patterns = []
        for i, row in enumerate(img_array):
            if len(np.unique(row)) < 5:  # Low variation
                row_patterns.append({
                    'row': int(i),
                    'pattern': 'uniform',
                    'values': [int(v) for v in np.unique(row)]
                })
        
        col_patterns = []
        for j in range(size[0]):
            col = img_array[:, j]
            if np.std(col) < 10:  # Low variation
                col_patterns.append({
                    'col': int(j),
                    'pattern': 'uniform',
                    'std': float(np.std(col))
                })
        
        # Check for symmetry
        h_symmetry = np.allclose(img_array, np.fliplr(img_array), atol=20)
        v_symmetry = np.allclose(img_array, np.flipud(img_array), atol=20)
        
        geometry = {
            'edge_count': {
                'horizontal': len(h_lines[0]),
                'vertical': len(v_lines[0])
            },
            'row_patterns': row_patterns[:10],
            'col_patterns': col_patterns[:10],
            'symmetry': {
                'horizontal': bool(h_symmetry),
                'vertical': bool(v_symmetry)
            },
            'gradients': {
                'mean_h_gradient': float(np.mean(horizontal_edges)),
                'mean_v_gradient': float(np.mean(vertical_edges))
            }
        }
        
        with open('geometry_analysis.json', 'w') as f:
            json.dump(geometry, f)
        
        print(f"[GEOMETRY] Analysis complete")
        
    except Exception as e:
        print(f"[GEOMETRY] Error: {e}")

if __name__ == "__main__":
    analyze_geometry()