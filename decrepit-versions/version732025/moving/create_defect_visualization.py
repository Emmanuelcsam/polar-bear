#!/usr/bin/env python3
"""
Create defect visualization using standard library only
Generates an HTML report with defect overlays
"""

import os
import sys
from pathlib import Path
import base64
import json

def create_html_visualization():
    """Create an HTML visualization with JavaScript overlay"""
    
    base_dir = Path(__file__).parent
    test_image_path = base_dir / "test_image" / "img(303).jpg"
    report_path = base_dir / "results" / "img (303)" / "3_detected" / "img (303)" / "img (303)_detailed.txt"
    
    # Read the test image as base64
    if not test_image_path.exists():
        print(f"Error: Test image not found at {test_image_path}")
        return
    
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Parse the detailed report
    regions = []
    defect_stats = {
        'scratches': 0,
        'digs': 0,
        'blobs': 0,
        'edge_irregularities': 0,
        'total_regions': 0
    }
    
    if report_path.exists():
        with open(report_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        in_region_section = False
        current_region = None
        
        # First, let's find the SPECIFIC DEFECTS DETECTED section
        defect_section_start = -1
        for i, line in enumerate(lines):
            if "SPECIFIC DEFECTS DETECTED" in line:
                defect_section_start = i
                break
        
        for i, line in enumerate(lines):
            # Parse defect statistics (they appear after SPECIFIC DEFECTS DETECTED)
            if defect_section_start > 0 and i > defect_section_start:
                if "Scratches:" in line:
                    try:
                        defect_stats['scratches'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif "Digs:" in line:
                    try:
                        defect_stats['digs'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif "Blobs:" in line:
                    try:
                        defect_stats['blobs'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif "Edge Irregularities:" in line:
                    try:
                        defect_stats['edge_irregularities'] = int(line.split(':')[1].strip())
                    except:
                        pass
            
            if "Total Regions Found:" in line:
                try:
                    defect_stats['total_regions'] = int(line.split(':')[1].strip())
                except:
                    pass
            
            # Parse region locations
            if "LOCAL ANOMALY REGIONS" in line:
                in_region_section = True
            elif "SPECIFIC DEFECTS DETECTED" in line:
                in_region_section = False
            elif in_region_section:
                if "Region" in line and ":" in line:
                    if current_region:
                        regions.append(current_region)
                    current_region = {}
                elif current_region is not None:
                    if "Location:" in line and "(" in line:
                        coords_str = line.split("(")[1].split(")")[0]
                        coords = [int(x.strip()) for x in coords_str.split(",")]
                        current_region['x'], current_region['y'], current_region['w'], current_region['h'] = coords
                    elif "Area:" in line:
                        current_region['area'] = int(line.split(':')[1].split()[0])
                    elif "Confidence:" in line:
                        current_region['confidence'] = float(line.split(':')[1].strip())
                    elif "Centroid:" in line:
                        cent_str = line.split("(")[1].split(")")[0]
                        current_region['centroid'] = [int(x.strip()) for x in cent_str.split(",")]
        
        if current_region:
            regions.append(current_region)
    
    # Create HTML with JavaScript visualization
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Defect Visualization - img(303).jpg</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
        .stat-box.fail {{
            border-left-color: #dc3545;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        #canvas-container {{
            position: relative;
            margin: 20px auto;
            text-align: center;
        }}
        canvas {{
            border: 1px solid #ddd;
            max-width: 100%;
            height: auto;
        }}
        .controls {{
            margin: 20px 0;
            text-align: center;
        }}
        button {{
            padding: 10px 20px;
            margin: 0 5px;
            border: none;
            border-radius: 5px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
            font-size: 16px;
        }}
        button:hover {{
            background-color: #0056b3;
        }}
        button.active {{
            background-color: #28a745;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f8f8;
            border-radius: 5px;
        }}
        .legend-item {{
            display: inline-block;
            margin: 0 15px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            vertical-align: middle;
            margin-right: 5px;
            border: 1px solid #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fiber Optic Defect Detection - img(303).jpg</h1>
        
        <div class="stats">
            <div class="stat-box fail">
                <div class="stat-label">Status</div>
                <div class="stat-value">FAIL</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Defects</div>
                <div class="stat-value">{defect_stats['scratches'] + defect_stats['digs'] + defect_stats['blobs'] + defect_stats['edge_irregularities']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Scratches</div>
                <div class="stat-value">{defect_stats['scratches']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Digs</div>
                <div class="stat-value">{defect_stats['digs']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Blobs</div>
                <div class="stat-value">{defect_stats['blobs']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Edge Irregularities</div>
                <div class="stat-value">{defect_stats['edge_irregularities']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Anomaly Regions</div>
                <div class="stat-value">{defect_stats['total_regions']}</div>
            </div>
        </div>
        
        <div class="controls">
            <button id="toggleBtn" onclick="toggleOverlay()">Show/Hide Defects</button>
            <button onclick="showAllRegions()">Show All Regions</button>
            <button onclick="showTop10()">Show Top 10</button>
            <button onclick="clearOverlay()">Clear Overlay</button>
        </div>
        
        <div class="legend">
            <strong>Legend:</strong>
            <span class="legend-item">
                <span class="legend-color" style="background-color: rgba(255, 0, 0, 0.3);"></span>
                High Confidence (>0.8)
            </span>
            <span class="legend-item">
                <span class="legend-color" style="background-color: rgba(255, 165, 0, 0.3);"></span>
                Medium Confidence (0.6-0.8)
            </span>
            <span class="legend-item">
                <span class="legend-color" style="background-color: rgba(255, 255, 0, 0.3);"></span>
                Low Confidence (<0.6)
            </span>
        </div>
        
        <div id="canvas-container">
            <canvas id="imageCanvas"></canvas>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background-color: #f8f8f8; border-radius: 5px;">
            <h3>Analysis Details</h3>
            <p><strong>Image Path:</strong> test_image/img(303).jpg</p>
            <p><strong>Dimensions:</strong> 864 x 1152 pixels</p>
            <p><strong>Analysis Date:</strong> 2025-07-02_15:51:57</p>
            <p><strong>Confidence:</strong> 100.0%</p>
            <p><strong>Recommendation:</strong> This fiber optic cable shows significant defects and should not pass quality control.</p>
        </div>
    </div>
    
    <script>
        const imageData = 'data:image/jpeg;base64,{image_base64}';
        const regions = {json.dumps(regions)};
        let showOverlay = true;
        let maxRegions = 10;
        
        const canvas = document.getElementById('imageCanvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = function() {{
            canvas.width = img.width;
            canvas.height = img.height;
            drawImage();
        }};
        
        img.src = imageData;
        
        function drawImage() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            
            if (showOverlay) {{
                drawDefects();
            }}
        }}
        
        function drawDefects() {{
            const regionsToShow = regions.slice(0, maxRegions);
            
            regionsToShow.forEach((region, index) => {{
                if (region.x !== undefined) {{
                    // Determine color based on confidence
                    let color;
                    if (region.confidence > 0.8) {{
                        color = 'rgba(255, 0, 0, 0.3)'; // Red
                    }} else if (region.confidence > 0.6) {{
                        color = 'rgba(255, 165, 0, 0.3)'; // Orange
                    }} else {{
                        color = 'rgba(255, 255, 0, 0.3)'; // Yellow
                    }}
                    
                    // Draw filled rectangle
                    ctx.fillStyle = color;
                    ctx.fillRect(region.x, region.y, region.w, region.h);
                    
                    // Draw border
                    ctx.strokeStyle = color.replace('0.3', '1');
                    ctx.lineWidth = 2;
                    ctx.strokeRect(region.x, region.y, region.w, region.h);
                    
                    // Draw label
                    ctx.fillStyle = 'red';
                    ctx.font = '14px Arial';
                    ctx.fillText(`#${{index + 1}} (${{region.confidence.toFixed(2)}})`, region.x, region.y - 5);
                }}
            }});
            
            // Draw summary text
            ctx.fillStyle = 'white';
            ctx.fillRect(10, 10, 250, 180);
            ctx.strokeStyle = 'black';
            ctx.strokeRect(10, 10, 250, 180);
            
            ctx.fillStyle = 'black';
            ctx.font = 'bold 16px Arial';
            ctx.fillText('Defect Summary', 20, 30);
            
            ctx.font = '14px Arial';
            ctx.fillText(`Total Defects: {defect_stats['scratches'] + defect_stats['digs'] + defect_stats['blobs'] + defect_stats['edge_irregularities']}`, 20, 55);
            ctx.fillText(`Scratches: {defect_stats['scratches']}`, 20, 75);
            ctx.fillText(`Digs: {defect_stats['digs']}`, 20, 95);
            ctx.fillText(`Blobs: {defect_stats['blobs']}`, 20, 115);
            ctx.fillText(`Edge Irregularities: {defect_stats['edge_irregularities']}`, 20, 135);
            ctx.fillText(`Regions shown: ${{Math.min(maxRegions, regions.length)}}/${{regions.length}}`, 20, 155);
            
            ctx.fillStyle = 'red';
            ctx.font = 'bold 14px Arial';
            ctx.fillText('STATUS: FAIL', 20, 175);
        }}
        
        function toggleOverlay() {{
            showOverlay = !showOverlay;
            drawImage();
        }}
        
        function showAllRegions() {{
            maxRegions = regions.length;
            showOverlay = true;
            drawImage();
        }}
        
        function showTop10() {{
            maxRegions = 10;
            showOverlay = true;
            drawImage();
        }}
        
        function clearOverlay() {{
            showOverlay = false;
            drawImage();
        }}
    </script>
</body>
</html>"""
    
    # Save HTML file
    output_path = base_dir / "img303_defect_visualization.html"
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ HTML visualization created: {output_path}")
    print(f"  Open this file in a web browser to see the interactive defect overlay")
    print(f"\n  Total defects found: {defect_stats['scratches'] + defect_stats['digs'] + defect_stats['blobs'] + defect_stats['edge_irregularities']}")
    print(f"  - Scratches: {defect_stats['scratches']}")
    print(f"  - Digs: {defect_stats['digs']}")
    print(f"  - Blobs: {defect_stats['blobs']}")
    print(f"  - Edge irregularities: {defect_stats['edge_irregularities']}")
    print(f"  - Total anomaly regions: {defect_stats['total_regions']}")
    
    # Also create a simple text-based visualization
    create_text_visualization()

def create_text_visualization():
    """Create a simple text-based visualization"""
    
    base_dir = Path(__file__).parent
    report_path = base_dir / "results" / "img (303)" / "3_detected" / "img (303)" / "img (303)_detailed.txt"
    
    # Create ASCII art representation
    width = 60
    height = 40
    grid = [['.' for _ in range(width)] for _ in range(height)]
    
    # Parse regions and map to grid
    if report_path.exists():
        with open(report_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        regions = []
        
        for line in lines:
            if "Location:" in line and "(" in line:
                try:
                    coords_str = line.split("(")[1].split(")")[0]
                    coords = [int(x.strip()) for x in coords_str.split(",")]
                    if len(coords) == 4:
                        regions.append(coords)
                except:
                    continue
        
        # Map regions to ASCII grid
        for x, y, w, h in regions[:20]:  # Show first 20 regions
            # Scale to grid size
            gx = int(x * width / 1152)
            gy = int(y * height / 864)
            gw = max(1, int(w * width / 1152))
            gh = max(1, int(h * height / 864))
            
            # Mark defect regions
            for dy in range(gh):
                for dx in range(gw):
                    if 0 <= gy + dy < height and 0 <= gx + dx < width:
                        grid[gy + dy][gx + dx] = '#'
    
    # Save text visualization
    output_path = base_dir / "img303_defect_map.txt"
    with open(output_path, 'w') as f:
        f.write("DEFECT MAP - img(303).jpg\n")
        f.write("="*60 + "\n")
        f.write("Legend: . = Normal area, # = Defect area\n")
        f.write("="*60 + "\n\n")
        
        for row in grid:
            f.write(''.join(row) + '\n')
        
        f.write("\n" + "="*60 + "\n")
        f.write("SUMMARY: 99 defects detected across 94 regions\n")
        f.write("STATUS: FAIL - Fiber optic cable has significant defects\n")
    
    print(f"\n✓ Text visualization created: {output_path}")

if __name__ == "__main__":
    print("="*60)
    print("CREATING DEFECT VISUALIZATION")
    print("="*60)
    print("\nThis script creates visualizations without external dependencies.")
    
    create_html_visualization()