"""
Real-time Location-Aware Fiber Inspection Pipeline
Streams defect data with physical coordinates to WebSocket/REST endpoints
"""

import cv2
import numpy as np
import json
import time
import threading
import asyncio
import websockets
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import serial
import pynmea2
import shared_config # Import the shared configuration module

# Import our AI detectors
from ai_segmenter_pytorch import AI_Segmenter
from anomaly_detector_pytorch import AI_AnomalyDetector
from do2mr_lei_detector import UnifiedDefectDetector

class CalibrationManager:
    """Manages coordinate transformations: pixels -> microns -> mm -> GPS"""
    
    def __init__(self, config: Dict):
        self.pixels_per_micron = config.get('pixels_per_micron', 0.65)
        self.origin_xy_pixels = config.get('origin_xy_pixels', [0, 0])
        self.physical_origin_mm = config.get('physical_origin_mm', [0, 0, 0])
        
    def px_to_world(self, xy_px: Tuple[int, int]) -> Tuple[List[float], List[float]]:
        """Convert pixel coordinates to microns and mm"""
        um = np.array(xy_px) / self.pixels_per_micron
        mm = um / 1000.0 + self.physical_origin_mm[:2]
        return um.tolist(), mm.tolist()
    
    def px_to_microns(self, value_px: float) -> float:
        """Convert pixel distance to microns"""
        return value_px / self.pixels_per_micron

class GPSReader:
    """Reads GPS data from serial NMEA device"""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baud: int = 9600):
        self.port = port
        self.baud = baud
        self._fix = None
        self._running = False
        self._thread = None
        
        try:
            self.serial = serial.Serial(port, baud, timeout=1)
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except:
            print(f"GPS not available on {port}")
            self.serial = None
    
    def _loop(self):
        """Background thread reading GPS data"""
        while self._running and self.serial:
            try:
                line = self.serial.readline().decode(errors='ignore')
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    msg = pynmea2.parse(line)
                    if msg.latitude and msg.longitude:
                        self._fix = (msg.longitude, msg.latitude)
            except:
                pass
    
    def last_fix(self) -> Optional[Tuple[float, float]]:
        """Get last GPS fix (lon, lat)"""
        return self._fix
    
    def is_ready(self) -> bool:
        """Check if GPS has a fix"""
        return self._fix is not None
    
    def close(self):
        """Clean up GPS reader"""
        self._running = False
        if self.serial:
            self.serial.close()

class RealtimeFiberAnalyzer:
    """Main real-time analysis engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.calibration = CalibrationManager(config.get('calibration', {}))
        
        # Initialize detectors based on config
        self.use_ai = config.get('use_ai_models', True)
        if self.use_ai:
            self.segmenter = AI_Segmenter("segmenter_best.pth") if Path("segmenter_best.pth").exists() else None
            self.anomaly_detector = AI_AnomalyDetector("cae_last.pth") if Path("cae_last.pth").exists() else None
        else:
            self.segmenter = None
            self.anomaly_detector = None
        
        # Classical detector as fallback
        self.classical_detector = UnifiedDefectDetector()
        
        # GPS support
        if config.get('calibration', {}).get('gps_enabled', False):
            gps_device = config.get('calibration', {}).get('gps_device', '/dev/ttyUSB0')
            self.gps = GPSReader(gps_device)
        else:
            self.gps = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame and return structured results
        
        Returns:
            Dictionary with defects, coordinates, and metadata
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Calculate FPS
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.last_fps_time
            self.fps = 30 / elapsed
            self.last_fps_time = time.time()
        
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        
        # Segmentation
        if self.segmenter:
            masks = self.segmenter.segment(frame)
            center, radii = self._extract_geometry(masks)
        else:
            # Fallback to Hough circles
            masks, center, radii = self._classical_segmentation(gray)
        
        # Defect detection
        if self.anomaly_detector and masks:
            fiber_mask = masks['core'] | masks['cladding']
            score_map, ai_defects = self.anomaly_detector.detect(frame, fiber_mask)
            defects = self._convert_ai_defects(ai_defects, masks)
        else:
            # Classical detection
            results = self.classical_detector.detect_all(gray, masks)
            defects = results['defects']
        
        # Add physical coordinates to each defect
        for defect in defects:
            # Pixel to physical coordinates
            px_coords = defect.get('centroid', [0, 0])
            um_coords, mm_coords = self.calibration.px_to_world(px_coords)
            
            defect['coordinates'] = {
                'pixels': px_coords,
                'microns': um_coords,
                'mm': mm_coords + [0],  # Add Z=0
            }
            
            # Add GPS if available
            if self.gps and self.gps.is_ready():
                defect['coordinates']['gps'] = list(self.gps.last_fix())
            
            # Convert size to microns
            if 'area' in defect:
                defect['area_um2'] = self.calibration.px_to_microns(defect['area'])
        
        # Quality assessment
        quality_score, pass_fail = self._assess_quality(defects, masks)
        
        # Processing time
        process_time = time.time() - start_time
        
        return {
            'utc': datetime.utcnow().isoformat() + 'Z',
            'frame_id': self.frame_count,
            'defects': defects,
            'quality_score': quality_score,
            'pass': pass_fail,
            'geometry': {
                'center': center,
                'radii': radii
            },
            'performance': {
                'fps': round(self.fps, 1),
                'process_time_ms': round(process_time * 1000, 1)
            }
        }
    
    def _extract_geometry(self, masks: Dict[str, np.ndarray]) -> Tuple[Tuple[int, int], Dict[str, float]]:
        """Extract center and radii from masks"""
        # Core center and radius
        ys, xs = np.where(masks['core'] > 0)
        if len(xs) > 10:
            cx, cy = int(xs.mean()), int(ys.mean())
            core_r = np.sqrt(((xs - cx)**2 + (ys - cy)**2).mean())
        else:
            cx = cy = core_r = 0
        
        # Cladding radius
        ys2, xs2 = np.where(masks['cladding'] > 0)
        if len(xs2) > 10 and cx > 0:
            clad_r = np.sqrt(((xs2 - cx)**2 + (ys2 - cy)**2).mean())
        else:
            clad_r = 0
        
        return (cx, cy), {'core': core_r, 'cladding': clad_r}
    
    def _classical_segmentation(self, gray: np.ndarray) -> Tuple[Dict, Tuple[int, int], Dict]:
        """Fallback segmentation using Hough circles"""
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 100,
            param1=50, param2=30, minRadius=50, maxRadius=200
        )
        
        if circles is None:
            return None, (0, 0), {}
        
        circles = np.uint16(np.around(circles))
        cx, cy, r_clad = circles[0, 0]
        r_core = r_clad // 5  # Approximate
        
        # Create masks
        h, w = gray.shape
        masks = {
            'core': np.zeros((h, w), dtype=np.uint8),
            'cladding': np.zeros((h, w), dtype=np.uint8),
            'ferrule': np.zeros((h, w), dtype=np.uint8)
        }
        cv2.circle(masks['core'], (cx, cy), r_core, 255, -1)
        cv2.circle(masks['cladding'], (cx, cy), r_clad, 255, -1)
        cv2.circle(masks['cladding'], (cx, cy), r_core, 0, -1)  # Remove core
        masks['ferrule'] = 255 - masks['core'] - masks['cladding']
        
        return masks, (cx, cy), {'core': float(r_core), 'cladding': float(r_clad)}
    
    def _convert_ai_defects(self, ai_defects: List[Dict], masks: Dict) -> List[Dict]:
        """Convert AI detector format to standard format"""
        converted = []
        for d in ai_defects:
            x, y, w, h = d['bbox']
            cx, cy = x + w//2, y + h//2
            
            # Determine zone
            zone = "FERRULE"
            if masks['core'][cy, cx] > 0:
                zone = "CORE"
            elif masks['cladding'][cy, cx] > 0:
                zone = "CLADDING"
            
            # Determine severity
            if d['confidence'] > 0.8 or d['area_px'] > 100:
                severity = "HIGH"
            elif d['confidence'] > 0.5:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            converted.append({
                'id': d['defect_id'],
                'type': 'ANOMALY',  # AI doesn't classify type
                'bbox': d['bbox'],
                'centroid': [cx, cy],
                'area': d['area_px'],
                'zone': zone,
                'severity': severity,
                'confidence': d['confidence']
            })
        
        return converted
    
    def _assess_quality(self, defects: List[Dict], masks: Optional[Dict]) -> Tuple[float, bool]:
        """Calculate quality score and pass/fail"""
        if not defects:
            return 100.0, True
        
        # Calculate weighted severity score
        severity_weights = {'LOW': 1, 'MEDIUM': 5, 'HIGH': 10}
        zone_weights = {'CORE': 3, 'CLADDING': 2, 'FERRULE': 1}
        
        total_penalty = 0
        for d in defects:
            severity = severity_weights.get(d.get('severity', 'LOW'), 1)
            zone = zone_weights.get(d.get('zone', 'FERRULE'), 1)
            total_penalty += severity * zone
        
        # Convert to 0-100 score
        quality_score = max(0, 100 - total_penalty)
        
        # Pass/fail based on critical defects
        has_core_defect = any(d['zone'] == 'CORE' for d in defects)
        pass_fail = quality_score >= 70 and not has_core_defect
        
        return quality_score, pass_fail

class RealtimeStreamer:
    """Handles WebSocket streaming of results"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.latest_result = None
        
    async def handler(self, websocket, path):
        """Handle new WebSocket connections"""
        self.clients.add(websocket)
        try:
            # Send latest result immediately
            if self.latest_result:
                await websocket.send(json.dumps(self.latest_result))
            
            # Keep connection alive
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    async def broadcast(self, data: Dict):
        """Broadcast data to all connected clients"""
        self.latest_result = data
        if self.clients:
            message = json.dumps(data)
            # Send to all clients concurrently
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def start_server(self):
        """Start WebSocket server"""
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

class RealtimePipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str = None):
        # Load configuration from shared_config.py
        self.current_config = shared_config.get_config()
        
        # Initialize analyzer with parameters from shared_config
        self.analyzer = RealtimeFiberAnalyzer(self.current_config)
        
        # Initialize streamer with parameters from shared_config
        stream_host = self.current_config.get('websocket_host', '0.0.0.0')
        stream_port = self.current_config.get('websocket_port', 8765)
        self.streamer = RealtimeStreamer(host=stream_host, port=stream_port)
        
        # Video source from shared_config
        self.source = self.current_config.get('video_source', 0) # Default to 0 (camera index)
        self.max_fps = self.current_config.get('max_fps', 10)
        self.display = self.current_config.get('display_output', True)
        
        self.status = "initialized" # Add a status variable

    def get_script_info(self):
        """Returns information about the script, its status, and exposed parameters."""
        return {
            "name": "Location Tracking Pipeline",
            "status": self.status,
            "parameters": {
                "video_source": self.source,
                "max_fps": self.max_fps,
                "display_output": self.display,
                "websocket_host": self.streamer.host,
                "websocket_port": self.streamer.port,
                "log_level": self.current_config.get("log_level"),
                "data_source": self.current_config.get("data_source"),
                "processing_enabled": self.current_config.get("processing_enabled"),
                "threshold_value": self.current_config.get("threshold_value")
            },
            "analyzer_info": self.analyzer.get_script_info() if hasattr(self.analyzer, 'get_script_info') else "N/A"
        }

    def set_script_parameter(self, key, value):
        """Sets a specific parameter for the script and updates shared_config."""
        if key in self.current_config:
            self.current_config[key] = value
            shared_config.set_config_value(key, value) # Update shared config
            
            # Apply changes if they affect the running script
            if key == "video_source":
                self.source = value
                # Note: Re-initializing video capture mid-run is complex, might require restart
            elif key == "max_fps":
                self.max_fps = value
            elif key == "display_output":
                self.display = value
            elif key == "websocket_host":
                self.streamer.host = value
            elif key == "websocket_port":
                self.streamer.port = value
            # Add more conditions here for other parameters that need immediate effect
            
            self.status = f"parameter '{key}' updated"
            return True
        return False
        
    def run(self):
        """Run the pipeline"""
        # Start WebSocket server in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        server_task = loop.create_task(self.streamer.start_server())
        
        # Start video capture in separate thread
        capture_thread = threading.Thread(target=self._capture_loop, args=(loop,))
        capture_thread.start()
        
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if self.analyzer.gps:
                self.analyzer.gps.close()
    
    def _capture_loop(self, loop):
        """Video capture and analysis loop"""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Failed to open video source: {self.source}")
            return
        
        frame_interval = 1.0 / self.max_fps
        last_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Throttle to max FPS
                current_time = time.time()
                if current_time - last_time < frame_interval:
                    continue
                last_time = current_time
                
                # Analyze frame
                results = self.analyzer.analyze_frame(frame)
                
                # Broadcast results
                asyncio.run_coroutine_threadsafe(
                    self.streamer.broadcast(results),
                    loop
                )
                
                # Display if enabled
                if self.display:
                    vis_frame = self._visualize(frame, results)
                    cv2.imshow('Fiber Inspection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            loop.call_soon_threadsafe(loop.stop)
    
    def _visualize(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Add overlays to frame"""
        vis = frame.copy()
        
        # Draw fiber geometry
        if results['geometry']['center'][0] > 0:
            cx, cy = results['geometry']['center']
            cv2.circle(vis, (cx, cy), int(results['geometry']['radii']['core']), 
                      (0, 255, 255), 2)
            cv2.circle(vis, (cx, cy), int(results['geometry']['radii']['cladding']), 
                      (255, 0, 255), 2)
        
        # Draw defects
        for d in results['defects']:
            x, y, w, h = d['bbox']
            color = {
                'HIGH': (0, 0, 255),
                'MEDIUM': (0, 165, 255),
                'LOW': (0, 255, 0)
            }.get(d['severity'], (255, 255, 255))
            
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{d['type'][:3]}-{d['severity'][0]}"
            cv2.putText(vis, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add status
        status_color = (0, 255, 0) if results['pass'] else (0, 0, 255)
        status = f"PASS" if results['pass'] else f"FAIL"
        cv2.putText(vis, f"{status} Q:{results['quality_score']:.0f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Add FPS
        cv2.putText(vis, f"FPS: {results['performance']['fps']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis

# Example client HTML
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Fiber Inspection Live View</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        #defects { margin-top: 20px; }
        .defect { padding: 5px; margin: 5px; border: 1px solid #ccc; }
        .HIGH { background-color: #ffcccc; }
        .MEDIUM { background-color: #ffffcc; }
        .LOW { background-color: #ccffcc; }
    </style>
</head>
<body>
    <h1>Real-time Fiber Inspection</h1>
    <div>
        <strong>Status:</strong> <span id="status">Connecting...</span>
        <strong>Quality:</strong> <span id="quality">-</span>
        <strong>FPS:</strong> <span id="fps">-</span>
    </div>
    <div id="defects"></div>
    
    <script>
        const ws = new WebSocket('ws://localhost:8765');
        
        ws.onopen = () => {
            document.getElementById('status').textContent = 'Connected';
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update status
            document.getElementById('status').textContent = data.pass ? 'PASS' : 'FAIL';
            document.getElementById('quality').textContent = data.quality_score.toFixed(1) + '%';
            document.getElementById('fps').textContent = data.performance.fps;
            
            // Update defects
            const defectsDiv = document.getElementById('defects');
            defectsDiv.innerHTML = '<h3>Defects:</h3>';
            
            data.defects.forEach(d => {
                const div = document.createElement('div');
                div.className = 'defect ' + d.severity;
                div.innerHTML = `
                    <strong>${d.type}</strong> in ${d.zone}<br>
                    Position: (${d.coordinates.microns[0].toFixed(1)}, 
                              ${d.coordinates.microns[1].toFixed(1)}) μm<br>
                    Severity: ${d.severity}
                `;
                defectsDiv.appendChild(div);
            });
        };
        
        ws.onerror = (error) => {
            document.getElementById('status').textContent = 'Error: ' + error;
        };
        
        ws.onclose = () => {
            document.getElementById('status').textContent = 'Disconnected';
        };
    </script>
</body>
</html>
"""

pipeline_instance = None

def get_script_info():
    """Returns information about the script, its status, and exposed parameters."""
    if pipeline_instance:
        return pipeline_instance.get_script_info()
    return {"name": "Location Tracking Pipeline", "status": "not_initialized", "parameters": {}}

def set_script_parameter(key, value):
    """Sets a specific parameter for the script and updates shared_config."""
    if pipeline_instance:
        return pipeline_instance.set_script_parameter(key, value)
    return False

if __name__ == "__main__":
    # Save example client
    with open("live_view.html", "w") as f:
        f.write(HTML_CLIENT)
    
    # Run pipeline
    pipeline_instance = RealtimePipeline()
    pipeline_instance.run()