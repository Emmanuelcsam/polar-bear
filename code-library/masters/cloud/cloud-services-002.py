import json
import time
import os
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import base64
from PIL import Image
import io
import numpy as np

class PixelAPIHandler(BaseHTTPRequestHandler):
    """HTTP API handler for pixel processing"""
    
    def do_GET(self):
        """Handle GET requests"""
        
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            status = {
                'status': 'running',
                'modules_available': self.get_available_modules(),
                'timestamp': time.time()
            }
            
            self.wfile.write(json.dumps(status).encode())
            
        elif parsed_path.path == '/results':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            results = self.collect_results()
            self.wfile.write(json.dumps(results).encode())
            
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        if self.path == '/analyze':
            try:
                # Parse JSON request
                request_data = json.loads(post_data.decode())
                
                # Process based on request type
                if request_data.get('type') == 'pixels':
                    result = self.analyze_pixels(request_data.get('data', []))
                elif request_data.get('type') == 'image':
                    result = self.analyze_image(request_data.get('data', ''))
                else:
                    result = {'error': 'Unknown analysis type'}
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            except Exception as e:
                self.send_error(400, f"Bad Request: {str(e)}")
                
        elif self.path == '/trigger':
            try:
                request_data = json.loads(post_data.decode())
                module = request_data.get('module', '')
                
                result = self.trigger_module(module)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            except Exception as e:
                self.send_error(400, f"Bad Request: {str(e)}")
                
        else:
            self.send_error(404, "Not Found")
    
    def get_available_modules(self):
        """Get list of available modules"""
        modules = []
        
        module_patterns = [
            'pixel_reader.py',
            'pattern_recognizer.py',
            'anomaly_detector.py',
            'neural_learner.py',
            'vision_processor.py',
            'gpu_accelerator.py',
            'ml_classifier.py'
        ]
        
        for module in module_patterns:
            if os.path.exists(module):
                modules.append(module.replace('.py', ''))
        
        return modules
    
    def collect_results(self):
        """Collect results from various JSON files"""
        results = {}
        
        result_files = [
            'pixel_data.json',
            'patterns.json',
            'anomalies.json',
            'neural_results.json',
            'vision_results.json',
            'gpu_results.json',
            'ml_report.json'
        ]
        
        for file in result_files:
            if os.path.exists(file):
                try:
                    with open(file, 'r') as f:
                        results[file.replace('.json', '')] = json.load(f)
                except:
                    pass
        
        return results
    
    def analyze_pixels(self, pixels):
        """Analyze pixel array"""
        
        if not pixels:
            return {'error': 'No pixel data provided'}
        
        # Save pixels for processing
        pixel_data = {
            'pixels': pixels,
            'size': [int(np.sqrt(len(pixels)))] * 2,
            'timestamp': time.time(),
            'source': 'network_api'
        }
        
        with open('pixel_data.json', 'w') as f:
            json.dump(pixel_data, f)
        
        # Basic analysis
        pixels_array = np.array(pixels)
        
        return {
            'received': len(pixels),
            'statistics': {
                'mean': float(np.mean(pixels_array)),
                'std': float(np.std(pixels_array)),
                'min': int(np.min(pixels_array)),
                'max': int(np.max(pixels_array))
            },
            'message': 'Data saved for processing'
        }
    
    def analyze_image(self, image_data):
        """Analyze base64 encoded image"""
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            
            # Save image
            image.save('network_received.jpg')
            
            # Extract pixels
            pixels = list(image.getdata())
            
            # Save as pixel data
            pixel_data = {
                'pixels': pixels,
                'size': image.size,
                'timestamp': time.time(),
                'source': 'network_api_image'
            }
            
            with open('pixel_data.json', 'w') as f:
                json.dump(pixel_data, f)
            
            return {
                'received': 'image',
                'size': image.size,
                'pixels': len(pixels),
                'saved_as': 'network_received.jpg'
            }
            
        except Exception as e:
            return {'error': f'Failed to process image: {str(e)}'}
    
    def trigger_module(self, module_name):
        """Trigger a specific module"""
        
        module_file = f"{module_name}.py"
        
        if not os.path.exists(module_file):
            return {'error': f'Module {module_name} not found'}
        
        # Run module in background
        def run_module():
            os.system(f"{os.sys.executable} {module_file}")
        
        thread = threading.Thread(target=run_module)
        thread.daemon = True
        thread.start()
        
        return {
            'triggered': module_name,
            'status': 'started',
            'timestamp': time.time()
        }
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        print(f"[NET_API] {self.address_string()} - {format % args}")

def start_api_server(port=8080):
    """Start the API server"""
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, PixelAPIHandler)
    
    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"[NET_API] Starting API server on port {port}")
    print(f"[NET_API] Local access: http://localhost:{port}")
    print(f"[NET_API] Network access: http://{local_ip}:{port}")
    print("[NET_API] Endpoints:")
    print("  GET  /status      - Get system status")
    print("  GET  /results     - Get all results")
    print("  POST /analyze     - Analyze pixel data or image")
    print("  POST /trigger     - Trigger a module")
    print("\n[NET_API] Press Ctrl+C to stop")
    
    # Save server info
    server_info = {
        'host': local_ip,
        'port': port,
        'started': time.time(),
        'endpoints': [
            'GET /status',
            'GET /results',
            'POST /analyze',
            'POST /trigger'
        ]
    }
    
    with open('network_api_info.json', 'w') as f:
        json.dump(server_info, f)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[NET_API] Shutting down server...")
        httpd.shutdown()

def test_api_client():
    """Test client for the API"""
    
    import requests
    
    base_url = "http://localhost:8080"
    
    print("[NET_API] Testing API client...")
    
    try:
        # Test status endpoint
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            print(f"[NET_API] Status: {response.json()}")
        
        # Test pixel analysis
        test_pixels = np.random.randint(0, 256, 100).tolist()
        response = requests.post(
            f"{base_url}/analyze",
            json={'type': 'pixels', 'data': test_pixels}
        )
        
        if response.status_code == 200:
            print(f"[NET_API] Analysis result: {response.json()}")
        
        # Test module trigger
        response = requests.post(
            f"{base_url}/trigger",
            json={'module': 'pattern_recognizer'}
        )
        
        if response.status_code == 200:
            print(f"[NET_API] Trigger result: {response.json()}")
            
    except Exception as e:
        print(f"[NET_API] Client test error: {e}")
        print("[NET_API] Make sure the server is running")

def create_remote_processor():
    """Create a remote processing client"""
    
    print("\n[NET_API] Remote Processor")
    print("This allows remote machines to submit data for processing")
    
    config = {
        'server_url': 'http://localhost:8080',
        'check_interval': 5,
        'auto_process': True
    }
    
    with open('remote_config.json', 'w') as f:
        json.dump(config, f)
    
    print("[NET_API] Remote configuration saved to remote_config.json")
    print("[NET_API] Edit server_url to point to the actual server")
    
    # Example remote submission script
    example_script = '''#!/usr/bin/env python3
import requests
import json
import base64
from PIL import Image
import io

# Load configuration
with open('remote_config.json', 'r') as f:
    config = json.load(f)

server_url = config['server_url']

# Submit an image
img = Image.open('test.jpg')
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
img_base64 = base64.b64encode(buffer.getvalue()).decode()

response = requests.post(
    f"{server_url}/analyze",
    json={'type': 'image', 'data': img_base64}
)

print(f"Result: {response.json()}")

# Trigger processing
response = requests.post(
    f"{server_url}/trigger",
    json={'module': 'pattern_recognizer'}
)

print(f"Processing: {response.json()}")

# Get results
response = requests.get(f"{server_url}/results")
results = response.json()

print(f"Found {len(results)} result files")
'''
    
    with open('remote_submit.py', 'w') as f:
        f.write(example_script)
    
    print("[NET_API] Example remote submission script saved to remote_submit.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'client':
        test_api_client()
    elif len(sys.argv) > 1 and sys.argv[1] == 'remote':
        create_remote_processor()
    else:
        # Start server
        port = 8080
        if len(sys.argv) > 1:
            try:
                port = int(sys.argv[1])
            except:
                pass
        
        start_api_server(port)