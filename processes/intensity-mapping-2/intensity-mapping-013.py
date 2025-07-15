#!/usr/bin/env python3
"""
Enhanced Intensity Reader with connector integration support
Can run independently or be controlled through connectors
"""
import sys
from PIL import Image
import json
import socket

# Configuration for connector communication
CONNECTOR_PORT = 10089

class IntensityReader:
    def __init__(self):
        self.last_results = []
        self.threshold = None
        self.connected = False
        
    def read_intensity(self, path, threshold=None, callback=None):
        """Read intensity values from an image"""
        if callback is None:
            callback = self.default_callback
            
        try:
            data = Image.open(path).convert("L").getdata()
            self.last_results = []
            
            for p in data:
                if threshold is None or p >= threshold:
                    callback(p)
                    self.last_results.append(p)
                    
            return self.last_results
        except Exception as e:
            print(f"Error reading image: {e}", file=sys.stderr)
            return []
            
    def default_callback(self, value):
        """Default callback to print values"""
        print(value)
        
    def get_statistics(self):
        """Get statistics from last read"""
        if not self.last_results:
            return None
            
        return {
            'count': len(self.last_results),
            'min': min(self.last_results),
            'max': max(self.last_results),
            'avg': sum(self.last_results) / len(self.last_results)
        }
        
    def notify_connector(self, message):
        """Send notification to connector if available"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect(('localhost', CONNECTOR_PORT))
            
            notification = {
                'source': 'intensity_reader',
                'type': 'notification',
                'message': message
            }
            sock.send(json.dumps(notification).encode())
            response = sock.recv(1024).decode()
            sock.close()
            return json.loads(response)
        except:
            # Connector not available, continue standalone
            return None

# Global instance for module-level functions
_reader = IntensityReader()

def read_intensity(path, threshold=None, callback=print):
    """Module-level function for backward compatibility"""
    return _reader.read_intensity(path, threshold, callback)
    
def get_statistics():
    """Get statistics from last read"""
    return _reader.get_statistics()

if __name__ == "__main__":
    v = sys.argv
    if len(v) < 2:
        print("Usage: intensity_reader.py <image_path> [threshold]")
        sys.exit(1)
        
    path = v[1]
    threshold = int(v[2]) if len(v) > 2 else None
    
    # Try to notify connector we're starting
    _reader.notify_connector(f"Starting intensity read on {path}")
    
    # Read intensity
    results = read_intensity(path, threshold)
    
    # Show statistics
    stats = get_statistics()
    if stats:
        print(f"\nStatistics: {json.dumps(stats, indent=2)}")
        
    # Notify connector we're done
    _reader.notify_connector(f"Completed intensity read: {stats}")