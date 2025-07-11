#!/usr/bin/env python3
"""
Enhanced Random Pixel Generator with connector integration support
Can run independently or be controlled through connectors
"""
import random
import time
import sys
import json
import socket
import threading

# Configuration
CONNECTOR_PORT = 10089

class PixelGenerator:
    def __init__(self):
        self.min_val = 0
        self.max_val = 255
        self.delay = 0
        self.running = False
        self.generated_count = 0
        self.last_values = []
        self.max_history = 100
        self.callbacks = []
        
    def set_parameters(self, min_val=None, max_val=None, delay=None):
        """Set generator parameters"""
        if min_val is not None:
            self.min_val = min_val
        if max_val is not None:
            self.max_val = max_val
        if delay is not None:
            self.delay = delay
            
    def get_parameters(self):
        """Get current parameters"""
        return {
            'min_val': self.min_val,
            'max_val': self.max_val,
            'delay': self.delay,
            'running': self.running,
            'generated_count': self.generated_count
        }
        
    def generate_single(self):
        """Generate a single pixel value"""
        value = random.randint(self.min_val, self.max_val)
        self.generated_count += 1
        
        # Keep history
        self.last_values.append(value)
        if len(self.last_values) > self.max_history:
            self.last_values.pop(0)
            
        return value
        
    def generate(self, min_val=None, max_val=None, delay=None, callback=None, count=None):
        """Generate pixel values continuously or for a specific count"""
        # Update parameters if provided
        self.set_parameters(min_val, max_val, delay)
        
        if callback is None:
            callback = print
            
        self.running = True
        generated = 0
        
        try:
            while self.running:
                value = self.generate_single()
                callback(value)
                
                # Notify all registered callbacks
                for cb in self.callbacks:
                    try:
                        cb(value)
                    except:
                        pass
                
                generated += 1
                if count and generated >= count:
                    break
                    
                if self.delay > 0:
                    time.sleep(self.delay)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            
    def stop(self):
        """Stop generation"""
        self.running = False
        
    def get_statistics(self):
        """Get generation statistics"""
        if not self.last_values:
            return None
            
        return {
            'generated_count': self.generated_count,
            'last_values_count': len(self.last_values),
            'min_generated': min(self.last_values),
            'max_generated': max(self.last_values),
            'avg_generated': sum(self.last_values) / len(self.last_values)
        }
        
    def register_callback(self, callback):
        """Register a callback for generated values"""
        self.callbacks.append(callback)
        
    def notify_connector(self, message):
        """Send notification to connector if available"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect(('localhost', CONNECTOR_PORT))
            
            notification = {
                'source': 'random_pixel_generator',
                'type': 'notification',
                'message': message,
                'stats': self.get_statistics()
            }
            sock.send(json.dumps(notification).encode())
            response = sock.recv(1024).decode()
            sock.close()
            return json.loads(response)
        except:
            # Connector not available, continue standalone
            return None

# Global instance
_generator = PixelGenerator()

# Module-level functions for backward compatibility
def generate(min_val=0, max_val=255, delay=0, callback=print):
    """Module-level generate function"""
    _generator.generate(min_val, max_val, delay, callback)
    
def set_parameters(min_val=None, max_val=None, delay=None):
    """Set generator parameters"""
    _generator.set_parameters(min_val, max_val, delay)
    
def get_parameters():
    """Get current parameters"""
    return _generator.get_parameters()
    
def get_statistics():
    """Get generation statistics"""
    return _generator.get_statistics()

if __name__ == "__main__":
    a = sys.argv
    
    # Parse command line arguments
    min_val = int(a[1]) if len(a) > 1 else 0
    max_val = int(a[2]) if len(a) > 2 else 255
    delay = float(a[3]) if len(a) > 3 else 0
    count = int(a[4]) if len(a) > 4 else None
    
    # Notify connector we're starting
    _generator.notify_connector(f"Starting generation with min={min_val}, max={max_val}, delay={delay}")
    
    # Start generation
    try:
        _generator.generate(min_val, max_val, delay, count=count)
    except KeyboardInterrupt:
        print("\nGeneration stopped by user")
        
    # Show final statistics
    stats = get_statistics()
    if stats:
        print(f"\nFinal statistics: {json.dumps(stats, indent=2)}")
        
    # Notify connector we're done
    _generator.notify_connector(f"Generation completed: {stats}")