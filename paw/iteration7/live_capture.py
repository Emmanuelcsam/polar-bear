import json
import time
import os
import numpy as np
from PIL import Image
import threading

# Try to import camera libraries
camera_available = False
camera_type = None

try:
    import cv2
    camera_available = True
    camera_type = 'opencv'
    print("[LIVE] OpenCV camera support available")
except ImportError:
    print("[LIVE] OpenCV not available, trying alternatives...")
    
    try:
        # Try PIL ImageGrab for screenshots as camera alternative
        from PIL import ImageGrab
        camera_available = True
        camera_type = 'screenshot'
        print("[LIVE] Using screenshot capture as camera alternative")
    except:
        print("[LIVE] No camera support available, will use simulation")

class LiveCapture:
    def __init__(self, source=0, fps=10):
        self.source = source
        self.fps = fps
        self.running = False
        self.frame_count = 0
        self.capture_stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'start_time': 0,
            'errors': 0
        }
        
        # Initialize capture
        self.camera = None
        if camera_available and camera_type == 'opencv':
            self.camera = cv2.VideoCapture(source)
            if not self.camera.isOpened():
                print("[LIVE] Camera not found, using simulation mode")
                self.camera = None
    
    def capture_frame(self):
        """Capture a single frame from camera or generate simulated data"""
        
        if camera_available and camera_type == 'opencv' and self.camera:
            # Real camera capture
            ret, frame = self.camera.read()
            if ret:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return gray
            else:
                self.capture_stats['errors'] += 1
                return None
                
        elif camera_available and camera_type == 'screenshot':
            # Screenshot capture (region of screen)
            try:
                # Capture small region of screen
                screenshot = ImageGrab.grab(bbox=(0, 0, 100, 100))
                gray = screenshot.convert('L')
                return np.array(gray)
            except:
                self.capture_stats['errors'] += 1
                return None
                
        else:
            # Simulation mode - generate dynamic test pattern
            return self.generate_simulated_frame()
    
    def generate_simulated_frame(self):
        """Generate simulated camera data with dynamic patterns"""
        size = (100, 100)
        frame = np.zeros(size, dtype=np.uint8)
        
        # Create time-varying patterns
        t = time.time()
        
        # Moving gradient
        for y in range(size[0]):
            for x in range(size[1]):
                # Circular wave pattern
                cx, cy = size[0]//2, size[1]//2
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                wave = int(128 + 127 * np.sin(dist/10 - t*2))
                
                # Add noise
                noise = np.random.randint(-20, 20)
                
                frame[y, x] = np.clip(wave + noise, 0, 255)
        
        # Add some random bright spots (simulated features)
        num_spots = np.random.randint(3, 8)
        for _ in range(num_spots):
            x, y = np.random.randint(10, 90, 2)
            radius = np.random.randint(2, 5)
            cv2.circle(frame, (x, y), radius, 255, -1) if camera_type == 'opencv' else None
            
            # Manual circle drawing if no OpenCV
            if camera_type != 'opencv':
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        if dx*dx + dy*dy <= radius*radius:
                            if 0 <= y+dy < size[0] and 0 <= x+dx < size[1]:
                                frame[y+dy, x+dx] = 255
        
        return frame
    
    def process_frame(self, frame):
        """Process frame and save to JSON for other modules"""
        
        # Convert to pixel list
        pixels = frame.flatten().tolist()
        
        # Calculate frame statistics
        stats = {
            'mean': float(np.mean(frame)),
            'std': float(np.std(frame)),
            'min': int(np.min(frame)),
            'max': int(np.max(frame)),
            'timestamp': time.time()
        }
        
        # Save as current frame data
        frame_data = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'pixels': pixels,
            'size': list(frame.shape),
            'stats': stats,
            'source': 'camera' if self.camera else 'simulation'
        }
        
        # Save to pixel_data.json (compatible with existing system)
        with open('pixel_data.json', 'w') as f:
            json.dump(frame_data, f)
        
        # Also save to live-specific file
        with open('live_frame.json', 'w') as f:
            json.dump(frame_data, f)
        
        # Save frame as image
        img = Image.fromarray(frame)
        img.save('live_current.jpg')
        
        self.capture_stats['frames_processed'] += 1
        
        # Trigger real-time analysis if significant change
        if hasattr(self, 'last_frame_mean'):
            change = abs(stats['mean'] - self.last_frame_mean)
            if change > 10:
                print(f"[LIVE] Significant change detected: {change:.1f}")
                self.trigger_analysis()
        
        self.last_frame_mean = stats['mean']
        
        return stats
    
    def trigger_analysis(self):
        """Trigger analysis modules for significant changes"""
        
        # Run pattern recognition in background
        if os.path.exists('pattern_recognizer.py'):
            thread = threading.Thread(
                target=lambda: os.system(f'{os.sys.executable} pattern_recognizer.py')
            )
            thread.daemon = True
            thread.start()
    
    def display_live_stats(self, stats):
        """Display live statistics"""
        elapsed = time.time() - self.capture_stats['start_time']
        actual_fps = self.capture_stats['frames_captured'] / elapsed if elapsed > 0 else 0
        
        print(f"\r[LIVE] Frame {self.frame_count} | "
              f"FPS: {actual_fps:.1f} | "
              f"Mean: {stats['mean']:.1f} | "
              f"Std: {stats['std']:.1f} | "
              f"Range: [{stats['min']}-{stats['max']}]", end='')
    
    def save_video_buffer(self, frames, filename='live_buffer.json'):
        """Save recent frames as a buffer"""
        
        buffer_data = {
            'frame_count': len(frames),
            'timestamps': [f['timestamp'] for f in frames],
            'stats': [f['stats'] for f in frames],
            'duration': frames[-1]['timestamp'] - frames[0]['timestamp'] if frames else 0
        }
        
        with open(filename, 'w') as f:
            json.dump(buffer_data, f)
    
    def start_capture(self, duration=None):
        """Start live capture"""
        self.running = True
        self.capture_stats['start_time'] = time.time()
        
        print(f"[LIVE] Starting live capture at {self.fps} FPS")
        print(f"[LIVE] Mode: {camera_type if camera_available else 'simulation'}")
        print("[LIVE] Press Ctrl+C to stop")
        
        frame_buffer = []
        frame_interval = 1.0 / self.fps
        next_frame_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                if current_time >= next_frame_time:
                    # Capture frame
                    frame = self.capture_frame()
                    
                    if frame is not None:
                        self.frame_count += 1
                        self.capture_stats['frames_captured'] += 1
                        
                        # Process frame
                        stats = self.process_frame(frame)
                        
                        # Store in buffer
                        frame_info = {
                            'frame_number': self.frame_count,
                            'timestamp': current_time,
                            'stats': stats
                        }
                        frame_buffer.append(frame_info)
                        
                        # Keep only recent frames
                        if len(frame_buffer) > 100:
                            frame_buffer.pop(0)
                        
                        # Display stats
                        self.display_live_stats(stats)
                        
                        # Save buffer periodically
                        if self.frame_count % 50 == 0:
                            self.save_video_buffer(frame_buffer)
                        
                        # Check duration
                        if duration and (current_time - self.capture_stats['start_time']) > duration:
                            break
                    
                    next_frame_time += frame_interval
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n[LIVE] Capture interrupted")
        
        finally:
            self.stop_capture()
    
    def stop_capture(self):
        """Stop capture and cleanup"""
        self.running = False
        
        if self.camera:
            self.camera.release()
        
        # Save final statistics
        elapsed = time.time() - self.capture_stats['start_time']
        
        final_stats = {
            'total_frames': self.capture_stats['frames_captured'],
            'processed_frames': self.capture_stats['frames_processed'],
            'errors': self.capture_stats['errors'],
            'duration': elapsed,
            'average_fps': self.capture_stats['frames_captured'] / elapsed if elapsed > 0 else 0,
            'mode': camera_type if camera_available else 'simulation'
        }
        
        with open('live_capture_stats.json', 'w') as f:
            json.dump(final_stats, f)
        
        print(f"\n[LIVE] Capture stopped")
        print(f"[LIVE] Captured {self.capture_stats['frames_captured']} frames")
        print(f"[LIVE] Average FPS: {final_stats['average_fps']:.1f}")
        print("[LIVE] Stats saved to live_capture_stats.json")

def monitor_live_feed():
    """Monitor live feed and trigger actions"""
    print("[LIVE] Starting live feed monitor...")
    
    last_processed = 0
    pattern_threshold = 50  # Pixel variance threshold
    
    while True:
        try:
            if os.path.exists('live_frame.json'):
                with open('live_frame.json', 'r') as f:
                    frame_data = json.load(f)
                
                # Check if new frame
                if frame_data['timestamp'] > last_processed:
                    last_processed = frame_data['timestamp']
                    
                    # Check for interesting patterns
                    stats = frame_data['stats']
                    
                    if stats['std'] > pattern_threshold:
                        print(f"\n[LIVE] High variance detected: {stats['std']:.1f}")
                        
                        # Trigger pattern recognition
                        if os.path.exists('pattern_recognizer.py'):
                            os.system(f'{os.sys.executable} pattern_recognizer.py')
                    
                    # Check frame rate
                    if 'frame_number' in frame_data and frame_data['frame_number'] % 100 == 0:
                        print(f"\n[LIVE] Milestone: {frame_data['frame_number']} frames processed")
            
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            break
        except:
            pass
    
    print("\n[LIVE] Monitor stopped")

if __name__ == "__main__":
    import sys
    
    # Parse simple arguments
    fps = 10
    duration = None
    mode = 'capture'
    
    for arg in sys.argv[1:]:
        if arg.startswith('fps='):
            fps = int(arg.split('=')[1])
        elif arg.startswith('duration='):
            duration = int(arg.split('=')[1])
        elif arg == 'monitor':
            mode = 'monitor'
    
    if mode == 'monitor':
        monitor_live_feed()
    else:
        # Start capture
        capture = LiveCapture(fps=fps)
        capture.start_capture(duration=duration)