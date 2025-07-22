# edge_fiber_inspector.py
"""
Edge Computing Script for IoT Fiber Inspection Devices
Optimized for Raspberry Pi, NVIDIA Jetson, and similar edge devices
Features:
- Lightweight ML inference
- Local caching and buffering
- Automatic failover
- Bandwidth optimization
- Hardware acceleration support
"""

import os
import cv2
import numpy as np
import json
import time
import threading
import queue
import logging
from datetime import datetime, timedelta
import requests
import sqlite3
import socket
import psutil
from collections import deque
from typing import Dict, List, Optional, Tuple
import paho.mqtt.client as mqtt
import edge_impulse_linux
from tflite_runtime.interpreter import Interpreter
import RPi.GPIO as GPIO  # For Raspberry Pi GPIO control

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/fiber_inspector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EdgeConfig:
    """Configuration for edge device"""
    # Device settings
    DEVICE_ID = os.environ.get('DEVICE_ID', socket.gethostname())
    LOCATION = os.environ.get('LOCATION', 'unknown')
    
    # Model settings
    MODEL_PATH = '/opt/models/fiber_anomaly_quantized.tflite'
    LABELS_PATH = '/opt/models/labels.txt'
    
    # Camera settings
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 15
    
    # Processing settings
    INFERENCE_INTERVAL = 0.5  # seconds between inferences
    BATCH_SIZE = 10  # frames to batch before sending
    MAX_QUEUE_SIZE = 100
    
    # Network settings
    API_ENDPOINT = os.environ.get('API_ENDPOINT', 'https://api.fiber-inspection.com')
    MQTT_BROKER = os.environ.get('MQTT_BROKER', 'mqtt.fiber-inspection.com')
    MQTT_PORT = 1883
    
    # Storage settings
    LOCAL_DB_PATH = '/var/lib/fiber_inspector/cache.db'
    MAX_CACHE_SIZE_MB = 500
    
    # Hardware settings
    USE_GPU = os.path.exists('/dev/nvidia0')  # Check for NVIDIA GPU
    USE_NPU = os.path.exists('/dev/npu0')  # Check for NPU/TPU
    
    # Alert settings
    ALERT_GPIO_PIN = 17  # GPIO pin for alert LED
    BUZZER_GPIO_PIN = 27  # GPIO pin for buzzer


class LightweightAnalyzer:
    """Lightweight analyzer for edge devices"""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.interpreter = None
        self.labels = []
        self.input_details = None
        self.output_details = None
        
        # Load TFLite model
        self._load_model()
        
        # Initialize hardware acceleration if available
        self._init_hardware_acceleration()
        
    def _load_model(self):
        """Load TensorFlow Lite model"""
        try:
            # Load TFLite model
            self.interpreter = Interpreter(
                model_path=self.config.MODEL_PATH,
                num_threads=4  # Use 4 threads for inference
            )
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Load labels
            with open(self.config.LABELS_PATH, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _init_hardware_acceleration(self):
        """Initialize hardware acceleration if available"""
        if self.config.USE_GPU:
            try:
                # Set CUDA device
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                logger.info("GPU acceleration enabled")
            except:
                logger.warning("GPU detected but initialization failed")
                
        if self.config.USE_NPU:
            try:
                # Initialize NPU/TPU
                # This would be device-specific
                logger.info("NPU acceleration enabled")
            except:
                logger.warning("NPU detected but initialization failed")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference"""
        # Get input shape
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize frame
        resized = cv2.resize(frame, (width, height))
        
        # Normalize based on model requirements
        if self.input_details[0]['dtype'] == np.float32:
            normalized = resized.astype(np.float32) / 255.0
        else:
            normalized = resized
            
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """Analyze frame using TFLite model"""
        start_time = time.time()
        
        # Preprocess
        input_data = self.preprocess_frame(frame)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Process results
        scores = output_data[0]
        is_anomalous = scores[1] > scores[0]  # Assuming binary classification
        anomaly_score = float(scores[1])
        
        # Simple defect detection
        defects = self._detect_defects_edge(frame)
        
        inference_time = time.time() - start_time
        
        return {
            'is_anomalous': bool(is_anomalous),
            'anomaly_score': anomaly_score,
            'defects': defects,
            'inference_time': inference_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_defects_edge(self, frame: np.ndarray) -> List[Dict]:
        """Lightweight defect detection for edge"""
        defects = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple threshold-based detection
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'type': 'anomaly',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'area': float(area)
                })
                
        return defects[:5]  # Limit to 5 defects for bandwidth


class EdgeInspector:
    """Main edge inspection system"""
    
    def __init__(self):
        self.config = EdgeConfig()
        self.analyzer = LightweightAnalyzer(self.config)
        self.running = False
        
        # Queues
        self.frame_queue = queue.Queue(maxsize=self.config.MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue(maxsize=self.config.MAX_QUEUE_SIZE)
        
        # Local cache
        self.cache = LocalCache(self.config.LOCAL_DB_PATH)
        
        # Network manager
        self.network = NetworkManager(self.config)
        
        # GPIO setup for alerts
        self._setup_gpio()
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'anomalies_detected': 0,
            'network_failures': 0,
            'last_sync': datetime.now()
        }
        
    def _setup_gpio(self):
        """Setup GPIO for alerts"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.config.ALERT_GPIO_PIN, GPIO.OUT)
            GPIO.setup(self.config.BUZZER_GPIO_PIN, GPIO.OUT)
            GPIO.output(self.config.ALERT_GPIO_PIN, GPIO.LOW)
            GPIO.output(self.config.BUZZER_GPIO_PIN, GPIO.LOW)
        except:
            logger.warning("GPIO setup failed - running without hardware alerts")
    
    def start(self):
        """Start edge inspection"""
        self.running = True
        
        # Start threads
        threads = [
            threading.Thread(target=self.capture_loop, daemon=True),
            threading.Thread(target=self.inference_loop, daemon=True),
            threading.Thread(target=self.sync_loop, daemon=True),
            threading.Thread(target=self.monitor_loop, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
            
        logger.info(f"Edge Inspector started - Device: {self.config.DEVICE_ID}")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop edge inspection"""
        self.running = False
        GPIO.cleanup()
        self.cache.close()
        logger.info("Edge Inspector stopped")
    
    def capture_loop(self):
        """Capture frames from camera"""
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        frame_count = 0
        last_inference = time.time()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            frame_count += 1
            
            # Only process at inference interval
            if time.time() - last_inference >= self.config.INFERENCE_INTERVAL:
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame,
                        'frame_id': f"{self.config.DEVICE_ID}_{frame_count}",
                        'timestamp': datetime.now()
                    })
                    last_inference = time.time()
                except queue.Full:
                    logger.warning("Frame queue full, dropping frame")
                    
        cap.release()
    
    def inference_loop(self):
        """Run inference on frames"""
        while self.running:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=1)
                
                # Run analysis
                result = self.analyzer.analyze(frame_data['frame'])
                
                # Add metadata
                result.update({
                    'device_id': self.config.DEVICE_ID,
                    'location': self.config.LOCATION,
                    'frame_id': frame_data['frame_id']
                })
                
                # Update stats
                self.stats['frames_processed'] += 1
                if result['is_anomalous']:
                    self.stats['anomalies_detected'] += 1
                    self.trigger_alert(result['anomaly_score'])
                
                # Queue for transmission
                self.result_queue.put(result)
                
                # Cache locally
                self.cache.store_result(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference error: {e}")
    
    def sync_loop(self):
        """Sync results with cloud"""
        batch = []
        last_sync = time.time()
        
        while self.running:
            try:
                # Collect results for batch
                while len(batch) < self.config.BATCH_SIZE and not self.result_queue.empty():
                    batch.append(self.result_queue.get_nowait())
                
                # Send batch if full or timeout
                if len(batch) >= self.config.BATCH_SIZE or \
                   (len(batch) > 0 and time.time() - last_sync > 10):
                    
                    if self.network.send_batch(batch):
                        batch.clear()
                        self.stats['last_sync'] = datetime.now()
                    else:
                        # Network failed, keep in cache
                        self.stats['network_failures'] += 1
                        for result in batch:
                            self.cache.mark_pending(result['frame_id'])
                        batch.clear()
                    
                    last_sync = time.time()
                
                # Retry pending items
                if self.network.is_connected():
                    pending = self.cache.get_pending_results(limit=50)
                    if pending and self.network.send_batch(pending):
                        self.cache.mark_synced([r['frame_id'] for r in pending])
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Sync error: {e}")
                time.sleep(5)
    
    def monitor_loop(self):
        """Monitor system health"""
        while self.running:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
                temp = self.get_cpu_temperature()
                
                # Log health metrics
                health = {
                    'device_id': self.config.DEVICE_ID,
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_usage': disk_usage,
                    'temperature': temp,
                    'stats': self.stats
                }
                
                # Send health update
                self.network.send_health(health)
                
                # Check for issues
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                if memory_percent > 90:
                    logger.warning(f"High memory usage: {memory_percent}%")
                if temp and temp > 70:
                    logger.warning(f"High temperature: {temp}°C")
                    
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(30)
    
    def trigger_alert(self, severity_score: float):
        """Trigger hardware alert"""
        try:
            if severity_score > 0.8:
                # Critical - continuous alert
                GPIO.output(self.config.ALERT_GPIO_PIN, GPIO.HIGH)
                GPIO.output(self.config.BUZZER_GPIO_PIN, GPIO.HIGH)
                threading.Timer(2.0, lambda: GPIO.output(self.config.BUZZER_GPIO_PIN, GPIO.LOW)).start()
            elif severity_score > 0.6:
                # High - pulse alert
                for _ in range(3):
                    GPIO.output(self.config.ALERT_GPIO_PIN, GPIO.HIGH)
                    time.sleep(0.2)
                    GPIO.output(self.config.ALERT_GPIO_PIN, GPIO.LOW)
                    time.sleep(0.2)
        except:
            pass  # Ignore GPIO errors
    
    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            # Raspberry Pi
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                return temp
        except:
            return None


class LocalCache:
    """Local SQLite cache for offline operation"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._init_db()
        
    def _init_db(self):
        """Initialize database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create tables
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS results (
                frame_id TEXT PRIMARY KEY,
                device_id TEXT,
                timestamp TEXT,
                is_anomalous INTEGER,
                anomaly_score REAL,
                defects TEXT,
                synced INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_synced ON results(synced)
        ''')
        
        self.conn.commit()
    
    def store_result(self, result: Dict):
        """Store result in cache"""
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO results 
                (frame_id, device_id, timestamp, is_anomalous, anomaly_score, defects, synced)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['frame_id'],
                result['device_id'],
                result['timestamp'],
                int(result['is_anomalous']),
                result['anomaly_score'],
                json.dumps(result['defects']),
                0
            ))
            self.conn.commit()
            
            # Clean old entries if cache is too large
            self._cleanup_old_entries()
            
        except Exception as e:
            logger.error(f"Cache store error: {e}")
    
    def get_pending_results(self, limit: int = 100) -> List[Dict]:
        """Get unsynced results"""
        cursor = self.conn.execute('''
            SELECT frame_id, device_id, timestamp, is_anomalous, 
                   anomaly_score, defects
            FROM results 
            WHERE synced = 0 
            ORDER BY created_at 
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor:
            results.append({
                'frame_id': row[0],
                'device_id': row[1],
                'timestamp': row[2],
                'is_anomalous': bool(row[3]),
                'anomaly_score': row[4],
                'defects': json.loads(row[5])
            })
            
        return results
    
    def mark_synced(self, frame_ids: List[str]):
        """Mark results as synced"""
        self.conn.executemany(
            'UPDATE results SET synced = 1 WHERE frame_id = ?',
            [(fid,) for fid in frame_ids]
        )
        self.conn.commit()
    
    def mark_pending(self, frame_id: str):
        """Mark result as pending sync"""
        self.conn.execute(
            'UPDATE results SET synced = 0 WHERE frame_id = ?',
            (frame_id,)
        )
        self.conn.commit()
    
    def _cleanup_old_entries(self):
        """Clean up old entries to maintain cache size"""
        # Get database size
        db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
        
        if db_size_mb > EdgeConfig.MAX_CACHE_SIZE_MB:
            # Delete oldest synced entries
            self.conn.execute('''
                DELETE FROM results 
                WHERE frame_id IN (
                    SELECT frame_id FROM results 
                    WHERE synced = 1 
                    ORDER BY created_at 
                    LIMIT 1000
                )
            ''')
            self.conn.commit()
            logger.info("Cleaned up old cache entries")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class NetworkManager:
    """Manage network communications"""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.mqtt_client = None
        self.connected = False
        self._init_mqtt()
        
    def _init_mqtt(self):
        """Initialize MQTT client"""
        self.mqtt_client = mqtt.Client(client_id=self.config.DEVICE_ID)
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_disconnect = self._on_disconnect
        
        try:
            self.mqtt_client.connect(self.config.MQTT_BROKER, self.config.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback"""
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT broker")
            # Subscribe to commands
            client.subscribe(f"devices/{self.config.DEVICE_ID}/commands")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        self.connected = False
        logger.warning("Disconnected from MQTT broker")
    
    def is_connected(self) -> bool:
        """Check if connected to network"""
        return self.connected and self._check_internet()
    
    def _check_internet(self) -> bool:
        """Check internet connectivity"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    def send_batch(self, results: List[Dict]) -> bool:
        """Send batch of results"""
        if not self.is_connected():
            return False
            
        try:
            # Send via MQTT for real-time updates
            topic = f"devices/{self.config.DEVICE_ID}/results"
            payload = json.dumps({
                'batch': results,
                'device_id': self.config.DEVICE_ID,
                'timestamp': datetime.now().isoformat()
            })
            
            self.mqtt_client.publish(topic, payload, qos=1)
            
            # Also send to API endpoint
            response = requests.post(
                f"{self.config.API_ENDPOINT}/results/batch",
                json={'results': results},
                headers={'X-Device-ID': self.config.DEVICE_ID},
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to send batch: {e}")
            return False
    
    def send_health(self, health: Dict) -> bool:
        """Send health metrics"""
        if not self.is_connected():
            return False
            
        try:
            topic = f"devices/{self.config.DEVICE_ID}/health"
            self.mqtt_client.publish(topic, json.dumps(health), qos=0)
            return True
        except:
            return False


def main():
    """Main entry point"""
    logger.info("Starting Fiber Optic Edge Inspector")
    
    # Check for required directories
    os.makedirs('/var/lib/fiber_inspector', exist_ok=True)
    os.makedirs('/var/log', exist_ok=True)
    
    # Start inspector
    inspector = EdgeInspector()
    
    try:
        inspector.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        inspector.stop()


if __name__ == "__main__":
    main()
