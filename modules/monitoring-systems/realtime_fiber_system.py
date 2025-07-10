# realtime_fiber_inspection.py
"""
Advanced Real-time Fiber Optic Inspection System
Features:
- Real-time video stream processing
- WebSocket communication for live updates
- Queue-based robust processing
- Edge device integration
- Automated quality control
- Performance monitoring
"""

import asyncio
import cv2
import numpy as np
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import websockets
import aiohttp
from dataclasses import dataclass, asdict
import redis
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import torch
import torchvision.transforms as transforms
from collections import deque
import firebase_admin
from firebase_admin import firestore, storage
from google.cloud import pubsub_v1
import grpc
from google.cloud import vision
import io
import base64
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Performance metrics
FRAMES_PROCESSED = Counter('fiber_frames_processed_total', 'Total frames processed')
ANOMALIES_DETECTED = Counter('fiber_anomalies_detected_total', 'Total anomalies detected')
PROCESSING_TIME = Histogram('fiber_processing_duration_seconds', 'Processing time per frame')
ACTIVE_STREAMS = Gauge('fiber_active_streams', 'Number of active inspection streams')
QUEUE_SIZE = Gauge('fiber_queue_size', 'Current processing queue size')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    """Data class for inspection results"""
    frame_id: str
    timestamp: datetime
    is_anomalous: bool
    anomaly_score: float
    severity: str
    defects: List[Dict]
    quality_score: float
    processing_time: float
    stream_id: str
    position: Optional[Dict] = None  # For fiber position tracking
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class StreamConfig:
    """Configuration for each inspection stream"""
    stream_id: str
    source: str  # URL, device ID, or file path
    fps: int = 30
    resolution: Tuple[int, int] = (1920, 1080)
    buffer_size: int = 10
    enable_recording: bool = True
    alert_threshold: float = 0.7
    preprocessing: Dict = None
    

class AdvancedFiberAnalyzer:
    """Enhanced analyzer with deep learning capabilities"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_torch_model(model_path)
        self.transform = self._get_transforms()
        
        # Feature extractor from original detection.py
        from detection import OmniFiberAnalyzer, OmniConfig
        self.traditional_analyzer = OmniFiberAnalyzer(OmniConfig())
        
    def _load_torch_model(self, model_path: str):
        """Load PyTorch model for real-time inference"""
        # For demo, using a pretrained ResNet modified for anomaly detection
        import torchvision.models as models
        
        model = models.resnet50(pretrained=True)
        # Modify for binary classification
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 2)  # Normal/Anomalous
        )
        
        # Load trained weights if available
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except:
            logger.warning("Could not load model weights, using pretrained")
            
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def analyze_frame_fast(self, frame: np.ndarray) -> Dict:
        """Fast analysis using deep learning model"""
        start_time = time.time()
        
        # Preprocess
        tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        # Inference
        outputs = self.model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        is_anomalous = probs[0, 1] > 0.5
        anomaly_score = float(probs[0, 1])
        
        # Quick defect detection using CV
        defects = self._detect_defects_fast(frame)
        
        processing_time = time.time() - start_time
        
        return {
            'is_anomalous': bool(is_anomalous),
            'anomaly_score': anomaly_score,
            'defects': defects,
            'processing_time': processing_time
        }
    
    def _detect_defects_fast(self, frame: np.ndarray) -> List[Dict]:
        """Fast defect detection using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        defects = []
        
        # Fast scratch detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 40:
                    defects.append({
                        'type': 'scratch',
                        'bbox': [min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)],
                        'confidence': min(length / 100, 1.0)
                    })
        
        # Fast blob detection
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)
        
        for kp in keypoints:
            if kp.size > 10:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                r = int(kp.size)
                defects.append({
                    'type': 'contamination',
                    'bbox': [x-r, y-r, 2*r, 2*r],
                    'confidence': min(kp.size / 50, 1.0)
                })
        
        return defects


class RealtimeProcessor:
    """Main real-time processing engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.streams: Dict[str, 'StreamProcessor'] = {}
        self.analyzer = AdvancedFiberAnalyzer(config.get('model_path', 'model.pth'))
        
        # Initialize connections
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Initialize Firestore
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        self.db = firestore.client()
        self.storage = storage.bucket()
        
        # Pub/Sub for distributed processing
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(
            config.get('project_id'), 
            config.get('topic_name', 'fiber-inspection')
        )
        
        # Processing queues
        self.frame_queue = asyncio.Queue(maxsize=1000)
        self.result_queue = asyncio.Queue(maxsize=1000)
        
        # Thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('threads', 8))
        self.process_pool = ProcessPoolExecutor(max_workers=config.get('processes', 4))
        
        # WebSocket connections
        self.websocket_clients = set()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
    async def start(self):
        """Start the real-time processing system"""
        logger.info("Starting Real-time Fiber Inspection System...")
        
        # Start Prometheus metrics server
        start_http_server(8000)
        
        # Start processing workers
        workers = [
            asyncio.create_task(self.frame_processor()),
            asyncio.create_task(self.result_processor()),
            asyncio.create_task(self.websocket_server()),
            asyncio.create_task(self.health_monitor()),
            asyncio.create_task(self.cache_manager()),
        ]
        
        # Start stream processors
        for stream_config in self.config.get('streams', []):
            await self.add_stream(StreamConfig(**stream_config))
        
        await asyncio.gather(*workers)
    
    async def add_stream(self, config: StreamConfig):
        """Add a new inspection stream"""
        if config.stream_id in self.streams:
            logger.warning(f"Stream {config.stream_id} already exists")
            return
            
        processor = StreamProcessor(config, self.frame_queue)
        self.streams[config.stream_id] = processor
        
        asyncio.create_task(processor.start())
        ACTIVE_STREAMS.inc()
        
        logger.info(f"Added stream: {config.stream_id}")
        
    async def remove_stream(self, stream_id: str):
        """Remove an inspection stream"""
        if stream_id in self.streams:
            await self.streams[stream_id].stop()
            del self.streams[stream_id]
            ACTIVE_STREAMS.dec()
            logger.info(f"Removed stream: {stream_id}")
    
    async def frame_processor(self):
        """Process frames from queue"""
        while True:
            try:
                # Get frame from queue
                frame_data = await self.frame_queue.get()
                QUEUE_SIZE.set(self.frame_queue.qsize())
                
                with PROCESSING_TIME.time():
                    # Analyze frame
                    result = await self._process_frame(frame_data)
                    
                    # Put result in queue
                    await self.result_queue.put(result)
                    
                FRAMES_PROCESSED.inc()
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                
    async def _process_frame(self, frame_data: Dict) -> InspectionResult:
        """Process a single frame"""
        frame = frame_data['frame']
        metadata = frame_data['metadata']
        
        # Run analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            self.thread_pool,
            self.analyzer.analyze_frame_fast,
            frame
        )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(analysis)
        
        # Determine severity
        severity = self._get_severity(analysis['anomaly_score'])
        
        # Create result
        result = InspectionResult(
            frame_id=metadata['frame_id'],
            timestamp=datetime.now(),
            is_anomalous=analysis['is_anomalous'],
            anomaly_score=analysis['anomaly_score'],
            severity=severity,
            defects=analysis['defects'],
            quality_score=quality_score,
            processing_time=analysis['processing_time'],
            stream_id=metadata['stream_id'],
            position=metadata.get('position')
        )
        
        # Track anomalies
        if result.is_anomalous:
            ANOMALIES_DETECTED.inc()
            
        return result
    
    def _calculate_quality_score(self, analysis: Dict) -> float:
        """Calculate overall quality score"""
        base_score = 100.0
        
        # Deduct for anomaly probability
        base_score -= analysis['anomaly_score'] * 50
        
        # Deduct for defects
        for defect in analysis['defects']:
            base_score -= defect['confidence'] * 10
            
        return max(0, min(100, base_score))
    
    def _get_severity(self, score: float) -> str:
        """Convert score to severity level"""
        if score < 0.3:
            return 'NORMAL'
        elif score < 0.5:
            return 'LOW'
        elif score < 0.7:
            return 'MEDIUM'
        elif score < 0.9:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    async def result_processor(self):
        """Process and distribute results"""
        while True:
            try:
                result = await self.result_queue.get()
                
                # Store in cache
                await self._cache_result(result)
                
                # Send to WebSocket clients
                await self._broadcast_result(result)
                
                # Store in Firestore if anomalous
                if result.is_anomalous:
                    await self._store_anomaly(result)
                    
                # Publish to Pub/Sub for downstream processing
                await self._publish_result(result)
                
                # Check for alerts
                await self._check_alerts(result)
                
            except Exception as e:
                logger.error(f"Result processing error: {e}")
    
    async def _cache_result(self, result: InspectionResult):
        """Cache result in Redis"""
        key = f"result:{result.stream_id}:{result.frame_id}"
        self.redis_client.setex(
            key, 
            300,  # 5 minute TTL
            json.dumps(result.to_dict())
        )
        
        # Update stream statistics
        stats_key = f"stats:{result.stream_id}"
        pipe = self.redis_client.pipeline()
        pipe.hincrby(stats_key, 'total_frames', 1)
        if result.is_anomalous:
            pipe.hincrby(stats_key, 'anomalies', 1)
        pipe.hset(stats_key, 'last_update', datetime.now().isoformat())
        pipe.hset(stats_key, 'avg_quality', result.quality_score)
        pipe.expire(stats_key, 3600)  # 1 hour TTL
        pipe.execute()
    
    async def _broadcast_result(self, result: InspectionResult):
        """Broadcast result to WebSocket clients"""
        if self.websocket_clients:
            message = json.dumps({
                'type': 'inspection_result',
                'data': result.to_dict()
            })
            
            # Send to all connected clients
            await asyncio.gather(
                *[client.send(message) for client in self.websocket_clients],
                return_exceptions=True
            )
    
    async def _store_anomaly(self, result: InspectionResult):
        """Store anomaly in Firestore"""
        try:
            # Store result
            doc_ref = self.db.collection('anomalies').document()
            doc_ref.set({
                **result.to_dict(),
                'created_at': firestore.SERVER_TIMESTAMP
            })
            
            # Update stream anomaly count
            stream_ref = self.db.collection('streams').document(result.stream_id)
            stream_ref.update({
                'anomaly_count': firestore.Increment(1),
                'last_anomaly': firestore.SERVER_TIMESTAMP
            })
            
        except Exception as e:
            logger.error(f"Failed to store anomaly: {e}")
    
    async def _publish_result(self, result: InspectionResult):
        """Publish result to Pub/Sub"""
        try:
            message = json.dumps(result.to_dict()).encode('utf-8')
            future = self.publisher.publish(self.topic_path, message)
            # Don't wait for publish to complete
        except Exception as e:
            logger.error(f"Failed to publish result: {e}")
    
    async def _check_alerts(self, result: InspectionResult):
        """Check if alerts need to be sent"""
        if result.severity in ['HIGH', 'CRITICAL']:
            await self._send_alert(result)
    
    async def _send_alert(self, result: InspectionResult):
        """Send alert for critical anomalies"""
        alert = {
            'type': 'FIBER_ANOMALY_ALERT',
            'severity': result.severity,
            'stream_id': result.stream_id,
            'timestamp': result.timestamp.isoformat(),
            'anomaly_score': result.anomaly_score,
            'defect_count': len(result.defects)
        }
        
        # Send to alert service
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    self.config.get('alert_webhook'),
                    json=alert,
                    timeout=aiohttp.ClientTimeout(total=5)
                )
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
    
    async def websocket_server(self):
        """WebSocket server for real-time updates"""
        async def handler(websocket, path):
            self.websocket_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.remove(websocket)
        
        server = await websockets.serve(
            handler,
            self.config.get('ws_host', 'localhost'),
            self.config.get('ws_port', 8765)
        )
        
        logger.info(f"WebSocket server started on {self.config.get('ws_host')}:{self.config.get('ws_port')}")
        await server.wait_closed()
    
    async def health_monitor(self):
        """Monitor system health"""
        while True:
            try:
                health = {
                    'timestamp': datetime.now().isoformat(),
                    'active_streams': len(self.streams),
                    'queue_size': self.frame_queue.qsize(),
                    'result_queue_size': self.result_queue.qsize(),
                    'websocket_clients': len(self.websocket_clients),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                }
                
                # Store health metrics
                self.redis_client.setex(
                    'system:health',
                    60,
                    json.dumps(health)
                )
                
                # Check for issues
                if health['cpu_percent'] > 90:
                    logger.warning(f"High CPU usage: {health['cpu_percent']}%")
                    
                if health['memory_percent'] > 90:
                    logger.warning(f"High memory usage: {health['memory_percent']}%")
                    
                if self.frame_queue.qsize() > 900:
                    logger.warning(f"Frame queue almost full: {self.frame_queue.qsize()}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def cache_manager(self):
        """Manage cache and cleanup"""
        while True:
            try:
                # Clean old results
                pattern = "result:*"
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
                    # Check TTL and clean if needed
                    if cursor == 0:
                        break
                
                # Archive old anomalies to cold storage
                cutoff = datetime.now() - timedelta(hours=24)
                old_anomalies = self.db.collection('anomalies')\
                    .where('timestamp', '<', cutoff)\
                    .limit(100)\
                    .get()
                
                for doc in old_anomalies:
                    # Archive to Cloud Storage
                    data = doc.to_dict()
                    blob_name = f"archive/{doc.id}.json"
                    blob = self.storage.blob(blob_name)
                    blob.upload_from_string(json.dumps(data))
                    
                    # Delete from Firestore
                    doc.reference.delete()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Cache manager error: {e}")
                await asyncio.sleep(3600)


class StreamProcessor:
    """Process individual video streams"""
    
    def __init__(self, config: StreamConfig, frame_queue: asyncio.Queue):
        self.config = config
        self.frame_queue = frame_queue
        self.running = False
        self.cap = None
        self.frame_count = 0
        self.recording = None
        self.position_tracker = PositionTracker()
        
    async def start(self):
        """Start processing stream"""
        self.running = True
        
        # Initialize video capture
        if self.config.source.startswith('rtsp://') or self.config.source.startswith('http://'):
            self.cap = cv2.VideoCapture(self.config.source)
        elif self.config.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.config.source))
        else:
            self.cap = cv2.VideoCapture(self.config.source)
            
        # Set capture properties
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        
        # Initialize recording if enabled
        if self.config.enable_recording:
            self._init_recording()
        
        # Start processing loop
        asyncio.create_task(self._process_loop())
        
        logger.info(f"Started stream processor for {self.config.stream_id}")
    
    def _init_recording(self):
        """Initialize video recording"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = f"recordings/{self.config.stream_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.recording = cv2.VideoWriter(
            filename,
            fourcc,
            self.config.fps,
            self.config.resolution
        )
    
    async def _process_loop(self):
        """Main processing loop"""
        skip_frames = 0
        frame_buffer = deque(maxlen=5)  # Keep last 5 frames for analysis
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame from {self.config.stream_id}")
                await asyncio.sleep(0.1)
                continue
                
            self.frame_count += 1
            
            # Apply preprocessing if configured
            if self.config.preprocessing:
                frame = self._preprocess_frame(frame, self.config.preprocessing)
            
            # Track fiber position
            position = self.position_tracker.track(frame)
            
            # Skip frames for performance (process every Nth frame)
            if skip_frames > 0:
                skip_frames -= 1
                continue
                
            # Add to buffer for motion detection
            frame_buffer.append(frame)
            
            # Check for motion or significant change
            if len(frame_buffer) >= 2:
                if not self._has_significant_change(frame_buffer[-2], frame_buffer[-1]):
                    skip_frames = 5  # Skip next 5 frames
                    continue
            
            # Prepare frame data
            frame_data = {
                'frame': frame,
                'metadata': {
                    'stream_id': self.config.stream_id,
                    'frame_id': f"{self.config.stream_id}_{self.frame_count}",
                    'frame_number': self.frame_count,
                    'timestamp': datetime.now().isoformat(),
                    'position': position
                }
            }
            
            # Add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame_data)
            except asyncio.QueueFull:
                logger.warning(f"Frame queue full for {self.config.stream_id}")
                skip_frames = 10  # Skip more frames if queue is full
            
            # Record if enabled
            if self.recording:
                self.recording.write(frame)
            
            # Control frame rate
            await asyncio.sleep(1.0 / self.config.fps)
    
    def _preprocess_frame(self, frame: np.ndarray, config: Dict) -> np.ndarray:
        """Apply preprocessing to frame"""
        # Resize if needed
        if 'resize' in config:
            frame = cv2.resize(frame, tuple(config['resize']))
            
        # Denoise
        if config.get('denoise', False):
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
        # Enhance contrast
        if config.get('enhance_contrast', False):
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            
        # Sharpen
        if config.get('sharpen', False):
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            frame = cv2.filter2D(frame, -1, kernel)
            
        return frame
    
    def _has_significant_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Check if there's significant change between frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Count changed pixels
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        
        # Return true if more than 1% of pixels changed
        return (changed_pixels / total_pixels) > 0.01
    
    async def stop(self):
        """Stop processing stream"""
        self.running = False
        
        if self.cap:
            self.cap.release()
            
        if self.recording:
            self.recording.release()
            
        logger.info(f"Stopped stream processor for {self.config.stream_id}")


class PositionTracker:
    """Track fiber position in frame"""
    
    def __init__(self):
        self.last_position = None
        self.tracker = None
        
    def track(self, frame: np.ndarray) -> Optional[Dict]:
        """Track fiber position in frame"""
        # For demo, using simple center detection
        # In production, use more sophisticated tracking
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find fiber center using HoughCircles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=200
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Take the first detected circle
            x, y, r = circles[0, 0]
            
            position = {
                'x': int(x),
                'y': int(y),
                'radius': int(r),
                'confidence': 0.8  # Simplified confidence
            }
            
            self.last_position = position
            return position
            
        return self.last_position


class PerformanceMonitor:
    """Monitor and optimize performance"""
    
    def __init__(self):
        self.metrics = {
            'fps': deque(maxlen=100),
            'processing_time': deque(maxlen=100),
            'queue_size': deque(maxlen=100),
            'memory_usage': deque(maxlen=100)
        }
        
    def update(self, metric: str, value: float):
        """Update performance metric"""
        if metric in self.metrics:
            self.metrics[metric].append(value)
            
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return stats


async def main():
    """Main entry point"""
    config = {
        'model_path': 'fiber_model.pth',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'project_id': 'your-project-id',
        'topic_name': 'fiber-inspection',
        'alert_webhook': 'https://your-alert-service.com/webhook',
        'ws_host': '0.0.0.0',
        'ws_port': 8765,
        'threads': 8,
        'processes': 4,
        'streams': [
            {
                'stream_id': 'camera_1',
                'source': '0',  # Webcam
                'fps': 30,
                'resolution': [1920, 1080],
                'preprocessing': {
                    'denoise': True,
                    'enhance_contrast': True
                }
            },
            {
                'stream_id': 'inspection_line_1',
                'source': 'rtsp://192.168.1.100:554/stream',
                'fps': 25,
                'resolution': [1280, 720],
                'preprocessing': {
                    'sharpen': True
                }
            }
        ]
    }
    
    processor = RealtimeProcessor(config)
    await processor.start()


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('recordings', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the system
    asyncio.run(main())
